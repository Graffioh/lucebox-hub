// pflash_drafter_ipc_daemon.cpp - PFlash drafter IPC daemon body.

#include "pflash_drafter_ipc.h"

#include "dflash27b.h"
#include "dflash_draft_ipc.h"
#include "qwen3/qwen3_drafter.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace dflash::common {

#if !defined(_WIN32)
namespace {

constexpr size_t PFLASH_CAPTURE_MAX_TOKENS = 140000;

bool resolve_capture_directory(
        const std::string & raw,
        std::filesystem::path & out,
        std::string & error) {
    std::error_code ec;
    const std::filesystem::path requested(raw);
    out = std::filesystem::canonical(requested, ec);
    if (ec || !std::filesystem::is_directory(out, ec)) {
        error = "capture request directory does not exist";
        return false;
    }
    if (out != requested.lexically_normal()) {
        error = "capture request directory must not contain symlinks";
        return false;
    }
    return true;
}

bool write_immutable_capture(
        const std::filesystem::path & path,
        const std::vector<uint16_t> & values,
        std::string & error) {
    const std::filesystem::path partial = path.string() + ".partial";
    const int fd = ::open(
        partial.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        error = "open capture partial failed: " + std::string(std::strerror(errno));
        return false;
    }
    bool ok = write_exact_fd(
        fd, values.data(), values.size() * sizeof(uint16_t));
    if (ok && ::fsync(fd) != 0) ok = false;
    if (::close(fd) != 0) ok = false;
    if (!ok) {
        error = "write capture partial failed";
        std::filesystem::remove(partial);
        return false;
    }
    if (::link(partial.c_str(), path.c_str()) != 0) {
        error = "publish capture failed: " + std::string(std::strerror(errno));
        std::filesystem::remove(partial);
        return false;
    }
    std::filesystem::remove(partial);
    return true;
}

} // namespace
#endif

int run_pflash_drafter_ipc_daemon(const char * drafter_path,
                                  int drafter_gpu,
                                  int stream_fd) {
#if defined(_WIN32)
    (void)drafter_path; (void)drafter_gpu; (void)stream_fd;
    std::fprintf(stderr, "PFlash drafter IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    if (!drafter_path || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=pflash-compress <drafter.gguf> "
            "--stream-fd=FD [--draft-gpu=N]\n");
        return 2;
    }

    DrafterContext ctx;
    if (!load_drafter(drafter_path, /*gpu_layers=*/999, std::max(0, drafter_gpu), ctx)) {
        std::fprintf(stderr, "[pflash-ipc-daemon] drafter load failed: %s\n",
                     dflash27b_last_error());
        stream_status(stream_fd, -1);
        return 1;
    }

    std::fprintf(stderr, "[pflash-ipc-daemon] ready gpu=%d\n", std::max(0, drafter_gpu));
    stream_status(stream_fd, 0);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") break;
        if (cmd == "capture_post_block12") {
            PFlashDrafterIpcCaptureCommand request;
            std::string error;
            if (!parse_pflash_drafter_ipc_capture_command(
                    line, request, error)) {
                std::fprintf(stderr, "[pflash-ipc-daemon] bad capture: %s\n",
                             error.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::filesystem::path request_dir;
            if (!resolve_capture_directory(
                    request.request_dir, request_dir, error)) {
                std::fprintf(stderr, "[pflash-ipc-daemon] bad capture directory: %s\n",
                             error.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            const std::filesystem::path input_path =
                request_dir / "input_ids.i32";
            const std::filesystem::path output_path =
                request_dir / "post_block12.bf16";
            std::error_code file_ec;
            const auto input_status =
                std::filesystem::symlink_status(input_path, file_ec);
            const uintmax_t input_bytes = file_ec ? 0 :
                std::filesystem::file_size(input_path, file_ec);
            if (file_ec || !std::filesystem::is_regular_file(input_status) ||
                input_bytes == 0 || input_bytes % sizeof(int32_t) != 0 ||
                input_bytes / sizeof(int32_t) > PFLASH_CAPTURE_MAX_TOKENS ||
                std::filesystem::exists(output_path, file_ec)) {
                std::fprintf(stderr,
                    "[pflash-ipc-daemon] capture input size or output path is invalid\n");
                stream_status(stream_fd, -1);
                continue;
            }
            auto input_ids = read_int32_file(input_path.string());
            if (input_ids.empty() ||
                request.score_query_end > (int)input_ids.size() ||
                std::any_of(input_ids.begin(), input_ids.end(), [&](int32_t id) {
                    return id < 0 || id >= ctx.weights.n_vocab;
                })) {
                std::fprintf(stderr,
                    "[pflash-ipc-daemon] capture token input or query span is invalid\n");
                stream_status(stream_fd, -1);
                continue;
            }
            std::vector<float> unused_scores;
            std::vector<uint16_t> capture;
            if (!forward_qwen3_drafter_model(
                    ctx.weights, input_ids, request.score_query_tokens,
                    unused_scores, request.score_query_end, &capture)) {
                std::fprintf(stderr, "[pflash-ipc-daemon] capture failed: %s\n",
                             dflash27b_last_error());
                stream_status(stream_fd, -1);
                continue;
            }
            const size_t expected =
                input_ids.size() * (size_t)ctx.weights.n_embd;
            if (capture.size() != expected ||
                !write_immutable_capture(output_path, capture, error)) {
                std::fprintf(stderr,
                    "[pflash-ipc-daemon] capture artifact failed: %s\n",
                    error.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            const int32_t token_count = (int32_t)input_ids.size();
            const int32_t hidden_size = (int32_t)ctx.weights.n_embd;
            if (!stream_status(stream_fd, 0) ||
                !write_exact_fd(stream_fd, &token_count, sizeof(token_count)) ||
                !write_exact_fd(stream_fd, &hidden_size, sizeof(hidden_size)) ||
                !write_exact_fd(stream_fd, input_ids.data(),
                                input_ids.size() * sizeof(int32_t))) {
                std::fprintf(stderr,
                    "[pflash-ipc-daemon] capture response write failed\n");
                break;
            }
            continue;
        }
        if (cmd == "compress" || cmd == "compress2") {
            PFlashDrafterIpcCompressCommand request;
            std::string parse_error;
            if (!parse_pflash_drafter_ipc_compress_command(line, request, parse_error)) {
                std::fprintf(stderr, "[pflash-ipc-daemon] bad compress: %s (%s)\n",
                             line.c_str(), parse_error.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            auto input_ids = read_int32_file(request.path);
            if (input_ids.empty()) {
                std::fprintf(stderr, "[pflash-ipc-daemon] read tokens failed: %s\n",
                             request.path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            auto compressed = drafter_score_and_compress(
                ctx, input_ids, request.keep_ratio, /*chunk_size=*/32,
                request.score_query_tokens, /*pool_kernel=*/13,
                request.score_query_end);
            if (compressed.empty()) {
                std::fprintf(stderr, "[pflash-ipc-daemon] compress returned empty\n");
                stream_status(stream_fd, -1);
                continue;
            }
            const int32_t n_out = (int32_t)compressed.size();
            if (!stream_status(stream_fd, 0) ||
                !write_exact_fd(stream_fd, &n_out, sizeof(n_out)) ||
                !write_exact_fd(stream_fd, compressed.data(),
                                compressed.size() * sizeof(int32_t))) {
                std::fprintf(stderr, "[pflash-ipc-daemon] stream write failed\n");
                break;
            }
            continue;
        }
        std::fprintf(stderr, "[pflash-ipc-daemon] unknown command: %s\n", line.c_str());
        stream_status(stream_fd, -1);
    }

    free_drafter(ctx);
    std::fprintf(stderr, "[pflash-ipc-daemon] stopped\n");
    return 0;
#endif
}

} // namespace dflash::common
