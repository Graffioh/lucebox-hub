// ggml-cuda/HIP F32 copy into a strided destination view.
//
// The PFlash Qwen3.5 scorer writes key logits chunk by chunk into a
// [S, n, H] buffer: ggml_cpy(scores[nk, n, H], view(logits, nk, n, H, offset b)).
// When S % key_chunk == 1 the last chunk is one key wide. A contiguous
// [1, n, H] source has nb01 == element size, so ggml-cuda took the tiled
// transpose kernel, which writes dst densely and ignores the view's strides:
// column S-1 stayed unwritten for rows 1.. and the values spilled into row 1.
// On the R9700 the unwritten memory held NaN and the request failed with
// "non-finite Qwen3.5 scoring-head scores" (65537 = 8 * 8192 + 1).
//
// This test runs the scorer's chunked-copy pattern and the bare [1, n, m]
// copy on the GPU backend and checks them bit for bit against a host
// reference and the CPU backend, for key counts with remainders 0, 1, 2 and
// chunk-1. It also keeps the legitimate transpose fast path covered (a
// transposed source into a contiguous destination).
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

uint32_t lcg_state = 0x2468ace1u;
float lcg_uniform() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float) ((lcg_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

size_t bit_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return a.size() + b.size();
    size_t mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++mm;
    }
    return mm;
}

// The scorer pattern: logits [S, n, H] prefilled with a sentinel, then one
// contiguous [nk, n, H] source per key chunk copied into the logits view at
// key offset b. Returns the whole logits buffer.
bool run_chunked(ggml_backend_t backend, int S, int n, int H, int chunk,
                 const std::vector<std::vector<float>> & parts, float sentinel,
                 std::vector<float> & out) {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, n, H);
    std::vector<ggml_tensor *> srcs;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int b = 0, c = 0; b < S; b += chunk, ++c) {
        const int nk = std::min(chunk, S - b);
        ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nk, n, H);
        srcs.push_back(src);
        ggml_tensor * dst = ggml_view_3d(ctx, logits, nk, n, H, logits->nb[1], logits->nb[2],
                                         (size_t) b * logits->nb[0]);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> fill((size_t) S * n * H, sentinel);
    ggml_backend_tensor_set(logits, fill.data(), 0, fill.size() * sizeof(float));
    for (size_t c = 0; c < srcs.size(); ++c) {
        ggml_backend_tensor_set(srcs[c], parts[c].data(), 0, parts[c].size() * sizeof(float));
    }
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if (ok) {
        out.resize(fill.size());
        ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// A transposed source (view of a [rows, cols] tensor) into a fresh
// contiguous tensor: the transpose fast path must stay exact.
bool run_transpose(ggml_backend_t backend, int rows, int cols, const std::vector<float> & in,
                   std::vector<float> & out) {
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
    ggml_tensor * t = ggml_cont(ctx, ggml_transpose(ctx, src));   // [cols, rows]
    ggml_set_output(t);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(src, in.data(), 0, in.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if (ok) {
        out.resize(in.size());
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::printf("test_cpy_strided_dst: no GPU device, skipped\n");
        return 77;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!cpu || !gpu) {
        std::fprintf(stderr, "test_cpy_strided_dst: backend init failed (cpu=%p gpu=%p)\n",
                     (void *) cpu, (void *) gpu);
        return 1;
    }
    int failures = 0;
    const float sentinel = std::numeric_limits<float>::quiet_NaN();
    const int chunk = 64;
    // n = query rows per tile (32 = the scorer's kTile, 1 = a single query
    // token), H = heads. S covers remainders 0, 1, 2 and chunk-1.
    const int Ss[] = {2 * chunk, 2 * chunk + 1, 2 * chunk + 2, 3 * chunk - 1, 1, chunk + 1};
    const int ns[] = {32, 7, 1};
    const int H = 8;
    for (const int S : Ss) {
        for (const int n : ns) {
            std::vector<std::vector<float>> parts;
            std::vector<float> expected((size_t) S * n * H, sentinel);
            for (int b = 0; b < S; b += chunk) {
                const int nk = std::min(chunk, S - b);
                std::vector<float> p((size_t) nk * n * H);
                for (auto & v : p) v = lcg_uniform();
                for (int h = 0; h < H; ++h) {
                    for (int q = 0; q < n; ++q) {
                        for (int k = 0; k < nk; ++k) {
                            expected[((size_t) h * n + q) * S + b + k] = p[((size_t) h * n + q) * nk + k];
                        }
                    }
                }
                parts.push_back(std::move(p));
            }
            std::vector<float> got_gpu, got_cpu;
            if (!run_chunked(gpu, S, n, H, chunk, parts, sentinel, got_gpu) ||
                !run_chunked(cpu, S, n, H, chunk, parts, sentinel, got_cpu)) {
                std::fprintf(stderr, "FAIL chunked S=%d n=%d: compute failed\n", S, n);
                ++failures;
                continue;
            }
            size_t nonfinite = 0;
            for (const float v : got_gpu) nonfinite += std::isfinite(v) ? 0 : 1;
            const size_t mm_ref = bit_mismatches(got_gpu, expected);
            const size_t mm_cpu = bit_mismatches(got_gpu, got_cpu);
            const bool pass = mm_ref == 0 && mm_cpu == 0 && nonfinite == 0;
            std::printf("%s chunked S=%d (S%%chunk=%d) n=%d H=%d: mismatches ref=%zu cpu=%zu nonfinite=%zu\n",
                        pass ? "ok  " : "FAIL", S, S % chunk, n, H, mm_ref, mm_cpu, nonfinite);
            failures += pass ? 0 : 1;
        }
    }
    const int shapes[][2] = {{256, 65}, {33, 1000}, {1, 64}, {64, 1}};
    for (const auto & s : shapes) {
        const int rows = s[0], cols = s[1];
        std::vector<float> in((size_t) rows * cols);
        for (auto & v : in) v = lcg_uniform();
        std::vector<float> expected((size_t) rows * cols);
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) expected[(size_t) r * cols + c] = in[(size_t) c * rows + r];
        }
        std::vector<float> got;
        const bool ok = run_transpose(gpu, rows, cols, in, got);
        const size_t mm = ok ? bit_mismatches(got, expected) : expected.size();
        const bool pass = ok && mm == 0;
        std::printf("%s transpose %dx%d: mismatches=%zu\n", pass ? "ok  " : "FAIL", rows, cols, mm);
        failures += pass ? 0 : 1;
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    if (failures) {
        std::fprintf(stderr, "test_cpy_strided_dst: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_cpy_strided_dst: all passed\n");
    return 0;
}
