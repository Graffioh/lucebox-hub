// Segment probe loader: the optional container head, on a CPU backend.
//
// Synthetic probe GGUFs (hidden 8, width 4, 5 conv taps) with and without the
// container tensors, loaded under PFLASH_SELECT_CONTAINER_HEADERS on and off;
// and the assembly vocabulary (newline tokens, cut marker ids) read from a
// tokenizer GGUF.

#include "CppUnitTestFramework.hpp"

#include "pflash/qwen35_drafter.h"
#include "pflash/pflash_selection.h"
#include "server/tokenizer.h"
#include "scoped_env.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

using luce::common::Qwen35DrafterState;

namespace {

constexpr int kHidden = 8;
constexpr int kWidth = 4;
constexpr int kTaps = 5;
constexpr const char * kSha = "sha-of-the-test-drafter";
constexpr const char * kContainerHeadersEnv = "PFLASH_SELECT_CONTAINER_HEADERS";

enum class Container { None, Full, NoThreshold, Partial };

std::string temp_path(const std::string & tag) {
    static int serial = 0;
    return (std::filesystem::temp_directory_path() /
            ("luce_pflash_probe_" + tag + "_" + std::to_string((long long) getpid()) +
             "_" + std::to_string(++serial) + ".gguf")).string();
}

std::string write_probe(Container container) {
    ggml_init_params ip{};
    ip.mem_size = 32 * ggml_tensor_overhead() + 64 * 1024;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "segmentprobe");
    gguf_set_val_str(g, "segmentprobe.schema", "qwen3_5_0_8b_segment_probe_v2");
    gguf_set_val_str(g, "segmentprobe.base_model", "Qwen/Qwen3.5-0.8B");
    gguf_set_val_str(g, "segmentprobe.runtime_gguf_sha256", kSha);
    gguf_set_val_str(g, "segmentprobe.feature_tap", "post_block14_residual_before_block15");
    gguf_set_val_f32(g, "segmentprobe.threshold", 0.9f);
    gguf_set_val_f32(g, "segmentprobe.norm_eps", 1e-5f);
    gguf_set_val_u32(g, "segmentprobe.min_segment", 1);
    gguf_set_val_u32(g, "segmentprobe.max_segment", 2048);
    gguf_set_val_u32(g, "segmentprobe.conv_taps", kTaps);
    float fill = 0.0f;
    const auto add = [&](const char * name, int64_t ne0, int64_t ne1 = 0) {
        ggml_tensor * t = ne1 > 0
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
            : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        float * data = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) data[i] = (fill += 0.01f);
        gguf_add_tensor(g, t);
    };
    add("segmentprobe.norm.weight", kHidden);
    add("segmentprobe.norm.bias", kHidden);
    add("segmentprobe.fc1.weight", kHidden, kWidth);
    add("segmentprobe.fc1.bias", kWidth);
    add("segmentprobe.fc2.weight", kWidth);
    add("segmentprobe.fc2.bias", 1);
    add("segmentprobe.conv.weight", kTaps);
    add("segmentprobe.conv.bias", 1);
    if (container != Container::None) {
        add("segmentprobe.container.fc2.weight", kWidth);
        if (container != Container::Partial) {
            add("segmentprobe.container.fc2.bias", 1);
            add("segmentprobe.container.conv.weight", kTaps);
            add("segmentprobe.container.conv.bias", 1);
        }
        if (container == Container::Full) {
            gguf_set_val_f32(g, "segmentprobe.container_threshold", 0.3f);
        }
    }
    const std::string path = temp_path("probe");
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

struct CpuProbeState {
    Qwen35DrafterState st;
    CpuProbeState() {
        st.weights.n_embd = kHidden;
        st.weights.backend = ggml_backend_cpu_init();
        st.gguf_sha256 = kSha;
    }
    ~CpuProbeState() {
        luce::common::free_qwen35_segment_probe(st);
        if (st.weights.backend) ggml_backend_free(st.weights.backend);
        st.weights.backend = nullptr;
    }
};

struct ProbeLoaderFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

} // namespace

TEST_CASE(ProbeLoaderFixture, probe_with_container_head_loads_it_only_when_requested) {
    const std::string path = write_probe(Container::Full);
    {
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
        CpuProbeState state;
        REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(state.st.probe_loaded);
        REQUIRE(state.st.probe_container_loaded);
        REQUIRE(state.st.probe_container_threshold == 0.3f);
        REQUIRE(state.st.probe_container_conv_w.size() == (size_t) kTaps);
        REQUIRE(state.st.probe_container_fc2_w != nullptr);
        // The container row landed on the backend as written: it follows the
        // unit tensors in the fill sequence.
        std::vector<float> row(kWidth);
        ggml_backend_tensor_get(state.st.probe_container_fc2_w, row.data(), 0,
                                row.size() * sizeof(float));
        const int before = 2 * kHidden + kHidden * kWidth + kWidth + kWidth + 1 + kTaps + 1;
        for (int i = 0; i < kWidth; ++i) {
            const float expected = 0.01f * (float) (before + i + 1);
            REQUIRE(std::abs(row[(size_t) i] - expected) < 1e-4f);
        }
    }
    {
        // Switch off: the container tensors are not even read.
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, nullptr};
        CpuProbeState state;
        REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(state.st.probe_loaded);
        REQUIRE(!state.st.probe_container_loaded);
        REQUIRE(state.st.probe_container_fc2_w == nullptr);
    }
    std::remove(path.c_str());
}

TEST_CASE(ProbeLoaderFixture, probe_without_container_head_leaves_headers_off) {
    const std::string path = write_probe(Container::None);
    luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
    CpuProbeState state;
    // Loads (the unit probe is intact), warns, and container headers stay off.
    REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
    REQUIRE(state.st.probe_loaded);
    REQUIRE(!state.st.probe_container_loaded);
    std::remove(path.c_str());
}

TEST_CASE(ProbeLoaderFixture, container_threshold_defaults_to_one_half) {
    const std::string path = write_probe(Container::NoThreshold);
    luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
    CpuProbeState state;
    REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
    REQUIRE(state.st.probe_container_loaded);
    REQUIRE(state.st.probe_container_threshold == 0.5f);
    std::remove(path.c_str());
}

TEST_CASE(ProbeLoaderFixture, partial_container_head_fails_closed_only_when_requested) {
    const std::string path = write_probe(Container::Partial);
    {
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
        CpuProbeState state;
        REQUIRE(!luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(!state.st.probe_loaded);
    }
    {
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "0"};
        CpuProbeState state;
        REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(!state.st.probe_container_loaded);
    }
    std::remove(path.c_str());
}

TEST_CASE(ProbeLoaderFixture, assembly_vocab_marks_newline_tokens_and_tokenizes_the_marker) {
    // Byte-level BPE strings: "Ċ" is '\n'.
    const std::vector<std::string> tokens{
        "<|im_start|>", "<|im_end|>", "\xC4\x8A", "[", ".", "]", "a", "a\xC4\x8A"};
    std::vector<const char *> token_ptrs;
    for (const auto & token : tokens) token_ptrs.push_back(token.c_str());
    std::vector<uint32_t> types{3, 3, 1, 1, 1, 1, 1, 1};
    gguf_context * g = gguf_init_empty();
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(), (int32_t) tokens.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_UINT32,
                      types.data(), (int32_t) types.size());
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen35");
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 0);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 1);
    const std::string path = temp_path("tokenizer");
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);

    Qwen35DrafterState st;
    REQUIRE(luce::common::load_qwen35_assembly_vocab(path, st));
    REQUIRE((st.newline_vocab == std::vector<uint8_t>{0, 0, 1, 0, 0, 0, 0, 1}));
    REQUIRE(!st.cut_marker_ids.empty());
    luce::common::Tokenizer tok;
    REQUIRE(tok.load_from_gguf(path.c_str()));
    REQUIRE(tok.decode(st.cut_marker_ids) == luce::pflash::kPFlashCutMarkerText);
    REQUIRE(st.cut_marker_ids.front() == 2);
    REQUIRE(st.cut_marker_ids.back() == 2);
    std::remove(path.c_str());
}
