// Segment probe loader: the optional record-start head, on a CPU backend.
//
// Synthetic probe GGUFs (hidden 8, width 4, record hidden 3, 5 unit and 7
// record conv taps) with and without the record tensors, loaded under
// PFLASH_SELECT_CONTAINER_HEADERS on and off;
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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

using luce::common::Qwen35DrafterState;

namespace {

constexpr int kHidden = 8;
constexpr int kWidth = 4;
constexpr int kTaps = 5;
constexpr int kRecordHidden = 3;
constexpr int kRecordTaps = 7;
constexpr const char * kSha = "sha-of-the-test-drafter";
constexpr const char * kContainerHeadersEnv = "PFLASH_SELECT_CONTAINER_HEADERS";

enum class Record { None, Full, NoThreshold, Partial, EvenTaps };

std::string temp_path(const std::string & tag) {
    static int serial = 0;
    return (std::filesystem::temp_directory_path() /
            ("luce_pflash_probe_" + tag + "_" + std::to_string((long long) getpid()) +
             "_" + std::to_string(++serial) + ".gguf")).string();
}

std::string write_probe(Record record, bool with_container = true) {
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
    if (record != Record::None) {
        add("segmentprobe.record.fc1.weight", kWidth, kRecordHidden);
        add("segmentprobe.record.fc1.bias", kRecordHidden);
        add("segmentprobe.record.fc2.weight", kRecordHidden);
        if (record != Record::Partial) {
            add("segmentprobe.record.fc2.bias", 1);
            add("segmentprobe.record.conv.weight",
                record == Record::EvenTaps ? kRecordTaps - 1 : kRecordTaps);
            add("segmentprobe.record.conv.bias", 1);
        }
        if (record == Record::Full) {
            gguf_set_val_f32(g, "segmentprobe.record_threshold", 0.3f);
        }
    }
    if (with_container) {
        // The mixed-level container head: never read by the runtime.
        add("segmentprobe.container.fc2.weight", kWidth);
        add("segmentprobe.container.fc2.bias", 1);
        add("segmentprobe.container.conv.weight", kTaps);
        add("segmentprobe.container.conv.bias", 1);
        gguf_set_val_f32(g, "segmentprobe.container_threshold", 0.4f);
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

TEST_CASE(ProbeLoaderFixture, probe_with_record_head_loads_it_only_when_requested) {
    const std::string path = write_probe(Record::Full);
    {
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
        CpuProbeState state;
        REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(state.st.probe_loaded);
        REQUIRE(state.st.probe_record_loaded);
        REQUIRE(state.st.probe_record_threshold == 0.3f);
        REQUIRE(state.st.probe_record_hidden == kRecordHidden);
        REQUIRE(state.st.probe_record_conv_w.size() == (size_t) kRecordTaps);
        REQUIRE(state.st.probe_record_fc1_w->ne[0] == kWidth);
        REQUIRE(state.st.probe_record_fc1_w->ne[1] == kRecordHidden);
        // The record rows landed on the backend as written: fc2 follows the
        // unit tensors, fc1 and its bias in the fill sequence.
        std::vector<float> row(kRecordHidden);
        ggml_backend_tensor_get(state.st.probe_record_fc2_w, row.data(), 0,
                                row.size() * sizeof(float));
        const int before = 2 * kHidden + kHidden * kWidth + kWidth + kWidth + 1 + kTaps + 1 +
            kWidth * kRecordHidden + kRecordHidden;
        for (int i = 0; i < kRecordHidden; ++i) {
            const float expected = 0.01f * (float) (before + i + 1);
            REQUIRE(std::abs(row[(size_t) i] - expected) < 1e-4f);
        }
    }
    {
        // Switch off: the record tensors are not even read.
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, nullptr};
        CpuProbeState state;
        REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(state.st.probe_loaded);
        REQUIRE(!state.st.probe_record_loaded);
        REQUIRE(state.st.probe_record_fc1_w == nullptr);
    }
    std::remove(path.c_str());
}

TEST_CASE(ProbeLoaderFixture, probe_without_record_head_leaves_headers_off) {
    // A container head alone is not used in its place.
    for (const bool with_container : {true, false}) {
        const std::string path = write_probe(Record::None, with_container);
        luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
        CpuProbeState state;
        // Loads (the unit probe is intact), warns, and headers stay off.
        REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
        REQUIRE(state.st.probe_loaded);
        REQUIRE(!state.st.probe_record_loaded);
        std::remove(path.c_str());
    }
}

TEST_CASE(ProbeLoaderFixture, record_threshold_defaults_to_one_half) {
    const std::string path = write_probe(Record::NoThreshold);
    luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
    CpuProbeState state;
    REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
    REQUIRE(state.st.probe_record_loaded);
    REQUIRE(state.st.probe_record_threshold == 0.5f);
    std::remove(path.c_str());
}

TEST_CASE(ProbeLoaderFixture, malformed_record_head_fails_closed_only_when_requested) {
    for (const Record record : {Record::Partial, Record::EvenTaps}) {
        const std::string path = write_probe(record);
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
            REQUIRE(!state.st.probe_record_loaded);
        }
        std::remove(path.c_str());
    }
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

TEST_CASE(ProbeLoaderFixture, record_logits_are_fc1_gelu_erf_fc2_on_the_trunk) {
    const std::string path = write_probe(Record::Full);
    luce_test::ScopedEnvVar headers{kContainerHeadersEnv, "1"};
    CpuProbeState state;
    REQUIRE(luce::common::load_qwen35_segment_probe(path, state.st));
    const auto & st = state.st;
    const auto read = [](ggml_tensor * t) {
        std::vector<float> v((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    };
    const auto w1 = read(st.probe_record_fc1_w);   // [width, hidden], width fastest
    const auto b1 = read(st.probe_record_fc1_b);
    const auto w2 = read(st.probe_record_fc2_w);
    const auto b2 = read(st.probe_record_fc2_b);

    constexpr int n = 3;
    std::vector<float> trunk((size_t) kWidth * n);
    for (size_t i = 0; i < trunk.size(); ++i) trunk[i] = std::sin(0.7f * (float) i) * 2.0f;

    ggml_init_params ip{};
    ip.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, n);
    std::memcpy(x->data, trunk.data(), trunk.size() * sizeof(float));
    // Weights live on the CPU backend buffer: the graph reads them in place.
    ggml_tensor * out = luce::common::qwen35_probe_record_logits(ctx, st, x);
    REQUIRE(ggml_nelements(out) == n);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    REQUIRE(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
    for (int t = 0; t < n; ++t) {
        double logit = b2[0];
        for (int h = 0; h < kRecordHidden; ++h) {
            double z = b1[(size_t) h];
            for (int c = 0; c < kWidth; ++c) {
                z += (double) w1[(size_t) h * kWidth + c] * trunk[(size_t) t * kWidth + c];
            }
            const double gelu = 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0)));
            logit += (double) w2[(size_t) h] * gelu;
        }
        REQUIRE(std::abs(((const float *) out->data)[t] - (float) logit) < 1e-4f);
    }
    ggml_free(ctx);
    std::remove(path.c_str());
}
