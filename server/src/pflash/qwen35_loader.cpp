// Qwen3.5-0.8B drafter loading: the drafter GGUF, the optional trained
// block-15 scoring head and the optional segment probe.
//
// The scorer is built on the Qwen3.5 target weights
// (load_target_gguf_partial).

#include "qwen35_drafter.h"

#include "common/gguf_inspect.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <string>
#include <vector>

namespace luce::common {

bool qwen35_head_block_available(const TargetWeights & w, std::string & error) {
    if (w.n_layer <= kQwen35HeadBlock || (size_t)kQwen35HeadBlock >= w.layers.size()) {
        error = "qwen35 scoring head needs at least 16 blocks";
        return false;
    }
    const TargetLayer & L = w.layers[(size_t)kQwen35HeadBlock];
    if (((kQwen35HeadBlock + 1) % w.full_attention_interval) != 0 ||
        !L.wq || !L.wk || !L.attn_norm || !L.q_norm || !L.k_norm) {
        error = "qwen35 scoring head block 15 is not a full-attention block";
        return false;
    }
    return true;
}

namespace {

static constexpr const char * kQwen35HeadSchema = "qwen3_5_0_8b_nope_qk_mass_v1";
static constexpr const char * kQwen35HeadBaseModel = "Qwen/Qwen3.5-0.8B";
static constexpr const char * kQwen35HeadFeatureTap =
    "post_block14_residual_before_block15";

static void free_qwen35_head(Qwen35DrafterState & st) {
    if (st.head_buf) { ggml_backend_buffer_free(st.head_buf); st.head_buf = nullptr; }
    if (st.head_ctx) { ggml_free(st.head_ctx); st.head_ctx = nullptr; }
    st.head_wq = st.head_wk = nullptr;
    st.head_loaded = false;
}

static void free_qwen35_segment_probe(Qwen35DrafterState & st) {
    if (st.probe_buf) { ggml_backend_buffer_free(st.probe_buf); st.probe_buf = nullptr; }
    if (st.probe_ctx) { ggml_free(st.probe_ctx); st.probe_ctx = nullptr; }
    st.probe_norm_w = st.probe_norm_b = st.probe_fc1_w = st.probe_fc1_b =
        st.probe_fc2_w = st.probe_fc2_b =
        st.probe_sub_fc2_w = st.probe_sub_fc2_b = nullptr;
    st.probe_conv_w.clear();
    st.probe_sub_conv_w.clear();
    st.probe_loaded = false;
    st.probe_line_aligned = false;
}

static bool qwen35_metadata_equals(gguf_context * g, const char * key,
                                   const std::string & expected) {
    const int id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
           expected == gguf_get_val_str(g, id);
}

// Optional trained head for the block-15 tap. Fails closed on any contract
// mismatch.
static bool load_qwen35_scoring_head(const std::string & path,
                                     Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    std::string block_error;
    if (!qwen35_head_block_available(w, block_error)) {
        set_last_error(block_error);
        return false;
    }
    if (st.gguf_sha256.empty()) {
        set_last_error("scoring head requires the drafter GGUF identity hash");
        return false;
    }
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        set_last_error("scoring head GGUF could not be opened: " + path);
        return false;
    }
    auto fail = [&](const std::string & message) {
        free_qwen35_head(st);
        gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    // GGUF contract of a scoring-head file: architecture `pflash_scoring_head`,
    // metadata and tensors under `scoringhead.*`.
    if (!qwen35_metadata_equals(g, "general.architecture", "pflash_scoring_head") ||
        !qwen35_metadata_equals(g, "scoringhead.schema", kQwen35HeadSchema) ||
        !qwen35_metadata_equals(g, "scoringhead.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "scoringhead.runtime_gguf_sha256", st.gguf_sha256) ||
        !qwen35_metadata_equals(g, "scoringhead.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("scoring head metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    struct Contract {
        const char * name;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"scoringhead.attn_q.weight", (int64_t)w.n_embd,
         (int64_t)w.n_head * w.n_embd_head_k, &st.head_wq},
        {"scoringhead.attn_k.weight", (int64_t)w.n_embd,
         (int64_t)w.n_head_kv * w.n_embd_head_k, &st.head_wk},
    };
    ggml_init_params head_params{};
    head_params.mem_size = 4 * ggml_tensor_overhead();
    head_params.no_alloc = true;
    st.head_ctx = ggml_init(head_params);
    if (!st.head_ctx) return fail("scoring head context allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = data_ctx ? ggml_get_tensor(data_ctx, contract.name) : nullptr;
        if (!source || source->type != GGML_TYPE_F32 || ggml_n_dims(source) != 2 ||
            source->ne[0] != contract.ne0 || source->ne[1] != contract.ne1) {
            return fail(std::string("scoring head tensor contract mismatch: ") +
                        contract.name);
        }
        *contract.destination =
            ggml_new_tensor_2d(st.head_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    st.head_buf = ggml_backend_alloc_ctx_tensors(st.head_ctx, w.backend);
    if (!st.head_buf) return fail("scoring head buffer allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    gguf_free(g);
    ggml_free(data_ctx);
    st.head_loaded = true;
    std::fprintf(stderr, "[qwen35-drafter] loaded scoring head: %s\n", path.c_str());
    std::fflush(stderr);
    return true;
}

static constexpr const char * kQwen35ProbeSchema = "qwen3_5_0_8b_segment_probe_v1";
static constexpr const char * kQwen35ProbeSchemaV2 = "qwen3_5_0_8b_segment_probe_v2";

static bool qwen35_metadata_f32(gguf_context * g, const char * key, float & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return false;
    out = gguf_get_val_f32(g, id);
    return true;
}

static bool qwen35_metadata_u32(gguf_context * g, const char * key, int & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return false;
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

// Optional segment probe for the block-14 tap: LayerNorm -> Linear -> GELU ->
// Linear on the GPU, a 5-tap smoothing on the CPU, sigmoid, cut above the
// threshold. Fails closed on any contract mismatch, like the head loader.
static bool load_qwen35_segment_probe(const std::string & path,
                                      Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    if (st.gguf_sha256.empty()) {
        set_last_error("segment probe requires the drafter GGUF identity hash");
        return false;
    }
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        set_last_error("segment probe GGUF could not be opened: " + path);
        return false;
    }
    auto fail = [&](const std::string & message) {
        free_qwen35_segment_probe(st);
        gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    if (!qwen35_metadata_equals(g, "general.architecture", "segmentprobe") ||
        (!qwen35_metadata_equals(g, "segmentprobe.schema", kQwen35ProbeSchema) &&
         !qwen35_metadata_equals(g, "segmentprobe.schema", kQwen35ProbeSchemaV2)) ||
        !qwen35_metadata_equals(g, "segmentprobe.base_model", kQwen35HeadBaseModel) ||
        !qwen35_metadata_equals(g, "segmentprobe.runtime_gguf_sha256", st.gguf_sha256) ||
        !qwen35_metadata_equals(g, "segmentprobe.feature_tap", kQwen35HeadFeatureTap)) {
        return fail("segment probe metadata does not match the loaded Qwen3.5-0.8B drafter");
    }
    if (!qwen35_metadata_f32(g, "segmentprobe.threshold", st.probe_threshold) ||
        !qwen35_metadata_f32(g, "segmentprobe.norm_eps", st.probe_norm_eps) ||
        !qwen35_metadata_u32(g, "segmentprobe.min_segment", st.probe_min_segment) ||
        !qwen35_metadata_u32(g, "segmentprobe.max_segment", st.probe_max_segment) ||
        !(st.probe_threshold > 0.0f && st.probe_threshold < 1.0f) ||
        st.probe_min_segment < 1 || st.probe_max_segment < st.probe_min_segment) {
        return fail("segment probe parameters are missing or out of range");
    }
    {
        // Optional oversize split rule; an unknown value refuses the probe.
        st.probe_line_aligned = false;
        const int id = gguf_find_key(g, "segmentprobe.oversize_split");
        if (id >= 0) {
            if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
                return fail("segmentprobe.oversize_split must be a string");
            }
            const std::string rule = gguf_get_val_str(g, id);
            if (rule == "line_aligned") {
                st.probe_line_aligned = true;
            } else if (rule != "best_score") {
                return fail("unknown segmentprobe.oversize_split: " + rule);
            }
        }
    }
    ggml_tensor * fc1 = data_ctx ? ggml_get_tensor(data_ctx, "segmentprobe.fc1.weight") : nullptr;
    if (!fc1 || fc1->type != GGML_TYPE_F32 || ggml_n_dims(fc1) != 2 ||
        fc1->ne[0] != w.n_embd || fc1->ne[1] < 1) {
        return fail("segment probe tensor contract mismatch: segmentprobe.fc1.weight");
    }
    st.probe_width = (int) fc1->ne[1];
    struct Contract {
        const char * name;
        int n_dims;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"segmentprobe.norm.weight", 1, (int64_t) w.n_embd, 1, &st.probe_norm_w},
        {"segmentprobe.norm.bias", 1, (int64_t) w.n_embd, 1, &st.probe_norm_b},
        {"segmentprobe.fc1.weight", 2, (int64_t) w.n_embd, (int64_t) st.probe_width, &st.probe_fc1_w},
        {"segmentprobe.fc1.bias", 1, (int64_t) st.probe_width, 1, &st.probe_fc1_b},
        // ggml drops trailing unit dimensions: the [width, 1] output row is 1-D.
        {"segmentprobe.fc2.weight", 1, (int64_t) st.probe_width, 1, &st.probe_fc2_w},
        {"segmentprobe.fc2.bias", 1, 1, 1, &st.probe_fc2_b},
    };
    ggml_init_params probe_params{};
    probe_params.mem_size = 8 * ggml_tensor_overhead();
    probe_params.no_alloc = true;
    st.probe_ctx = ggml_init(probe_params);
    if (!st.probe_ctx) return fail("segment probe context allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        if (!source || source->type != GGML_TYPE_F32 ||
            ggml_n_dims(source) != contract.n_dims ||
            source->ne[0] != contract.ne0 ||
            (contract.n_dims == 2 && source->ne[1] != contract.ne1)) {
            return fail(std::string("segment probe tensor contract mismatch: ") + contract.name);
        }
        *contract.destination = contract.n_dims == 1
            ? ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, contract.ne0)
            : ggml_new_tensor_2d(st.probe_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    ggml_tensor * conv_w = ggml_get_tensor(data_ctx, "segmentprobe.conv.weight");
    ggml_tensor * conv_b = ggml_get_tensor(data_ctx, "segmentprobe.conv.bias");
    if (!conv_w || conv_w->type != GGML_TYPE_F32 || ggml_n_dims(conv_w) != 1 ||
        conv_w->ne[0] < 1 || conv_w->ne[0] % 2 != 1 ||
        !conv_b || conv_b->type != GGML_TYPE_F32 || ggml_n_dims(conv_b) != 1 || conv_b->ne[0] != 1) {
        return fail("segment probe tensor contract mismatch: segmentprobe.conv");
    }
    st.probe_conv_w.assign((const float *) conv_w->data,
                           (const float *) conv_w->data + conv_w->ne[0]);
    st.probe_conv_b = ((const float *) conv_b->data)[0];
    // Optional sub-unit head (schema v2): scores feed only the oversize
    // split rule's interior argmax. All four tensors ship together or none.
    ggml_tensor * sub_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.fc2.weight");
    ggml_tensor * sub_b_src = nullptr;
    if (sub_src) {
        sub_b_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.fc2.bias");
        ggml_tensor * sub_cw_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.conv.weight");
        ggml_tensor * sub_cb_src = ggml_get_tensor(data_ctx, "segmentprobe.subunit.conv.bias");
        if (sub_src->type != GGML_TYPE_F32 || ggml_n_dims(sub_src) != 1 ||
            sub_src->ne[0] != (int64_t) st.probe_width ||
            !sub_b_src || sub_b_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(sub_b_src) != 1 || sub_b_src->ne[0] != 1 ||
            !sub_cw_src || sub_cw_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(sub_cw_src) != 1 || sub_cw_src->ne[0] != conv_w->ne[0] ||
            !sub_cb_src || sub_cb_src->type != GGML_TYPE_F32 ||
            ggml_n_dims(sub_cb_src) != 1 || sub_cb_src->ne[0] != 1) {
            return fail("segment probe tensor contract mismatch: segmentprobe.subunit");
        }
        st.probe_sub_fc2_w = ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, st.probe_width);
        ggml_set_name(st.probe_sub_fc2_w, "segmentprobe.subunit.fc2.weight");
        st.probe_sub_fc2_b = ggml_new_tensor_1d(st.probe_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(st.probe_sub_fc2_b, "segmentprobe.subunit.fc2.bias");
        st.probe_sub_conv_w.assign((const float *) sub_cw_src->data,
                                   (const float *) sub_cw_src->data + sub_cw_src->ne[0]);
        st.probe_sub_conv_b = ((const float *) sub_cb_src->data)[0];
    }
    st.probe_buf = ggml_backend_alloc_ctx_tensors(st.probe_ctx, w.backend);
    if (!st.probe_buf) return fail("segment probe buffer allocation failed");
    for (const auto & contract : contracts) {
        ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
        ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
    }
    if (st.probe_sub_fc2_w) {
        ggml_backend_tensor_set(st.probe_sub_fc2_w, sub_src->data, 0, ggml_nbytes(sub_src));
        ggml_backend_tensor_set(st.probe_sub_fc2_b, sub_b_src->data, 0, ggml_nbytes(sub_b_src));
    }
    gguf_free(g);
    ggml_free(data_ctx);
    st.probe_loaded = true;
    std::fprintf(stderr,
        "[qwen35-drafter] loaded segment probe: %s (width %d, threshold %.3f, "
        "segments %d-%d tokens, %zu conv taps%s%s)\n",
        path.c_str(), st.probe_width, st.probe_threshold,
        st.probe_min_segment, st.probe_max_segment, st.probe_conv_w.size(),
        st.probe_sub_fc2_w ? ", sub-unit head" : "",
        st.probe_line_aligned ? ", line-aligned oversize split" : "");
    std::fflush(stderr);
    return true;
}

static void free_qwen35_request_probe(Qwen35DrafterState & st) {
    if (st.request_buf) { ggml_backend_buffer_free(st.request_buf); st.request_buf = nullptr; }
    if (st.request_ctx) { ggml_free(st.request_ctx); st.request_ctx = nullptr; }
    st.request_norm_w = st.request_norm_b = st.request_fc1_w = st.request_fc1_b =
        st.request_fc2_w = st.request_fc2_b = nullptr;
    st.request_width = 0;
    st.request_loaded = false;
    st.request_stub = false;
    st.request_turns = false;
    st.request_v5 = false;
    st.request_checkpoint_sha256.clear();
    st.request_policy = luce::pflash::PFlashRequestPolicy{};
}

static void free_qwen35_stage2(Qwen35DrafterState & st) {
    luce::pflash::pflash_stage2_device_free(st.stage2_dev);
    st.stage2 = luce::pflash::PFlashStage2Weights{};
    st.stage2_w_in.clear();
    st.stage2_loaded = false;
}

static constexpr const char * kQwen35RequestProbeSchema = "qwen3_5_0_8b_request_probe_v1";

// Request probe (PFLASH_SELECT_QUERY_PROBE=1): its own LayerNorm -> fc1 ->
// GELU(erf) -> fc2 trunk on the block-14 tap, from a GGUF of its own
// (``path``), or -- ``stub`` -- an all-zero trunk of width 256 with
// tau = +inf, under which every request falls back to today's windows.
// Fails closed on any contract mismatch, like the segment probe.
static bool load_qwen35_request_probe(const char * path, bool stub,
                                      Qwen35DrafterState & st) {
    const TargetWeights & w = st.weights;
    ggml_context * data_ctx = nullptr;
    gguf_context * g = nullptr;
    auto fail = [&](const std::string & message) {
        free_qwen35_request_probe(st);
        if (g) gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        set_last_error(message);
        return false;
    };
    if (stub) {
        st.request_width = 256;
        st.request_norm_eps = 1e-5f;
        st.request_tau = std::numeric_limits<float>::infinity();
    } else {
        if (!path || !*path) {
            return fail("PFLASH_SELECT_QUERY_PROBE=1 needs PFLASH_REQUEST_PROBE_GGUF");
        }
        if (st.gguf_sha256.empty()) {
            return fail("request probe requires the drafter GGUF identity hash");
        }
        gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &data_ctx };
        g = gguf_init_from_file(path, params);
        if (!g) return fail(std::string("request probe GGUF could not be opened: ") + path);
        if (!qwen35_metadata_equals(g, "general.architecture", "requestprobe") ||
            !qwen35_metadata_equals(g, "requestprobe.schema", kQwen35RequestProbeSchema) ||
            !qwen35_metadata_equals(g, "requestprobe.base_model", kQwen35HeadBaseModel) ||
            !qwen35_metadata_equals(g, "requestprobe.runtime_gguf_sha256", st.gguf_sha256) ||
            !qwen35_metadata_equals(g, "requestprobe.feature_tap", kQwen35HeadFeatureTap)) {
            return fail("request probe metadata does not match the loaded Qwen3.5-0.8B drafter");
        }
        if (!qwen35_metadata_f32(g, "requestprobe.norm_eps", st.request_norm_eps) ||
            !(st.request_norm_eps > 0.0f) ||
            !qwen35_metadata_f32(g, "requestprobe.tau", st.request_tau) ||
            std::isnan(st.request_tau)) {
            return fail("request probe metadata requestprobe.norm_eps / requestprobe.tau "
                        "is missing or out of range");
        }
        if (!qwen35_metadata_equals(g, "requestprobe.extraction_version",
                                    luce::pflash::kRequestExtractionVersion)) {
            return fail(std::string("request probe metadata requestprobe.extraction_version "
                        "is missing or is not ") + luce::pflash::kRequestExtractionVersion +
                        " (the runtime's request-run extraction rule)");
        }
        // V2-A3 candidate rule: opted in by the GGUF; absent = human messages only. cand_v5 (STAGE2-CONTRACT.md
        // section 2) is applied only on an exact match too.
        st.request_turns = false;
        st.request_v5 = false;
        if (gguf_find_key(g, luce::pflash::kRequestCandidateRuleKey) >= 0) {
            if (qwen35_metadata_equals(g, luce::pflash::kRequestCandidateRuleKey,
                                       luce::pflash::kRequestCandidateRule)) {
                st.request_turns = true;
            } else if (qwen35_metadata_equals(g, luce::pflash::kRequestCandidateRuleKey,
                                              luce::pflash::kRequestCandidateRuleV5)) {
                st.request_v5 = true;
            } else {
                return fail(std::string("request probe metadata ") +
                            luce::pflash::kRequestCandidateRuleKey + " is not " +
                            luce::pflash::kRequestCandidateRule + " or " + luce::pflash::kRequestCandidateRuleV5 +
                            " (the runtime's candidate rules)");
            }
        }
        // Provenance; the pairing key of a stage-2 GGUF (read, never checked here).
        st.request_checkpoint_sha256.clear();
        {
            const int64_t id = gguf_find_key(g, luce::pflash::kRequestCheckpointShaKey);
            if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING) {
                st.request_checkpoint_sha256 = gguf_get_val_str(g, id);
            }
        }
        ggml_tensor * fc1 = ggml_get_tensor(data_ctx, "requestprobe.fc1.weight");
        if (!fc1 || fc1->type != GGML_TYPE_F32 || ggml_n_dims(fc1) != 2 ||
            fc1->ne[0] != w.n_embd || fc1->ne[1] < 1) {
            return fail("request probe tensor contract mismatch: requestprobe.fc1.weight");
        }
        st.request_width = (int) fc1->ne[1];
    }
    struct Contract {
        const char * name;
        int n_dims;
        int64_t ne0;
        int64_t ne1;
        ggml_tensor ** destination;
    };
    const Contract contracts[] = {
        {"requestprobe.norm.weight", 1, (int64_t) w.n_embd, 1, &st.request_norm_w},
        {"requestprobe.norm.bias", 1, (int64_t) w.n_embd, 1, &st.request_norm_b},
        {"requestprobe.fc1.weight", 2, (int64_t) w.n_embd, (int64_t) st.request_width, &st.request_fc1_w},
        {"requestprobe.fc1.bias", 1, (int64_t) st.request_width, 1, &st.request_fc1_b},
        {"requestprobe.fc2.weight", 1, (int64_t) st.request_width, 1, &st.request_fc2_w},
        {"requestprobe.fc2.bias", 1, 1, 1, &st.request_fc2_b},
    };
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    st.request_ctx = ggml_init(ip);
    if (!st.request_ctx) return fail("request probe context allocation failed");
    for (const auto & contract : contracts) {
        if (!stub) {
            ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
            if (!source || source->type != GGML_TYPE_F32 ||
                ggml_n_dims(source) != contract.n_dims ||
                source->ne[0] != contract.ne0 ||
                (contract.n_dims == 2 && source->ne[1] != contract.ne1)) {
                return fail(std::string("request probe tensor contract mismatch: ") + contract.name);
            }
        }
        *contract.destination = contract.n_dims == 1
            ? ggml_new_tensor_1d(st.request_ctx, GGML_TYPE_F32, contract.ne0)
            : ggml_new_tensor_2d(st.request_ctx, GGML_TYPE_F32, contract.ne0, contract.ne1);
        ggml_set_name(*contract.destination, contract.name);
    }
    st.request_buf = ggml_backend_alloc_ctx_tensors(st.request_ctx, w.backend);
    if (!st.request_buf) return fail("request probe buffer allocation failed");
    for (const auto & contract : contracts) {
        if (stub) {
            ggml_backend_tensor_memset(*contract.destination, 0, 0, ggml_nbytes(*contract.destination));
        } else {
            ggml_tensor * source = ggml_get_tensor(data_ctx, contract.name);
            ggml_backend_tensor_set(*contract.destination, source->data, 0, ggml_nbytes(source));
        }
    }
    // v2 policy: GGUF metadata (optional keys), then the JSON file and the pin variable (config).
    luce::pflash::PFlashRequestPolicy policy;
    if (g) {
        const struct { const char * key; int * dst; } ints[] = {
            {"requestprobe.policy.min_prompt_tokens", &policy.min_prompt_tokens},
            {"requestprobe.policy.max_prompt_tokens", &policy.max_prompt_tokens},
            {"requestprobe.policy.max_runs", &policy.max_runs},
            {"requestprobe.policy.recency_pin_tokens", &policy.recency_pin_tokens},
        };
        for (const auto & k : ints) {
            if (gguf_find_key(g, k.key) >= 0 && !qwen35_metadata_u32(g, k.key, *k.dst)) {
                return fail(std::string("request probe metadata ") + k.key + " must be a u32");
            }
        }
        const struct { const char * key; float * dst; } floats[] = {
            {"requestprobe.policy.min_mean_margin", &policy.min_mean_margin},
            {"requestprobe.policy.min_max_margin", &policy.min_max_margin},
            {"requestprobe.policy.min_latest_share", &policy.min_latest_share},
            {"requestprobe.policy.max_latest_share", &policy.max_latest_share},
        };
        for (const auto & k : floats) {
            if (gguf_find_key(g, k.key) >= 0 && !qwen35_metadata_f32(g, k.key, *k.dst)) {
                return fail(std::string("request probe metadata ") + k.key + " must be an f32");
            }
        }
    }
    std::string policy_error;
    if (const char * file = std::getenv("PFLASH_REQUEST_PROBE_POLICY")) {
        std::ifstream in(file);
        std::stringstream text;
        text << in.rdbuf();
        if (!*file || !in || !luce::pflash::pflash_request_policy_apply_json(text.str(), policy, policy_error)) {
            return fail(std::string("PFLASH_REQUEST_PROBE_POLICY ") + file + ": " +
                        (policy_error.empty() ? "cannot read the file" : policy_error));
        }
    }
    if (const char * raw = std::getenv("PFLASH_SELECT_QUERY_PROBE_RECENCY_PIN")) {
        char * end = nullptr;
        const long k = std::strtol(raw, &end, 10);
        if (!*raw || *end || k < 0 || k > 4096) {
            return fail("PFLASH_SELECT_QUERY_PROBE_RECENCY_PIN must be an integer in [0, 4096]");
        }
        policy.recency_pin_tokens = (int) k;
    }
    if (!luce::pflash::pflash_request_policy_valid(policy, policy_error)) return fail(policy_error);
    st.request_policy = policy;
    if (g) gguf_free(g);
    if (data_ctx) ggml_free(data_ctx);
    st.request_stub = stub;
    st.request_loaded = true;
    std::fprintf(stderr, "[qwen35-drafter] request probe candidate rule: %s\n",
                 st.request_v5 ? luce::pflash::kRequestCandidateRuleV5
                 : st.request_turns ? luce::pflash::kRequestCandidateRule : "human messages only");
    std::fprintf(stderr,
        "[qwen35-drafter] loaded request probe: %s (width %d, tau %g; policy: prompt [%d,%d], "
        "min mean/max margin %g/%g, latest share [%g,%g], max runs %d, recency pin %d)\n",
        stub ? "STUB all-zero trunk, every request falls back" : path,
        st.request_width, (double) st.request_tau, policy.min_prompt_tokens, policy.max_prompt_tokens,
        (double) policy.min_mean_margin, (double) policy.min_max_margin, (double) policy.min_latest_share,
        (double) policy.max_latest_share, policy.max_runs, policy.recency_pin_tokens);
    std::fflush(stderr);
    return true;
}

// Stage 2 (PFLASH_REQUEST_STAGE2_GGUF, STAGE2-CONTRACT.md section 3): only with a cand_v5 request probe and the
// scoring head (a_h reads its Q/K). Every pin is checked by pflash_stage2_load; the drafter's block-15 norms the
// look-back multiplies by must be F32 at the head's shapes. Fails closed.
static bool load_qwen35_stage2(const char * path, const std::string & drafter_path,
                               const std::string & head_path, Qwen35DrafterState & st) {
    free_qwen35_stage2(st);
    const TargetWeights & w = st.weights;
    const auto fail = [&](const std::string & message) {
        free_qwen35_stage2(st);
        set_last_error(message);
        return false;
    };
    if (!path || !*path) return fail(std::string(luce::pflash::kStage2GgufEnv) + " is empty");
    if (!st.request_loaded || st.request_stub || !st.request_v5) {
        return fail(std::string(luce::pflash::kStage2GgufEnv) +
                    " needs PFLASH_SELECT_QUERY_PROBE=1 with a cand_v5 request probe (requestprobe.candidate_rule " +
                    luce::pflash::kRequestCandidateRuleV5 + ")");
    }
    if (!st.head_loaded || head_path.empty()) {
        return fail(std::string(luce::pflash::kStage2GgufEnv) + " needs PFLASH_SCORING_HEAD_GGUF (a_h uses its Q/K)");
    }
    if (w.n_embd != luce::pflash::kStage2Hidden || w.n_head != luce::pflash::kStage2Heads ||
        w.n_head_kv != luce::pflash::kStage2KvHeads || w.n_embd_head_k != luce::pflash::kStage2HeadDim) {
        return fail("stage 2: the drafter's block-15 geometry is not the u314 head's (1024 / 8 / 2 / 256)");
    }
    const TargetLayer & L = w.layers[(size_t) kQwen35HeadBlock];
    const struct { ggml_tensor * t; int64_t ne0; const char * name; } norms[] = {
        {L.attn_norm, w.n_embd, "blk.15.attn_norm.weight"},
        {L.q_norm, w.n_embd_head_k, "blk.15.attn_q_norm.weight"},
        {L.k_norm, w.n_embd_head_k, "blk.15.attn_k_norm.weight"},
    };
    for (const auto & n : norms) {
        if (!n.t || n.t->type != GGML_TYPE_F32 || ggml_nelements(n.t) != n.ne0) {
            return fail(std::string("stage 2: drafter tensor ") + n.name + " is not F32 at the expected shape");
        }
    }
    // The drafter's own epsilon key (the loader's 1e-6 default is not a pin).
    float drafter_eps = 0.0f;
    {
        gguf_init_params params{ /*no_alloc=*/ true, /*ctx=*/ nullptr };
        gguf_context * g = gguf_init_from_file(drafter_path.c_str(), params);
        if (!g) return fail("stage 2: the drafter GGUF could not be reopened for its metadata");
        const int64_t id = gguf_find_key(g, "qwen35.attention.layer_norm_rms_epsilon");
        const bool ok = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32;
        if (ok) drafter_eps = gguf_get_val_f32(g, id);
        gguf_free(g);
        if (!ok) return fail("stage 2: the drafter GGUF has no f32 qwen35.attention.layer_norm_rms_epsilon");
        if (std::memcmp(&drafter_eps, &w.rms_eps, sizeof(float)) != 0) {
            return fail("stage 2: the drafter's loaded rms_eps differs from its GGUF key");
        }
    }
    const auto head_identity = read_gguf_metadata(head_path, /*compute_sha256=*/ true);
    if (!head_identity.ok || head_identity.sha256.empty()) {
        return fail("stage 2: the scoring-head GGUF could not be hashed");
    }
    luce::pflash::PFlashStage2Pins pins;
    pins.drafter_sha256 = st.gguf_sha256;
    pins.head_sha256 = head_identity.sha256;
    pins.stage1_checkpoint_sha256 = st.request_checkpoint_sha256;
    pins.stage1_candidate_rule = luce::pflash::kRequestCandidateRuleV5;
    pins.drafter_rms_eps = drafter_eps;
    std::string error;
    if (!luce::pflash::pflash_stage2_load(path, pins, st.stage2, error)) return fail(error);
    if (!luce::pflash::pflash_stage2_device_upload(w.backend, st.stage2, st.stage2_dev, error)) return fail(error);
    st.stage2_w_in.resize((size_t) w.n_embd);
    ggml_backend_tensor_get(L.attn_norm, st.stage2_w_in.data(), 0, st.stage2_w_in.size() * sizeof(float));
    st.stage2_loaded = true;
    std::fprintf(stderr,
        "[qwen35-drafter] loaded request stage 2: %s (tau2 %g, tau_lb %g, tau_fb %g, tau_source %s%s)\n",
        path, (double) st.stage2.tau2, (double) st.stage2.tau_lb, (double) st.stage2.tau_fb,
        st.stage2.tau_source.c_str(),
        st.stage2.tau_source == luce::pflash::kStage2TauSourceDev ? ": PROVISIONAL thresholds, not gate-eligible" : "");
    std::fprintf(stderr,
        "[qwen35-drafter] request stage 2: drafter scoring sessions DISABLED (every scoring call runs on fresh F32 "
        "rows, as PFLASH_DRAFTER_SESSIONS=0; the variable is ignored); PFLASH_CHAT_HISTORY_QUERIES windows are "
        "dropped (stage 2 alone decides look-back)\n");
    std::fflush(stderr);
    return true;
}

} // namespace

bool load_qwen35_drafter(const std::string & gguf_path,
                         DrafterContext & out) {
    auto * st = new Qwen35DrafterState();
    // The scorer never needs logits, and tied-embedding Qwen3.5-0.8B
    // exports omit output.weight, so skip the lm_head entirely.
    TargetLoadPlan plan;
    plan.load_output = false;
    if (!load_target_gguf_partial(gguf_path, out.backend, plan, st->weights)) {
        delete st;
        return false;
    }
    const char * head_path = std::getenv("PFLASH_SCORING_HEAD_GGUF");
    const char * probe_path = std::getenv("PFLASH_SEGMENT_PROBE_GGUF");
    const bool request_wanted = luce::pflash::pflash_query_probe_requested();
    const bool request_stub = request_wanted && luce::pflash::pflash_query_probe_stub_requested();
    const char * request_path = std::getenv("PFLASH_REQUEST_PROBE_GGUF");
    if (head_path || probe_path || (request_wanted && !request_stub)) {
        const auto identity = read_gguf_metadata(gguf_path, /*compute_sha256=*/ true);
        st->gguf_sha256 = identity.ok ? identity.sha256 : std::string();
    }
    if (probe_path) {
        if (!*probe_path || !load_qwen35_segment_probe(probe_path, *st)) {
            if (!*probe_path) {
                set_last_error("PFLASH_SEGMENT_PROBE_GGUF is empty");
            }
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: segment probe load failed, "
                "refusing to serve without it\n");
            std::fflush(stderr);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (request_wanted && !load_qwen35_request_probe(request_path, request_stub, *st)) {
        std::fprintf(stderr,
            "[qwen35-drafter] ERROR: PFLASH_SELECT_QUERY_PROBE=1 and the request probe "
            "did not load (%s), refusing to serve without it\n", luce_last_error());
        std::fflush(stderr);
        free_qwen35_segment_probe(*st);
        free_target_weights(st->weights);
        delete st;
        return false;
    }
    if (head_path) {
        if (!*head_path || !load_qwen35_scoring_head(head_path, *st)) {
            if (!*head_path) {
                set_last_error("PFLASH_SCORING_HEAD_GGUF is empty");
            }
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: scoring head load failed, "
                "refusing to serve without it\n");
            std::fflush(stderr);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (const char * stage2_path = std::getenv(luce::pflash::kStage2GgufEnv)) {
        if (!load_qwen35_stage2(stage2_path, gguf_path, head_path ? head_path : "", *st)) {
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: %s is set and request stage 2 did not load (%s), "
                "refusing to serve without it\n", luce::pflash::kStage2GgufEnv, luce_last_error());
            std::fflush(stderr);
            free_qwen35_head(*st);
            free_qwen35_segment_probe(*st);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (st->probe_loaded && st->probe_line_aligned) {
        st->probe_vocab = std::make_unique<Tokenizer>();
        if (!st->probe_vocab->load_from_gguf(gguf_path.c_str())) {
            set_last_error("line-aligned segment probe: drafter vocabulary load failed");
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: the line-aligned probe split needs the "
                "drafter vocabulary, refusing to serve without it\n");
            std::fflush(stderr);
            free_qwen35_stage2(*st);
            free_qwen35_head(*st);
            free_qwen35_segment_probe(*st);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    if (luce::pflash::pflash_fold_requested()) {
        st->fold_vocab = std::make_unique<Tokenizer>();
        if (!st->fold_vocab->load_from_gguf(gguf_path.c_str())) {
            set_last_error("PFLASH_SELECT_STRATEGY=fold: drafter vocabulary load failed");
            std::fprintf(stderr,
                "[qwen35-drafter] ERROR: fold selection needs the drafter "
                "vocabulary, refusing to serve without it\n");
            std::fflush(stderr);
            free_qwen35_stage2(*st);
            free_qwen35_head(*st);
            free_qwen35_segment_probe(*st);
            free_qwen35_request_probe(*st);
            free_target_weights(st->weights);
            delete st;
            return false;
        }
    }
    out.state = st;
    out.loaded = true;
    std::fprintf(stderr,
        "[drafter] loaded qwen35: n_layer=%d n_head=%d n_head_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d gpu=%d\n",
        st->weights.n_layer, st->weights.n_head, st->weights.n_head_kv,
        st->weights.n_embd, st->weights.n_ff, st->weights.n_embd_head_k,
        st->weights.n_vocab, out.gpu);
    std::fflush(stderr);
    return true;
}

void free_qwen35_drafter_state(DrafterContext & ctx) {
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    for (auto & session : st->sessions) {
        if (session) free_qwen35_scoring_session(*session);
    }
    st->sessions.clear();
    free_qwen35_stage2(*st);
    free_qwen35_head(*st);
    free_qwen35_segment_probe(*st);
    free_qwen35_request_probe(*st);
    free_target_weights(st->weights);
    delete st;
    ctx.state = nullptr;
}

} // namespace luce::common
