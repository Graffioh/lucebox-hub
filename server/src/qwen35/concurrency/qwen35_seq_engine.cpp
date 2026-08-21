// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "qwen35_roctx.h"
#include "graph_builders.h"
#include "attn_masks.h"
#include "prefill_helpers.h"
#include "common/sampler.h"
#include "common/ddtree.h"
#include "common/geometric_draft_topk_cuda.h"
#include "common/speculation/adapters/dflash2_speculator.h"
#include "common/concurrency/chain_spec_shapes.h"
#include "common/speculation/spec_cost_profile.h"
#include "internal.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace dflash::common {

namespace {

constexpr const char * kSpecProfileContextMethod =
    "synthetic-zero-kv-zero-features-v1";

// Denser than pure power-of-2: reduces padding waste at non-power-of-2
// live counts (e.g. C=5 uses bucket=6 at 17% waste vs bucket=8 at 37.5%).
int decode_bucket_width(int live_count) {
    static constexpr int buckets[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    for (int b : buckets)
        if (b >= live_count) return b;
    return 64;
}

bool chain_direct_commit_enabled() {
    const char * value = std::getenv("DFLASH_CHAIN_DURABLE_REPLAY");
    return !(value && std::atoi(value) != 0);
}

double initial_prediction_realized_tokens(
        const SpecPlan & plan, const SeqEngine::StepResult & result) {
    double realized = 0.0;
    for (const SpecPlanScore & score : plan.ordered) {
        if (!score.admitted ||
            score.source == SpecScoreSource::Unavailable) {
            continue;
        }
        const auto output = std::find_if(
            result.decode.begin(), result.decode.end(),
            [&](const SeqEngine::DecodeOutput & item) {
                return item.slot == score.slot;
            });
        if (output == result.decode.end() || output->failed)
            return std::numeric_limits<double>::quiet_NaN();
        realized += 1.0 + static_cast<double>(output->spec_accepted_tokens);
    }
    return realized;
}

void log_spec_gate_plan(
        const SpecPlan & plan,
        double initial_realized_tokens,
        double measured_us) {
    int fresh = 0;
    int initial = 0;
    int unavailable = plan.unavailable_count;
    for (const SpecPlanScore & score : plan.ordered) {
        switch (score.source) {
            case SpecScoreSource::Fresh: ++fresh; break;
            case SpecScoreSource::Initial: ++initial; break;
            case SpecScoreSource::Unavailable: ++unavailable; break;
        }
    }

    std::fprintf(stderr, "[spec-gate] C=%d k=%d scores=[",
                 plan.concurrency, plan.admitted_count);
    for (size_t i = 0; i < plan.ordered.size(); ++i) {
        const SpecPlanScore & score = plan.ordered[i];
        std::fprintf(stderr, "%s%llu:%.3f/%s/%s%s",
            i == 0 ? "" : ",",
            static_cast<unsigned long long>(score.request_id),
            score.expected_yield,
            spec_score_source_name(score.source),
            score.score_kind.c_str(),
            score.admitted ? "*" : "");
    }
    std::fprintf(stderr, "] decisions=[");
    for (size_t i = 0; i < plan.ordered.size(); ++i) {
        const SpecPlanScore & score = plan.ordered[i];
        std::fprintf(stderr, "%s%llu:%s",
            i == 0 ? "" : ",",
            static_cast<unsigned long long>(score.request_id),
            score.admitted ? "speculation" : "ar");
    }
    std::fprintf(stderr,
        "] sources=fresh:%d,initial:%d,unavailable:%d "
        "initial_tokens=%.3f/",
        fresh, initial, unavailable, plan.initial_predicted_tokens);
    if (std::isfinite(initial_realized_tokens)) {
        std::fprintf(stderr, "%.3f", initial_realized_tokens);
    } else {
        std::fprintf(stderr, "n/a");
    }
    std::fprintf(stderr,
        " profiled_cost=%.1fus cost_scale=%.3f"
        " G(k)=%.6f G(0)=%.6f predicted_cost=%.1fus",
        plan.profiled_cost, plan.cost_scale,
        plan.goodput, plan.ar_goodput, plan.predicted_cost);
    if (std::isfinite(measured_us)) {
        std::fprintf(stderr, " measured=%.1fus\n", measured_us);
    } else {
        std::fprintf(stderr, " measured=ar-path\n");
    }
}

void log_spec_epoch(
        const SpecCohortEpoch & epoch,
        const SpeculationGate & gate) {
    const SpecPlan & plan = epoch.plan;
    std::fprintf(stderr,
        "[spec-epoch] {\"epoch_id\":%llu,\"request_ids\":[",
        static_cast<unsigned long long>(epoch.id));
    for (size_t i = 0; i < epoch.request_ids.size(); ++i) {
        std::fprintf(stderr, "%s%llu", i == 0 ? "" : ",",
            static_cast<unsigned long long>(epoch.request_ids[i]));
    }
    std::fprintf(stderr, "],\"selected_request_ids\":[");
    for (size_t i = 0; i < plan.admitted_request_ids.size(); ++i) {
        std::fprintf(stderr, "%s%llu", i == 0 ? "" : ",",
            static_cast<unsigned long long>(plan.admitted_request_ids[i]));
    }
    std::fprintf(stderr, "],\"requests\":[");
    for (size_t i = 0; i < plan.ordered.size(); ++i) {
        const SpecPlanScore & score = plan.ordered[i];
        const bool failed = gate.evaluation_failed(score.request_id);
        const double initial = gate.initial_score(score.request_id);
        const std::string kind = gate.initial_score_kind(score.request_id);
        const std::vector<double> & hazards =
            gate.initial_hazards(score.request_id);
        std::fprintf(stderr,
            "%s{\"request_id\":%llu,\"slot\":%d,"
            "\"activation_score\":",
            i == 0 ? "" : ",",
            static_cast<unsigned long long>(score.request_id), score.slot);
        if (std::isfinite(initial)) std::fprintf(stderr, "%.6f", initial);
        else std::fprintf(stderr, "null");
        std::fprintf(stderr,
            ",\"score_kind\":\"%s\",\"expected_yield\":",
            kind.c_str());
        if (std::isfinite(initial))
            std::fprintf(stderr, "%.6f", score.expected_yield);
        else std::fprintf(stderr, "null");
        std::fprintf(stderr, ",\"hazards\":");
        if (std::isfinite(initial)) {
            std::fprintf(stderr, "[");
            for (size_t j = 0; j < hazards.size(); ++j)
                std::fprintf(stderr, "%s%.8g", j == 0 ? "" : ",", hazards[j]);
            std::fprintf(stderr, "]");
        } else {
            std::fprintf(stderr, "null");
        }
        const char * evaluation = failed ? "failed"
            : std::isfinite(initial) ? "scored" : "unavailable";
        const char * reason = failed ? "activation_evaluation_failed"
            : score.execution_unsupported ? "execution_unsupported"
            : score.admitted ? "selected_by_joint_goodput"
            : "ar_counterfactual_won";
        std::fprintf(stderr,
            ",\"evaluation\":\"%s\",\"route\":\"%s\","
            "\"reason\":\"%s\"}",
            evaluation, score.admitted ? "speculation" : "ar", reason);
    }
    std::fprintf(stderr,
        "],\"profiled_cost_us\":%.1f,\"cost_scale\":%.6f,"
        "\"predicted_cost_us\":%.1f,\"goodput\":%.9f,"
        "\"ar_goodput\":%.9f}\n",
        plan.profiled_cost, plan.cost_scale, plan.predicted_cost,
        plan.goodput, plan.ar_goodput);
}

void log_spec_evaluation_fallback(
        uint64_t request_id,
        int slot,
        const std::string & kind,
        const char * reason) {
    const std::string cause = reason
        ? reason : "activation_evaluation_failed";
    std::fprintf(stderr,
        "[spec-evaluation] {\"request_id\":%llu,\"slot\":%d,"
        "\"score_kind\":\"%s\",\"evaluation\":\"failed\","
        "\"reason\":\"%s\"}\n",
        static_cast<unsigned long long>(request_id), slot, kind.c_str(),
        cause.c_str());
}

uint64_t file_size_or_zero(const char * path) {
    if (!path || !*path) return 0;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return 0;
    const std::streamoff size = file.tellg();
    return size > 0 ? static_cast<uint64_t>(size) : 0;
}

int configured_chain_verify_depth(int maximum) {
    const char * value = std::getenv("DFLASH_SPEC_CHAIN_DEPTH");
    if (!value || !*value) {
        return resolve_chain_verify_depth(0, maximum);
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    const bool integer = end != value && end && *end == '\0' &&
        parsed >= std::numeric_limits<int>::min() &&
        parsed <= std::numeric_limits<int>::max();
    const int resolved = integer
        ? resolve_chain_verify_depth(static_cast<int>(parsed), maximum)
        : 0;
    if (resolved != 0) return resolved;

    std::fprintf(stderr,
        "[parallel-chain] ignoring invalid DFLASH_SPEC_CHAIN_DEPTH=%s; "
        "expected root-inclusive depth 2..%d, using %d\n",
        value, maximum, maximum);
    return resolve_chain_verify_depth(0, maximum);
}

} // namespace

Qwen35SeqEngine::Qwen35SeqEngine(
        Qwen35Backend & backend, PagedKvPool & pool, int max_ctx,
        int64_t scratch_row, int tree_width, int tree_scratch_base,
        int tree_scratch_stride, SpecMode spec_mode, int max_prefills,
        int mixed_prefill_tokens, int long_mixed_prefill_tokens,
        int long_prefill_threshold, int idle_prefill_tokens,
        int prefill_quantum)
    : max_prefills_(std::max(1, max_prefills)),
      mixed_prefill_tokens_(std::max(1, mixed_prefill_tokens)),
      long_mixed_prefill_tokens_(std::max(1, long_mixed_prefill_tokens)),
      long_prefill_threshold_(std::max(1, long_prefill_threshold)),
      idle_prefill_tokens_(std::max(1, idle_prefill_tokens)),
      prefill_quantum_(std::max(1, prefill_quantum)), b_(backend),
      slots_(pool, max_ctx, std::max(1, tree_width),
             backend.paged_kv_residency_.get()),
      scratch_row_(scratch_row), tree_width_(tree_width),
      chain_verify_depth_(configured_chain_verify_depth(tree_width)),
      tree_scratch_base_(tree_scratch_base),
      tree_scratch_stride_(tree_scratch_stride), spec_mode_(spec_mode) {
    if (spec_mode_ == SpecMode::chain &&
        chain_verify_depth_ >= 2 && chain_verify_depth_ != tree_width_) {
        std::fprintf(stderr,
            "[parallel-chain] verify_depth=%d draft_width=%d "
            "(DFLASH_SPEC_CHAIN_DEPTH)\n",
            chain_verify_depth_, tree_width_);
    }
    const int n_slots = slots_.slot_count();
    slot_draft_kv_.resize((size_t)n_slots);
    prepared_chain_drafts_.resize((size_t)n_slots);
    last_activation_estimate_.resize((size_t)n_slots);

    adaptive_fallback_ar_.assign((size_t)n_slots, 0);
    if (spec_mode_ == SpecMode::chain && b_.dw_.selector.enabled) {
        DFlash2BenefitModelSignature signature;
        signature.target_layers = b_.w_.n_layer;
        signature.target_hidden = b_.w_.n_embd;
        signature.target_vocab = b_.w_.n_vocab;
        signature.draft_layers = b_.dw_.n_layer;
        signature.draft_hidden = b_.dw_.n_embd;
        signature.draft_block_size = b_.dw_.block_size;
        signature.selector_rank = b_.dw_.selector.rank;
        signature.selector_top_k = b_.dw_.selector.top_k;
        signature.selector_vocab = b_.dw_.selector.pred_cb
            ? static_cast<int>(b_.dw_.selector.pred_cb->ne[1]) : 0;
        signature.conv_kernel_size = b_.dw_.conv_kernel_size;
        signature.conv_group_size = b_.dw_.conv_group_size;
        signature.target_file_size = file_size_or_zero(b_.cfg_.target_path);
        signature.draft_file_size = file_size_or_zero(b_.cfg_.draft_path);

        std::string config_error;
        DFlash2BenefitConfig config =
            DFlash2BenefitProvider::config_from_environment(config_error);
        if (config_error.empty()) {
            speculator_ = std::make_unique<DFlash2Speculator>(
                b_.dw_, b_.draft_backend_, b_.w_.output,
                signature, config);
        }
        if (!config_error.empty() || !speculator_is_ready(speculator_.get())) {
            const std::string error = !config_error.empty()
                ? config_error
                : speculator_ ? speculator_->error()
                              : "adapter construction failed";
            speculator_.reset();
            adaptive_fallback_reason_ =
                speculator_fallback_reason(speculator_.get());
            std::fprintf(stderr,
                "[parallel-chain] no speculator adapter: %s; "
                "adaptive requests will use AR fallback\n",
                error.c_str());
        } else {
            std::fprintf(stderr,
                "[parallel-chain] speculator=%s lm_weight=%.3f "
                "hazard_scale=%.3f yield_scale=%.3f signature=%s\n",
                speculator_->score_kind().c_str(),
                config.lm_log_weight, config.hazard_scale,
                config.yield_scale, signature.str().c_str());
        }
    } else if (spec_mode_ == SpecMode::chain) {
        adaptive_fallback_reason_ =
            speculator_fallback_reason(speculator_.get());
        std::fprintf(stderr,
            "[parallel-chain] no speculator adapter for loaded drafter; "
            "adaptive requests will use AR fallback\n");
    }

    // The concurrent DDTree stack is gated to a local same-device drafter.
    // Build metadata-only BF16 views over each slot's disjoint target feature
    // ring; draft_kv_begin_step converts only newly committed rows to its F32
    // append input instead of syncing the entire 200 MiB ring per round.
    capture_features_ = tree_width_ > 0 && b_.cache_.target_feat &&
        b_.cache_.target_feat_cap > 0 && b_.cfg_.draft_path &&
        !b_.cfg_.remote_draft.enabled() && !b_.split_gpus_ &&
        b_.cfg_.draft_gpu == b_.cfg_.device.gpu;
    if (!capture_features_) return;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_slots + 1);
    ip.no_alloc = true;
    feature_view_ctx_ = ggml_init(ip);
    if (!feature_view_ctx_) {
        capture_features_ = false;
        return;
    }

    const int cap = b_.cache_.target_feat_cap;
    const int64_t fc_in =
        (int64_t)b_.w_.n_capture_layers * b_.w_.n_embd;
    slot_feature_mirrors_.resize((size_t)n_slots);
    for (int slot = 0; slot < n_slots; ++slot) {
        DraftFeatureMirror & mirror = slot_feature_mirrors_[(size_t)slot];
        mirror.target_feat = ggml_view_2d(
            feature_view_ctx_, b_.cache_.target_feat, fc_in, cap,
            b_.cache_.target_feat->nb[1],
            (size_t)slot * (size_t)cap * b_.cache_.target_feat->nb[1]);
        mirror.device = b_.cfg_.draft_gpu;
        mirror.target_device = b_.cfg_.device.gpu;
        mirror.cap = cap;
        mirror.n_target_layers = b_.w_.n_capture_layers;
        mirror.hidden_size = b_.w_.n_embd;
        mirror.storage_type = b_.cache_.target_feat->type;
    }
}

Qwen35SeqEngine::~Qwen35SeqEngine() {
    draft_kv_batch_free(batch_draft_graph_);
    for (std::unique_ptr<DraftKvState> & state : dummy_draft_kv_) {
        if (state) draft_kv_free(*state);
    }
    dummy_draft_kv_.clear();
    for (std::unique_ptr<DraftKvState> & state : slot_draft_kv_) {
        if (state) draft_kv_free(*state);
    }
    slot_draft_kv_.clear();
    for (DraftFeatureMirror & mirror : slot_feature_mirrors_) {
        draft_feature_mirror_free(mirror);
    }
    slot_feature_mirrors_.clear();
    if (feature_view_ctx_) {
        ggml_free(feature_view_ctx_);
        feature_view_ctx_ = nullptr;
    }
}

bool Qwen35SeqEngine::spec_gate_debug_enabled() const {
    const char * value = std::getenv("DFLASH_SPEC_GATE_LOG");
    return value && std::atoi(value) != 0;
}

bool Qwen35SeqEngine::step_timing_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("DFLASH_STEP_TIMING");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

bool Qwen35SeqEngine::profile_spec_costs(int context_tokens) {
    adaptive_fallback_reason_ = "cost_profile_unavailable";
    speculation_gate_.reset();
    spec_cohort_epoch_.reset();
    next_spec_cohort_epoch_id_ = 1;
    if (spec_mode_ != SpecMode::chain || !capture_features_ ||
        !activation_scoring_available() || tree_width_ <= 1 || tree_width_ > 16 ||
        resolve_chain_verify_depth(chain_verify_depth_, tree_width_) == 0 ||
        slots_.residency_active()) {
        std::fprintf(stderr,
            "[spec-profile] disabled: chain/features unavailable or "
            "concurrent KVFlash residency active; adaptive capability "
            "unavailable\n");
        return false;
    }

    const int n_slots = slots_.slot_count();
    const int T = tree_width_;
    const int V = chain_verify_depth_for_round();
    const int hidden = b_.w_.n_embd;
    const int n_head_kv = b_.w_.n_head_kv;
    const int max_profile_ctx = std::min(
        slots_.max_context() - T, b_.cache_.target_feat_cap);
    if (n_slots < 1 || max_profile_ctx < 1) return false;
    const int ctx_tokens = std::clamp(context_tokens, 1, max_profile_ctx);
    const SpecProfileGrid grid = build_spec_profile_grid(
        n_slots, V, V, [](int lanes) {
            return chain_decode_bucket_width(lanes);
        });
    const bool profile_batched = batched_drafting_enabled();

    std::ostringstream identity;
    identity << "qwen35-spec-cost-v1"
             << "|slots=" << n_slots
             << "|tree=" << T
             << "|verify=" << V
             << "|ctx=" << ctx_tokens
             << "|max_ctx=" << slots_.max_context()
             << "|kq_pad=" << b_.cfg_.kq_stride_pad
             << "|draft_mode=" << (profile_batched ? "batched" : "serial")
             << "|context_method=" << kSpecProfileContextMethod
             << "|commit_mode="
             << (chain_direct_commit_enabled() ? "direct" : "replay")
             << "|speculator=" << speculator_->score_kind()
             << "|target_device=" << placement_device_name(b_.cfg_.device)
             << "|draft_device=" << b_.cfg_.draft_gpu;
    if (ggml_backend_dev_t device =
            ggml_backend_get_device(b_.target_backend_)) {
        identity << "|device_name=" << ggml_backend_dev_name(device)
                 << "|device_description="
                 << ggml_backend_dev_description(device);
    }
    auto append_file_identity = [&](const char * label, const char * path) {
        identity << '|' << label << '=' << (path ? path : "");
        if (!path || !*path) return;
        std::error_code ec;
        const uintmax_t size = std::filesystem::file_size(path, ec);
        identity << ":size=" << (ec ? 0 : size);
        ec.clear();
        const auto modified = std::filesystem::last_write_time(path, ec);
        identity << ":mtime=" << (ec ? 0 : modified.time_since_epoch().count());
    };
    append_file_identity("target", b_.cfg_.target_path);
    append_file_identity("draft", b_.cfg_.draft_path);
    auto append_grid = [&](const char * label, const std::vector<int> & values) {
        identity << '|' << label << '=';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) identity << ',';
            identity << values[i];
        }
    };
    append_grid("tree_rows", grid.tree_rows);
    append_grid("step_rows", grid.step_rows);
    append_grid("draft_lanes", grid.draft_lanes);
    const std::string profile_identity = identity.str();
    const std::string profile_cache_path =
        spec_cost_profile_cache_path(profile_identity);

    auto install_profile = [&](const SpecCostTables & tables) {
        SpecStepGeometry geometry;
        geometry.tree_width = V;
        geometry.bucket = [](int lanes) {
            return chain_decode_bucket_width(lanes);
        };
        speculation_gate_ = std::make_unique<SpeculationGate>(
            tables, geometry, V,
            [](const char * table, int requested, int profiled) {
                std::fprintf(stderr,
                    "[spec-gate] %s_cost index %d outside profile; "
                    "clamped to %d\n",
                    table, requested, profiled);
            }, SpecGateConfig{}, chain_direct_commit_enabled());
        if (!speculation_gate_->valid()) {
            speculation_gate_.reset();
            return false;
        }
        adaptive_fallback_reason_.clear();
        return true;
    };

    SpecCostTables cached_tables;
    std::string cache_error;
    if (load_spec_cost_profile(
            profile_cache_path, profile_identity,
            cached_tables, cache_error) &&
        install_profile(cached_tables)) {
        std::fprintf(stderr,
            "[spec-profile] loaded %s context=%d context_method=%s "
            "mode=%s-draft speculator=%s\n",
            profile_cache_path.c_str(), ctx_tokens,
            kSpecProfileContextMethod,
            profile_batched ? "batched" : "serial",
            cached_tables.speculator_id.c_str());
        return true;
    }
    if (!profile_cache_path.empty() &&
        cache_error != "profile cache miss") {
        std::fprintf(stderr, "[spec-profile] cache ignored: %s\n",
                     cache_error.c_str());
    }

    std::string profile_error;
    std::vector<int> synthetic_slots;
    synthetic_slots.reserve((size_t)n_slots);
    auto cleanup = [&]() {
        for (int slot : synthetic_slots) {
            if (slot >= 0 && slot < (int)slot_draft_kv_.size() &&
                slot_draft_kv_[(size_t)slot]) {
                draft_kv_reset(*slot_draft_kv_[(size_t)slot]);
            }
            if (slots_.is_active(slot)) slots_.retire(slot);
            if (slot >= 0 && slot < (int)last_activation_estimate_.size()) {
                last_activation_estimate_[(size_t)slot] = {};
            }
        }
    };

    const int32_t profile_token = b_.w_.mask_token_id >= 0
        ? b_.w_.mask_token_id : 0;
    std::vector<int32_t> prompt((size_t)ctx_tokens, profile_token);
    const SamplerCfg greedy{};
    for (int lane = 0; lane < n_slots; ++lane) {
        AdmitResult admitted = admit(
            std::numeric_limits<uint64_t>::max() - (uint64_t)lane,
            prompt, greedy);
        if (admitted.status != AdmitResult::Status::admitted) {
            profile_error = admitted.error.empty()
                ? "synthetic slot admission failed" : admitted.error;
            cleanup();
            std::fprintf(stderr, "[spec-profile] %s\n", profile_error.c_str());
            return false;
        }
        synthetic_slots.push_back(admitted.slot);
        Qwen35SlotManager::PrefillChunk chunk =
            slots_.append_prefill(admitted.slot, ctx_tokens);
        if (!chunk.ok ||
            !upload_block_table_delta(admitted.slot, chunk.first_new_block,
                                      chunk.new_blocks.data(),
                                      chunk.new_blocks.size())) {
            profile_error = "synthetic paged context allocation failed";
            cleanup();
            std::fprintf(stderr, "[spec-profile] %s\n", profile_error.c_str());
            return false;
        }
        slots_.commit_prefill(admitted.slot);
    }
    for (ggml_tensor * tensor : b_.cache_.attn_k) {
        if (tensor) ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    for (ggml_tensor * tensor : b_.cache_.attn_v) {
        if (tensor) ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    if (b_.cache_.target_feat) {
        ggml_backend_tensor_memset(
            b_.cache_.target_feat, 0, 0, ggml_nbytes(b_.cache_.target_feat));
    }
    seq_lens_.assign((size_t)n_slots, ctx_tokens);
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());
    ggml_backend_synchronize(b_.target_backend_);

    int prepared_tree_rows = -1;
    auto tree_runner = [&](int total_rows) -> double {
        if (!profile_error.empty())
            return std::numeric_limits<double>::infinity();
        if (prepared_tree_rows != total_rows) {
            if (total_rows <= 0 || total_rows % V != 0) {
                profile_error = "invalid tree profiling shape";
                return std::numeric_limits<double>::infinity();
            }
            const int bucket = total_rows / V;
            const int live = std::min(bucket, n_slots);
            StepGraph & sg = b_.sg_;
            if (!build_target_step_paged_tree(
                    sg, b_.w_, b_.cache_, b_.target_backend_,
                    V, bucket, ctx_tokens,
                    tree_scratch_base_, tree_scratch_stride_,
                    b_.cfg_.kq_stride_pad)) {
                profile_error = "tree profiling graph build failed";
                return std::numeric_limits<double>::infinity();
            }

            std::vector<int32_t> tokens((size_t)total_rows, profile_token);
            std::vector<float> embeddings((size_t)hidden * total_rows);
            std::vector<int32_t> parents((size_t)total_rows, -1);
            std::vector<int32_t> sizes((size_t)bucket, 0);
            std::vector<int32_t> active((size_t)bucket, -1);
            std::vector<int32_t> state((size_t)bucket, 0);
            std::vector<int32_t> queries((size_t)total_rows, -1);
            std::vector<int32_t> positions((size_t)4 * total_rows, 0);
            std::vector<int64_t> rows(
                (size_t)n_head_kv * total_rows, scratch_row_);
            if (!b_.w_.embedder.embed(
                    tokens.data(), total_rows, embeddings.data())) {
                profile_error = "tree profiling embedding failed";
                return std::numeric_limits<double>::infinity();
            }
            for (int lane = 0; lane < live; ++lane) {
                const int slot = synthetic_slots[(size_t)lane];
                sizes[(size_t)lane] = V;
                active[(size_t)lane] = slot;
                state[(size_t)lane] = slot;
                for (int node = 0; node < V; ++node) {
                    const int row = lane * V + node;
                    parents[(size_t)row] = node == 0 ? -1 : node - 1;
                    queries[(size_t)row] = slot;
                    for (int axis = 0; axis < 3; ++axis) {
                        positions[(size_t)axis * total_rows + row] =
                            ctx_tokens + node;
                    }
                    for (int head = 0; head < n_head_kv; ++head) {
                        rows[(size_t)head * total_rows + row] =
                            (int64_t)tree_scratch_base_ +
                            (int64_t)slot * tree_scratch_stride_ + node;
                    }
                }
            }
            ggml_backend_tensor_set(sg.inp_embed, embeddings.data(), 0,
                                    sizeof(float) * embeddings.size());
            ggml_backend_tensor_set(sg.positions, positions.data(), 0,
                                    sizeof(int32_t) * positions.size());
            ggml_backend_tensor_set(sg.parent_ids, parents.data(), 0,
                                    sizeof(int32_t) * parents.size());
            ggml_backend_tensor_set(sg.tree_sizes, sizes.data(), 0,
                                    sizeof(int32_t) * sizes.size());
            if (detail::target_paged_tree_active_slots_need_upload(sg)) {
                ggml_backend_tensor_set(sg.active_slot_ids, active.data(), 0,
                                        sizeof(int32_t) * active.size());
            }
            ggml_backend_tensor_set(sg.state_slot_ids, state.data(), 0,
                                    sizeof(int32_t) * state.size());
            ggml_backend_tensor_set(sg.paged_query_seq_ids, queries.data(), 0,
                                    sizeof(int32_t) * queries.size());
            ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                                    sizeof(int64_t) * rows.size());
            prepared_tree_rows = total_rows;
        }
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(b_.target_backend_, b_.sg_.gf) !=
            GGML_STATUS_SUCCESS) {
            profile_error = "tree profiling compute failed";
            return std::numeric_limits<double>::infinity();
        }
        ggml_backend_synchronize(b_.target_backend_);
        // The serving path rebuilds this graph for every decode iteration.
        // Rebuild between samples too: replaying the same captured target
        // graph is not a supported lifecycle on the HIP graph backend.
        prepared_tree_rows = -1;
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };

    int prepared_step_rows = -1;
    auto step_runner = [&](int total_rows) -> double {
        if (!profile_error.empty())
            return std::numeric_limits<double>::infinity();
        if (prepared_step_rows != total_rows) {
            if (total_rows <= 0 || total_rows > n_slots * V) {
                profile_error = "invalid durable-step profiling shape";
                return std::numeric_limits<double>::infinity();
            }
            std::vector<QwenPrefillSegment> segments;
            int offset = 0;
            for (int lane = 0; offset < total_rows; ++lane) {
                const int length = std::min(V, total_rows - offset);
                segments.push_back({
                    offset, length, synthetic_slots[(size_t)lane]});
                offset += length;
            }
            StepGraph & sg = b_.sg_;
            if (!build_target_step(
                    sg, b_.w_, b_.cache_, b_.target_backend_,
                    0, total_rows, false, true, false, 0, 0,
                    b_.cfg_.kq_stride_pad, false, false, false, true,
                    1, 0, ctx_tokens + V,
                    total_rows, segments.data(), (int)segments.size(),
                    (int)segments.size(), false) ||
                !sg.kv_write_rows || !sg.target_feat_rows ||
                !sg.paged_query_seq_ids || !sg.paged_query_positions ||
                !sg.logits_row_indices) {
                profile_error = "durable-step profiling graph build failed";
                return std::numeric_limits<double>::infinity();
            }

            std::vector<int32_t> tokens((size_t)total_rows, profile_token);
            std::vector<float> embeddings((size_t)hidden * total_rows);
            std::vector<int32_t> positions((size_t)4 * total_rows, 0);
            std::vector<int64_t> rows(
                (size_t)n_head_kv * total_rows, scratch_row_);
            std::vector<int32_t> queries((size_t)total_rows, -1);
            std::vector<int32_t> feature_rows(
                (size_t)total_rows,
                b_.cache_.target_feat_cap * n_slots);
            std::vector<int32_t> logits_rows;
            logits_rows.reserve(segments.size());
            if (!b_.w_.embedder.embed(
                    tokens.data(), total_rows, embeddings.data())) {
                profile_error = "durable-step profiling embedding failed";
                return std::numeric_limits<double>::infinity();
            }
            for (const QwenPrefillSegment & segment : segments) {
                for (int j = 0; j < segment.n_tokens; ++j) {
                    const int row = segment.token_offset + j;
                    queries[(size_t)row] = segment.seq_slot;
                    for (int axis = 0; axis < 3; ++axis) {
                        positions[(size_t)axis * total_rows + row] =
                            ctx_tokens + j;
                    }
                    for (int head = 0; head < n_head_kv; ++head) {
                        rows[(size_t)head * total_rows + row] =
                            (int64_t)tree_scratch_base_ +
                            (int64_t)segment.seq_slot * tree_scratch_stride_ + j;
                    }
                }
                logits_rows.push_back(segment.token_offset + segment.n_tokens - 1);
            }
            ggml_backend_tensor_set(sg.inp_embed, embeddings.data(), 0,
                                    sizeof(float) * embeddings.size());
            ggml_backend_tensor_set(sg.positions, positions.data(), 0,
                                    sizeof(int32_t) * positions.size());
            ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                                    sizeof(int64_t) * rows.size());
            ggml_backend_tensor_set(sg.paged_query_seq_ids, queries.data(), 0,
                                    sizeof(int32_t) * queries.size());
            ggml_backend_tensor_set(sg.paged_query_positions,
                                    positions.data(), 0,
                                    sizeof(int32_t) * total_rows);
            ggml_backend_tensor_set(sg.target_feat_rows, feature_rows.data(), 0,
                                    sizeof(int32_t) * feature_rows.size());
            ggml_backend_tensor_set(sg.logits_row_indices, logits_rows.data(), 0,
                                    sizeof(int32_t) * logits_rows.size());
            prepared_step_rows = total_rows;
        }
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(b_.target_backend_, b_.sg_.gf) !=
            GGML_STATUS_SUCCESS) {
            profile_error = "durable-step profiling compute failed";
            return std::numeric_limits<double>::infinity();
        }
        ggml_backend_synchronize(b_.target_backend_);
        prepared_step_rows = -1;
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };

    auto draft_runner = [&](int lanes) -> double {
        if (!profile_error.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        std::vector<StepInput> profile_inputs;
        std::vector<uint8_t> selected;
        profile_inputs.reserve(static_cast<size_t>(lanes));
        selected.assign(static_cast<size_t>(lanes), 1);
        for (int lane = 0; lane < lanes; ++lane) {
            profile_inputs.push_back({
                synthetic_slots[static_cast<size_t>(lane)],
                profile_token,
                true,
                SpeculationPolicy::Always,
            });
        }
        const auto start = std::chrono::steady_clock::now();
        if (!prepare_chain_drafts(
                profile_inputs, selected,
                /*force_serial=*/!profile_batched,
                /*fail_fast_batch=*/profile_batched)) {
            profile_error = "draft profiling adapter proposal failed";
            return std::numeric_limits<double>::infinity();
        }
        for (const StepInput & input : profile_inputs) {
            const PreparedChainDraft & prepared =
                prepared_chain_drafts_[static_cast<size_t>(input.slot)];
            if (!prepared.valid ||
                static_cast<int>(prepared.tokens.size()) != T) {
                profile_error = "draft profiling proposal shape failed";
                return std::numeric_limits<double>::infinity();
            }
        }
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };

    SpecCostProfileResult profiled = SpecCostProfiler{}.profile(
        grid, tree_runner, step_runner, draft_runner,
        speculator_->score_kind(), 5);
    cleanup();
    if (!profiled.ok() || !profile_error.empty()) {
        std::fprintf(stderr, "[spec-profile] failed: %s%s\n",
            profile_error.c_str(), profiled.error.c_str());
        return false;
    }

    SpecCostTables tables = std::move(profiled.tables);
    if (!install_profile(tables)) return false;
    if (!save_spec_cost_profile(
            profile_cache_path, profile_identity, tables, cache_error)) {
        std::fprintf(stderr, "[spec-profile] cache save failed: %s\n",
                     cache_error.c_str());
    } else if (!profile_cache_path.empty()) {
        std::fprintf(stderr, "[spec-profile] saved %s\n",
                     profile_cache_path.c_str());
    }

    auto print_table = [](const char * name, const SpecCostSeries & series) {
        std::fprintf(stderr, "[spec-profile] %s", name);
        for (size_t i = 0; i < series.indices.size(); ++i) {
            std::fprintf(stderr, "%s%d:%.1fus",
                i == 0 ? " " : ",", series.indices[i], series.costs[i]);
        }
        std::fprintf(stderr, "\n");
    };
    std::fprintf(stderr,
        "[spec-profile] context=%d context_method=%s reps=5 mode=%s-draft "
        "speculator=%s\n",
        ctx_tokens,
        kSpecProfileContextMethod,
        profile_batched ? "batched" : "serial",
        tables.speculator_id.c_str());
    print_table("tree_cost", tables.tree_cost);
    print_table("step_cost", tables.step_cost);
    print_table("draft_cost", tables.draft_cost);
    return true;
}

DraftFeatureMirror * Qwen35SeqEngine::slot_feature_mirror(int slot) {
    if (!capture_features_ || slot < 0 ||
        slot >= (int)slot_feature_mirrors_.size()) {
        return nullptr;
    }
    return &slot_feature_mirrors_[(size_t)slot];
}

DraftKvState * Qwen35SeqEngine::ensure_slot_draft_kv(int slot) {
    DraftFeatureMirror * mirror = slot_feature_mirror(slot);
    if (!mirror || slot < 0 || slot >= (int)slot_draft_kv_.size()) {
        return nullptr;
    }
    std::unique_ptr<DraftKvState> & state = slot_draft_kv_[(size_t)slot];
    if (state && state->gf && state->built_for == (const void *)&b_.dw_) {
        return state.get();
    }
    if (state) draft_kv_free(*state);
    state = std::make_unique<DraftKvState>();
    const int cap = std::min(
        mirror->cap, std::max(1, b_.cfg_.draft_ctx_max));
    if (!draft_kv_init(*state, b_.dw_, b_.draft_backend_, cap, nullptr)) {
        draft_kv_free(*state);
        state.reset();
        return nullptr;
    }
    return state.get();
}
bool Qwen35SeqEngine::batched_drafting_enabled() const {
    const char * value = std::getenv("DFLASH_SPEC_BATCHED_DRAFT");
    return !value || std::atoi(value) != 0;
}

bool Qwen35SeqEngine::activation_scoring_available() const {
    return spec_mode_ == SpecMode::chain && capture_features_ &&
           speculator_is_ready(speculator_.get());
}

std::string Qwen35SeqEngine::chain_activation_score_kind() const {
    return speculator_is_ready(speculator_.get())
        ? speculator_->score_kind() : kUnspecifiedScoreKind;
}

bool Qwen35SeqEngine::activation_scoring_enabled() const {
    const char * value = std::getenv("DFLASH_SPEC_ACTIVATION_SCORE");
    return !value || std::atoi(value) != 0;
}

bool Qwen35SeqEngine::prepare_chain_drafts(
        const std::vector<StepInput> & inputs,
        const std::vector<uint8_t> & selected,
        bool force_serial,
        bool fail_fast_batch) {
    if (selected.size() != inputs.size() ||
        !speculator_is_ready(speculator_.get())) return false;

    // Accumulate the full drafting wall (draft graph compute + fused
    // selector + readbacks) into the round's [step-timing] attribution,
    // including early-failure paths.
    struct DraftTimer {
        Qwen35SeqEngine * engine;
        std::chrono::steady_clock::time_point start;
        int lanes;
        ~DraftTimer() {
            if (!engine) return;
            engine->round_draft_us_ +=
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count();
            engine->round_draft_lanes_ += lanes;
        }
    };
    DraftTimer draft_timer{
        this,
        std::chrono::steady_clock::now(),
        (int)std::count_if(
            selected.begin(), selected.end(),
            [](uint8_t value) { return value != 0; })};

    const int T = tree_width_;
    const int hidden = b_.w_.n_embd;
    const uint32_t requirements = speculator_->input_requirements();
    const bool need_prenorm =
        (requirements & SpeculatorInputPrenorm) != 0;
    struct Lane {
        size_t input_index = 0;
        int slot = -1;
        int32_t seed = -1;
        DraftKvState * state = nullptr;
        DraftFeatureMirror * mirror = nullptr;
    };
    std::vector<Lane> lanes;
    lanes.reserve(inputs.size());
    std::vector<int32_t> noise((size_t)T, b_.w_.mask_token_id);
    std::vector<float> noise_embed((size_t)hidden * T);

    // Proposal validity is current-block-specific. Do not clear the last
    // published activation score before the draft succeeds: bootstrap must either
    // publish a finite score or fail, and a later draft failure must not erase
    // the immutable activation score already owned by the gate.
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!selected[i]) continue;
        const int slot = inputs[i].slot;
        if (slot < 0 || slot >= (int)prepared_chain_drafts_.size()) continue;
        prepared_chain_drafts_[(size_t)slot].valid = false;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!selected[i]) continue;
        const StepInput & in = inputs[i];
        if (!chain_proposal_input_capable(in)) return false;
        DraftKvState * state = ensure_slot_draft_kv(in.slot);
        DraftFeatureMirror * mirror = slot_feature_mirror(in.slot);
        if (!state || !mirror ||
            !draft_kv_begin_step(
                *state, b_.dw_, b_.draft_backend_, *mirror,
                slots_.slot(in.slot).cur_pos)) {
            return false;
        }
        noise[0] = in.token;
        std::fill(
            noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
        if (!b_.w_.embedder.embed(
                noise.data(), T, noise_embed.data())) {
            return false;
        }
        ggml_backend_tensor_set(
            state->inp_embed, noise_embed.data(), 0,
            sizeof(float) * noise_embed.size());
        lanes.push_back({i, in.slot, in.token, state, mirror});
    }
    if (lanes.empty()) return true;

    std::vector<SpecProposal> proposals;
    auto invoke_adapter = [&](
            const std::vector<std::vector<float>> & hidden_blocks,
             const std::vector<std::vector<float>> & prenorm_blocks,
             const std::vector<int32_t> & seeds) {
        SpeculatorBatchInput adapter_input;
        adapter_input.lane_count = static_cast<int>(seeds.size());
        adapter_input.requested_depth = T;
        adapter_input.seed_tokens = seeds;
        for (const std::vector<float> & block : hidden_blocks) {
            adapter_input.hidden_by_lane.push_back(block.data());
        }
        if (need_prenorm) {
            for (const std::vector<float> & block : prenorm_blocks) {
                adapter_input.prenorm_by_lane.push_back(block.data());
            }
        }
        return speculator_input_satisfies(adapter_input, requirements) &&
               speculator_->propose(adapter_input, proposals);
    };
    bool used_batch = false;
    const bool try_batched = !force_serial && batched_drafting_enabled();
    if (try_batched) {
        const int bucket =
            chain_decode_bucket_width((int)lanes.size());
        std::vector<DraftKvState *> batch_states;
        std::vector<int32_t> seeds;
        batch_states.reserve((size_t)bucket);
        seeds.reserve((size_t)bucket);
        for (const Lane & lane : lanes) {
            batch_states.push_back(lane.state);
            seeds.push_back(lane.seed);
        }

        const int dummy_count = bucket - (int)lanes.size();
        const int cap = std::min(
            lanes[0].mirror->cap,
            std::max(1, b_.cfg_.draft_ctx_max));
        while ((int)dummy_draft_kv_.size() < dummy_count) {
            auto dummy = std::make_unique<DraftKvState>();
            if (!draft_kv_init(
                    *dummy, b_.dw_, b_.draft_backend_, cap, nullptr)) {
                draft_kv_free(*dummy);
                break;
            }
            dummy_draft_kv_.push_back(std::move(dummy));
        }
        if ((int)dummy_draft_kv_.size() >= dummy_count) {
            noise[0] = lanes[0].seed;
            std::fill(
                noise.begin() + 1, noise.end(),
                b_.w_.mask_token_id);
            bool dummy_ok = b_.w_.embedder.embed(
                noise.data(), T, noise_embed.data());
            for (int i = 0; dummy_ok && i < dummy_count; ++i) {
                DraftKvState * dummy =
                    dummy_draft_kv_[(size_t)i].get();
                dummy_ok = draft_kv_begin_step(
                    *dummy, b_.dw_, b_.draft_backend_,
                    *lanes[0].mirror, 1);
                if (dummy_ok) {
                    ggml_backend_tensor_set(
                        dummy->inp_embed, noise_embed.data(), 0,
                        sizeof(float) * noise_embed.size());
                    batch_states.push_back(dummy);
                    seeds.push_back(lanes[0].seed);
                }
            }
            if (dummy_ok) {
                std::vector<std::vector<float>> hidden_blocks;
                std::vector<std::vector<float>> prenorm_blocks;
                used_batch = draft_kv_batch_compute(
                    batch_draft_graph_, b_.dw_, b_.draft_backend_,
                    batch_states, need_prenorm,
                    hidden_blocks, prenorm_blocks) &&
                    invoke_adapter(hidden_blocks, prenorm_blocks, seeds) &&
                    proposals.size() >= lanes.size();
            }
        }
        if (!used_batch) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr,
                    "[draft-kv-batch] unavailable; using serial fallback\n");
            }
        }
    }

    if (!used_batch && try_batched && fail_fast_batch) return false;

    if (!used_batch && try_batched) {
        // A failed backend compute can leave a subset of the packed draft
        // graph's cache writes visible. Rebuild every real lane from its
        // captured target-feature ring before entering the serial fallback.
        for (const Lane & lane : lanes) {
            draft_kv_reset(*lane.state);
            if (!draft_kv_begin_step(
                    *lane.state, b_.dw_, b_.draft_backend_, *lane.mirror,
                    slots_.slot(lane.slot).cur_pos)) {
                return false;
            }
            noise[0] = lane.seed;
            std::fill(
                noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
            if (!b_.w_.embedder.embed(
                    noise.data(), T, noise_embed.data())) {
                return false;
            }
            ggml_backend_tensor_set(
                lane.state->inp_embed, noise_embed.data(), 0,
                sizeof(float) * noise_embed.size());
        }
    }

    if (!used_batch) {
        std::vector<std::vector<float>> hidden_blocks(
            lanes.size(),
            std::vector<float>(
                static_cast<size_t>(hidden) * static_cast<size_t>(T)));
        std::vector<std::vector<float>> prenorm_blocks;
        if (need_prenorm) {
            prenorm_blocks.assign(
                lanes.size(),
                std::vector<float>(
                    static_cast<size_t>(hidden) * static_cast<size_t>(T)));
        }
        std::vector<int32_t> seeds;
        seeds.reserve(lanes.size());
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            DraftKvState * state = lanes[lane].state;
            if (ggml_backend_graph_compute(
                    b_.draft_backend_, state->gf) !=
                GGML_STATUS_SUCCESS) {
                return false;
            }
            ggml_backend_tensor_get_async(
                b_.draft_backend_, state->hidden_states,
                hidden_blocks[lane].data(), 0,
                sizeof(float) * hidden_blocks[lane].size());
            if (need_prenorm) {
                ggml_backend_tensor_get_async(
                    b_.draft_backend_, state->hidden_prenorm,
                    prenorm_blocks[lane].data(), 0,
                    sizeof(float) * prenorm_blocks[lane].size());
            }
            seeds.push_back(lanes[lane].seed);
        }
        ggml_backend_synchronize(b_.draft_backend_);
        if (!invoke_adapter(hidden_blocks, prenorm_blocks, seeds) ||
            proposals.size() != lanes.size()) {
            return false;
        }
    }

    if (proposals.size() < lanes.size()) return false;
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
        SpecProposal & proposal = proposals[lane];
        if (!proposal.error.empty() ||
            static_cast<int>(proposal.tokens.size()) != T ||
            !std::isfinite(proposal.estimate.expected_yield) ||
            static_cast<int>(
                proposal.estimate.conditional_hazards.size()) < T - 1) {
            if (!proposal.error.empty()) {
                std::fprintf(stderr,
                    "[spec-gate] activation evaluation failed "
                    "request=%llu slot=%d kind=%s: %s\n",
                    static_cast<unsigned long long>(
                        slots_.slot(lanes[lane].slot).request_id),
                    lanes[lane].slot, speculator_->score_kind().c_str(),
                    proposal.error.c_str());
            }
            return false;
        }

        const Lane & info = lanes[lane];
        PreparedChainDraft & prepared =
            prepared_chain_drafts_[static_cast<size_t>(info.slot)];
        prepared.valid = true;
        prepared.generated = slots_.slot(info.slot).generated_tokens();
        prepared.root = info.seed;
        prepared.tokens = std::move(proposal.tokens);
        prepared.estimate = proposal.estimate;
        prepared.debug_depth_fields =
            std::move(proposal.debug_depth_fields);

        ActivationEstimate & published =
            last_activation_estimate_[static_cast<size_t>(info.slot)];
        if (!std::isfinite(published.expected_yield)) {
            published = std::move(proposal.estimate);
        }
    }
    return true;
}

bool Qwen35SeqEngine::ddtree_eligible(const StepPlan & plan) const {
    if (spec_mode_ != SpecMode::ddtree || tree_width_ <= 1 ||
        !capture_features_ || !plan.prefills.empty() ||
        plan.decode.empty() || b_.dw_.block_size <= 1 ||
        b_.cfg_.ddtree_budget + 1 != tree_width_) {
        return false;
    }
    const int min_floor = []() {
        const char * value = std::getenv("DFLASH_MIN_TOKENS");
        return value ? std::max(0, std::atoi(value)) : 0;
    }();
    for (const StepInput & in : plan.decode) {
        if (!in.allow_speculation ||
            in.speculation_policy == SpeculationPolicy::Never ||
            in.slot < 0 ||
            in.slot >= slots_.slot_count() ||
            !slots_.slot(in.slot).decoding() ||
            (in.speculation_policy != SpeculationPolicy::Always &&
             !slots_.ddtree_speculation_allowed(in.slot)) ||
            slots_.slot(in.slot).sampler.needs_logit_processing() ||
            slots_.slot(in.slot).cur_pos < 1 ||
            slots_.slot(in.slot).cur_pos >= slots_.max_context()) {
            return false;
        }
        const Qwen35Slot & seq = slots_.slot(in.slot);
        const int generated = seq.generated_tokens();
        if (generated < min_floor) return false;
    }
    return true;
}
bool Qwen35SeqEngine::chain_proposal_input_capable(
        const StepInput & in) const {
    const uint32_t supported_inputs =
        SpeculatorInputHidden | SpeculatorInputPrenorm;
    const bool adapter_capable =
        speculator_is_ready(speculator_.get()) &&
        speculator_->max_block_size() == tree_width_ &&
        (speculator_->input_requirements() & ~supported_inputs) == 0;
    return spec_mode_ == SpecMode::chain && capture_features_ &&
           tree_width_ > 1 && tree_width_ <= 16 &&
           resolve_chain_verify_depth(
               chain_verify_depth_for_round(), tree_width_) != 0 &&
           b_.dw_.block_size == tree_width_ && adapter_capable &&
           in.slot >= 0 && in.slot < slots_.slot_count() &&
           slots_.slot(in.slot).decoding() &&
           slots_.slot(in.slot).cur_pos >= 1 &&
           slots_.slot(in.slot).cur_pos < slots_.max_context();
}

bool Qwen35SeqEngine::chain_activation_input_scoreable(
        const StepInput & in) const {
    return chain_proposal_input_capable(in) &&
           activation_scoring_available();
}

bool Qwen35SeqEngine::chain_spec_request_capable(
        const StepInput & in) const {
    return chain_proposal_input_capable(in) &&
           in.allow_speculation &&
           in.speculation_policy != SpeculationPolicy::Never &&
           !slots_.slot(in.slot).sampler.needs_logit_processing();
}

bool Qwen35SeqEngine::chain_spec_input_eligible(
        const StepInput & in) const {
    return chain_spec_request_capable(in);
}
SeqEngine::StepResult Qwen35SeqEngine::step_chain_spec(
        const StepPlan & plan, const std::vector<uint8_t> & admitted,
        std::chrono::steady_clock::time_point round_started) {
    StepResult result;
    const std::vector<StepInput> & inputs = plan.decode;
    if (admitted.size() != inputs.size() || !plan.prefills.empty()) {
        result.error = "invalid chain speculation admission plan";
        return result;
    }

    const int T = tree_width_;
    const int V = chain_verify_depth_for_round();
    if (resolve_chain_verify_depth(V, T) == 0) {
        result.error =
            "invalid root-inclusive speculative chain verify depth";
        return result;
    }
    const int hidden = b_.w_.n_embd;
    const int n_head_kv = b_.w_.n_head_kv;
    const int n_slots = slots_.slot_count();
    const int min_tokens = []() {
        const char * value = std::getenv("DFLASH_MIN_TOKENS");
        return value ? std::max(0, std::atoi(value)) : 0;
    }();
    const int requested_spec_count = static_cast<int>(std::count_if(
        admitted.begin(), admitted.end(),
        [](uint8_t value) { return value != 0; }));
    if (requested_spec_count == 0) {
        result.error = "empty chain speculation admission plan";
        return result;
    }

    // Optional per-phase wall attribution ([step-timing]). Timestamps mark
    // phase boundaries; graph computes are synchronous on this backend, so
    // each *_exec span covers upload + compute up to its trailing sync.
    const bool timing = step_timing_enabled();
    using timing_clock = std::chrono::steady_clock;
    const auto t_round_start = round_started;
    timing_clock::time_point t_verify_build_start, t_verify_build_end,
        t_verify_exec_end, t_posterior_end, t_commit_end,
        t_replay_build_end, t_replay_exec_end, t_sample_end;
    auto span_us = [](timing_clock::time_point from,
                      timing_clock::time_point to) {
        return std::chrono::duration<double, std::micro>(to - from).count();
    };

    struct Proposal {
        size_t input_index = 0;
        int slot = -1;
        int32_t root = -1;
        DDTree tree;
        std::vector<int32_t> flat;
        std::vector<int> accepted;
        std::vector<int32_t> path;
        std::vector<std::string> debug_depth_fields;
        int32_t verify_bonus = -1;
        int32_t pending = -1;
    };
    struct ArLane {
        size_t input_index = 0;
        int slot = -1;
        int32_t token = -1;
        int position = -1;
        int64_t physical_row = -1;
        int32_t pending = -1;
    };

    std::vector<Proposal> proposals;
    proposals.reserve(static_cast<size_t>(requested_spec_count));
    std::vector<int> proposal_for_input(inputs.size(), -1);
    std::vector<uint8_t> active_admitted = admitted;
    std::vector<uint8_t> retried(inputs.size(), 0);
    std::vector<std::string> proposal_errors(inputs.size());

    auto clean_proposal_lane = [&](size_t i) {
        const int slot = inputs[i].slot;
        if (slot >= 0 &&
            slot < static_cast<int>(prepared_chain_drafts_.size())) {
            prepared_chain_drafts_[static_cast<size_t>(slot)].valid = false;
        }
        if (slot >= 0 && slot < static_cast<int>(slot_draft_kv_.size()) &&
            slot_draft_kv_[static_cast<size_t>(slot)]) {
            draft_kv_reset(*slot_draft_kv_[static_cast<size_t>(slot)]);
        }
    };
    auto fail_proposal_lane = [&](size_t i, const char * error) {
        clean_proposal_lane(i);
        active_admitted[i] = 0;
        proposal_errors[i] = error;
        std::fprintf(stderr,
            "[spec-proposal-failure] request_id=%llu slot=%d error=%s\n",
            (unsigned long long)slots_.slot(inputs[i].slot).request_id,
            inputs[i].slot, error);
    };
    auto retry_proposal_lane = [&](size_t i) {
        retried[i] = 1;
        clean_proposal_lane(i);
        std::vector<uint8_t> selected(inputs.size(), 0);
        selected[i] = 1;
        if (prepare_chain_drafts(inputs, selected, /*force_serial=*/true)) {
            return true;
        }
        fail_proposal_lane(
            i, "chain proposal preparation failed after clean retry");
        return false;
    };

    std::vector<uint8_t> need_prepare(inputs.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const StepInput & in = inputs[i];
        const bool hard_eligible = chain_spec_input_eligible(in);
        if (!admitted[i]) continue;
        if (!hard_eligible) {
            fail_proposal_lane(
                i, "epoch-selected speculation request became ineligible");
            continue;
        }
        const PreparedChainDraft & prepared =
            prepared_chain_drafts_[(size_t)in.slot];
        const Qwen35Slot & seq = slots_.slot(in.slot);
        need_prepare[i] =
            !prepared.valid ||
            prepared.generated != seq.generated_tokens() ||
            prepared.root != in.token ||
            (int)prepared.tokens.size() != T;
    }
    if (std::any_of(
            need_prepare.begin(), need_prepare.end(),
            [](uint8_t value) { return value != 0; }) &&
        !prepare_chain_drafts(
            inputs, need_prepare, /*force_serial=*/false,
            /*fail_fast_batch=*/true)) {
        // The packed prepare has no target-side effects. Reset its drafter
        // state, then retry each affected request once through the serial path
        // so one broken lane cannot fail or demote healthy peers.
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (need_prepare[i]) clean_proposal_lane(i);
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (need_prepare[i] && active_admitted[i]) {
                retry_proposal_lane(i);
            }
        }
    }

    auto take_prepared_proposal = [&](size_t i, Proposal & proposal) {
        const StepInput & in = inputs[i];
        PreparedChainDraft & prepared =
            prepared_chain_drafts_[(size_t)in.slot];
        if (!prepared.valid ||
            prepared.generated != slots_.slot(in.slot).generated_tokens() ||
            prepared.root != in.token ||
            (int)prepared.tokens.size() != T) {
            return false;
        }

        Proposal next;
        next.input_index = i;
        next.slot = in.slot;
        next.root = in.token;
        next.debug_depth_fields =
            std::move(prepared.debug_depth_fields);
        next.flat = std::move(prepared.tokens);
        prepared.valid = false;
        if (!truncate_chain_proposal(next.flat, V)) return false;
        const size_t verified_signal_depths =
            static_cast<size_t>(V - 1);
        if (next.debug_depth_fields.size() > verified_signal_depths) {
            next.debug_depth_fields.resize(verified_signal_depths);
        }
        next.tree = make_chain_verify_tree(next.flat);
        if (next.tree.n_nodes + 1 != V) return false;
        proposal = std::move(next);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!active_admitted[i]) continue;
        Proposal proposal;
        if (!take_prepared_proposal(i, proposal)) {
            if (!retried[i] && retry_proposal_lane(i) &&
                take_prepared_proposal(i, proposal)) {
                // The clean retry repaired a stale or malformed proposal.
            } else {
                if (active_admitted[i]) {
                    fail_proposal_lane(
                        i, "chain proposal remained invalid after clean retry");
                }
                continue;
            }
        }
        proposal_for_input[i] =
            static_cast<int>(proposals.size());
        proposals.push_back(std::move(proposal));
    }
    const int spec_count = static_cast<int>(proposals.size());
    const bool direct_commit = chain_direct_commit_enabled();
    auto lane_disposition = [&](size_t i) {
        return chain_lane_disposition(
            active_admitted[i] != 0, !proposal_errors[i].empty());
    };

    std::vector<ArLane> ar_lanes;
    ar_lanes.reserve(inputs.size() - static_cast<size_t>(spec_count));
    std::vector<int> ar_for_input(inputs.size(), -1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (lane_disposition(i) != ChainLaneDisposition::AR) continue;
        ArLane lane;
        lane.input_index = i;
        lane.slot = inputs[i].slot;
        lane.token = inputs[i].token;
        lane.position = slots_.slot(lane.slot).cur_pos;
        ar_for_input[i] = static_cast<int>(ar_lanes.size());
        ar_lanes.push_back(lane);
    }
    const int ar_count = static_cast<int>(ar_lanes.size());
    const int tree_lane_count = spec_count;
    const int tree_bucket = tree_lane_count > 0
        ? chain_decode_bucket_width(tree_lane_count) : 0;

    // Compact direct graphs write AR K/V and recurrent state durably in the
    // fused launch. Allocate their physical rows now, but keep history and
    // cur_pos staged until the target graph and all promotions succeed.
    if (direct_commit) {
        for (ArLane & lane : ar_lanes) {
            const Qwen35SlotManager::StepAppend app =
                slots_.append_token(lane.slot, lane.token);
            if (!app.ok) {
                result.error = app.busy
                    ? "paged KV pool exhausted during compact AR staging"
                    : "compact AR K/V staging failed";
                return result;
            }
            const bool table_ok = slots_.residency_active() ||
                app.new_block < 0 ||
                upload_block_table_delta(
                    lane.slot, app.new_block_index, &app.new_block, 1);
            if (!table_ok) {
                result.error = "compact AR block-table update failed";
                return result;
            }
            lane.position = app.position;
            lane.physical_row = app.physical_row;
        }
    }
    StepGraph & tree_sg = b_.sg_;
    std::vector<int32_t> posterior;

    int replay_total = 0;
    if (tree_lane_count > 0) {
    // Launch 1: scratch-only packed path-tree verification.
    int max_prefix = 1;
    for (const Proposal & proposal : proposals) {
        max_prefix = std::max(
            max_prefix, slots_.slot(proposal.slot).cur_pos);
    }
    if (direct_commit) {
        for (const ArLane & ar : ar_lanes) {
            max_prefix = std::max(max_prefix, ar.position + 1);
        }
    }
    t_verify_build_start = timing_clock::now();
    if (!build_target_step_paged_tree(
            tree_sg, b_.w_, b_.cache_, b_.target_backend_,
            V, tree_bucket, max_prefix,
            tree_scratch_base_, tree_scratch_stride_,
            b_.cfg_.kq_stride_pad, direct_commit ? ar_count : 0,
            direct_commit)) {
        result.error = "packed chain speculation verify graph build failed";
        return result;
    }
    t_verify_build_end = timing_clock::now();

    const int spec_tree_rows = V * tree_bucket;
    const int spec_row_offset = direct_commit ? ar_count : 0;
    const int total_tree = spec_row_offset + spec_tree_rows;
    std::vector<int32_t> flat_tokens(static_cast<size_t>(total_tree), 0);
    std::vector<int32_t> parents(
        static_cast<size_t>(spec_tree_rows), -1);
    std::vector<int32_t> sizes(static_cast<size_t>(tree_bucket), 0);
    const int mapped_slot_count = spec_row_offset + tree_bucket;
    std::vector<int32_t> tree_slots(
        static_cast<size_t>(mapped_slot_count), -1);
    std::vector<int32_t> tree_state_slots(
        static_cast<size_t>(mapped_slot_count), 0);
    std::vector<int32_t> query_slots(static_cast<size_t>(total_tree), -1);
    std::vector<int32_t> query_positions(
        direct_commit ? static_cast<size_t>(total_tree) : 0, -1);
    std::vector<int64_t> tree_rows(
        static_cast<size_t>(total_tree) * n_head_kv, scratch_row_);
    std::vector<int32_t> tree_positions(
        static_cast<size_t>(4) * total_tree, 0);
    std::vector<float> tree_embed(
        static_cast<size_t>(hidden) * total_tree, 0.0f);
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);

    if (direct_commit) {
        for (int ar_index = 0; ar_index < ar_count; ++ar_index) {
            const ArLane & ar = ar_lanes[static_cast<size_t>(ar_index)];
            const int row = ar_index;
            tree_slots[static_cast<size_t>(ar_index)] = ar.slot;
            tree_state_slots[static_cast<size_t>(ar_index)] = ar.slot;
            seq_lens_[static_cast<size_t>(ar.slot)] = ar.position + 1;
            flat_tokens[static_cast<size_t>(row)] = ar.token;
            query_slots[static_cast<size_t>(row)] = ar.slot;
            query_positions[static_cast<size_t>(row)] = ar.position;
            tree_positions[static_cast<size_t>(0) * total_tree + row] =
                ar.position;
            tree_positions[static_cast<size_t>(1) * total_tree + row] =
                ar.position;
            tree_positions[static_cast<size_t>(2) * total_tree + row] =
                ar.position;
            for (int head = 0; head < n_head_kv; ++head) {
                tree_rows[static_cast<size_t>(head) * total_tree + row] =
                    ar.physical_row;
            }
        }
    }

    for (int lane = 0; lane < spec_count; ++lane) {
        const Proposal & proposal = proposals[static_cast<size_t>(lane)];
        const int tree_base = lane * V;
        const int row_base = spec_row_offset + tree_base;
        const int mapped_lane = spec_row_offset + lane;
        sizes[static_cast<size_t>(lane)] = V;
        tree_slots[static_cast<size_t>(mapped_lane)] = proposal.slot;
        tree_state_slots[static_cast<size_t>(mapped_lane)] = proposal.slot;
        seq_lens_[static_cast<size_t>(proposal.slot)] =
            slots_.slot(proposal.slot).cur_pos;
        for (int node = 0; node < V; ++node) {
            const int tree_row = tree_base + node;
            const int row = row_base + node;
            flat_tokens[static_cast<size_t>(row)] =
                proposal.flat[static_cast<size_t>(node)];
            parents[static_cast<size_t>(tree_row)] = node == 0
                ? -1 : proposal.tree.parents[static_cast<size_t>(node)];
            query_slots[static_cast<size_t>(row)] = proposal.slot;
            const int depth = node == 0
                ? 0 : proposal.tree.depths[static_cast<size_t>(node) - 1];
            const int position =
                slots_.slot(proposal.slot).cur_pos + depth;
            tree_positions[static_cast<size_t>(0) * total_tree + row] =
                position;
            tree_positions[static_cast<size_t>(1) * total_tree + row] =
                position;
            tree_positions[static_cast<size_t>(2) * total_tree + row] =
                position;
            for (int head = 0; head < n_head_kv; ++head) {
                tree_rows[static_cast<size_t>(head) * total_tree + row] =
                    static_cast<int64_t>(tree_scratch_base_) +
                    static_cast<int64_t>(proposal.slot) *
                        tree_scratch_stride_ + node;
            }
        }
    }

    if (!b_.w_.embedder.embed(
            flat_tokens.data(), total_tree, tree_embed.data())) {
        result.error = "packed chain speculation embedding failed";
        return result;
    }
    ggml_backend_tensor_set(tree_sg.inp_embed, tree_embed.data(), 0,
                            sizeof(float) * tree_embed.size());
    ggml_backend_tensor_set(tree_sg.positions, tree_positions.data(), 0,
                            sizeof(int32_t) * tree_positions.size());
    ggml_backend_tensor_set(tree_sg.parent_ids, parents.data(), 0,
                            sizeof(int32_t) * parents.size());
    ggml_backend_tensor_set(tree_sg.tree_sizes, sizes.data(), 0,
                            sizeof(int32_t) * sizes.size());
    if (detail::target_paged_tree_active_slots_need_upload(tree_sg)) {
        ggml_backend_tensor_set(
            tree_sg.active_slot_ids, tree_slots.data(), 0,
            sizeof(int32_t) * tree_slots.size());
    }
    ggml_backend_tensor_set(
        tree_sg.state_slot_ids, tree_state_slots.data(), 0,
        sizeof(int32_t) * tree_state_slots.size());
    ggml_backend_tensor_set(
        tree_sg.paged_query_seq_ids, query_slots.data(), 0,
        sizeof(int32_t) * query_slots.size());
    if (tree_sg.paged_query_positions) {
        ggml_backend_tensor_set(
            tree_sg.paged_query_positions, query_positions.data(), 0,
            sizeof(int32_t) * query_positions.size());
    }
    ggml_backend_tensor_set(tree_sg.kv_write_rows, tree_rows.data(), 0,
                            sizeof(int64_t) * tree_rows.size());
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());
    if (ggml_backend_graph_compute(b_.target_backend_, tree_sg.gf) !=
        GGML_STATUS_SUCCESS) {
        result.error = "packed chain speculation verify compute failed";
        return result;
    }
    t_verify_exec_end = timing_clock::now();

    posterior.assign(static_cast<size_t>(total_tree), -1);
    ggml_backend_tensor_get(
        tree_sg.argmax_tokens, posterior.data(), 0,
        sizeof(int32_t) * posterior.size());
    t_posterior_end = timing_clock::now();

    for (int lane = 0; lane < spec_count; ++lane) {
        Proposal & proposal = proposals[static_cast<size_t>(lane)];
        const int32_t * lane_posterior =
            posterior.data() + static_cast<size_t>(spec_row_offset + lane * V);
        proposal.accepted = follow_verified_tree(
            proposal.tree, lane_posterior, proposal.verify_bonus);
        const int room =
            slots_.max_context() - slots_.slot(proposal.slot).cur_pos;
        truncate_verified_path(
            proposal.accepted, static_cast<size_t>(std::max(0, room)),
            lane_posterior, proposal.verify_bonus);
        if (proposal.accepted.empty()) {
            result.error = "chain accepted path has no context headroom";
            return result;
        }
        proposal.path.reserve(proposal.accepted.size());
        for (int flat_index : proposal.accepted) {
            proposal.path.push_back(flat_index == 0
                ? proposal.root
                : proposal.tree.token_ids[
                    static_cast<size_t>(flat_index) - 1]);
        }
        const size_t safe_prefix = chain_min_tokens_safe_prefix(
            proposal.path,
            slots_.slot(proposal.slot).generated_tokens(),
            min_tokens,
            [&](int32_t token) { return token_is_eos(token); });
        proposal.path.resize(safe_prefix);
        proposal.accepted.resize(safe_prefix);
        if (!proposal.debug_depth_fields.empty()) {
            static const bool selector_log_enabled = []() {
                const char * value =
                    std::getenv("DFLASH_DFLASH2_SELECTOR_LOG");
                return value && std::atoi(value) != 0;
            }();
            if (selector_log_enabled) {
                const Qwen35Slot & sequence = slots_.slot(proposal.slot);
                const size_t accepted_depth =
                    proposal.path.empty() ? 0 : proposal.path.size() - 1;
                std::fprintf(stderr,
                    "[spec-selector] {\"request_id\":%llu,\"slot\":%d,"
                    "\"score_kind\":\"%s\",\"generated\":%d,"
                    "\"accepted_depth\":%zu,\"depths\":[",
                    static_cast<unsigned long long>(sequence.request_id),
                    proposal.slot, chain_activation_score_kind().c_str(),
                    sequence.generated_tokens(), accepted_depth);
                for (size_t depth = 0;
                     depth < proposal.debug_depth_fields.size(); ++depth) {
                    std::fprintf(stderr,
                        "%s{\"depth\":%zu,\"accepted\":%s,%s}",
                        depth == 0 ? "" : ",", depth + 1,
                        depth < accepted_depth ? "true" : "false",
                        proposal.debug_depth_fields[depth].c_str());
                }
                std::fprintf(stderr, "]}\n");
            }
        }
        replay_total += static_cast<int>(proposal.path.size());
    }
    } else {
        const auto no_verify = timing_clock::now();
        t_verify_build_start = no_verify;
        t_verify_build_end = no_verify;
        t_verify_exec_end = no_verify;
        t_posterior_end = no_verify;
    }

    // Stage accepted path segments and all non-admitted AR peers. Nothing is
    // published to slot history until the combined durable graph succeeds.
    std::vector<QwenPrefillSegment> replay_segments;
    std::vector<int32_t> replay_tokens;
    std::vector<int32_t> replay_slots;
    std::vector<int32_t> replay_positions;
    std::vector<int64_t> replay_physical;
    replay_segments.reserve(static_cast<size_t>(spec_count));
    replay_tokens.reserve(static_cast<size_t>(replay_total));
    replay_slots.reserve(static_cast<size_t>(replay_total));
    replay_positions.reserve(static_cast<size_t>(replay_total));
    replay_physical.reserve(static_cast<size_t>(replay_total));
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);
    int max_kv_len = 1;
    int replay_offset = 0;

    for (Proposal & proposal : proposals) {
        const Qwen35SlotManager::StepAppend app = slots_.append_tokens(
            proposal.slot, proposal.path.data(),
            static_cast<int>(proposal.path.size()));
        const bool table_ok = slots_.residency_active() ||
            upload_block_table_delta(
                proposal.slot, app.first_new_block,
                app.new_blocks.data(), app.new_blocks.size());
        if (!app.ok || app.physical_rows.size() != proposal.path.size() ||
            !table_ok) {
            result.error = app.busy
                ? "paged KV pool exhausted during chain speculation commit"
                : "chain accepted-path K/V append failed";
            return result;
        }
        replay_segments.push_back({
            replay_offset, static_cast<int>(proposal.path.size()),
            proposal.slot,
        });
        for (size_t row = 0; row < proposal.path.size(); ++row) {
            replay_tokens.push_back(proposal.path[row]);
            replay_slots.push_back(proposal.slot);
            replay_positions.push_back(app.position + static_cast<int>(row));
            replay_physical.push_back(app.physical_rows[row]);
        }
        replay_offset += static_cast<int>(proposal.path.size());
        const int seq_len =
            app.position + static_cast<int>(proposal.path.size());
        seq_lens_[static_cast<size_t>(proposal.slot)] = seq_len;
        max_kv_len = std::max(max_kv_len, seq_len);
    }

    for (int ar_index = 0; ar_index < ar_count; ++ar_index) {
        ArLane & lane = ar_lanes[static_cast<size_t>(ar_index)];
        const StepInput & in = inputs[lane.input_index];
        if (!direct_commit) {
            const Qwen35SlotManager::StepAppend app =
                slots_.append_token(in.slot, in.token);
            if (!app.ok) {
                result.error = app.busy
                    ? "paged KV pool exhausted during mixed AR commit"
                    : "mixed AR K/V append failed";
                return result;
            }
            const bool table_ok = slots_.residency_active() ||
                app.new_block < 0 ||
                upload_block_table_delta(
                    in.slot, app.new_block_index, &app.new_block, 1);
            if (!table_ok) {
                result.error = "mixed AR block-table update failed";
                return result;
            }
            lane.position = app.position;
            lane.physical_row = app.physical_row;
        }
        seq_lens_[static_cast<size_t>(in.slot)] = lane.position + 1;
        max_kv_len = std::max(max_kv_len, lane.position + 1);
    }
    if (spec_count == 0 && ar_count == 0) {
        result.decode.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            DecodeOutput out;
            out.slot = inputs[i].slot;
            out.failed = true;
            out.error = proposal_errors[i].empty()
                ? "chain proposal lane made no progress"
                : proposal_errors[i];
            result.decode.push_back(std::move(out));
        }
        return result;
    }
    if (!upload_all_active_block_tables()) {
        result.error = "chain mixed-step block-table refresh failed";
        return result;
    }
    t_commit_end = timing_clock::now();

    if (direct_commit) {
        const int spec_tree_rows = V * tree_bucket;
        const int spec_row_offset = ar_count;
        const int total_tree = spec_row_offset + spec_tree_rows;
        std::vector<int32_t> accepted_prefixes(
            static_cast<size_t>(tree_bucket), 0);
        std::vector<int64_t> commit_rows(
            static_cast<size_t>(spec_tree_rows), -1);
        std::vector<int32_t> feature_commit_rows(
            static_cast<size_t>(total_tree), -1);
        const int feature_cap = b_.cache_.target_feat_cap;
        int replay_cursor = 0;
        for (int lane = 0; lane < spec_count; ++lane) {
            const Proposal & proposal = proposals[static_cast<size_t>(lane)];
            if (proposal.path.size() != proposal.accepted.size()) {
                result.error = "direct commit path/acceptance size mismatch";
                return result;
            }
            accepted_prefixes[static_cast<size_t>(lane)] =
                static_cast<int32_t>(proposal.path.size());
            for (size_t depth = 0; depth < proposal.accepted.size(); ++depth) {
                const int node = proposal.accepted[depth];
                // DFlash2 proposals are chains. A branching tree needs an
                // indexed journal-commit kernel rather than prefix commit.
                if (node != static_cast<int>(depth)) {
                    result.error =
                        "direct commit requires a contiguous chain acceptance";
                    return result;
                }
                const int flat = lane * V + node;
                const int source_row = spec_row_offset + flat;
                if (replay_cursor >= static_cast<int>(replay_physical.size())) {
                    result.error = "direct commit replay cursor overflow";
                    return result;
                }
                commit_rows[static_cast<size_t>(flat)] =
                    replay_physical[static_cast<size_t>(replay_cursor)];
                feature_commit_rows[static_cast<size_t>(source_row)] =
                    proposal.slot * feature_cap +
                    replay_positions[static_cast<size_t>(replay_cursor)] %
                        feature_cap;
                ++replay_cursor;
            }
        }
        if (replay_cursor != replay_total) {
            result.error = "direct commit replay cursor mismatch";
            return result;
        }
        for (int ar_index = 0; ar_index < ar_count; ++ar_index) {
            const ArLane & ar = ar_lanes[static_cast<size_t>(ar_index)];
            feature_commit_rows[static_cast<size_t>(ar_index)] =
                ar.slot * feature_cap + ar.position % feature_cap;
        }
        std::vector<int32_t> commit_slots(
            static_cast<size_t>(tree_bucket), -1);
        for (int lane = 0; lane < spec_count; ++lane) {
            commit_slots[static_cast<size_t>(lane)] =
                proposals[static_cast<size_t>(lane)].slot;
        }
        ggml_backend_tensor_set(
            tree_sg.accepted_prefixes, accepted_prefixes.data(), 0,
            accepted_prefixes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            tree_sg.commit_slot_ids, commit_slots.data(), 0,
            commit_slots.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            tree_sg.commit_rows, commit_rows.data(), 0,
            commit_rows.size() * sizeof(int64_t));
        ggml_backend_tensor_set(
            tree_sg.feature_commit_rows, feature_commit_rows.data(), 0,
            feature_commit_rows.size() * sizeof(int32_t));

        const size_t n_delta = b_.cache_.ssm_state.size();
        if (tree_sg.delta_captures.size() != n_delta ||
            b_.cache_.conv_state.size() != n_delta ||
            !tree_sg.tree_features || !b_.cache_.target_feat) {
            result.error = "direct commit capture set incomplete";
            return result;
        }
        std::vector<const ggml_tensor *> journals;
        std::vector<ggml_tensor *> states;
        std::vector<const ggml_tensor *> conv_inputs;
        std::vector<ggml_tensor *> conv_states;
        journals.reserve(n_delta);
        states.reserve(n_delta);
        conv_inputs.reserve(n_delta);
        conv_states.reserve(n_delta);
        for (size_t layer = 0; layer < n_delta; ++layer) {
            const DeltaNetCapture & capture =
                tree_sg.delta_captures[layer];
            if (!capture.transition_journal || !capture.conv_input ||
                !b_.cache_.ssm_state[layer] ||
                !b_.cache_.conv_state[layer]) {
                result.error = "direct commit layer capture incomplete";
                return result;
            }
            journals.push_back(capture.transition_journal);
            states.push_back(b_.cache_.ssm_state[layer]);
            conv_inputs.push_back(capture.conv_input);
            conv_states.push_back(b_.cache_.conv_state[layer]);
        }
        std::vector<ggml_tensor *> cache_tensors;
        cache_tensors.reserve(
            b_.cache_.attn_k.size() + b_.cache_.attn_v.size());
        for (ggml_tensor * tensor : b_.cache_.attn_k) {
            if (tensor) cache_tensors.push_back(tensor);
        }
        for (ggml_tensor * tensor : b_.cache_.attn_v) {
            if (tensor) cache_tensors.push_back(tensor);
        }
        if (cache_tensors.empty()) {
            result.error = "direct commit K/V cache set empty";
            return result;
        }

        t_replay_build_end = timing_clock::now();
        if (!ggml_backend_cuda_tree_cache_commit_many(
                cache_tensors.data(),
                static_cast<int>(cache_tensors.size()),
                tree_sg.commit_rows, tree_sg.commit_slot_ids,
                tree_scratch_base_, tree_scratch_stride_)) {
            result.error = "direct commit K/V promotion failed";
            return result;
        }
        if (!ggml_backend_cuda_tree_feature_commit(
                tree_sg.tree_features, b_.cache_.target_feat,
                tree_sg.feature_commit_rows)) {
            result.error = "direct commit target-feature promotion failed";
            return result;
        }
        if (!ggml_backend_cuda_gdn_transition_journal_commit_many(
                journals.data(), states.data(), conv_inputs.data(),
                conv_states.data(), static_cast<int>(n_delta),
                tree_sg.accepted_prefixes, tree_sg.commit_slot_ids)) {
            result.error = "direct commit recurrent-state promotion failed";
            return result;
        }
        ggml_backend_tensor_set(
            b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
            sizeof(int32_t) * seq_lens_.size());
        t_replay_exec_end = timing_clock::now();

        std::vector<int> write_slots;
        write_slots.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (chain_lane_executes(lane_disposition(i))) {
                write_slots.push_back(inputs[i].slot);
            }
        }
        if (!commit_residency_writes(write_slots)) {
            result.error = "direct commit residency write failed";
            return result;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (chain_lane_executes(lane_disposition(i))) {
                slots_.commit_step(inputs[i].slot);
            }
        }
        for (int lane = 0; lane < spec_count; ++lane) {
            Proposal & proposal = proposals[static_cast<size_t>(lane)];
            const int graph_row = spec_row_offset + lane * V +
                proposal.accepted.back();
            proposal.pending = sample_graph_row(
                proposal.slot, graph_row,
                &posterior[static_cast<size_t>(graph_row)], &logits_buf_);
            if (proposal.pending < 0) {
                result.error = "direct commit speculative sampling failed";
                return result;
            }
        }
        for (int ar_index = 0; ar_index < ar_count; ++ar_index) {
            ArLane & ar = ar_lanes[static_cast<size_t>(ar_index)];
            const int graph_row = ar_index;
            ar.pending = sample_graph_row(
                ar.slot, graph_row,
                &posterior[static_cast<size_t>(graph_row)], &logits_buf_);
            if (ar.pending < 0) {
                result.error = "direct commit AR sampling failed";
                return result;
            }
        }
        t_sample_end = timing_clock::now();

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!chain_lane_executes(lane_disposition(i))) continue;
            std::string reselect_error;
            if (!maybe_reselect_residency(inputs[i].slot, reselect_error)) {
                result.error = reselect_error.empty()
                    ? "KVFlash reselect failed" : reselect_error;
                return result;
            }
        }
        result.decode.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            DecodeOutput out;
            out.slot = inputs[i].slot;
            const ChainLaneDisposition disposition = lane_disposition(i);
            if (disposition == ChainLaneDisposition::Failed) {
                out.failed = true;
                out.error = proposal_errors[i];
                result.decode.push_back(std::move(out));
                continue;
            }
            if (disposition == ChainLaneDisposition::Speculation) {
                Proposal & proposal =
                    proposals[static_cast<size_t>(proposal_for_input[i])];
                out.token = proposal.pending;
                out.spec_steps = 1;
                out.spec_accepted_tokens = proposal.path.size() > 1
                    ? static_cast<uint64_t>(proposal.path.size() - 1) : 0;
                out.target_forwards = 1;
                out.committed_tokens.assign(
                    proposal.path.begin() + 1, proposal.path.end());
            } else {
                ArLane & ar =
                    ar_lanes[static_cast<size_t>(ar_for_input[i])];
                out.token = ar.pending;
                out.target_forwards = 1;
            }
            attach_residency_telemetry(out);
            result.decode.push_back(std::move(out));
        }
        if (timing) {
            const auto t_round_end = timing_clock::now();
            std::fprintf(stderr,
                "[step-timing] {\"path\":\"spec-direct\",\"live\":%d,"
                "\"k\":%d,\"tree_bucket\":%d,\"tree_rows\":%d,"
                "\"replay_rows\":0,\"ar_lanes\":%d,\"ar_bucket\":0,"
                "\"max_kv_len\":%d,\"draft_us\":%.1f,"
                "\"draft_lanes\":%d,\"pre_us\":%.1f,"
                "\"verify_build_us\":%.1f,\"verify_exec_us\":%.1f,"
                "\"posterior_read_us\":%.1f,\"commit_cpu_us\":%.1f,"
                "\"replay_build_us\":%.1f,\"replay_exec_us\":%.1f,"
                "\"sample_read_us\":%.1f,\"finish_us\":%.1f,"
                "\"total_us\":%.1f,\"accepted_tokens\":%d,"
                "\"emitted_tokens\":%d,\"target_forwards\":%d}\n",
                spec_count + ar_count, spec_count, tree_bucket,
                total_tree, ar_count, max_kv_len,
                round_draft_us_, round_draft_lanes_,
                span_us(t_round_start, t_verify_build_start),
                span_us(t_verify_build_start, t_verify_build_end),
                span_us(t_verify_build_end, t_verify_exec_end),
                span_us(t_verify_exec_end, t_posterior_end),
                span_us(t_posterior_end, t_commit_end),
                span_us(t_commit_end, t_replay_build_end),
                span_us(t_replay_build_end, t_replay_exec_end),
                span_us(t_replay_exec_end, t_sample_end),
                span_us(t_sample_end, t_round_end),
                span_us(t_round_start, t_round_end),
                replay_total - spec_count, replay_total + ar_count,
                spec_count + ar_count);
        }
        return result;
    }

    // Launch 2: accepted path segments + compact AR rows in the same builder
    // combination already used by mixed prefill/decode.
    const int ar_bucket = chain_decode_bucket_width(ar_count);
    const int n_total = replay_total + ar_bucket;
    const int gather_rows = spec_count + ar_bucket;
    const bool has_replay = replay_total > 0;
    StepGraph & durable_sg = b_.sg_;
    if (!build_target_step(
            durable_sg, b_.w_, b_.cache_, b_.target_backend_,
            0, n_total, false, true, false, 0, 0,
            b_.cfg_.kq_stride_pad, false, false, false, true,
            ar_bucket > 0 ? ar_bucket : 1, 0, max_kv_len,
            replay_total, replay_segments.data(),
            static_cast<int>(replay_segments.size()),
            gather_rows, ar_bucket > 0) ||
        !durable_sg.kv_write_rows || !durable_sg.target_feat_rows ||
        (has_replay &&
         (!durable_sg.paged_query_seq_ids ||
          !durable_sg.paged_query_positions)) ||
        (ar_bucket > 0 &&
         (!durable_sg.active_slot_ids || !durable_sg.state_slot_ids)) ||
        !durable_sg.logits_row_indices || !durable_sg.argmax_tokens) {
        result.error = "chain mixed commit/AR graph build failed";
        return result;
    }
    t_replay_build_end = timing_clock::now();

    std::vector<int32_t> durable_tokens(static_cast<size_t>(n_total), 0);
    std::copy(replay_tokens.begin(), replay_tokens.end(),
              durable_tokens.begin());
    for (int lane = 0; lane < ar_count; ++lane) {
        durable_tokens[static_cast<size_t>(replay_total + lane)] =
            ar_lanes[static_cast<size_t>(lane)].token;
    }
    embed_buf_.resize(static_cast<size_t>(hidden) * n_total);
    if (!b_.w_.embedder.embed(
            durable_tokens.data(), n_total, embed_buf_.data())) {
        result.error = "chain mixed commit/AR embedding failed";
        return result;
    }
    ggml_backend_tensor_set(
        durable_sg.inp_embed, embed_buf_.data(), 0,
        sizeof(float) * embed_buf_.size());

    pos_buf_.assign(static_cast<size_t>(4) * n_total, 0);
    for (int row = 0; row < replay_total; ++row) {
        const int position = replay_positions[static_cast<size_t>(row)];
        pos_buf_[static_cast<size_t>(0) * n_total + row] = position;
        pos_buf_[static_cast<size_t>(1) * n_total + row] = position;
        pos_buf_[static_cast<size_t>(2) * n_total + row] = position;
    }
    for (int lane = 0; lane < ar_count; ++lane) {
        const int row = replay_total + lane;
        const int position = ar_lanes[static_cast<size_t>(lane)].position;
        pos_buf_[static_cast<size_t>(0) * n_total + row] = position;
        pos_buf_[static_cast<size_t>(1) * n_total + row] = position;
        pos_buf_[static_cast<size_t>(2) * n_total + row] = position;
    }
    ggml_backend_tensor_set(durable_sg.positions, pos_buf_.data(), 0,
                            sizeof(int32_t) * pos_buf_.size());

    rows_buf_.assign(
        static_cast<size_t>(n_total) * n_head_kv, scratch_row_);
    for (int head = 0; head < n_head_kv; ++head) {
        for (int row = 0; row < replay_total; ++row) {
            rows_buf_[static_cast<size_t>(head) * n_total + row] =
                replay_physical[static_cast<size_t>(row)];
        }
        for (int lane = 0; lane < ar_count; ++lane) {
            rows_buf_[static_cast<size_t>(head) * n_total +
                      replay_total + lane] =
                ar_lanes[static_cast<size_t>(lane)].physical_row;
        }
    }
    ggml_backend_tensor_set(
        durable_sg.kv_write_rows, rows_buf_.data(), 0,
        sizeof(int64_t) * rows_buf_.size());

    query_slot_ids_.assign(static_cast<size_t>(n_total), -1);
    query_positions_.assign(static_cast<size_t>(n_total), -1);
    for (int row = 0; row < replay_total; ++row) {
        query_slot_ids_[static_cast<size_t>(row)] =
            replay_slots[static_cast<size_t>(row)];
        query_positions_[static_cast<size_t>(row)] =
            replay_positions[static_cast<size_t>(row)];
    }
    for (int lane = 0; lane < ar_count; ++lane) {
        const int row = replay_total + lane;
        query_slot_ids_[static_cast<size_t>(row)] =
            ar_lanes[static_cast<size_t>(lane)].slot;
        query_positions_[static_cast<size_t>(row)] =
            ar_lanes[static_cast<size_t>(lane)].position;
    }
    logits_rows_.clear();
    logits_rows_.reserve(static_cast<size_t>(gather_rows));
    int path_end = 0;
    for (const Proposal & proposal : proposals) {
        path_end += static_cast<int>(proposal.path.size());
        logits_rows_.push_back(path_end - 1);
    }
    for (int lane = 0; lane < ar_bucket; ++lane) {
        logits_rows_.push_back(replay_total + lane);
    }
    if (has_replay) {
        ggml_backend_tensor_set(
            durable_sg.paged_query_seq_ids, query_slot_ids_.data(), 0,
            sizeof(int32_t) * query_slot_ids_.size());
        ggml_backend_tensor_set(
            durable_sg.paged_query_positions, query_positions_.data(), 0,
            sizeof(int32_t) * query_positions_.size());
    }
    ggml_backend_tensor_set(
        durable_sg.logits_row_indices, logits_rows_.data(), 0,
        sizeof(int32_t) * logits_rows_.size());

    active_slot_ids_.assign(static_cast<size_t>(ar_bucket), -1);
    state_slot_ids_.assign(static_cast<size_t>(ar_bucket), 0);
    for (int lane = 0; lane < ar_count; ++lane) {
        active_slot_ids_[static_cast<size_t>(lane)] =
            ar_lanes[static_cast<size_t>(lane)].slot;
        state_slot_ids_[static_cast<size_t>(lane)] =
            ar_lanes[static_cast<size_t>(lane)].slot;
    }
    if (ar_bucket > 0) {
        ggml_backend_tensor_set(
            durable_sg.active_slot_ids, active_slot_ids_.data(), 0,
            sizeof(int32_t) * active_slot_ids_.size());
        ggml_backend_tensor_set(
            durable_sg.state_slot_ids, state_slot_ids_.data(), 0,
            sizeof(int32_t) * state_slot_ids_.size());
    }

    const int feature_cap = b_.cache_.target_feat_cap;
    const int dead_feature_row = feature_cap * n_slots;
    feature_rows_.assign(
        static_cast<size_t>(n_total), dead_feature_row);
    for (int row = 0; row < replay_total; ++row) {
        feature_rows_[static_cast<size_t>(row)] =
            replay_slots[static_cast<size_t>(row)] * feature_cap +
            replay_positions[static_cast<size_t>(row)] % feature_cap;
    }
    for (int lane = 0; lane < ar_count; ++lane) {
        const ArLane & ar = ar_lanes[static_cast<size_t>(lane)];
        feature_rows_[static_cast<size_t>(replay_total + lane)] =
            ar.slot * feature_cap + ar.position % feature_cap;
    }
    ggml_backend_tensor_set(
        durable_sg.target_feat_rows, feature_rows_.data(), 0,
        sizeof(int32_t) * feature_rows_.size());
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());

    if (ggml_backend_graph_compute(b_.target_backend_, durable_sg.gf) !=
        GGML_STATUS_SUCCESS) {
        result.error = "chain mixed commit/AR compute failed";
        return result;
    }
    t_replay_exec_end = timing_clock::now();

    argmax_buf_.assign(static_cast<size_t>(gather_rows), -1);
    ggml_backend_tensor_get_async(
        b_.target_backend_, durable_sg.argmax_tokens,
        argmax_buf_.data(), 0,
        sizeof(int32_t) * argmax_buf_.size());
    ggml_backend_synchronize(b_.target_backend_);
    for (int lane = 0; lane < spec_count; ++lane) {
        if (argmax_buf_[static_cast<size_t>(lane)] < 0) {
            result.error = "chain durable replay produced invalid token";
            return result;
        }
    }
    for (int lane = 0; lane < ar_count; ++lane) {
        if (argmax_buf_[static_cast<size_t>(spec_count + lane)] < 0) {
            result.error = "mixed AR durable step produced invalid token";
            return result;
        }
    }

    std::vector<int> write_slots;
    write_slots.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (chain_lane_executes(lane_disposition(i))) {
            write_slots.push_back(inputs[i].slot);
        }
    }
    if (!commit_residency_writes(write_slots)) {
        result.error = "chain mixed-step KV write commit failed";
        return result;
    }

    // Publish the fed root/path before sampling the next token, matching the
    // ordinary AR path's penalty history, RNG, and min-token-floor semantics.
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (chain_lane_executes(lane_disposition(i))) {
            slots_.commit_step(inputs[i].slot);
        }
    }
    for (int lane = 0; lane < spec_count; ++lane) {
        Proposal & proposal = proposals[static_cast<size_t>(lane)];
        proposal.pending = sample_graph_row(
            proposal.slot, lane,
            &argmax_buf_[static_cast<size_t>(lane)], &logits_buf_);
        if (proposal.pending < 0) {
            result.error = "chain durable replay sampling failed";
            return result;
        }
    }
    for (int lane = 0; lane < ar_count; ++lane) {
        ArLane & ar = ar_lanes[static_cast<size_t>(lane)];
        const int gathered_row = spec_count + lane;
        ar.pending = sample_graph_row(
            ar.slot, gathered_row,
            &argmax_buf_[static_cast<size_t>(gathered_row)], &logits_buf_);
        if (ar.pending < 0) {
            result.error = "mixed AR durable sampling failed";
            return result;
        }
    }
    t_sample_end = timing_clock::now();

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!chain_lane_executes(lane_disposition(i))) continue;
        std::string reselect_error;
        if (!maybe_reselect_residency(inputs[i].slot, reselect_error)) {
            result.error = reselect_error.empty()
                ? "KVFlash reselect failed" : reselect_error;
            return result;
        }
    }

    result.decode.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        DecodeOutput out;
        out.slot = inputs[i].slot;
        const ChainLaneDisposition disposition = lane_disposition(i);
        if (disposition == ChainLaneDisposition::Failed) {
            out.failed = true;
            out.error = proposal_errors[i];
            result.decode.push_back(std::move(out));
            continue;
        }
        if (disposition == ChainLaneDisposition::Speculation) {
            Proposal & proposal =
                proposals[static_cast<size_t>(proposal_for_input[i])];
            out.token = proposal.pending;
            out.spec_steps = 1;
            out.spec_accepted_tokens =
                proposal.path.size() > 1
                    ? static_cast<uint64_t>(proposal.path.size() - 1)
                    : 0;
            out.target_forwards = 2;
            out.committed_tokens.assign(
                proposal.path.begin() + 1, proposal.path.end());
        } else {
            ArLane & ar =
                ar_lanes[static_cast<size_t>(ar_for_input[i])];
            out.token = ar.pending;
            out.target_forwards = 1;
        }
        attach_residency_telemetry(out);
        result.decode.push_back(std::move(out));
    }
    if (timing) {
        const auto t_round_end = timing_clock::now();
        std::fprintf(stderr,
            "[step-timing] {\"path\":\"spec\",\"live\":%d,\"k\":%d,"
            "\"tree_bucket\":%d,\"tree_rows\":%d,\"replay_rows\":%d,"
            "\"ar_lanes\":%d,\"ar_bucket\":%d,\"max_kv_len\":%d,"
            "\"draft_us\":%.1f,\"draft_lanes\":%d,"
            "\"pre_us\":%.1f,\"verify_build_us\":%.1f,"
            "\"verify_exec_us\":%.1f,\"posterior_read_us\":%.1f,"
            "\"commit_cpu_us\":%.1f,\"replay_build_us\":%.1f,"
            "\"replay_exec_us\":%.1f,\"sample_read_us\":%.1f,"
            "\"finish_us\":%.1f,\"total_us\":%.1f,"
            "\"accepted_tokens\":%d,\"emitted_tokens\":%d,"
            "\"target_forwards\":%d}\n",
            spec_count + ar_count, spec_count,
            tree_bucket, V * tree_bucket, replay_total,
            ar_count, ar_bucket, max_kv_len,
            round_draft_us_, round_draft_lanes_,
            span_us(t_round_start, t_verify_build_start),
            span_us(t_verify_build_start, t_verify_build_end),
            span_us(t_verify_build_end, t_verify_exec_end),
            span_us(t_verify_exec_end, t_posterior_end),
            span_us(t_posterior_end, t_commit_end),
            span_us(t_commit_end, t_replay_build_end),
            span_us(t_replay_build_end, t_replay_exec_end),
            span_us(t_replay_exec_end, t_sample_end),
            span_us(t_sample_end, t_round_end),
            span_us(t_round_start, t_round_end),
            replay_total - spec_count, replay_total + ar_count,
            2 * spec_count + ar_count);
    }
    return result;
}


std::optional<SeqEngine::StepResult> Qwen35SeqEngine::step_ddtree(
        const StepPlan & plan) {
    StepResult result;
    const int active = (int)plan.decode.size();
    const int bucket = decode_bucket_width(active);
    const int T = tree_width_;
    const int q_len = b_.dw_.block_size;
    const int hidden = b_.w_.n_embd;
    const int n_head_kv = b_.w_.n_head_kv;
    const int n_slots = slots_.slot_count();
    const int K = b_.cfg_.ddtree_budget > q_len - 1 ? 8 : 1;

    struct Proposal {
        int slot = -1;
        int32_t root = -1;
        DDTree tree;
        std::vector<int32_t> flat;
        std::vector<int> accepted;
        int32_t bonus = -1;
    };
    std::vector<Proposal> proposals;
    proposals.reserve((size_t)active);
    std::vector<int32_t> noise((size_t)q_len, b_.w_.mask_token_id);
    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<float> logits((size_t)b_.w_.n_vocab * q_len);
    std::vector<float> top_lp((size_t)q_len * K);
    std::vector<int32_t> top_ids((size_t)q_len * K);

    auto proposal_fallback = [&]() -> std::optional<StepResult> {
        // begin_step updates persistent drafter bookkeeping before compute.
        // A failed proposal graph may therefore leave only part of that
        // cache valid. Reset all participating draft rings so a later
        // speculative round rebuilds them from committed target features.
        for (const StepInput & in : plan.decode) {
            if (in.slot >= 0 && in.slot < (int)slot_draft_kv_.size() &&
                slot_draft_kv_[(size_t)in.slot]) {
                draft_kv_reset(*slot_draft_kv_[(size_t)in.slot]);
            }
        }
        return std::nullopt;
    };

    if (!build_lm_head_projection_step(
            b_.proj_sg_, b_.w_, b_.target_backend_, q_len)) {
        return proposal_fallback();
    }

    // Proposal is sequential by slot: immutable draft weights are shared,
    // while each slot owns an independent persistent context-KV ring.
    for (const StepInput & in : plan.decode) {
        DraftKvState * draft = ensure_slot_draft_kv(in.slot);
        DraftFeatureMirror * mirror = slot_feature_mirror(in.slot);
        if (!draft || !mirror) return proposal_fallback();
        noise[0] = in.token;
        std::fill(noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
        if (!b_.w_.embedder.embed(
                noise.data(), q_len, noise_embed.data()) ||
            !draft_kv_begin_step(*draft, b_.dw_, b_.draft_backend_,
                                 *mirror, slots_.slot(in.slot).cur_pos)) {
            return proposal_fallback();
        }
        ggml_backend_tensor_set(
            draft->inp_embed, noise_embed.data(), 0,
            sizeof(float) * noise_embed.size());
        if (ggml_backend_graph_compute(b_.draft_backend_, draft->gf) !=
            GGML_STATUS_SUCCESS) {
            return proposal_fallback();
        }
        // The draft and target backends own separate HIP streams even when
        // both are placed on hip:0.  Projection consumes the draft hidden
        // state on the target stream, so establish the producer/consumer
        // ordering explicitly before the cross-backend tensor copy.  Without
        // this barrier the first tree proposal can race stale hidden rows and
        // collapse acceptance to the one-token fallback.
        ggml_backend_synchronize(b_.draft_backend_);
        ggml_backend_tensor_copy(
            draft->hidden_states, b_.proj_sg_.hidden_input);
        if (ggml_backend_graph_compute(
                b_.target_backend_, b_.proj_sg_.gf) != GGML_STATUS_SUCCESS) {
            return proposal_fallback();
        }
        bool topk_ready = false;
#ifdef DFLASH27B_HAVE_DRAFT_TOPK
        topk_ready = geometric_extract_draft_topk_cuda(
            b_.proj_sg_.logits->data, q_len, b_.w_.n_vocab, K,
            top_lp.data(), top_ids.data(), b_.cfg_.ddtree_temp);
#endif
        if (!topk_ready) {
            ggml_backend_tensor_get(
                b_.proj_sg_.logits, logits.data(), 0,
                sizeof(float) * logits.size());
            extract_draft_topk(
                logits.data(), q_len, b_.w_.n_vocab, K,
                top_lp.data(), top_ids.data(), b_.cfg_.ddtree_temp);
        }

        Proposal p;
        p.slot = in.slot;
        p.root = in.token;
        p.tree = build_ddtree(
            top_lp.data() + K, top_ids.data() + K,
            q_len - 1, K, b_.cfg_.ddtree_budget,
            b_.cfg_.ddtree_chain_seed);
        p.flat.assign((size_t)T, 0);
        p.flat[0] = in.token;
        for (int node = 0; node < p.tree.n_nodes; ++node) {
            p.flat[(size_t)node + 1] = p.tree.token_ids[(size_t)node];
        }
        proposals.push_back(std::move(p));
    }

    StepGraph & tree_sg = b_.sg_;
    int max_prefix = 1;
    for (const Proposal & p : proposals) {
        max_prefix = std::max(max_prefix, slots_.slot(p.slot).cur_pos);
    }
    if (!build_target_step_paged_tree(
            tree_sg, b_.w_, b_.cache_, b_.target_backend_, T, bucket,
            max_prefix, tree_scratch_base_, tree_scratch_stride_,
            b_.cfg_.kq_stride_pad)) {
        result.error = "packed DDTree verify graph build failed";
        return result;
    }

    const int total_tree = T * bucket;
    std::vector<int32_t> flat_tokens((size_t)total_tree, 0);
    std::vector<int32_t> parents((size_t)total_tree, -1);
    std::vector<int32_t> sizes((size_t)bucket, 0);
    // Negative active IDs identify bucket padding. Recurrent gathers cannot
    // index a negative slab, so padded trees use slot 0 only for their
    // read-only base-state gather; tree_size=0/query_slot=-1 keeps all of
    // their attention/output rows inactive and tree mode never persists it.
    std::vector<int32_t> tree_slots((size_t)bucket, -1);
    std::vector<int32_t> tree_state_slots((size_t)bucket, 0);
    std::vector<int32_t> query_slots((size_t)total_tree, -1);
    std::vector<int64_t> tree_rows(
        (size_t)total_tree * n_head_kv, scratch_row_);
    std::vector<int32_t> tree_pos((size_t)4 * total_tree, 0);
    std::vector<float> tree_embed((size_t)hidden * total_tree, 0.0f);
    seq_lens_.assign((size_t)n_slots, 0);

    for (int s = 0; s < active; ++s) {
        const Proposal & p = proposals[(size_t)s];
        const int base = s * T;
        sizes[(size_t)s] = p.tree.n_nodes + 1;
        tree_slots[(size_t)s] = p.slot;
        tree_state_slots[(size_t)s] = p.slot;
        seq_lens_[(size_t)p.slot] = slots_.slot(p.slot).cur_pos;
        for (int node = 0; node < sizes[(size_t)s]; ++node) {
            const int row = base + node;
            flat_tokens[(size_t)row] = p.flat[(size_t)node];
            parents[(size_t)row] = node == 0 ? -1 :
                p.tree.parents[(size_t)node];
            query_slots[(size_t)row] = p.slot;
            const int depth = node == 0 ? 0 :
                p.tree.depths[(size_t)node - 1];
            const int pos = slots_.slot(p.slot).cur_pos + depth;
            tree_pos[(size_t)0 * total_tree + row] = pos;
            tree_pos[(size_t)1 * total_tree + row] = pos;
            tree_pos[(size_t)2 * total_tree + row] = pos;
            for (int h = 0; h < n_head_kv; ++h) {
                tree_rows[(size_t)h * total_tree + row] =
                    (int64_t)tree_scratch_base_ +
                    (int64_t)p.slot * tree_scratch_stride_ + node;
            }
        }
    }
    if (!b_.w_.embedder.embed(
            flat_tokens.data(), total_tree, tree_embed.data())) {
        result.error = "packed DDTree embedding failed";
        return result;
    }
    ggml_backend_tensor_set(tree_sg.inp_embed, tree_embed.data(), 0,
                            sizeof(float) * tree_embed.size());
    ggml_backend_tensor_set(tree_sg.positions, tree_pos.data(), 0,
                            sizeof(int32_t) * tree_pos.size());
    ggml_backend_tensor_set(tree_sg.parent_ids, parents.data(), 0,
                            sizeof(int32_t) * parents.size());
    ggml_backend_tensor_set(tree_sg.tree_sizes, sizes.data(), 0,
                            sizeof(int32_t) * sizes.size());
    // Mapped-tree DeltaNet uses active_slot_ids only as a topology marker;
    // gallocr may therefore optimize away its backing buffer. The actual
    // state/attention mappings below are live graph inputs and remain
    // mandatory. Upload the marker only if a future topology consumes it.
    if (detail::target_paged_tree_active_slots_need_upload(tree_sg)) {
        ggml_backend_tensor_set(tree_sg.active_slot_ids, tree_slots.data(), 0,
                                sizeof(int32_t) * tree_slots.size());
    }
    ggml_backend_tensor_set(tree_sg.state_slot_ids, tree_state_slots.data(), 0,
                            sizeof(int32_t) * tree_state_slots.size());
    ggml_backend_tensor_set(tree_sg.paged_query_seq_ids, query_slots.data(), 0,
                            sizeof(int32_t) * query_slots.size());
    ggml_backend_tensor_set(tree_sg.kv_write_rows, tree_rows.data(), 0,
                            sizeof(int64_t) * tree_rows.size());
    ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
                            sizeof(int32_t) * seq_lens_.size());
    if (ggml_backend_graph_compute(b_.target_backend_, tree_sg.gf) !=
        GGML_STATUS_SUCCESS) {
        result.error = "packed DDTree verify compute failed";
        return result;
    }
    std::vector<int32_t> posterior((size_t)total_tree, -1);
    ggml_backend_tensor_get(tree_sg.argmax_tokens, posterior.data(), 0,
                            sizeof(int32_t) * posterior.size());

    int replay_total = 0;
    for (int s = 0; s < active; ++s) {
        Proposal & p = proposals[(size_t)s];
        p.accepted = follow_verified_tree(
            p.tree, posterior.data() + (size_t)s * T, p.bonus);
        const int room = slots_.max_context() - slots_.slot(p.slot).cur_pos;
        truncate_verified_path(
            p.accepted, (size_t)std::max(0, room),
            posterior.data() + (size_t)s * T, p.bonus);
        if (p.accepted.empty()) {
            result.error = "DDTree accepted path has no context headroom";
            return result;
        }
        replay_total += (int)p.accepted.size();
    }

    std::vector<QwenPrefillSegment> replay_segments;
    std::vector<int32_t> replay_tokens;
    std::vector<int32_t> replay_slots;
    std::vector<int32_t> replay_positions;
    std::vector<int64_t> replay_rows;
    std::vector<int32_t> replay_logits_rows;
    replay_segments.reserve((size_t)active);
    replay_tokens.reserve((size_t)replay_total);
    replay_slots.reserve((size_t)replay_total);
    replay_positions.reserve((size_t)replay_total);
    replay_rows.assign((size_t)replay_total * n_head_kv, scratch_row_);
    replay_logits_rows.reserve((size_t)active);
    seq_lens_.assign((size_t)n_slots, 0);

    int replay_offset = 0;
    for (Proposal & p : proposals) {
        std::vector<int32_t> path;
        path.reserve(p.accepted.size());
        for (int dfs : p.accepted) {
            path.push_back(dfs == 0 ? p.root :
                p.tree.token_ids[(size_t)dfs - 1]);
        }
        const Qwen35SlotManager::StepAppend app = slots_.append_tokens(
            p.slot, path.data(), (int)path.size());
        const bool table_ok = slots_.residency_active() ||
            upload_block_table_delta(p.slot, app.first_new_block,
                app.new_blocks.data(), app.new_blocks.size());
        if (!app.ok || app.physical_rows.size() != path.size() ||
            !table_ok) {
            result.error = app.busy
                ? "paged KV pool exhausted during DDTree replay"
                : "DDTree replay K/V append failed";
            return result;
        }
        replay_segments.push_back(
            {replay_offset, (int)path.size(), p.slot});
        for (size_t i = 0; i < path.size(); ++i) {
            replay_tokens.push_back(path[i]);
            replay_slots.push_back(p.slot);
            replay_positions.push_back(app.position + (int)i);
            for (int h = 0; h < n_head_kv; ++h) {
                replay_rows[(size_t)h * replay_total + replay_offset + i] =
                    app.physical_rows[i];
            }
        }
        replay_offset += (int)path.size();
        replay_logits_rows.push_back(replay_offset - 1);
        seq_lens_[(size_t)p.slot] = app.position + (int)path.size();
    }

    if (!upload_all_active_block_tables()) {
        result.error = "DDTree replay block-table refresh failed";
        return result;
    }

    StepGraph & replay_sg = b_.sg_;
    if (!build_target_step(
            replay_sg, b_.w_, b_.cache_, b_.target_backend_,
            0, replay_total, false, true, false, 0, 0,
            b_.cfg_.kq_stride_pad, false, false, false, true,
            1, 0, *std::max_element(seq_lens_.begin(), seq_lens_.end()),
            replay_total, replay_segments.data(),
            (int)replay_segments.size(), active, false) ||
        !replay_sg.target_feat_rows || !replay_sg.paged_query_seq_ids ||
        !replay_sg.paged_query_positions || !replay_sg.logits_row_indices ||
        !replay_sg.argmax_tokens) {
        result.error = "DDTree accepted-path replay graph build failed";
        return result;
    }
    embed_buf_.resize((size_t)hidden * replay_total);
    if (!b_.w_.embedder.embed(
            replay_tokens.data(), replay_total, embed_buf_.data())) {
        result.error = "DDTree replay embedding failed";
        return result;
    }
    pos_buf_.assign((size_t)4 * replay_total, 0);
    feature_rows_.resize((size_t)replay_total);
    const int cap = b_.cache_.target_feat_cap;
    for (int row = 0; row < replay_total; ++row) {
        const int pos = replay_positions[(size_t)row];
        pos_buf_[(size_t)0 * replay_total + row] = pos;
        pos_buf_[(size_t)1 * replay_total + row] = pos;
        pos_buf_[(size_t)2 * replay_total + row] = pos;
        feature_rows_[(size_t)row] =
            replay_slots[(size_t)row] * cap + pos % cap;
    }
    ggml_backend_tensor_set(replay_sg.inp_embed, embed_buf_.data(), 0,
                            sizeof(float) * embed_buf_.size());
    ggml_backend_tensor_set(replay_sg.positions, pos_buf_.data(), 0,
                            sizeof(int32_t) * pos_buf_.size());
    ggml_backend_tensor_set(replay_sg.kv_write_rows, replay_rows.data(), 0,
                            sizeof(int64_t) * replay_rows.size());
    ggml_backend_tensor_set(replay_sg.paged_query_seq_ids,
                            replay_slots.data(), 0,
                            sizeof(int32_t) * replay_slots.size());
    ggml_backend_tensor_set(replay_sg.paged_query_positions,
                            replay_positions.data(), 0,
                            sizeof(int32_t) * replay_positions.size());
    ggml_backend_tensor_set(replay_sg.logits_row_indices,
                            replay_logits_rows.data(), 0,
                            sizeof(int32_t) * replay_logits_rows.size());
    ggml_backend_tensor_set(replay_sg.target_feat_rows,
                            feature_rows_.data(), 0,
                            sizeof(int32_t) * feature_rows_.size());
    ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
                            sizeof(int32_t) * seq_lens_.size());
    if (ggml_backend_graph_compute(b_.target_backend_, replay_sg.gf) !=
        GGML_STATUS_SUCCESS) {
        result.error = "DDTree accepted-path replay compute failed";
        return result;
    }
    std::vector<int> replay_write_slots;
    replay_write_slots.reserve(proposals.size());
    for (const Proposal & p : proposals) replay_write_slots.push_back(p.slot);
    if (!commit_residency_writes(replay_write_slots)) {
        result.error = "DDTree replay KV write commit failed";
        return result;
    }

    // The replay is the durable target forward: its recurrent/KV/feature
    // state is what the next step consumes. Use its posterior rather than
    // the tree-verify posterior so the pending scalar remains exact even if
    // the two graph shapes differ numerically.
    std::vector<int32_t> replay_next((size_t)active, -1);
    ggml_backend_tensor_get(
        replay_sg.argmax_tokens, replay_next.data(), 0,
        sizeof(int32_t) * replay_next.size());
    for (int s = 0; s < active; ++s) {
        if (replay_next[(size_t)s] < 0) {
            result.error = "DDTree replay produced an invalid pending token";
            return result;
        }
        proposals[(size_t)s].bonus = replay_next[(size_t)s];
    }

    for (Proposal & p : proposals) {
        slots_.commit_step(p.slot);
        std::string reselect_error;
        if (!maybe_reselect_residency(p.slot, reselect_error)) {
            result.error = reselect_error.empty()
                ? "KVFlash reselect failed" : reselect_error;
            return result;
        }
    }

    uint64_t cohort_emitted = 0;
    for (const Proposal & p : proposals) {
        // accepted contains the replay root plus accepted children. The
        // output emits those children plus one separately computed pending
        // scalar, so accepted.size() is this request's emitted yield.
        cohort_emitted += (uint64_t)p.accepted.size();
    }
    const bool suspend_cohort =
        Qwen35SlotManager::ddtree_cohort_should_suspend(
            cohort_emitted, active);
    std::vector<bool> newly_suspended((size_t)slots_.slot_count(), false);
    for (const Proposal & p : proposals) {
        newly_suspended[(size_t)p.slot] =
            slots_.record_ddtree_sample(p.slot, suspend_cohort);
    }

    result.decode.reserve((size_t)active);
    for (Proposal & p : proposals) {
        DecodeOutput out;
        out.slot = p.slot;
        out.token = p.bonus;
        out.ddtree_steps = 1;
        const int accepted_children = (int)p.accepted.size() - 1;
        out.ddtree_accepted_tokens = (uint64_t)accepted_children;
        out.target_forwards = 2;
        for (size_t i = 1; i < p.accepted.size(); ++i) {
            const int dfs = p.accepted[i];
            out.committed_tokens.push_back(
                p.tree.token_ids[(size_t)dfs - 1]);
        }
        if (newly_suspended[(size_t)p.slot]) {
            out.ddtree_suspensions = 1;
            const Qwen35Slot & seq = slots_.slot(p.slot);
            std::fprintf(stderr,
                "[parallel-ddtree] adaptive suspend request=%llu slot=%d "
                "sample=%llu emitted=%d accepted_children=%d "
                "cohort_emitted=%llu cohort_size=%d target_forwards=2 "
                "floor=%d\n",
                (unsigned long long)seq.request_id, p.slot,
                (unsigned long long)seq.ddtree_sampled_steps,
                accepted_children + 1, accepted_children,
                (unsigned long long)cohort_emitted, active,
                Qwen35SlotManager::kDdtreeMinEmittedTokens);
        }
        attach_residency_telemetry(out);
        result.decode.push_back(std::move(out));
    }
    return result;
}

bool Qwen35SeqEngine::token_is_eos(int32_t token) const {
    return b_.token_is_eos(token);
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    AdmitResult result = slots_.admit(request_id, prompt, sampler);
    if (result.status == AdmitResult::Status::admitted) {
        reset_recurrent_slot(b_.cache_, result.slot);
        if (slots_.residency_active()) {
            slots_.slot(result.slot).kvflash_last_reselect_generated =
                -std::max(1, b_.kvflash_tau_);
        }
        if (result.slot >= 0 && result.slot < (int)slot_draft_kv_.size() &&
            slot_draft_kv_[(size_t)result.slot]) {
            draft_kv_reset(*slot_draft_kv_[(size_t)result.slot]);
        }
        if (result.slot >= 0 &&
            result.slot < (int)last_activation_estimate_.size()) {
            last_activation_estimate_[(size_t)result.slot] = {};
            if (result.slot <
                (int)prepared_chain_drafts_.size()) {
                prepared_chain_drafts_[(size_t)result.slot].valid = false;
            }
        }
        if (result.slot >= 0 &&
            result.slot < (int)adaptive_fallback_ar_.size()) {
            adaptive_fallback_ar_[(size_t)result.slot] = 0;
        }
    }
    return result;
}

int32_t Qwen35SeqEngine::sample_graph_row(
        int slot, int logits_row, const int32_t * cached_argmax,
        std::vector<float> * logits_scratch) {
    const TargetWeights & w = b_.w_;
    const int vocab = w.n_vocab;
    Qwen35Slot & seq = slots_.slot(slot);
    int32_t token = -1;
    if (seq.sampler.needs_logit_processing()) {
        std::vector<float> local_logits;
        std::vector<float> & logits = logits_scratch
            ? *logits_scratch
            : local_logits;
        if (logits.empty()) logits.resize((size_t)vocab);
        ggml_backend_tensor_get_async(
            b_.target_backend_, b_.sg_.logits, logits.data(),
            (size_t)logits_row * (size_t)vocab * sizeof(float),
            sizeof(float) * (size_t)vocab);
        ggml_backend_synchronize(b_.target_backend_);
        token = sample_logits(logits.data(), vocab, seq.sampler,
                              seq.sample_history, seq.rng);
    } else if (cached_argmax) {
        token = *cached_argmax;
    } else {
        ggml_backend_tensor_get_async(
            b_.target_backend_, b_.sg_.argmax_tokens, &token,
            (size_t)logits_row * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_synchronize(b_.target_backend_);
    }
    return b_.apply_min_tokens_floor(
        token, seq.generated_tokens(),
        (size_t)logits_row * (size_t)vocab * sizeof(float));
}

bool Qwen35SeqEngine::upload_block_table_delta(
        int slot, int first_block, const int32_t * blocks, size_t count) {
    if (count == 0) return true;
    ggml_tensor * table = b_.cache_.paged_block_table;
    if (!table || slot < 0 || slot >= table->ne[1] || first_block < 0 ||
        (uint64_t)first_block + count > (uint64_t)table->ne[0]) {
        return false;
    }
    // `blocks` commonly points into a temporary PrefillChunk vector or a
    // stack-local StepAppend. Keep this tiny metadata write synchronous so
    // the backend never observes a source whose lifetime has ended.
    ggml_backend_tensor_set(
        table, blocks,
        (size_t)slot * table->nb[1] +
            (size_t)first_block * sizeof(int32_t),
        count * sizeof(int32_t));
    return true;
}

bool Qwen35SeqEngine::upload_all_active_block_tables() {
    if (!slots_.residency_active()) return true;
    ggml_tensor * table = b_.cache_.paged_block_table;
    if (!table) return false;
    std::vector<int32_t> column((size_t)table->ne[0], -1);
    std::vector<int32_t> snapshot;
    for (int slot = 0; slot < slots_.slot_count(); ++slot) {
        if (!slots_.is_active(slot)) continue;
        std::fill(column.begin(), column.end(), -1);
        if (!slots_.block_table_snapshot(slot, snapshot) ||
            snapshot.size() > column.size()) {
            return false;
        }
        std::copy(snapshot.begin(), snapshot.end(), column.begin());
        ggml_backend_tensor_set(
            table, column.data(), (size_t)slot * table->nb[1],
            sizeof(int32_t) * column.size());
    }
    return true;
}

bool Qwen35SeqEngine::commit_residency_writes(
        const std::vector<int> & slots) {
    if (!slots_.residency_active()) return true;
    // Graph completion is not a host barrier for every backend. Pending pages
    // become evictable only after all target writes are device-complete.
    ggml_backend_synchronize(b_.target_backend_);
    for (int slot : slots) {
        if (!slots_.commit_residency_writes(slot)) return false;
    }
    return true;
}

bool Qwen35SeqEngine::maybe_reselect_residency(
        int slot, std::string & error) {
    if (!slots_.residency_active()) return true;
    Qwen35Slot & seq = slots_.slot(slot);
    const int generated = seq.generated_tokens();
    const int tau = std::max<int>(
        b_.kvflash_tau_, (int)(seq.sample_history.size() / 45));
    if (generated - seq.kvflash_last_reselect_generated < tau) return true;

    b_.kvflash_ensure_scorer();
    std::vector<float> scores;
    const std::vector<float> * score_ptr = nullptr;
    if (b_.kvflash_scorer_) {
        if (!b_.kvflash_scorer_->score_chunks(
                seq.sample_history, PAGED_BLOCK_SIZE, scores)) {
            // Short histories and recoverable drafter failures are expected
            // scorer outcomes. Preserve service with the pager's explicit
            // recency/LRU policy; only residency or transfer errors below are
            // fatal to the request.
            std::fprintf(stderr,
                "[parallel-kvflash] scorer unavailable for slot %d; using LRU\n",
                slot);
        } else {
            const size_t blocks = (seq.sample_history.size() +
                PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;
            scores.resize(blocks, scores.empty() ? 0.0f : scores.back());
            score_ptr = &scores;
        }
    }
    if (!slots_.reselect_residency(slot, score_ptr, &error)) return false;
    seq.kvflash_last_reselect_generated = generated;
    return upload_all_active_block_tables();
}

void Qwen35SeqEngine::attach_residency_telemetry(DecodeOutput & out) {
    slots_.take_residency_telemetry(out.slot, out);
}

void Qwen35SeqEngine::fail_prefill(
        int slot, std::vector<PrefillOutput> & prefill_outputs,
        const char * log_message, const char * client_message) {
    if (!slots_.is_prefilling(slot)) return;
    std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                 log_message, slot);
    PrefillOutput out;
    out.slot = slot;
    out.status = PrefillOutput::Status::failed;
    out.error = client_message;
    prefill_outputs.push_back(std::move(out));
}

Qwen35SeqEngine::PrefillStage Qwen35SeqEngine::stage_prefill_chunk(
        int slot, int max_tokens,
        std::vector<PrefillOutput> & prefill_outputs) {
    PrefillStage stage;
    if (!slots_.is_prefilling(slot)) return stage;

    Qwen35Slot & seq = slots_.slot(slot);
    stage.kv_pos = seq.cur_pos;
    stage.chunk = std::min(
        max_tokens, seq.prompt_len - stage.kv_pos);
    if (stage.chunk <= 0) return PrefillStage{};
    stage.commit = stage.kv_pos + stage.chunk >= seq.prompt_len;

    Qwen35SlotManager::PrefillChunk chunk =
        slots_.append_prefill(slot, stage.chunk);
    if (!chunk.ok || chunk.rows.size() != (size_t)stage.chunk) {
        fail_prefill(slot, prefill_outputs, "prefill K/V allocation failed",
                     "prefill K/V allocation failed");
        return PrefillStage{};
    }
    const bool table_ok = slots_.residency_active() || upload_block_table_delta(
              slot, chunk.first_new_block, chunk.new_blocks.data(),
              chunk.new_blocks.size());
    if (!table_ok) {
        fail_prefill(
            slot, prefill_outputs, "prefill block-table delta exceeds device capacity",
            "prefill block-table update failed");
        return PrefillStage{};
    }

    stage.rows = std::move(chunk.rows);
    stage.embeddings.resize((size_t)b_.w_.n_embd * stage.chunk);
    if (!b_.w_.embedder.embed(
            seq.sample_history.data() + stage.kv_pos, stage.chunk,
            stage.embeddings.data())) {
        fail_prefill(slot, prefill_outputs, "prefill embed failed",
                     "prefill embedding failed");
        return PrefillStage{};
    }
    stage.ready = true;
    return stage;
}

SeqEngine::StepResult Qwen35SeqEngine::step(const StepPlan & plan) {
    StepResult result;
    std::vector<DecodeOutput> & decode_outputs = result.decode;
    std::vector<PrefillOutput> & prefill_outputs = result.prefills;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    if ((int)inputs.size() != slots_.decoding_count()) {
        return fail_step("decode plan does not cover every live slot");
    }
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & in : inputs) {
        if (in.slot < 0 || in.slot >= n_slots || in.token < 0 ||
            decode_seen[(size_t)in.slot] ||
            !slots_.is_active(in.slot) || slots_.is_prefilling(in.slot)) {
            return fail_step("invalid or duplicate decode row in step plan");
        }
        decode_seen[(size_t)in.slot] = 1;
    }

    const StepPlanLimits limits = step_plan_limits((int)inputs.size());
    if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
        return fail_step("prefill plan exceeds engine sequence capacity");
    }
    int planned_prefill_tokens = 0;
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens <= 0 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) {
            return fail_step("invalid or duplicate prefill slice in step plan");
        }
        prefill_seen[(size_t)slice.slot] = 1;
        planned_prefill_tokens += slice.max_tokens;
        if (planned_prefill_tokens > limits.max_prefill_tokens_total) {
            return fail_step("prefill plan exceeds engine total-token capacity");
        }
    }
    if (inputs.empty() && plan.prefills.empty()) return result;

    if (ddtree_eligible(plan)) {
        std::optional<StepResult> speculative = step_ddtree(plan);
        if (speculative) return std::move(*speculative);
        // Proposal setup failed before target/cache mutation. Preserve service
        // by taking the existing packed AR path for this iteration.
    }

    // One common wall-clock origin makes pure AR, adaptive k=0, and admitted
    // speculative rounds directly comparable and feeds the online cost model
    // even when diagnostic phase logging is disabled.
    const bool timing = step_timing_enabled();
    using timing_clock = std::chrono::steady_clock;
    const bool gate_cost_timing =
        spec_mode_ == SpecMode::chain &&
        speculation_gate_ != nullptr;
    const auto decode_round_started = timing || gate_cost_timing
        ? timing_clock::now() : timing_clock::time_point{};
    std::optional<SpecPlan> pending_ar_gate_plan;
    std::vector<uint8_t> spec_service_ar(inputs.size(), 0);

    if (spec_mode_ == SpecMode::chain && !inputs.empty()) {
        // New chain round: restart the [step-timing] draft attribution.
        round_draft_us_ = 0.0;
        round_draft_lanes_ = 0;
        const auto chain_started = decode_round_started;
        std::vector<uint8_t> admitted(inputs.size(), 0);
        SpecPlan gate_plan;
        bool have_gate_plan = false;

        const char * force_value = std::getenv("DFLASH_SPEC_GATE_FORCE");
        const std::string force = force_value ? force_value : "";

        if (speculation_gate_) {
            std::vector<SpecCandidate> candidates;
            candidates.reserve(inputs.size());
            const bool use_activation_score = activation_scoring_enabled();
            for (const StepInput & in : inputs) {
                const Qwen35Slot & seq = slots_.slot(in.slot);
                SpeculationPolicy policy = in.speculation_policy;
                if (policy == SpeculationPolicy::Adaptive) {
                    if (force == "all") policy = SpeculationPolicy::Always;
                    if (force == "none") policy = SpeculationPolicy::Never;
                }
                // The activation-score-off arm is an explicit AR ablation. It must
                // not pay the one-time activation draft.
                if (!use_activation_score &&
                    policy == SpeculationPolicy::Adaptive) {
                    policy = SpeculationPolicy::Never;
                }
                const bool scoreable =
                    chain_activation_input_scoreable(in);
                const bool can_speculate =
                    chain_spec_request_capable(in);
                double activation_score =
                    std::numeric_limits<double>::quiet_NaN();
                std::vector<double> activation_hazards;
                if (use_activation_score && scoreable && in.slot >= 0 &&
                    in.slot < (int)last_activation_estimate_.size() &&
                    std::isfinite(last_activation_estimate_[(size_t)in.slot].expected_yield)) {
                    activation_score =
                        last_activation_estimate_[(size_t)in.slot].expected_yield;
                    activation_hazards = last_activation_estimate_[
                        (size_t)in.slot].conditional_hazards;
                }
                candidates.push_back({
                    seq.request_id, in.slot, policy,
                    scoreable, can_speculate, activation_score,
                    std::move(activation_hazards),
                    chain_activation_score_kind(),
                });
            }
            const bool cohort_changed =
                !spec_cohort_epoch_.has_value() ||
                !spec_cohort_epoch_->matches(candidates);
            bool published_epoch = false;
            if (cohort_changed) {
                gate_plan = speculation_gate_->plan(
                    (int)inputs.size(), candidates, (int)inputs.size());
            } else {
                gate_plan = spec_cohort_epoch_->plan;
            }
            have_gate_plan = true;
            if (!gate_plan.valid) {
                return fail_step(gate_plan.error.empty()
                    ? "adaptive speculation gate failed" : gate_plan.error);
            }
            auto replan_with_published_score = [&]() {
                for (SpecCandidate & candidate : candidates) {
                    if (!candidate.scoreable ||
                        candidate.policy == SpeculationPolicy::Never) {
                        continue;
                    }
                    const int slot = candidate.slot;
                    if (slot >= 0 &&
                        slot < (int)last_activation_estimate_.size() &&
                        std::isfinite(last_activation_estimate_[(size_t)slot].expected_yield)) {
                        candidate.activation_yield =
                            last_activation_estimate_[(size_t)slot].expected_yield;
                        candidate.conditional_hazards =
                            last_activation_estimate_[(size_t)slot].conditional_hazards;
                    }
                }
                gate_plan = speculation_gate_->plan(
                    (int)inputs.size(), candidates, (int)inputs.size());
                return gate_plan.valid;
            };

            auto reset_evaluation_lane = [&](int slot) {
                if (slot >= 0 &&
                    slot < (int)prepared_chain_drafts_.size()) {
                    prepared_chain_drafts_[(size_t)slot] = {};
                }
                if (slot >= 0 && slot < (int)slot_draft_kv_.size() &&
                    slot_draft_kv_[(size_t)slot]) {
                    draft_kv_reset(*slot_draft_kv_[(size_t)slot]);
                }
                if (slot >= 0 &&
                    slot < (int)last_activation_estimate_.size()) {
                    last_activation_estimate_[(size_t)slot] = {};
                }
            };
            auto commit_evaluation_fallback =
                    [&](const SpecPendingEvaluation & evaluation) {
                reset_evaluation_lane(evaluation.slot);
                if (speculation_gate_->record_evaluation_failure(
                        evaluation.request_id)) {
                    const std::string kind =
                        chain_activation_score_kind();
                    const char * reason =
                        "activation_evaluation_failed";
                    log_spec_evaluation_fallback(
                        evaluation.request_id, evaluation.slot,
                        kind, reason);
                }
            };

            // Resolve every one-time evaluation action before one immediate
            // replan. A packed bootstrap failure is retried once per lane so
            // one broken request is marked evaluation-failed without poisoning a
            // healthy scored peer or the cohort.
            if (!gate_plan.pending_evaluations.empty()) {
                std::vector<SpecPendingEvaluation> score_evaluations;
                std::vector<uint8_t> bootstrap(inputs.size(), 0);
                for (const SpecPendingEvaluation & evaluation :
                     gate_plan.pending_evaluations) {
                    if (evaluation.action ==
                        SpecEvaluationAction::FallbackAR) {
                        commit_evaluation_fallback(evaluation);
                        continue;
                    }
                    score_evaluations.push_back(evaluation);
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        if (inputs[i].slot == evaluation.slot) {
                            bootstrap[i] = 1;
                        }
                    }
                }

                if (use_activation_score && !score_evaluations.empty()) {
                    const bool batch_scored = prepare_chain_drafts(
                        inputs, bootstrap, /*force_serial=*/false,
                        /*fail_fast_batch=*/true);
                    if (!batch_scored) {
                        for (const SpecPendingEvaluation & evaluation :
                             score_evaluations) {
                            reset_evaluation_lane(evaluation.slot);
                        }
                        for (const SpecPendingEvaluation & evaluation :
                             score_evaluations) {
                            std::vector<uint8_t> one(inputs.size(), 0);
                            for (size_t i = 0; i < inputs.size(); ++i) {
                                if (inputs[i].slot == evaluation.slot) {
                                    one[i] = 1;
                                }
                            }
                            const bool lane_scored = prepare_chain_drafts(
                                inputs, one, /*force_serial=*/true);
                            if (!lane_scored || evaluation.slot < 0 ||
                                evaluation.slot >=
                                    (int)last_activation_estimate_.size() ||
                                !std::isfinite(last_activation_estimate_[
                                    (size_t)evaluation.slot].expected_yield)) {
                                commit_evaluation_fallback(evaluation);
                            }
                        }
                    } else {
                        for (const SpecPendingEvaluation & evaluation :
                             score_evaluations) {
                            if (evaluation.slot < 0 ||
                                evaluation.slot >=
                                    (int)last_activation_estimate_.size() ||
                                !std::isfinite(last_activation_estimate_[
                                    (size_t)evaluation.slot].expected_yield)) {
                                commit_evaluation_fallback(evaluation);
                            }
                        }
                    }
                } else {
                    for (const SpecPendingEvaluation & evaluation :
                         score_evaluations) {
                        commit_evaluation_fallback(evaluation);
                    }
                }

                if (!replan_with_published_score()) {
                    return fail_step(gate_plan.error.empty()
                        ? "adaptive speculation gate failed" : gate_plan.error);
                }
                // This guard converts any unexpectedly unpublished score into
                // the same request-local fallback rather than repeating the
                // cold evaluation forever or failing a mixed cohort.
                if (!gate_plan.pending_evaluations.empty()) {
                    const std::vector<SpecPendingEvaluation> unresolved =
                        gate_plan.pending_evaluations;
                    for (const SpecPendingEvaluation & evaluation : unresolved) {
                        commit_evaluation_fallback(evaluation);
                    }
                    if (!replan_with_published_score()) {
                        return fail_step(gate_plan.error.empty()
                            ? "adaptive speculation gate failed"
                            : gate_plan.error);
                    }
                }
                if (!gate_plan.pending_evaluations.empty()) {
                    return fail_step(
                        "adaptive activation score did not resolve");
                }
            }
            if (cohort_changed) {
                SpecCohortEpoch epoch;
                epoch.id = next_spec_cohort_epoch_id_++;
                epoch.request_ids = SpecCohortEpoch::ids(candidates);
                epoch.plan = gate_plan;
                spec_cohort_epoch_ = std::move(epoch);
                log_spec_epoch(*spec_cohort_epoch_, *speculation_gate_);
                published_epoch = true;
            }

            // Discard bootstrap proposals for lanes routed to AR in the new
            // epoch. Selected lanes keep that first proposal as useful work.
            for (const SpecPlanScore & score : gate_plan.ordered) {
                if (!published_epoch || score.admitted) continue;
                const int slot = score.slot;
                if (slot >= 0 &&
                    slot < (int)prepared_chain_drafts_.size()) {
                    prepared_chain_drafts_[(size_t)slot] = {};
                }
                if (slot >= 0 && slot < (int)slot_draft_kv_.size() &&
                    slot_draft_kv_[(size_t)slot]) {
                    draft_kv_reset(*slot_draft_kv_[(size_t)slot]);
                }
            }

            // Apply the cached epoch subset. Planned prefills use a
            // telemetered AR service round without changing the epoch plan.
            for (const SpecPlanScore & score : gate_plan.ordered) {
                if (!score.admitted) continue;
                const int slot = score.slot;
                bool found = false;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (inputs[i].slot == slot) {
                        admitted[i] = 1;
                        found = true;
                    }
                }
                if (!found) {
                    return fail_step(
                        "speculation gate admitted a missing decode lane");
                }
            }
        } else {
            // Forced modes remain available without a cost profile. A normal
            // Adaptive request fails closed once, request-locally, and then
            // remains AR for its entire slot lifetime without failing UX.
            for (size_t i = 0; i < inputs.size(); ++i) {
                SpeculationPolicy policy = inputs[i].speculation_policy;
                if (policy == SpeculationPolicy::Adaptive &&
                    force == "all") {
                    policy = SpeculationPolicy::Always;
                } else if (policy == SpeculationPolicy::Adaptive &&
                           force == "none") {
                    policy = SpeculationPolicy::Never;
                } else if (policy == SpeculationPolicy::Adaptive) {
                    policy = SpeculationPolicy::Never;
                    const int slot = inputs[i].slot;
                    if (slot >= 0 &&
                        slot < (int)adaptive_fallback_ar_.size() &&
                        !adaptive_fallback_ar_[(size_t)slot]) {
                        adaptive_fallback_ar_[(size_t)slot] = 1;
                        const uint64_t request_id =
                            slots_.slot(slot).request_id;
                        const char * reason =
                            adaptive_fallback_reason_.empty()
                                ? "cost_profile_unavailable"
                                : adaptive_fallback_reason_.c_str();
                        log_spec_evaluation_fallback(
                            request_id, slot,
                            chain_activation_score_kind(), reason);
                    }
                }
                admitted[i] = policy == SpeculationPolicy::Always;
            }
        }

        const bool any_admitted = std::any_of(
            admitted.begin(), admitted.end(),
            [](uint8_t value) { return value != 0; });
        if (any_admitted && plan.prefills.empty()) {
            StepResult speculative =
                step_chain_spec(plan, admitted, decode_round_started);
            const double measured_us =
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - chain_started).count();
            const bool spec_completed = speculative.error.empty();
            bool proposal_failed = false;
            std::vector<int> accepted_lengths(inputs.size(), 0);
            if (spec_completed) {
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (!admitted[i]) continue;
                    const auto output = std::find_if(
                        speculative.decode.begin(), speculative.decode.end(),
                        [&](const DecodeOutput & item) {
                            return item.slot == inputs[i].slot;
                        });
                    if (output == speculative.decode.end() || output->failed ||
                        output->spec_steps == 0) {
                        proposal_failed = true;
                        break;
                    }
                    accepted_lengths[i] =
                        1 + static_cast<int>(output->spec_accepted_tokens);
                }
            }
            const bool cost_sample_valid =
                have_gate_plan && spec_completed && !proposal_failed;
            if (cost_sample_valid) {
                const ChainLaunchShape executed = chain_launch_shape(
                    admitted, accepted_lengths,
                    chain_verify_depth_for_round());
                const bool direct_commit = chain_direct_commit_enabled();
                const int priced_tree_rows = direct_commit
                    ? executed.tree_rows +
                        static_cast<int>(inputs.size()) -
                        executed.spec_lanes
                    : executed.tree_rows;
                speculation_gate_->observe_cost(
                    {static_cast<int>(inputs.size()), executed.spec_lanes,
                     priced_tree_rows,
                     direct_commit ? 0 : executed.commit_rows,
                     round_draft_lanes_},
                    measured_us);
            }
            if (have_gate_plan && spec_gate_debug_enabled() &&
                spec_completed && !proposal_failed) {
                const double realized_tokens = spec_completed
                    ? initial_prediction_realized_tokens(gate_plan, speculative)
                    : std::numeric_limits<double>::quiet_NaN();
                log_spec_gate_plan(
                    gate_plan, realized_tokens,
                    cost_sample_valid
                        ? measured_us
                        : std::numeric_limits<double>::quiet_NaN());
            }
            return speculative;
        }
        if (any_admitted) {
            // Chain verification cannot share a target graph with prompt work.
            // Use the existing packed AR+prefill graph for this service round
            // so selected prompts make immediate progress. Routing remains
            // fixed for the current epoch; the per-request metric distinguishes this bounded
            // scheduling suspension from speculative execution.
            spec_service_ar = admitted;
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (!admitted[i]) continue;
                const int slot = inputs[i].slot;
                if (slot >= 0 &&
                    slot < (int)prepared_chain_drafts_.size()) {
                    prepared_chain_drafts_[(size_t)slot] = {};
                }
                if (slot >= 0 && slot < (int)slot_draft_kv_.size() &&
                    slot_draft_kv_[(size_t)slot]) {
                    draft_kv_reset(*slot_draft_kv_[(size_t)slot]);
                }
            }
        }
        if (!any_admitted && have_gate_plan) {
            if (gate_plan.admitted_count == 0 && plan.prefills.empty()) {
                pending_ar_gate_plan = gate_plan;
            }
        }
    }
    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden = w.n_embd;
    const int n_head_kv = w.n_head_kv;
    timing_clock::time_point t_ar_build_start, t_ar_build_end,
        t_ar_exec_start, t_ar_exec_end, t_ar_read_end;

    decode_outputs.reserve(inputs.size());
    prefill_outputs.reserve(plan.prefills.size());
    output_rows_.clear();
    live_tokens_.clear();
    live_positions_.clear();
    live_physical_rows_.clear();
    live_slot_ids_.clear();
    output_rows_.reserve(inputs.size());
    live_tokens_.reserve(inputs.size());
    live_positions_.reserve(inputs.size());
    live_physical_rows_.reserve(inputs.size());
    live_slot_ids_.reserve(inputs.size());

    int max_kv_len = 1;
    for (const StepInput & in : inputs) {
        DecodeOutput out;
        out.slot = in.slot;
        out.failed = true;
        int compact_row = -1;
        const Qwen35SlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
        if (!app.ok) {
            out.error = app.busy
                ? "paged KV pool exhausted during decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "decode K/V append failed";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
        const bool table_ok = slots_.residency_active() || app.new_block < 0 ||
            upload_block_table_delta(in.slot, app.new_block_index,
                                     &app.new_block, 1);
        if (!table_ok) {
            out.error = "decode block-table entry exceeds device capacity";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
        compact_row = (int)live_tokens_.size();
        live_tokens_.push_back(in.token);
        live_positions_.push_back(app.position);
        live_physical_rows_.push_back(app.physical_row);
        live_slot_ids_.push_back(in.slot);
        max_kv_len = std::max(max_kv_len, app.position + 1);
        out.failed = false;
        decode_outputs.push_back(std::move(out));
        output_rows_.push_back(compact_row);
    }

    std::vector<PrefillStage> prefills;
    prefills.reserve(plan.prefills.size());
    for (const PrefillSlice & slice : plan.prefills) {
        const size_t outputs_before = prefill_outputs.size();
        PrefillStage prefill =
            stage_prefill_chunk(slice.slot, slice.max_tokens, prefill_outputs);
        if (!prefill.ready) {
            if (prefill_outputs.size() == outputs_before) {
                fail_prefill(
                    slice.slot, prefill_outputs,
                    "prefill made no progress despite reserved capacity",
                    "prefill scheduler made no progress");
            }
            return fail_step("selected prefill work made no progress");
        }
        prefills.push_back(std::move(prefill));
    }
    if (!upload_all_active_block_tables()) {
        return fail_step("active KVFlash block-table refresh failed");
    }

    const int live_count = (int)live_tokens_.size();
    const bool with_decode = live_count > 0;
    const int decode_bucket = with_decode ? decode_bucket_width(live_count) : 0;

    dec_tokens_.assign((size_t)decode_bucket, 0);
    dec_rows_.assign((size_t)decode_bucket * n_head_kv, scratch_row_);
    active_slot_ids_.assign((size_t)decode_bucket, -1);
    state_slot_ids_.assign((size_t)decode_bucket, 0);
    seq_lens_.assign((size_t)n_slots, 0);
    for (int row = 0; row < live_count; ++row) {
        dec_tokens_[(size_t)row] = live_tokens_[(size_t)row];
        const int pos = live_positions_[(size_t)row];
        active_slot_ids_[(size_t)row] = live_slot_ids_[(size_t)row];
        state_slot_ids_[(size_t)row] = live_slot_ids_[(size_t)row];
        seq_lens_[(size_t)live_slot_ids_[(size_t)row]] = pos + 1;
        for (int h = 0; h < n_head_kv; ++h) {
            dec_rows_[(size_t)h * decode_bucket + row] =
                live_physical_rows_[(size_t)row];
        }
    }

    int n_prefill = 0;
    int n_commits = 0;
    std::vector<QwenPrefillSegment> segments;
    segments.reserve(prefills.size());
    for (size_t i = 0; i < prefills.size(); ++i) {
        const PrefillStage & prefill = prefills[i];
        const int slot = plan.prefills[i].slot;
        segments.push_back({n_prefill, prefill.chunk, slot});
        n_prefill += prefill.chunk;
        n_commits += prefill.commit ? 1 : 0;
        max_kv_len = std::max(max_kv_len, prefill.kv_pos + prefill.chunk);
        seq_lens_[(size_t)slot] = prefill.kv_pos + prefill.chunk;
    }
    const bool with_prefill = n_prefill > 0;
    const int n_total = n_prefill + decode_bucket;
    const Qwen35RoctxMetadata roctx_metadata{
        live_count, decode_bucket, n_prefill, (int)segments.size(),
        n_total, max_kv_len};
    const Qwen35RoctxRange roctx_step("qwen35.concurrent_step", roctx_metadata);
    const int gather_rows = with_prefill
        ? (with_decode ? n_commits + decode_bucket
                       : std::max(1, n_commits))
        : 0;

    if (timing) t_ar_build_start = timing_clock::now();
    bool built = false;
    if (with_prefill) {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0, /*n_tokens=*/n_total,
            /*with_mask=*/false, /*capture=*/capture_features_,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0, /*logits_tail_rows=*/0,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/with_decode ? decode_bucket : 1,
            /*seq_slot=*/0,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/n_prefill,
            segments.data(), (int)segments.size(), gather_rows,
            /*compact_slots=*/with_decode);
    } else {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0, /*n_tokens=*/decode_bucket,
            /*with_mask=*/false, /*capture=*/capture_features_,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0, /*logits_tail_rows=*/0,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/decode_bucket,
            /*seq_slot=*/0,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/0,
            /*prefill_segments=*/nullptr,
            /*n_prefill_segments=*/0,
            /*n_logits_rows=*/0,
            /*compact_slots=*/true);
    }
    if (!built || !sg.kv_write_rows ||
        (capture_features_ && !sg.target_feat_rows) ||
        (with_prefill &&
         (!sg.paged_query_seq_ids || !sg.paged_query_positions ||
          !sg.logits_row_indices))) {
        return fail_step("packed prefill/decode graph build failed");
    }
    if (timing) t_ar_build_end = timing_clock::now();

    embed_buf_.resize((size_t)hidden * n_total);
    int token_offset = 0;
    for (const PrefillStage & prefill : prefills) {
        std::copy(prefill.embeddings.begin(), prefill.embeddings.end(),
                  embed_buf_.begin() + (size_t)hidden * token_offset);
        token_offset += prefill.chunk;
    }
    if (with_decode &&
        !w.embedder.embed(
            dec_tokens_.data(), decode_bucket,
            embed_buf_.data() + (size_t)hidden * n_prefill)) {
        return fail_step("decode embedding failed");
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.inp_embed, embed_buf_.data(), 0,
        sizeof(float) * (size_t)hidden * n_total);

    pos_buf_.assign((size_t)4 * n_total, 0);
    token_offset = 0;
    for (const PrefillStage & prefill : prefills) {
        fill_qwen35_mrope_positions(
            pos_buf_.data(), n_total, token_offset,
            prefill.kv_pos, prefill.chunk);
        token_offset += prefill.chunk;
    }
    if (with_decode) {
        for (int row = 0; row < live_count; ++row) {
            const int pos = live_positions_[(size_t)row];
            const int packed_row = n_prefill + row;
            pos_buf_[(size_t)0 * n_total + packed_row] = pos;
            pos_buf_[(size_t)1 * n_total + packed_row] = pos;
            pos_buf_[(size_t)2 * n_total + packed_row] = pos;
        }
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.positions, pos_buf_.data(), 0,
        sizeof(int32_t) * pos_buf_.size());

    rows_buf_.assign((size_t)n_total * n_head_kv, scratch_row_);
    for (int h = 0; h < n_head_kv; ++h) {
        token_offset = 0;
        for (const PrefillStage & prefill : prefills) {
            for (int i = 0; i < prefill.chunk; ++i) {
                rows_buf_[(size_t)h * n_total + token_offset + i] =
                    prefill.rows[(size_t)i];
            }
            token_offset += prefill.chunk;
        }
        for (int row = 0; row < decode_bucket; ++row) {
            rows_buf_[(size_t)h * n_total + n_prefill + row] =
                dec_rows_[(size_t)h * decode_bucket + row];
        }
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.kv_write_rows, rows_buf_.data(), 0,
        sizeof(int64_t) * rows_buf_.size());

    if (capture_features_) {
        const int cap = b_.cache_.target_feat_cap;
        const int dead_row = cap * n_slots;
        feature_rows_.assign((size_t)n_total, dead_row);
        int feature_offset = 0;
        for (size_t i = 0; i < prefills.size(); ++i) {
            const PrefillStage & prefill = prefills[i];
            const int slot = plan.prefills[i].slot;
            for (int row = 0; row < prefill.chunk; ++row) {
                feature_rows_[(size_t)(feature_offset + row)] =
                    slot * cap + (prefill.kv_pos + row) % cap;
            }
            feature_offset += prefill.chunk;
        }
        for (int row = 0; row < live_count; ++row) {
            feature_rows_[(size_t)(n_prefill + row)] =
                live_slot_ids_[(size_t)row] * cap +
                live_positions_[(size_t)row] % cap;
        }
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.target_feat_rows, feature_rows_.data(), 0,
            sizeof(int32_t) * feature_rows_.size());
    }

    if (with_prefill) {
        query_slot_ids_.assign((size_t)n_total, -1);
        query_positions_.assign((size_t)n_total, -1);
        logits_rows_.clear();
        logits_rows_.reserve((size_t)gather_rows);
        token_offset = 0;
        for (size_t i = 0; i < prefills.size(); ++i) {
            const PrefillStage & prefill = prefills[i];
            const int slot = plan.prefills[i].slot;
            for (int row = 0; row < prefill.chunk; ++row) {
                query_slot_ids_[(size_t)(token_offset + row)] = slot;
                query_positions_[(size_t)(token_offset + row)] =
                    prefill.kv_pos + row;
            }
            if (prefill.commit) {
                logits_rows_.push_back(token_offset + prefill.chunk - 1);
            }
            token_offset += prefill.chunk;
        }
        for (int row = 0; row < live_count; ++row) {
            query_slot_ids_[(size_t)(n_prefill + row)] =
                live_slot_ids_[(size_t)row];
            query_positions_[(size_t)(n_prefill + row)] =
                live_positions_[(size_t)row];
        }
        for (int row = 0; row < decode_bucket; ++row) {
            logits_rows_.push_back(n_prefill + row);
        }
        if (logits_rows_.empty()) {
            logits_rows_.push_back(n_total - 1);
        }
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.paged_query_seq_ids,
            query_slot_ids_.data(), 0,
            sizeof(int32_t) * query_slot_ids_.size());
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.paged_query_positions,
            query_positions_.data(), 0,
            sizeof(int32_t) * query_positions_.size());
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.logits_row_indices,
            logits_rows_.data(), 0,
            sizeof(int32_t) * logits_rows_.size());
    }
    if (with_decode) {
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.active_slot_ids,
            active_slot_ids_.data(), 0,
            sizeof(int32_t) * active_slot_ids_.size());
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.state_slot_ids,
            state_slot_ids_.data(), 0,
            sizeof(int32_t) * state_slot_ids_.size());
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, b_.cache_.paged_kv_seq_lens,
        seq_lens_.data(), 0, sizeof(int32_t) * seq_lens_.size());

    if (timing) t_ar_exec_start = timing_clock::now();
    ggml_status st = GGML_STATUS_FAILED;
    {
        const Qwen35RoctxRange roctx_compute(
            "qwen35.graph_compute", roctx_metadata);
        st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    }
    if (st != GGML_STATUS_SUCCESS) {
        return fail_step("packed prefill/decode compute failed");
    }
    if (timing) t_ar_exec_end = timing_clock::now();

    const int decode_row0 = with_prefill ? n_commits : 0;
    const int argmax_rows = with_prefill ? gather_rows : decode_bucket;
    argmax_buf_.assign((size_t)argmax_rows, -1);
    ggml_backend_tensor_get_async(
        b_.target_backend_, sg.argmax_tokens, argmax_buf_.data(), 0,
        sizeof(int32_t) * argmax_buf_.size());
    {
        const Qwen35RoctxRange roctx_sync(
            "qwen35.argmax_readback", roctx_metadata);
        ggml_backend_synchronize(b_.target_backend_);
    }
    if (timing) t_ar_read_end = timing_clock::now();
    std::vector<int> write_slots;
    write_slots.reserve(live_slot_ids_.size() + prefills.size());
    write_slots.insert(write_slots.end(),
                       live_slot_ids_.begin(), live_slot_ids_.end());
    for (size_t i = 0; i < prefills.size(); ++i) {
        write_slots.push_back(plan.prefills[i].slot);
    }
    if (!commit_residency_writes(write_slots)) {
        return fail_step("KVFlash pending write commit failed");
    }

    for (size_t oi = 0; oi < inputs.size(); ++oi) {
        DecodeOutput & out = decode_outputs[oi];
        if (out.failed) continue;
        slots_.commit_step(out.slot);
        const int row = decode_row0 + output_rows_[oi];
        out.spec_service_ar_steps =
            spec_service_ar[oi] ? 1 : 0;
        out.token = sample_graph_row(
            out.slot, row, &argmax_buf_[(size_t)row], &logits_buf_);
    }

    for (DecodeOutput & out : decode_outputs) {
        if (out.failed) continue;
        std::string reselect_error;
        if (!maybe_reselect_residency(out.slot, reselect_error)) {
            return fail_step(reselect_error.empty()
                ? "KVFlash reselect failed" : reselect_error);
        }
        out.target_forwards = 1;
        attach_residency_telemetry(out);
    }

    int commit_row = 0;
    for (size_t i = 0; i < prefills.size(); ++i) {
        const int slot = plan.prefills[i].slot;
        PrefillOutput out;
        out.slot = slot;
        if (prefills[i].commit) {
            out.status = PrefillOutput::Status::completed;
            out.token = sample_graph_row(
                slot, commit_row, &argmax_buf_[(size_t)commit_row],
                &logits_buf_);
            ++commit_row;
            slots_.commit_prefill(slot);
        }
        prefill_outputs.push_back(std::move(out));
    }
    if (plan.prefills.empty() && live_count > 0 &&
        (timing || pending_ar_gate_plan.has_value())) {
        const auto t_ar_end = timing_clock::now();
        auto span_us = [](timing_clock::time_point from,
                          timing_clock::time_point to) {
            return std::chrono::duration<double, std::micro>(
                to - from).count();
        };
        if (pending_ar_gate_plan) {
            const double measured_us =
                span_us(decode_round_started, t_ar_end);
            speculation_gate_->observe_cost(
                {live_count, 0, 0, decode_bucket, round_draft_lanes_},
                measured_us);
            if (spec_gate_debug_enabled()) {
                log_spec_gate_plan(
                    *pending_ar_gate_plan, 0.0,
                    measured_us);
            }
        }
        if (timing) {
            std::fprintf(stderr,
                "[step-timing] {\"path\":\"ar\",\"live\":%d,\"k\":0,"
                "\"decode_bucket\":%d,\"n_prefill\":0,\"max_kv_len\":%d,"
                "\"draft_us\":%.1f,\"draft_lanes\":%d,"
                "\"pre_us\":%.1f,\"graph_build_us\":%.1f,"
                "\"graph_prepare_us\":%.1f,\"graph_exec_us\":%.1f,"
                "\"sample_read_us\":%.1f,\"finish_us\":%.1f,"
                "\"total_us\":%.1f,\"accepted_tokens\":0,"
                "\"emitted_tokens\":%d,\"target_forwards\":%d}\n",
                live_count, decode_bucket, max_kv_len,
                round_draft_us_, round_draft_lanes_,
                span_us(decode_round_started, t_ar_build_start),
                span_us(t_ar_build_start, t_ar_build_end),
                span_us(t_ar_build_end, t_ar_exec_start),
                span_us(t_ar_exec_start, t_ar_exec_end),
                span_us(t_ar_exec_end, t_ar_read_end),
                span_us(t_ar_read_end, t_ar_end),
                span_us(decode_round_started, t_ar_end),
                live_count, live_count);
        }
    }
    return result;
}

void Qwen35SeqEngine::retire(int slot) {
    if (!slots_.is_active(slot)) return;
    const uint64_t request_id = slots_.slot(slot).request_id;
    if (speculation_gate_) speculation_gate_->forget(request_id);
    if (slot >= 0 && slot < (int)last_activation_estimate_.size()) {
        last_activation_estimate_[(size_t)slot] = {};
        if (slot < (int)prepared_chain_drafts_.size()) {
            prepared_chain_drafts_[(size_t)slot].valid = false;
        }
    }
    if (slot >= 0 && slot < (int)adaptive_fallback_ar_.size()) {
        adaptive_fallback_ar_[(size_t)slot] = 0;
    }
    slots_.retire(slot);
}

}  // namespace dflash::common
