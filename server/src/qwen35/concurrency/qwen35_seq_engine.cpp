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
#include "internal.h"
#include "common/gpu_runtime_compat.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
extern "C++" to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);


namespace dflash::common {

namespace {

// Denser than pure power-of-2: reduces padding waste at non-power-of-2
// live counts (e.g. C=5 uses bucket=6 at 17% waste vs bucket=8 at 37.5%).
int decode_bucket_width(int live_count) {
    static constexpr int buckets[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    for (int b : buckets)
        if (b >= live_count) return b;
    return 64;
}

int forced_speculative_requests() {
    static const int forced = []() {
        const char * value = std::getenv(
            "DFLASH_ADAPTIVE_VERIFY_FORCE_SPECULATIVE_REQUESTS");
        return value ? std::max(0, std::atoi(value)) : -1;
    }();
    return forced;
}

} // namespace

Qwen35SeqEngine::Qwen35SeqEngine(
        Qwen35Backend & backend, PagedKvPool & pool, int max_ctx,
        int64_t scratch_row, int tree_width, int tree_scratch_base,
        int tree_scratch_stride, int max_prefills,
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
      speculation_gate_(SpecGateConfig{}, std::max(1, tree_width)),
      scratch_row_(scratch_row), tree_width_(tree_width),
      tree_scratch_base_(tree_scratch_base),
      tree_scratch_stride_(tree_scratch_stride) {
    const int n_slots = slots_.slot_count();
    slot_draft_kv_.resize((size_t)n_slots);

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

bool Qwen35SeqEngine::ddtree_available(const StepPlan & plan) const {
    return tree_width_ > 1 && capture_features_ && plan.prefills.empty() &&
        !plan.decode.empty() && b_.dw_.block_size > 1 &&
        b_.cfg_.ddtree_budget + 1 == tree_width_;
}

bool Qwen35SeqEngine::ddtree_input_eligible(const StepInput & in) const {
    if (!in.allow_speculation || in.slot < 0 ||
        in.slot >= slots_.slot_count() ||
        !slots_.slot(in.slot).decoding() ||
        slots_.slot(in.slot).sampler.needs_logit_processing() ||
        slots_.slot(in.slot).cur_pos < 1 ||
        slots_.slot(in.slot).cur_pos >= slots_.max_context()) {
        return false;
    }
    const int min_floor = []() {
        const char * value = std::getenv("DFLASH_MIN_TOKENS");
        return value ? std::max(0, std::atoi(value)) : 0;
    }();
    return slots_.slot(in.slot).generated_tokens() >= min_floor;
}

std::optional<SeqEngine::StepResult> Qwen35SeqEngine::step_ddtree(
        const StepPlan & speculative_plan, const StepPlan & ar_plan) {
    StepResult result;
    const int active = (int)speculative_plan.decode.size();
    const int total_active = active + (int)ar_plan.decode.size();
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
        for (const StepInput & in : speculative_plan.decode) {
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
    for (const StepInput & in : speculative_plan.decode) {
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

    const bool target_is_meta = b_.cache_.ssm_state.empty() ||
        !b_.cache_.ssm_state.front() ||
        ggml_backend_buft_is_meta(ggml_backend_buffer_get_type(
            b_.cache_.ssm_state.front()->buffer));
    const bool direct_commit =
        !target_is_meta && bucket <= b_.cache_.tree_capture_lanes &&
        T == b_.cache_.tree_capture_width &&
        b_.cache_.target_feat_tree_scratch_base > 0;
    struct DirectArStage {
        Qwen35SlotManager::StepAppend append;
    };
    std::vector<DirectArStage> direct_ar;
    if (direct_commit) {
        direct_ar.resize(ar_plan.decode.size());
        for (size_t i = 0; i < ar_plan.decode.size(); ++i) {
            const StepInput & in = ar_plan.decode[i];
            DirectArStage & staged = direct_ar[i];
            staged.append = slots_.append_token(in.slot, in.token);
            const bool table_ok = staged.append.ok &&
                (slots_.residency_active() ||
                 upload_block_table_delta(
                     in.slot, staged.append.first_new_block,
                     staged.append.new_blocks.data(),
                     staged.append.new_blocks.size()));
            if (!table_ok || staged.append.physical_rows.size() != 1) {
                result.error = staged.append.busy
                    ? "paged KV pool exhausted during mixed DDTree/AR step"
                    : "mixed DDTree/AR append failed";
                return result;
            }
        }
        if (!direct_ar.empty() && !upload_all_active_block_tables()) {
            result.error =
                "mixed DDTree/AR block-table refresh failed";
            return result;
        }
    }

    StepGraph & tree_sg = b_.sg_;
    int max_prefix = 1;
    for (const Proposal & p : proposals) {
        max_prefix = std::max(max_prefix, slots_.slot(p.slot).cur_pos);
    }
    for (const DirectArStage & staged : direct_ar) {
        max_prefix = std::max(max_prefix, staged.append.position + 1);
    }
    const int n_ar = (int)direct_ar.size();
    if (!build_target_step_paged_tree(
            tree_sg, b_.w_, b_.cache_, b_.target_backend_, T, bucket,
            max_prefix, tree_scratch_base_, tree_scratch_stride_,
            b_.cfg_.kq_stride_pad, direct_commit, n_ar)) {
        result.error = "packed DDTree verify graph build failed";
        return result;
    }

    const int total_tree = T * bucket;
    const int total_packed = total_tree + n_ar;
    std::vector<int32_t> tree_feature_rows;
    if (direct_commit) {
        tree_feature_rows.resize((size_t)total_packed);
        for (int row = 0; row < total_tree; ++row) {
            tree_feature_rows[(size_t)row] =
                b_.cache_.target_feat_tree_scratch_base + row;
        }
    }
    std::vector<int32_t> flat_tokens((size_t)total_packed, 0);
    std::vector<int32_t> parents((size_t)total_tree, -1);
    std::vector<int32_t> sizes((size_t)bucket, 0);
    // Negative active IDs identify bucket padding. Recurrent gathers cannot
    // index a negative slab, so padded trees use slot 0 only for their
    // read-only base-state gather; tree_size=0/query_slot=-1 keeps all of
    // their attention/output rows inactive and tree mode never persists it.
    std::vector<int32_t> tree_slots((size_t)bucket, -1);
    std::vector<int32_t> tree_state_slots((size_t)bucket, 0);
    std::vector<int32_t> ar_slots((size_t)n_ar, -1);
    std::vector<int32_t> ar_state_slots((size_t)n_ar, 0);
    std::vector<int32_t> query_slots((size_t)total_packed, -1);
    std::vector<int32_t> query_positions((size_t)total_packed, -1);
    std::vector<int64_t> tree_rows(
        (size_t)total_packed * n_head_kv, scratch_row_);
    std::vector<int32_t> tree_pos((size_t)4 * total_packed, 0);
    std::vector<float> tree_embed((size_t)hidden * total_packed, 0.0f);
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
            tree_pos[(size_t)0 * total_packed + row] = pos;
            tree_pos[(size_t)1 * total_packed + row] = pos;
            tree_pos[(size_t)2 * total_packed + row] = pos;
            for (int h = 0; h < n_head_kv; ++h) {
                tree_rows[(size_t)h * total_packed + row] =
                    (int64_t)tree_scratch_base_ +
                    (int64_t)p.slot * tree_scratch_stride_ + node;
            }
        }
    }
    const int feature_cap = b_.cache_.target_feat_cap;
    for (int i = 0; i < n_ar; ++i) {
        const StepInput & in = ar_plan.decode[(size_t)i];
        const Qwen35SlotManager::StepAppend & app =
            direct_ar[(size_t)i].append;
        const int row = total_tree + i;
        flat_tokens[(size_t)row] = in.token;
        ar_slots[(size_t)i] = in.slot;
        ar_state_slots[(size_t)i] = in.slot;
        query_slots[(size_t)row] = in.slot;
        query_positions[(size_t)row] = app.position;
        seq_lens_[(size_t)in.slot] = app.position + 1;
        tree_pos[(size_t)0 * total_packed + row] = app.position;
        tree_pos[(size_t)1 * total_packed + row] = app.position;
        tree_pos[(size_t)2 * total_packed + row] = app.position;
        tree_feature_rows[(size_t)row] =
            in.slot * feature_cap + app.position % feature_cap;
        for (int h = 0; h < n_head_kv; ++h) {
            tree_rows[(size_t)h * total_packed + row] =
                app.physical_rows[0];
        }
    }
    if (!b_.w_.embedder.embed(
            flat_tokens.data(), total_packed, tree_embed.data())) {
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
    if (n_ar > 0) {
        ggml_backend_tensor_set(
            tree_sg.ar_active_slot_ids, ar_slots.data(), 0,
            sizeof(int32_t) * ar_slots.size());
        ggml_backend_tensor_set(
            tree_sg.ar_state_slot_ids, ar_state_slots.data(), 0,
            sizeof(int32_t) * ar_state_slots.size());
        ggml_backend_tensor_set(
            tree_sg.paged_query_positions, query_positions.data(), 0,
            sizeof(int32_t) * query_positions.size());
    }
    ggml_backend_tensor_set(tree_sg.paged_query_seq_ids, query_slots.data(), 0,
                            sizeof(int32_t) * query_slots.size());
    ggml_backend_tensor_set(tree_sg.kv_write_rows, tree_rows.data(), 0,
                            sizeof(int64_t) * tree_rows.size());
    if (direct_commit) {
        ggml_backend_tensor_set(
            tree_sg.target_feat_rows, tree_feature_rows.data(), 0,
            sizeof(int32_t) * tree_feature_rows.size());
    }
    ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
                            sizeof(int32_t) * seq_lens_.size());
    if (ggml_backend_graph_compute(b_.target_backend_, tree_sg.gf) !=
        GGML_STATUS_SUCCESS) {
        result.error = "packed DDTree verify compute failed";
        return result;
    }
    std::vector<int32_t> posterior((size_t)total_packed, -1);
    ggml_backend_tensor_get(tree_sg.argmax_tokens, posterior.data(), 0,
                            sizeof(int32_t) * posterior.size());
    std::vector<int32_t> direct_ar_next((size_t)n_ar, -1);
    for (int i = 0; i < n_ar; ++i) {
        const int row = total_tree + i;
        direct_ar_next[(size_t)i] = sample_graph_row(
            ar_plan.decode[(size_t)i].slot, row,
            &posterior[(size_t)row], &logits_buf_);
        if (direct_ar_next[(size_t)i] < 0) {
            result.error =
                "mixed DDTree/AR graph produced an invalid AR token";
            return result;
        }
    }

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
    if (direct_commit) {
        struct DirectAppend {
            Qwen35SlotManager::StepAppend append;
            std::vector<int32_t> tokens;
        };
        std::vector<DirectAppend> appends((size_t)active);
        std::vector<int> write_slots;
        write_slots.reserve((size_t)total_active);
        for (const StepInput & in : ar_plan.decode) {
            write_slots.push_back(in.slot);
        }

        for (int s = 0; s < active; ++s) {
            Proposal & p = proposals[(size_t)s];
            DirectAppend & staged = appends[(size_t)s];
            staged.tokens.reserve(p.accepted.size());
            for (int dfs : p.accepted) {
                staged.tokens.push_back(dfs == 0 ? p.root :
                    p.tree.token_ids[(size_t)dfs - 1]);
            }
            staged.append = slots_.append_tokens(
                p.slot, staged.tokens.data(), (int)staged.tokens.size());
            const bool table_ok = staged.append.ok &&
                (slots_.residency_active() ||
                 upload_block_table_delta(
                     p.slot, staged.append.first_new_block,
                     staged.append.new_blocks.data(),
                     staged.append.new_blocks.size()));
            if (!table_ok ||
                staged.append.physical_rows.size() != staged.tokens.size()) {
                result.error = staged.append.busy
                    ? "paged KV pool exhausted during DDTree direct commit"
                    : "DDTree direct commit K/V append failed";
                return result;
            }
            write_slots.push_back(p.slot);
        }

        const size_t n_delta = b_.cache_.ssm_state.size();
        if (tree_sg.delta_captures.size() != n_delta ||
            b_.cache_.conv_state.size() != n_delta) {
            result.error = "DDTree direct commit capture count mismatch";
            return result;
        }
        const auto to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        if (!to_fp32) {
            result.error = "DDTree direct commit has no F16 state converter";
            return result;
        }

        cudaStream_t stream = nullptr;
        auto copy_2d = [&](void * dst, size_t dst_pitch,
                           const void * src, size_t src_pitch,
                           size_t width, size_t height) {
            return cudaMemcpy2DAsync(
                       dst, dst_pitch, src, src_pitch, width, height,
                       cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
        };
        auto copy_row = [&](ggml_tensor * tensor, int64_t src_row,
                            int64_t dst_row) {
            if (!tensor || src_row < 0 || dst_row < 0 ||
                src_row >= tensor->ne[1] || dst_row >= tensor->ne[1]) {
                return false;
            }
            const size_t row_bytes = tensor->nb[1];
            for (int h = 0; h < (int)tensor->ne[2]; ++h) {
                const char * src = (const char *)tensor->data +
                    (size_t)h * tensor->nb[2] +
                    (size_t)src_row * tensor->nb[1];
                char * dst = (char *)tensor->data +
                    (size_t)h * tensor->nb[2] +
                    (size_t)dst_row * tensor->nb[1];
                if (cudaMemcpyAsync(
                        dst, src, row_bytes, cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
                    return false;
                }
            }
            return true;
        };

        const int conv_kernel = b_.w_.ssm_d_conv;
        if (conv_kernel < 2) {
            result.error = "DDTree direct commit has invalid conv kernel";
            return result;
        }
        for (size_t il = 0; il < n_delta; ++il) {
            const DeltaNetCapture & cap = tree_sg.delta_captures[il];
            ggml_tensor * state = b_.cache_.ssm_state[il];
            ggml_tensor * conv = b_.cache_.conv_state[il];
            if (!cap.ssm_intermediate_states || !cap.conv_input ||
                !state || !conv ||
                cap.ssm_intermediate_states->type != GGML_TYPE_F16 ||
                cap.ssm_intermediate_states->ne[3] < T * bucket ||
                cap.conv_input->ne[2] < bucket) {
                result.error = "DDTree direct commit capture layout mismatch";
                return result;
            }
            const int64_t state_elems =
                state->ne[0] * state->ne[1] * state->ne[2];
            for (int s = 0; s < active; ++s) {
                const Proposal & p = proposals[(size_t)s];
                const int deepest = p.accepted.back();
                const int capture_row = s * T + deepest;
                const char * state_src =
                    (const char *)cap.ssm_intermediate_states->data +
                    (size_t)capture_row *
                        cap.ssm_intermediate_states->nb[3];
                float * state_dst = (float *)((char *)state->data +
                    (size_t)p.slot * state->nb[3]);
                to_fp32(state_src, state_dst, state_elems, stream);
                if (cudaPeekAtLastError() != cudaSuccess) {
                    result.error = "DDTree direct commit SSM conversion failed";
                    return result;
                }

                std::vector<int> ancestry((size_t)conv_kernel - 1);
                ancestry.back() = deepest;
                for (int k = conv_kernel - 3; k >= 0; --k) {
                    const int next = ancestry[(size_t)k + 1];
                    ancestry[(size_t)k] =
                        next >= 0 ? p.tree.parents[(size_t)next] : next - 1;
                }
                for (int k = 0; k < conv_kernel - 1; ++k) {
                    const int source_col =
                        conv_kernel - 1 + ancestry[(size_t)k];
                    if (source_col < 0 ||
                        source_col >= cap.conv_input->ne[0]) {
                        result.error =
                            "DDTree direct commit conv ancestry is invalid";
                        return result;
                    }
                    const char * conv_src =
                        (const char *)cap.conv_input->data +
                        (size_t)s * cap.conv_input->nb[2] +
                        (size_t)source_col *
                            ggml_element_size(cap.conv_input);
                    char * conv_dst = (char *)conv->data +
                        (size_t)p.slot * conv->nb[2] +
                        (size_t)k * ggml_element_size(conv);
                    if (!copy_2d(
                            conv_dst, conv->nb[1], conv_src,
                            cap.conv_input->nb[1],
                            ggml_element_size(conv), conv->ne[1])) {
                        result.error =
                            "DDTree direct commit conv state copy failed";
                        return result;
                    }
                }
            }
        }

        for (int s = 0; s < active; ++s) {
            const Proposal & p = proposals[(size_t)s];
            const DirectAppend & staged = appends[(size_t)s];
            for (size_t d = 0; d < p.accepted.size(); ++d) {
                const int dfs = p.accepted[d];
                const int64_t src_row =
                    (int64_t)tree_scratch_base_ +
                    (int64_t)p.slot * tree_scratch_stride_ + dfs;
                const int64_t dst_row = staged.append.physical_rows[d];
                for (size_t il = 0; il < b_.cache_.attn_k.size(); ++il) {
                    if (!copy_row(b_.cache_.attn_k[il], src_row, dst_row) ||
                        !copy_row(b_.cache_.attn_v[il], src_row, dst_row)) {
                        result.error =
                            "DDTree direct commit paged K/V copy failed";
                        return result;
                    }
                }

                ggml_tensor * feat = b_.cache_.target_feat;
                const int src_feat =
                    b_.cache_.target_feat_tree_scratch_base + s * T + dfs;
                const int dst_feat = p.slot * b_.cache_.target_feat_cap +
                    (staged.append.position + (int)d) %
                        b_.cache_.target_feat_cap;
                if (!feat || src_feat < 0 || dst_feat < 0 ||
                    src_feat >= feat->ne[1] || dst_feat >= feat->ne[1] ||
                    cudaMemcpyAsync(
                        (char *)feat->data +
                            (size_t)dst_feat * feat->nb[1],
                        (const char *)feat->data +
                            (size_t)src_feat * feat->nb[1],
                        feat->nb[1], cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
                    result.error =
                        "DDTree direct commit target feature copy failed";
                    return result;
                }
            }
        }

        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            result.error = "DDTree direct commit synchronization failed";
            return result;
        }
        if (!commit_residency_writes(write_slots)) {
            result.error = "DDTree direct commit residency write failed";
            return result;
        }
        for (int slot : write_slots) slots_.commit_step(slot);
        if (!upload_all_active_block_tables()) {
            result.error = "DDTree direct commit block-table refresh failed";
            return result;
        }


        for (int slot : write_slots) {
            std::string reselect_error;
            if (!maybe_reselect_residency(slot, reselect_error)) {
                result.error = reselect_error.empty()
                    ? "KVFlash reselect failed" : reselect_error;
                return result;
            }
        }

        result.decode.reserve((size_t)total_active);
        for (Proposal & p : proposals) {
            if (p.bonus < 0 || p.bonus >= b_.w_.n_vocab) {
                result.error =
                    "DDTree direct commit produced an invalid pending token";
                return result;
            }
            DecodeOutput out;
            out.slot = p.slot;
            out.token = p.bonus;
            out.ddtree_steps = 1;
            out.ddtree_accepted_tokens =
                (uint64_t)((int)p.accepted.size() - 1);
            out.target_forwards = 1;
            for (size_t i = 1; i < p.accepted.size(); ++i) {
                const int dfs = p.accepted[i];
                out.committed_tokens.push_back(
                    p.tree.token_ids[(size_t)dfs - 1]);
            }
            attach_residency_telemetry(out);
            result.decode.push_back(std::move(out));
        }
        for (size_t i = 0; i < ar_plan.decode.size(); ++i) {
            DecodeOutput out;
            out.slot = ar_plan.decode[i].slot;
            out.token = direct_ar_next[i];
            out.target_forwards = 1;
            attach_residency_telemetry(out);
            result.decode.push_back(std::move(out));
        }
        static const bool direct_commit_diag =
            std::getenv("DFLASH_DDTREE_DIRECT_COMMIT_DIAG") != nullptr;
        if (direct_commit_diag) {
            std::fprintf(stderr,
                "[parallel-ddtree] mode=one-pass-tree-ar speculative=%d ar=%zu\n",
                active, ar_plan.decode.size());
        }
        return result;
    }
    replay_total += (int)ar_plan.decode.size();

    std::vector<QwenPrefillSegment> replay_segments;
    std::vector<int32_t> replay_tokens;
    std::vector<int32_t> replay_slots;
    std::vector<int32_t> replay_positions;
    std::vector<int64_t> replay_rows;
    std::vector<int32_t> replay_logits_rows;
    replay_segments.reserve((size_t)total_active);
    replay_tokens.reserve((size_t)replay_total);
    replay_slots.reserve((size_t)replay_total);
    replay_positions.reserve((size_t)replay_total);
    replay_rows.assign((size_t)replay_total * n_head_kv, scratch_row_);
    replay_logits_rows.reserve((size_t)total_active);
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
    for (const StepInput & in : ar_plan.decode) {
        const Qwen35SlotManager::StepAppend app = slots_.append_tokens(
            in.slot, &in.token, 1);
        const bool table_ok = slots_.residency_active() ||
            upload_block_table_delta(in.slot, app.first_new_block,
                app.new_blocks.data(), app.new_blocks.size());
        if (!app.ok || app.physical_rows.size() != 1 || !table_ok) {
            result.error = app.busy
                ? "paged KV pool exhausted during mixed DDTree/AR replay"
                : "mixed DDTree/AR replay K/V append failed";
            return result;
        }
        replay_segments.push_back({replay_offset, 1, in.slot});
        replay_tokens.push_back(in.token);
        replay_slots.push_back(in.slot);
        replay_positions.push_back(app.position);
        for (int h = 0; h < n_head_kv; ++h) {
            replay_rows[(size_t)h * replay_total + replay_offset] =
                app.physical_rows[0];
        }
        ++replay_offset;
        replay_logits_rows.push_back(replay_offset - 1);
        seq_lens_[(size_t)in.slot] = app.position + 1;
    }
    if (replay_offset != replay_total ||
        (int)replay_segments.size() != total_active ||
        (int)replay_logits_rows.size() != total_active) {
        result.error = "mixed DDTree/AR replay staging mismatch";
        return result;
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
            (int)replay_segments.size(), total_active, false) ||
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
    replay_write_slots.reserve((size_t)total_active);
    for (const Proposal & p : proposals) replay_write_slots.push_back(p.slot);
    for (const StepInput & in : ar_plan.decode) {
        replay_write_slots.push_back(in.slot);
    }
    if (!commit_residency_writes(replay_write_slots)) {
        result.error = "mixed DDTree/AR replay KV write commit failed";
        return result;
    }
    for (int slot : replay_write_slots) slots_.commit_step(slot);

    // The replay is the one durable target forward for both routes. Its
    // recurrent/KV/feature state is what the next step consumes.
    std::vector<int32_t> replay_next((size_t)total_active, -1);
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
    std::vector<int32_t> ar_next(ar_plan.decode.size(), -1);
    for (size_t i = 0; i < ar_plan.decode.size(); ++i) {
        const int logits_row = active + (int)i;
        ar_next[i] = sample_graph_row(
            ar_plan.decode[i].slot, logits_row,
            &replay_next[(size_t)logits_row], &logits_buf_);
        if (ar_next[i] < 0) {
            result.error = "mixed replay produced an invalid AR token";
            return result;
        }
    }

    for (int slot : replay_write_slots) {
        std::string reselect_error;
        if (!maybe_reselect_residency(slot, reselect_error)) {
            result.error = reselect_error.empty()
                ? "KVFlash reselect failed" : reselect_error;
            return result;
        }
    }

    result.decode.reserve((size_t)total_active);
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
        attach_residency_telemetry(out);
        result.decode.push_back(std::move(out));
    }
    for (size_t i = 0; i < ar_plan.decode.size(); ++i) {
        DecodeOutput out;
        out.slot = ar_plan.decode[i].slot;
        out.token = ar_next[i];
        out.target_forwards = 1;
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

    if (!ddtree_available(plan)) return step_regular(plan);

    const char * adaptive = std::getenv("DFLASH_DDTREE_ADAPTIVE");
    const bool adaptive_enabled = !(adaptive && std::atoi(adaptive) == 0);
    const int C = static_cast<int>(inputs.size());
    const int k_cap = std::min(
        C, std::max(0, b_.cache_.tree_capture_lanes));

    // DDTree's adapter has no cold prior: normal speculative rounds provide
    // both accepted yield and the hardware cost. DSpark can later populate
    // prior_accept from its calibrated confidence head without gate changes.
    std::vector<SpecCandidate> candidates;
    candidates.reserve(inputs.size());
    for (const StepInput & in : inputs) {
        const Qwen35Slot & seq = slots_.slot(in.slot);
        candidates.push_back({
            in.slot,
            seq.request_id,
            in.speculation_policy,
            ddtree_input_eligible(in),
            std::numeric_limits<double>::quiet_NaN(),
            seq.generated_tokens(),
        });
    }

    const int oracle_limit = forced_speculative_requests();
    const bool gate_active = adaptive_enabled && oracle_limit < 0;
    std::vector<int> spec_slots;
    if (gate_active) {
        spec_slots = speculation_gate_.plan(C, candidates, k_cap);
    } else if (oracle_limit >= 0) {
        // Explicit Always requests remain mandatory even when the benchmark
        // oracle asks for a smaller synthetic speculative subbatch.
        for (const SpecCandidate & candidate : candidates) {
            if (candidate.eligible &&
                candidate.policy == SpeculationPolicy::Always) {
                spec_slots.push_back(candidate.slot);
            }
        }
        const int target = std::max(
            oracle_limit, static_cast<int>(spec_slots.size()));
        for (const SpecCandidate & candidate : candidates) {
            if ((int)spec_slots.size() >= target) break;
            if (!candidate.eligible ||
                candidate.policy != SpeculationPolicy::Adaptive) {
                continue;
            }
            spec_slots.push_back(candidate.slot);
        }
    } else {
        // Burn-in/parity mode: preserve fixed speculation for every eligible
        // request except an explicit per-request Never override.
        for (const SpecCandidate & candidate : candidates) {
            if (candidate.eligible &&
                candidate.policy != SpeculationPolicy::Never) {
                spec_slots.push_back(candidate.slot);
            }
        }
    }
    if (gate_active && (int)spec_slots.size() > k_cap) {
        return fail_step(
            "forced speculation requests exceed DDTree executor capacity");
    }

    std::vector<uint8_t> selected((size_t)n_slots, 0);
    for (int slot : spec_slots) {
        if (slot >= 0 && slot < n_slots) selected[(size_t)slot] = 1;
    }

    StepPlan speculative_plan;
    StepPlan ar_plan;
    speculative_plan.decode.reserve(spec_slots.size());
    ar_plan.decode.reserve(inputs.size() - spec_slots.size());
    for (const StepInput & in : inputs) {
        const bool eligible = ddtree_input_eligible(in);
        if (eligible && selected[(size_t)in.slot]) {
            speculative_plan.decode.push_back(in);
        } else {
            ar_plan.decode.push_back(in);
        }
    }

    using Clock = std::chrono::steady_clock;
    StepResult routed_result;
    const int speculative_count =
        static_cast<int>(speculative_plan.decode.size());
    const auto started = Clock::now();
    if (speculative_count > 0) {
        std::optional<StepResult> mixed =
            step_ddtree(speculative_plan, ar_plan);
        if (!mixed) {
            // Proposal setup failed before target/cache mutation. Preserve
            // service with one ordinary packed step and retry speculation on a
            // later iteration. Do not learn from this contaminated timing.
            return step_regular(plan);
        }
        if (!mixed->ok()) return std::move(*mixed);
        routed_result = std::move(*mixed);
    } else {
        routed_result = step_regular(ar_plan);
        if (!routed_result.ok()) return routed_result;
    }
    const double route_us = std::max(
        1.0, std::chrono::duration<double, std::micro>(
                 Clock::now() - started).count());

    if (gate_active) {
        if (speculative_count == 0) {
            speculation_gate_.observe_ar(C, route_us);
        } else {
            std::vector<std::pair<std::uint64_t, int>> accepted_yields;
            accepted_yields.reserve((size_t)speculative_count);
            for (const DecodeOutput & out : routed_result.decode) {
                if (out.failed || out.slot < 0 || out.slot >= n_slots ||
                    !selected[(size_t)out.slot]) {
                    continue;
                }
                accepted_yields.emplace_back(
                    slots_.slot(out.slot).request_id,
                    static_cast<int>(out.ddtree_accepted_tokens + 1));
            }
            speculation_gate_.observe_spec(
                C, speculative_count, route_us, accepted_yields);
        }
    }

    std::vector<DecodeOutput> by_slot((size_t)n_slots);
    std::vector<uint8_t> present((size_t)n_slots, 0);
    auto collect = [&](std::vector<DecodeOutput> & outputs) {
        for (DecodeOutput & out : outputs) {
            if (out.slot < 0 || out.slot >= n_slots ||
                present[(size_t)out.slot]) {
                return false;
            }
            present[(size_t)out.slot] = 1;
            by_slot[(size_t)out.slot] = std::move(out);
        }
        return true;
    };
    if (!collect(routed_result.decode)) {
        return fail_step("adaptive decode produced duplicate slot output");
    }
    result.decode.reserve(inputs.size());
    for (const StepInput & in : inputs) {
        if (!present[(size_t)in.slot]) {
            return fail_step("adaptive decode omitted a live slot output");
        }
        result.decode.push_back(std::move(by_slot[(size_t)in.slot]));
    }
    return result;
}

SeqEngine::StepResult Qwen35SeqEngine::step_regular(const StepPlan & plan) {
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
    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden = w.n_embd;
    const int n_head_kv = w.n_head_kv;

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

    ggml_status st = GGML_STATUS_FAILED;
    {
        const Qwen35RoctxRange roctx_compute(
            "qwen35.graph_compute", roctx_metadata);
        st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    }
    if (st != GGML_STATUS_SUCCESS) {
        return fail_step("packed prefill/decode compute failed");
    }

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
    return result;
}

void Qwen35SeqEngine::retire(int slot) {
    if (!slots_.is_active(slot)) return;
    speculation_gate_.forget(slots_.slot(slot).request_id);
    slots_.retire(slot);
}

}  // namespace dflash::common
