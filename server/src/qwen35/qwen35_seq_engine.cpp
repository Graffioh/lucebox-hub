// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "graph_builders.h"
#include "attn_masks.h"
#include "prefill_staging.h"
#include "common/sampler.h"
#include "internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

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

// A full Qwen staging set consists of K/V plus the recurrent SSM/conv slabs.
// Count only complete extra sets so a partially initialized cache fails safe.
int staging_set_capacity(const TargetCache & cache) {
    if (cache.n_seq_slots <= 1) return 1;
    if (cache.staging_k.empty() ||
        cache.staging_v.empty() || cache.staging_ssm_state.empty() ||
        cache.staging_conv_state.empty()) {
        return 0;
    }
    const size_t n_extra = std::min({
        cache.staging_k_extra.size(),
        cache.staging_v_extra.size(),
        cache.staging_ssm_state_extra.size(),
        cache.staging_conv_state_extra.size(),
    });
    return 1 + (int)n_extra;
}


} // namespace

bool Qwen35SeqEngine::token_is_eos(int32_t token) const {
    return b_.token_is_eos(token);
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    const int lease_capacity =
        std::min(max_prefills_, staging_set_capacity(b_.cache_));
    std::vector<uint8_t> staging_in_use((size_t)lease_capacity, 0);
    for (const auto & pending : pending_) {
        if (!pending) continue;
        const int idx = pending->staging_idx;
        if (idx >= 0 && idx < lease_capacity) {
            staging_in_use[(size_t)idx] = 1;
        }
    }
    const auto free_lease = std::find(
        staging_in_use.begin(), staging_in_use.end(), (uint8_t)0);
    if (free_lease == staging_in_use.end()) {
        return {false, true, -1, "all prefill staging lanes are busy"};
    }
    const int staging_idx = (int)(free_lease - staging_in_use.begin());
    AdmitResult r =
        slots_.admit(request_id, prompt, sampler);
    if (!r.ok) return r;
    const int slot = r.slot;

    if ((size_t)slot >= pending_.size() || pending_[(size_t)slot]) {
        slots_.retire(slot);
        return {false, false, -1, "invalid prefill slot state"};
    }

    PendingPrefill pp;
    pp.slot = slot;
    pp.staging_idx = staging_idx;
    pp.prompt = prompt;
    pending_[(size_t)slot] = std::move(pp);

    return r;
}

int32_t Qwen35SeqEngine::sample_graph_row(
        int slot, int logits_row, const int32_t * cached_argmax,
        std::vector<float> * logits_scratch) {
    const TargetWeights & w = b_.w_;
    const int vocab = w.n_vocab;
    SeqSlot & seq = slots_.slot(slot);
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
        token, (int)seq.sample_history.size(),
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

void Qwen35SeqEngine::fail_pending_prefill(
        int slot, std::vector<StepOutput> & outputs,
        const char * log_message, const char * client_message) {
    if (slot < 0 || (size_t)slot >= pending_.size() || !pending_[(size_t)slot]) return;
    std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                 log_message, slot);
    StepOutput out;
    out.slot = slot;
    out.failed = true;
    out.error = client_message;
    out.prefill_done = true;
    outputs.push_back(std::move(out));
    pending_[(size_t)slot].reset();
}

Qwen35SeqEngine::PrefillStage Qwen35SeqEngine::stage_prefill_chunk(
        int slot, int max_tokens, int staging_idx,
        std::vector<StepOutput> & outputs) {
    PrefillStage stage;
    if (slot < 0 || (size_t)slot >= pending_.size() || !pending_[(size_t)slot])
        return stage;

    PendingPrefill & pending = *pending_[(size_t)slot];
    stage.kv_pos = pending.progress;
    stage.staging_idx = staging_idx;
    if (stage.kv_pos == 0) {
        reset_prefill_staging(b_.cache_, staging_idx);
    }
    stage.chunk = std::min(
        max_tokens, (int)pending.prompt.size() - stage.kv_pos);
    if (stage.chunk <= 0) return PrefillStage{};
    stage.commit = stage.kv_pos + stage.chunk >= (int)pending.prompt.size();

    SeqSlotManager::PrefillChunk chunk =
        slots_.append_prefill(slot, stage.chunk);
    if (chunk.busy) return PrefillStage{};
    if (!chunk.ok || chunk.rows.size() != (size_t)stage.chunk) {
        fail_pending_prefill(slot, outputs, "prefill K/V allocation failed",
                             "prefill K/V allocation failed");
        return PrefillStage{};
    }
    if (!upload_block_table_delta(
            slot, chunk.first_new_block, chunk.new_blocks.data(),
            chunk.new_blocks.size())) {
        fail_pending_prefill(
            slot, outputs, "prefill block-table delta exceeds device capacity",
            "prefill block-table update failed");
        return PrefillStage{};
    }

    stage.rows = std::move(chunk.rows);
    stage.embeddings.resize((size_t)b_.w_.n_embd * stage.chunk);
    if (!b_.w_.embedder.embed(
            pending.prompt.data() + stage.kv_pos, stage.chunk,
            stage.embeddings.data())) {
        fail_pending_prefill(slot, outputs, "prefill embed failed",
                             "prefill embedding failed");
        return PrefillStage{};
    }
    stage.ready = true;
    return stage;
}

bool Qwen35SeqEngine::run_prefill_graph(
        const PrefillStage & prefill, int prefill_slot,
        std::vector<StepOutput> & outputs) {
    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden = w.n_embd;
    const int chunk = prefill.chunk;
    const bool commit = prefill.commit;
    const int kv_pos = prefill.kv_pos;
    const int staging_idx = prefill.staging_idx;

    const bool with_mask =
        (b_.cfg_.kq_stride_pad > KQ_MASK_PAD) || (chunk > 1);
    bool built = build_target_step(
        sg, w, b_.cache_, b_.target_backend_,
        /*kv_start=*/kv_pos, /*n_tokens=*/chunk,
        with_mask, /*capture=*/false,
        /*capture_delta_intermediate=*/false,
        /*fa_window=*/0,
        /*logits_tail_rows=*/1,
        b_.cfg_.kq_stride_pad,
        /*capture_moe_router=*/false,
        /*kvflash_mask=*/false,
        /*capture_qk=*/false,
        /*paged_attention=*/false,
        /*n_seqs=*/1,
        /*seq_slot=*/prefill_slot,
        /*paged_prefill=*/true,
        /*paged_max_kv_len=*/0,
        /*n_prefill_tokens=*/0,
        /*prefill_commit=*/commit,
        /*compact_slots=*/false,
        /*staging_idx=*/staging_idx);
    if (!built || !sg.kv_write_rows) {
        fail_pending_prefill(prefill_slot, outputs,
                             "prefill graph build failed",
                             "prefill graph build failed");
        return false;
    }

    const int n_total = chunk;
    embed_buf_.resize((size_t)hidden * n_total);
    std::copy(prefill.embeddings.begin(), prefill.embeddings.end(),
              embed_buf_.begin());
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.inp_embed, embed_buf_.data(), 0,
        sizeof(float) * (size_t)hidden * n_total);

    pos_buf_.resize((size_t)4 * n_total);
    fill_qwen35_mrope_positions(pos_buf_.data(), kv_pos, n_total);
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.positions, pos_buf_.data(), 0,
        sizeof(int32_t) * pos_buf_.size());

    rows_buf_.resize((size_t)n_total * w.n_head_kv);
    for (int h = 0; h < w.n_head_kv; h++)
        for (int i = 0; i < chunk; i++)
            rows_buf_[(size_t)h * n_total + i] = prefill.rows[(size_t)i];
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.kv_write_rows, rows_buf_.data(), 0,
        sizeof(int64_t) * rows_buf_.size());

    upload_qwen35_causal_mask(
        sg.attn_mask, kv_pos, chunk, b_.cfg_.kq_stride_pad);

    const auto st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        fail_pending_prefill(prefill_slot, outputs,
                             "prefill compute failed",
                             "prefill compute failed");
        return false;
    }
    return true;
}

SeqEngine::StepResult Qwen35SeqEngine::step(const StepPlan & plan) {
    StepResult result;
    std::vector<StepOutput> & outputs = result.outputs;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        result.status = StepResult::Status::failed;
        result.outputs.clear();
        result.prefill_progress.clear();
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

    const StepPlanLimits limits =
        step_plan_limits((int)inputs.size());
    const int slice_limit = limits.max_prefill_tokens_per_sequence;
    if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
        return fail_step("prefill plan exceeds engine sequence capacity");
    }
    int prefill_tokens_total = 0;
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens <= 0 || slice.max_tokens > slice_limit ||
            prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !pending_[(size_t)slice.slot]) {
            return fail_step("invalid or duplicate prefill slice in step plan");
        }
        prefill_seen[(size_t)slice.slot] = 1;
        prefill_tokens_total += slice.max_tokens;
        if (prefill_tokens_total > limits.max_prefill_tokens_total) {
            return fail_step(
                "prefill plan exceeds engine total-token capacity");
        }
    }
    if (inputs.empty() && plan.prefills.empty()) {
        result.status = StepResult::Status::idle;
        return result;
    }

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden    = w.n_embd;
    const int n_head_kv = w.n_head_kv;

    outputs.reserve(inputs.size() + (size_t)max_prefills_);
    output_rows_.clear();        output_rows_.reserve(inputs.size());
    live_tokens_.clear();        live_tokens_.reserve(inputs.size());
    live_positions_.clear();     live_positions_.reserve(inputs.size());
    live_physical_rows_.clear(); live_physical_rows_.reserve(inputs.size());
    live_slot_ids_.clear();      live_slot_ids_.reserve(inputs.size());

    int max_kv_len = 1;
    for (const StepInput & in : inputs) {
        StepOutput out;
        out.slot = in.slot;
        out.failed = true;
        int compact_row = -1;
        if (in.slot < 0 || in.slot >= n_slots) {
            out.error = "invalid decode slot";
            outputs.push_back(out);
            output_rows_.push_back(compact_row);
            continue;
        }
        const SeqSlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
        if (!app.ok) {
            out.error = app.busy
                ? "paged KV pool exhausted during decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "decode K/V append failed";
            outputs.push_back(out);
            output_rows_.push_back(compact_row);
            continue;
        }
        if (app.new_block >= 0) {
            if (!upload_block_table_delta(
                    in.slot, app.new_block_index, &app.new_block, 1)) {
                out.error = "decode block-table entry exceeds device capacity";
                outputs.push_back(out);
                output_rows_.push_back(compact_row);
                continue;
            }
        }
        compact_row = (int)live_tokens_.size();
        live_tokens_.push_back(in.token);
        live_positions_.push_back(app.position);
        live_physical_rows_.push_back(app.physical_row);
        live_slot_ids_.push_back(in.slot);
        max_kv_len = std::max(max_kv_len, app.position + 1);
        out.failed = false;
        outputs.push_back(out);
        output_rows_.push_back(compact_row);
    }
    const int live_count = (int)live_tokens_.size();
    const bool with_decode = live_count > 0;
    // Prefill membership, ordering, and token limits come from the common
    // scheduler. Staging identity remains request-owned below this boundary.
    prefill_slots_.clear();
    prefill_token_limits_.clear();
    for (const PrefillSlice & slice : plan.prefills) {
        prefill_slots_.push_back(slice.slot);
        prefill_token_limits_.push_back(slice.max_tokens);
    }

    auto advance_prefill_only = [&](int pslot, int max_tokens) {
        const int staging_idx = pending_[(size_t)pslot]->staging_idx;
        const size_t outputs_before = outputs.size();
        PrefillStage prefill = stage_prefill_chunk(
            pslot, max_tokens, staging_idx, outputs);
        if (!prefill.ready) {
            if (outputs.size() == outputs_before) {
                fail_pending_prefill(
                    pslot, outputs,
                    "prefill made no progress despite reserved capacity",
                    "prefill scheduler made no progress");
            }
            return false;
        }
        if (!run_prefill_graph(prefill, pslot, outputs)) return false;

        // Queue the scalar readback before the one mandatory drain so
        // committing greedy prefills do not introduce a second sync.
        int32_t argmax = -1;
        if (prefill.commit) {
            ggml_backend_tensor_get_async(
                b_.target_backend_, b_.sg_.argmax_tokens, &argmax, 0,
                sizeof(argmax));
        }
        ggml_backend_synchronize(b_.target_backend_);

        PendingPrefill & pending = *pending_[(size_t)pslot];
        pending.progress += prefill.chunk;
        result.prefill_progress.push_back({pslot, prefill.chunk});
        if (prefill.commit) {
            StepOutput out;
            out.slot = pslot;
            out.prefill_done = true;
            out.token = sample_graph_row(
                pslot, /*logits_row=*/0, &argmax, &logits_buf_);
            slots_.commit_prefill(pslot, (int)pending.prompt.size());
            outputs.push_back(out);
            pending_[(size_t)pslot].reset();
        }
        return true;
    };

    // With no live decode rows, advance up to K independent prefills.
    if (!with_decode) {
        const int n_prefills_this_step =
            std::min((int)prefill_slots_.size(), max_prefills_);
        for (int pi = 0; pi < n_prefills_this_step; ++pi) {
            advance_prefill_only(
                prefill_slots_[(size_t)pi],
                prefill_token_limits_[(size_t)pi]);
        }
        if (outputs.empty() && result.prefill_progress.empty()) {
            return fail_step("selected prefill work made no progress");
        }
        result.status = StepResult::Status::progressed;
        return result;
    }

    // Keep K prefills moving while decode is active: one gets a standalone
    // staging set, and the FIFO head is fused into the decode graph below.
    if (max_prefills_ > 1 && prefill_slots_.size() > 1) {
        advance_prefill_only(prefill_slots_[1], prefill_token_limits_[1]);
    }

    // ── Build and compute the fused/decode graph ───────────────────────
    const int decode_bucket = decode_bucket_width(live_count);

    dec_tokens_.assign((size_t)decode_bucket, 0);
    dec_pos_.assign((size_t)4 * decode_bucket, 0);
    dec_rows_.assign((size_t)decode_bucket * n_head_kv, scratch_row_);
    active_slot_ids_.assign((size_t)decode_bucket, -1);
    state_slot_ids_.assign((size_t)decode_bucket, 0);
    seq_lens_.assign((size_t)n_slots, 0);
    for (int row = 0; row < live_count; ++row) {
        dec_tokens_[(size_t)row] = live_tokens_[(size_t)row];
        const int pos = live_positions_[(size_t)row];
        dec_pos_[(size_t)4 * row + 0] = pos;
        dec_pos_[(size_t)4 * row + 1] = pos;
        dec_pos_[(size_t)4 * row + 2] = pos;
        active_slot_ids_[(size_t)row] = live_slot_ids_[(size_t)row];
        state_slot_ids_[(size_t)row] = live_slot_ids_[(size_t)row];
        seq_lens_[(size_t)live_slot_ids_[(size_t)row]] = pos + 1;
        for (int h = 0; h < n_head_kv; ++h) {
            dec_rows_[(size_t)h * decode_bucket + row] =
                live_physical_rows_[(size_t)row];
        }
    }
    PrefillStage prefill;
    int prefill_slot = -1;
    if (!prefill_slots_.empty()) {
        prefill_slot = prefill_slots_.front();
        const size_t outputs_before = outputs.size();
        prefill = stage_prefill_chunk(
            prefill_slot, prefill_token_limits_.front(),
            pending_[(size_t)prefill_slot]->staging_idx, outputs);
        if (!prefill.ready && outputs.size() == outputs_before) {
            fail_pending_prefill(
                prefill_slot, outputs,
                "prefill made no progress despite reserved capacity",
                "prefill scheduler made no progress");
        }
    }
    bool with_prefill = prefill.ready;
    const int chunk = prefill.chunk;
    const bool commit = prefill.commit;
    const int kv_pos = prefill.kv_pos;

    auto fail_fused_prefill = [&]() {
        fail_pending_prefill(prefill_slot, outputs,
                             "fused prefill graph build failed",
                             "prefill graph build failed");
        with_prefill = false;
    };

    bool built = false;
    if (with_prefill) {
        const bool with_mask =
            (b_.cfg_.kq_stride_pad > KQ_MASK_PAD) || (chunk > 1);
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/kv_pos,
            /*n_tokens=*/chunk + decode_bucket,
            with_mask, /*capture=*/false,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0,
            /*logits_tail_rows=*/decode_bucket + (commit ? 1 : 0),
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/decode_bucket,
            /*seq_slot=*/prefill_slot,
            /*paged_prefill=*/true,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/chunk,
            /*prefill_commit=*/commit,
            /*compact_slots=*/true,
            /*staging_idx=*/prefill.staging_idx);
        if (!built || !sg.kv_write_rows) fail_fused_prefill();
    }
    if (!with_prefill) {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0, /*n_tokens=*/decode_bucket,
            /*with_mask=*/false, /*capture=*/false,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0,
            /*logits_tail_rows=*/0,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/decode_bucket,
            /*seq_slot=*/0,
            /*paged_prefill=*/false,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/0,
            /*prefill_commit=*/false,
            /*compact_slots=*/true);
    }
    if (!built || !sg.kv_write_rows) {
        std::fprintf(stderr, "[parallel] fused/decode build failed\n");
        for (StepOutput & out : outputs) out.failed = true;
        return fail_step("fused/decode graph build failed");
    }

    const int n_prefill = with_prefill ? chunk : 0;
    const int n_decode = decode_bucket;
    const int n_total = n_prefill + n_decode;
    embed_buf_.resize((size_t)hidden * n_total);
    if (n_prefill > 0) {
        std::copy(prefill.embeddings.begin(), prefill.embeddings.end(),
                  embed_buf_.begin());
    }
    if (!w.embedder.embed(dec_tokens_.data(), n_decode,
                          embed_buf_.data() + (size_t)hidden * n_prefill)) {
        for (StepOutput & out : outputs) out.failed = true;
        return fail_step("decode embedding failed");
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.inp_embed, embed_buf_.data(), 0,
        sizeof(float) * (size_t)hidden * n_total);

    pos_buf_.resize((size_t)4 * n_total);
    if (n_prefill > 0) {
        fill_qwen35_mrope_positions(pos_buf_.data(), kv_pos, n_prefill);
    }
    std::copy(dec_pos_.begin(), dec_pos_.begin() + (size_t)4 * n_decode,
              pos_buf_.begin() + (size_t)4 * n_prefill);
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.positions, pos_buf_.data(), 0,
        sizeof(int32_t) * pos_buf_.size());

    rows_buf_.assign((size_t)n_total * n_head_kv, scratch_row_);
    for (int h = 0; h < n_head_kv; h++) {
        for (int i = 0; i < n_prefill; i++) {
            rows_buf_[(size_t)h * n_total + i] =
                prefill.rows[(size_t)i];
        }
        for (int s = 0; s < n_decode; s++) {
            rows_buf_[(size_t)h * n_total + n_prefill + s] =
                dec_rows_[(size_t)h * decode_bucket + s];
        }
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.kv_write_rows, rows_buf_.data(), 0,
        sizeof(int64_t) * rows_buf_.size());

    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.active_slot_ids,
        active_slot_ids_.data(), 0,
        sizeof(int32_t) * active_slot_ids_.size());

    if (n_prefill > 0) {
        upload_qwen35_causal_mask(
            sg.attn_mask, kv_pos, n_prefill, b_.cfg_.kq_stride_pad);
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.state_slot_ids,
        state_slot_ids_.data(), 0,
        sizeof(int32_t) * state_slot_ids_.size());
    ggml_backend_tensor_set_async(
        b_.target_backend_, b_.cache_.paged_kv_seq_lens,
        seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());

    const auto st =
        ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[parallel] fused/decode compute failed\n");
        for (StepOutput & out : outputs) out.failed = true;
        return fail_step("fused/decode compute failed");
    }

    // A committing fused graph exposes [prefill tail | decode rows].
    // Non-committing prefill chunks expose only the decode rows.
    const int dec_row0 = (n_prefill > 0 && commit) ? 1 : 0;
    argmax_buf_.assign((size_t)(dec_row0 + n_decode), -1);
    ggml_backend_tensor_get_async(
        b_.target_backend_, sg.argmax_tokens, argmax_buf_.data(), 0,
        sizeof(int32_t) * argmax_buf_.size());
    ggml_backend_synchronize(b_.target_backend_);
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        StepOutput & out = outputs[oi];
        if (out.failed || out.prefill_done) continue;
        slots_.commit_step(out.slot);
        const int row = dec_row0 + output_rows_[oi];
        out.token = sample_graph_row(
            out.slot, row, &argmax_buf_[(size_t)row], &logits_buf_);
    }

    if (with_prefill) {
        PendingPrefill & pending = *pending_[(size_t)prefill_slot];
        pending.progress += n_prefill;
        result.prefill_progress.push_back({prefill_slot, n_prefill});
        if (commit) {
            StepOutput out;
            out.slot = prefill_slot;
            out.prefill_done = true;
            out.token = sample_graph_row(
                prefill_slot, /*logits_row=*/0, &argmax_buf_[0],
                &logits_buf_);
            slots_.commit_prefill(
                prefill_slot, (int)pending.prompt.size());
            outputs.push_back(out);
            pending_[(size_t)prefill_slot].reset();
        }
    }
    result.status = StepResult::Status::progressed;
    return result;
}

void Qwen35SeqEngine::retire(int slot) {
    if (slot >= 0 && (size_t)slot < pending_.size())
        pending_[(size_t)slot].reset();
    if (!slots_.is_active(slot)) return;
    slots_.retire(slot);
}

}  // namespace dflash::common
