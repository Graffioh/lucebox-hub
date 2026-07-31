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

int decode_bucket_width(int live_count) {
    int width = 1;
    while (width < live_count) width <<= 1;
    return width;
}

} // namespace

bool Qwen35SeqEngine::token_is_eos(int32_t token) const {
    return b_.token_is_eos(token);
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    if (pending_prefill_) {
        AdmitResult r;
        r.busy = true;
        r.error = "another admission is still prefilling";
        return r;
    }

    AdmitResult r =
        slots_.admit(request_id, (int)prompt.size(), sampler);
    if (!r.ok) return r;
    const int slot = r.slot;

    // The staging K/V and recurrent slabs carry this sequence across its
    // incremental prefill chunks. The slot's device length remains zero
    // until the final chunk commits.
    reset_prefill_staging(b_.cache_);

    PendingPrefill pp;
    pp.slot = slot;
    pp.prompt = prompt;
    pending_prefill_ = std::move(pp);

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
        ggml_backend_tensor_get(
            b_.sg_.logits, logits.data(),
            (size_t)logits_row * (size_t)vocab * sizeof(float),
            sizeof(float) * (size_t)vocab);
        token = sample_logits(logits.data(), vocab, seq.sampler,
                              seq.sample_history, seq.rng);
    } else if (cached_argmax) {
        token = *cached_argmax;
    } else {
        ggml_backend_tensor_get(
            b_.sg_.argmax_tokens, &token,
            (size_t)logits_row * sizeof(int32_t), sizeof(int32_t));
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
    ggml_backend_tensor_set(
        table, blocks,
        (size_t)slot * table->nb[1] +
            (size_t)first_block * sizeof(int32_t),
        count * sizeof(int32_t));
    return true;
}

void Qwen35SeqEngine::fail_pending_prefill(
        std::vector<StepOutput> & outputs, const char * log_message,
        const char * client_message) {
    if (!pending_prefill_) return;
    std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                 log_message, pending_prefill_->slot);
    StepOutput out;
    out.slot = pending_prefill_->slot;
    out.failed = true;
    out.error = client_message;
    out.prefill_done = true;
    outputs.push_back(std::move(out));
    pending_prefill_.reset();
}

Qwen35SeqEngine::PrefillStage Qwen35SeqEngine::stage_prefill_chunk(
        std::vector<StepOutput> & outputs) {
    PrefillStage stage;
    if (!pending_prefill_) return stage;

    PendingPrefill & pending = *pending_prefill_;
    static const int prefill_ubatch = qwen35_prefill_ubatch(512);
    stage.kv_pos = pending.progress;
    stage.chunk = std::min(
        prefill_ubatch, (int)pending.prompt.size() - stage.kv_pos);
    stage.commit = stage.kv_pos + stage.chunk >= (int)pending.prompt.size();

    SeqSlotManager::PrefillChunk chunk =
        slots_.append_prefill(pending.slot, stage.chunk);
    if (chunk.busy) return PrefillStage{};
    if (!chunk.ok || chunk.rows.size() != (size_t)stage.chunk) {
        fail_pending_prefill(outputs, "prefill K/V allocation failed",
                             "prefill K/V allocation failed");
        return PrefillStage{};
    }
    if (!upload_block_table_delta(
            pending.slot, chunk.first_new_block, chunk.new_blocks.data(),
            chunk.new_blocks.size())) {
        fail_pending_prefill(
            outputs, "prefill block-table delta exceeds device capacity",
            "prefill block-table update failed");
        return PrefillStage{};
    }

    stage.rows = std::move(chunk.rows);
    stage.embeddings.resize((size_t)b_.w_.n_embd * stage.chunk);
    if (!b_.w_.embedder.embed(
            pending.prompt.data() + stage.kv_pos, stage.chunk,
            stage.embeddings.data())) {
        fail_pending_prefill(outputs, "prefill embed failed",
                             "prefill embedding failed");
        return PrefillStage{};
    }
    stage.ready = true;
    return stage;
}

bool Qwen35SeqEngine::step(
        const std::vector<StepInput> & inputs,
        std::vector<StepOutput> & outputs) {
    outputs.clear();
    const int n_slots = slots_.slot_count();

    // Every decoding slot must appear. The prefilling slot is deliberately
    // absent: its state advances through the staging segment.
    if ((int)inputs.size() != slots_.decoding_count()) {
        std::fprintf(stderr,
            "[parallel] step got %zu inputs for %d decoding slots\n",
            inputs.size(), slots_.decoding_count());
        return false;
    }
    if (inputs.empty() && !pending_prefill_) return true;

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden    = w.n_embd;
    const int n_head_kv = w.n_head_kv;

    outputs.reserve(inputs.size() + 1);
    std::vector<int> output_rows;
    output_rows.reserve(inputs.size());
    std::vector<int32_t> live_tokens;
    std::vector<int32_t> live_positions;
    std::vector<int64_t> live_physical_rows;
    std::vector<int32_t> live_slot_ids;
    live_tokens.reserve(inputs.size());
    live_positions.reserve(inputs.size());
    live_physical_rows.reserve(inputs.size());
    live_slot_ids.reserve(inputs.size());
    int max_kv_len = 1;
    for (const StepInput & in : inputs) {
        StepOutput out;
        out.slot = in.slot;
        out.failed = true;
        int compact_row = -1;
        if (in.slot < 0 || in.slot >= n_slots) {
            out.error = "invalid decode slot";
            outputs.push_back(out);
            output_rows.push_back(compact_row);
            continue;
        }
        // Cache-row allocation, the context guard, and the penalty
        // history all live in the manager.
        const SeqSlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
        if (!app.ok) {
            out.error = app.busy
                ? "paged KV pool exhausted during decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "decode K/V append failed";
            outputs.push_back(out);
            output_rows.push_back(compact_row);
            continue;
        }
        if (app.new_block >= 0) {
            if (!upload_block_table_delta(
                    in.slot, app.new_block_index, &app.new_block, 1)) {
                out.error = "decode block-table entry exceeds device capacity";
                outputs.push_back(out);
                output_rows.push_back(compact_row);
                continue;
            }
        }
        compact_row = (int)live_tokens.size();
        live_tokens.push_back(in.token);
        live_positions.push_back(app.position);
        live_physical_rows.push_back(app.physical_row);
        live_slot_ids.push_back(in.slot);
        max_kv_len = std::max(max_kv_len, app.position + 1);
        out.failed = false;
        outputs.push_back(out);
        output_rows.push_back(compact_row);
    }
    const int live_count = (int)live_tokens.size();
    const bool with_decode = live_count > 0;
    if (!with_decode && !pending_prefill_) return true;
    const int decode_bucket = with_decode ? decode_bucket_width(live_count) : 0;

    // Pack live sequences first and pad only to the next graph bucket. Paged
    // attention and GDN use these ids to reach physical metadata/state.
    std::vector<int32_t> dec_tokens((size_t)decode_bucket, 0);
    std::vector<int32_t> dec_pos((size_t)4 * decode_bucket, 0);
    std::vector<int64_t> dec_rows(
        (size_t)decode_bucket * n_head_kv, scratch_row_);
    std::vector<int32_t> active_slot_ids((size_t)decode_bucket, -1);
    std::vector<int32_t> state_slot_ids((size_t)decode_bucket, 0);
    // Device kv lengths, rebuilt per step from the live rows. The kernel
    // reads this only through active_slot_ids, so entries for idle,
    // prefilling, or retired slots are never consumed and stay 0. The fed
    // token's row is written and attended in this same step: position + 1.
    std::vector<int32_t> seq_lens((size_t)n_slots, 0);
    for (int row = 0; row < live_count; ++row) {
        dec_tokens[(size_t)row] = live_tokens[(size_t)row];
        const int pos = live_positions[(size_t)row];
        dec_pos[(size_t)4 * row + 0] = pos;
        dec_pos[(size_t)4 * row + 1] = pos;
        dec_pos[(size_t)4 * row + 2] = pos;
        active_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        state_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        seq_lens[(size_t)live_slot_ids[(size_t)row]] = pos + 1;
        for (int h = 0; h < n_head_kv; ++h) {
            dec_rows[(size_t)h * decode_bucket + row] =
                live_physical_rows[(size_t)row];
        }
    }

    PrefillStage prefill = stage_prefill_chunk(outputs);
    bool with_prefill = prefill.ready;
    const int chunk = prefill.chunk;
    const bool commit = prefill.commit;
    const int kv_pos = prefill.kv_pos;

    // A prefill graph build failure fails only the admission and falls back
    // to the established decode graph for existing streams.
    auto fail_pending_build = [&]() {
        fail_pending_prefill(outputs, "prefill step build failed",
                             "prefill graph build failed");
        with_prefill = false;
    };

    bool built = false;
    if (with_prefill && with_decode) {
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
            /*seq_slot=*/pending_prefill_->slot,
            /*paged_prefill=*/true,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/chunk,
            /*prefill_commit=*/commit,
            /*compact_slots=*/true);
        if (!built || !sg.kv_write_rows) fail_pending_build();
    } else if (with_prefill) {
        const bool with_mask =
            (b_.cfg_.kq_stride_pad > KQ_MASK_PAD) || (chunk > 1);
        built = build_target_step(
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
            /*seq_slot=*/pending_prefill_->slot,
            /*paged_prefill=*/true,
            /*paged_max_kv_len=*/0,
            /*n_prefill_tokens=*/0,
            /*prefill_commit=*/commit);
        if (!built || !sg.kv_write_rows) {
            fail_pending_build();
            return true;
        }
    }
    if (!with_prefill && with_decode) {
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
        if (!built || !sg.kv_write_rows) {
            std::fprintf(stderr, "[parallel] decode build failed\n");
            for (StepOutput & out : outputs) out.failed = true;
            return false;
        }
    }
    if (!with_prefill && !with_decode) return true;

    // Token axis: [prefill chunk | compact bucketed decode rows].
    const int n_prefill = with_prefill ? chunk : 0;
    const int n_decode  = with_decode ? decode_bucket : 0;
    const int n_total   = n_prefill + n_decode;

    std::vector<float> embed_buf((size_t)hidden * n_total);
    if (n_prefill > 0) {
        std::copy(prefill.embeddings.begin(), prefill.embeddings.end(),
                  embed_buf.begin());
    }
    if (n_decode > 0 &&
        !w.embedder.embed(dec_tokens.data(), n_decode,
                          embed_buf.data() + (size_t)hidden * n_prefill)) {
        for (StepOutput & out : outputs) out.failed = true;
        return false;
    }
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                            sizeof(float) * (size_t)hidden * n_total);

    std::vector<int32_t> pos_buf((size_t)4 * n_total, 0);
    fill_qwen35_mrope_positions(pos_buf.data(), kv_pos, n_prefill);
    if (n_decode > 0) {
        std::copy(dec_pos.begin(), dec_pos.end(),
                  pos_buf.begin() + (size_t)4 * n_prefill);
    }
    ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                            sizeof(int32_t) * pos_buf.size());

    std::vector<int64_t> rows((size_t)n_total * n_head_kv, scratch_row_);
    for (int h = 0; h < n_head_kv; h++) {
        for (int i = 0; i < n_prefill; i++) {
            rows[(size_t)h * n_total + i] =
                prefill.rows[(size_t)i];
        }
        for (int s = 0; s < n_decode; s++) {
            rows[(size_t)h * n_total + n_prefill + s] =
                dec_rows[(size_t)h * decode_bucket + s];
        }
    }
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());
    if (n_prefill > 0) {
        upload_qwen35_causal_mask(
            sg.attn_mask, kv_pos, n_prefill, b_.cfg_.kq_stride_pad);
    }
    if (n_decode > 0) {
        ggml_backend_tensor_set(
            sg.active_slot_ids, active_slot_ids.data(), 0,
            sizeof(int32_t) * active_slot_ids.size());
        ggml_backend_tensor_set(
            sg.state_slot_ids, state_slot_ids.data(), 0,
            sizeof(int32_t) * state_slot_ids.size());
        ggml_backend_tensor_set(
            b_.cache_.paged_kv_seq_lens, seq_lens.data(), 0,
            sizeof(int32_t) * seq_lens.size());
    }

    const auto st =
        ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[parallel] step compute failed (chunk=%d live=%d)\n",
            chunk, with_decode ? 1 : 0);
        for (StepOutput & out : outputs) out.failed = true;
        return false;
    }

    // Fused final chunks expose [prefill tail | decode rows]; otherwise the
    // decode rows start at zero in the logits-tail tensor.
    const int dec_row0 = (n_prefill > 0 && commit) ? 1 : 0;
    if (n_decode > 0) {
        std::vector<int32_t> argmax((size_t)(dec_row0 + decode_bucket), -1);
        ggml_backend_tensor_get(sg.argmax_tokens, argmax.data(), 0,
                                sizeof(int32_t) * argmax.size());
        std::vector<float> logits_buf;
        for (size_t oi = 0; oi < outputs.size(); ++oi) {
            StepOutput & out = outputs[oi];
            if (out.failed) continue;
            slots_.commit_step(out.slot);
            const int row = dec_row0 + output_rows[oi];
            out.token = sample_graph_row(
                out.slot, row, &argmax[(size_t)row], &logits_buf);
        }
    }

    if (with_prefill) {
        PendingPrefill & pending = *pending_prefill_;
        pending.progress += chunk;
        if (commit) {
            StepOutput out;
            out.slot = pending.slot;
            out.prefill_done = true;
            // The committing graph leaves the prompt's final logits at row 0.
            out.token = sample_graph_row(pending.slot, /*logits_row=*/0);
            slots_.commit_prefill(pending.slot, (int)pending.prompt.size());
            outputs.push_back(out);
            pending_prefill_.reset();
        }
    }
    return true;
}

void Qwen35SeqEngine::retire(int slot) {
    if (pending_prefill_ && pending_prefill_->slot == slot) {
        pending_prefill_.reset();
    }
    if (!slots_.is_active(slot)) return;
    slots_.retire(slot);
}

}  // namespace dflash::common
