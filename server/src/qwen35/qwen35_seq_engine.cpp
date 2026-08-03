// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "graph_builders.h"
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
    AdmitResult r =
        slots_.admit(request_id, (int)prompt.size(), sampler);
    if (!r.ok) return r;
    const int slot = r.slot;

    // The prompt's chunked prefill advances its recurrent state in this
    // slot's own slab, so clear whatever the previous occupant left there.
    // K/V needs no reset: chunk rows are written into freshly allocated pool
    // blocks and read back causally through the block table.
    reset_recurrent_slot(b_.cache_, slot);

    PendingPrefill pp;
    pp.prompt = prompt;
    pending_prefills_[(size_t)slot] = std::move(pp);

    return r;
}

bool Qwen35SeqEngine::prefill_pending() const {
    for (const auto & p : pending_prefills_) {
        if (p) return true;
    }
    return false;
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
        int slot, std::vector<StepOutput> & outputs,
        const char * log_message, const char * client_message) {
    if (slot < 0 || slot >= (int)pending_prefills_.size() ||
        !pending_prefills_[(size_t)slot]) {
        return;
    }
    std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                 log_message, slot);
    StepOutput out;
    out.slot = slot;
    out.failed = true;
    out.error = client_message;
    out.prefill_done = true;
    outputs.push_back(std::move(out));
    pending_prefills_[(size_t)slot].reset();
}

std::vector<Qwen35SeqEngine::SelectedPrefill>
Qwen35SeqEngine::select_prefill_chunks(std::vector<StepOutput> & outputs) {
    std::vector<SelectedPrefill> selected;
    const int n_slots = slots_.slot_count();
    int n_pending = 0;
    for (const auto & p : pending_prefills_) n_pending += p ? 1 : 0;
    if (n_pending == 0) return selected;

    // Divide one global token budget across the pending prompts: an equal
    // share recomputed against the residual budget, so a short prompt's
    // leftover flows to the prompts scanned after it, and at least one
    // token per prompt so progress never stalls. The starting slot rotates
    // each step, keeping the split fair when the shares are uneven.
    static const int prefill_ubatch = qwen35_prefill_ubatch(512);
    int budget = prefill_ubatch;
    int left = n_pending;
    const int start = prefill_cursor_ % std::max(1, n_slots);
    for (int scan = 0; scan < n_slots && budget > 0 && left > 0; ++scan) {
        const int slot = (start + scan) % n_slots;
        auto & pending = pending_prefills_[(size_t)slot];
        if (!pending) continue;
        const int remaining = (int)pending->prompt.size() - pending->progress;
        const int share = std::max(1, budget / left);
        const int chunk = std::min(share, remaining);
        --left;
        if (chunk <= 0) {
            fail_pending_prefill(slot, outputs, "invalid prefill progress",
                                 "prefill state corrupted");
            continue;
        }

        SeqSlotManager::PrefillChunk pc = slots_.append_prefill(slot, chunk);
        if (pc.busy) {
            // Pool temporarily out of blocks: keep the partial progress and
            // the budget share; retried next step.
            continue;
        }
        if (!pc.ok || pc.rows.size() != (size_t)chunk) {
            fail_pending_prefill(slot, outputs,
                                 "prefill K/V allocation failed",
                                 "prefill K/V allocation failed");
            continue;
        }
        if (!upload_block_table_delta(
                slot, pc.first_new_block, pc.new_blocks.data(),
                pc.new_blocks.size())) {
            fail_pending_prefill(
                slot, outputs,
                "prefill block-table delta exceeds device capacity",
                "prefill block-table update failed");
            continue;
        }

        SelectedPrefill sel;
        sel.slot = slot;
        sel.kv_pos = pending->progress;
        sel.chunk = chunk;
        sel.commit = pending->progress + chunk >= (int)pending->prompt.size();
        sel.rows = std::move(pc.rows);
        selected.push_back(std::move(sel));
        budget -= chunk;
    }
    prefill_cursor_ = (start + 1) % std::max(1, n_slots);
    return selected;
}

bool Qwen35SeqEngine::step(
        const std::vector<StepInput> & inputs,
        std::vector<StepOutput> & outputs) {
    outputs.clear();
    const int n_slots = slots_.slot_count();

    // Every decoding slot must appear. The prefilling slot is deliberately
    // absent: its rows advance through the leading prefill segment.
    if ((int)inputs.size() != slots_.decoding_count()) {
        std::fprintf(stderr,
            "[parallel] step got %zu inputs for %d decoding slots\n",
            inputs.size(), slots_.decoding_count());
        return false;
    }
    if (inputs.empty() && !prefill_pending()) return true;

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
    if (!with_decode && !prefill_pending()) return true;
    const int decode_bucket = with_decode ? decode_bucket_width(live_count) : 0;

    // Pack live sequences first and pad only to the next graph bucket. Paged
    // attention and GDN use these ids to reach physical metadata/state.
    std::vector<int32_t> dec_tokens((size_t)decode_bucket, 0);
    std::vector<int32_t> dec_pos((size_t)4 * decode_bucket, 0);
    std::vector<int64_t> dec_rows(
        (size_t)decode_bucket * n_head_kv, scratch_row_);
    std::vector<int32_t> active_slot_ids((size_t)decode_bucket, -1);
    std::vector<int32_t> state_slot_ids((size_t)decode_bucket, 0);
    // Device kv lengths, rebuilt per step from the live rows plus the
    // prefilling slot's chunk (set after staging below). The kernel reads
    // an entry only through a row's seq id, so idle and retired slots stay
    // 0. The fed token's row is written and attended in this same step:
    // position + 1.
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

    std::vector<SelectedPrefill> selected = select_prefill_chunks(outputs);
    bool with_prefill = !selected.empty();
    int n_prefill = 0;
    int n_commits = 0;
    std::vector<QwenPrefillSegment> segments;
    segments.reserve(selected.size());
    for (const SelectedPrefill & sel : selected) {
        segments.push_back({n_prefill, sel.chunk, sel.slot});
        n_prefill += sel.chunk;
        n_commits += sel.commit ? 1 : 0;
        // Chunk rows are written and causally read in this same step, so
        // the launch bound and each prefilling slot's device length must
        // cover them. Decode rows never see a partial sequence: their
        // extents clamp to their own positions, and prompts never see each
        // other: each row's seq id selects its own block-table column.
        max_kv_len = std::max(max_kv_len, sel.kv_pos + sel.chunk);
        seq_lens[(size_t)sel.slot] = sel.kv_pos + sel.chunk;
    }

    // A prefill graph build failure fails the in-flight admissions and
    // falls back to the established decode graph for existing streams.
    auto fail_pending_build = [&]() {
        for (const SelectedPrefill & sel : selected) {
            fail_pending_prefill(sel.slot, outputs,
                                 "prefill step build failed",
                                 "prefill graph build failed");
        }
        selected.clear();
        segments.clear();
        n_prefill = 0;
        n_commits = 0;
        with_prefill = false;
    };
    // The logits gather (multi-prompt steps): committing rows lead in
    // selected order, decode rows follow. Non-committing pure-prefill steps
    // keep one dummy row so the output contract stays small.
    const int gather_rows = with_decode
        ? n_commits + decode_bucket
        : std::max(1, n_commits);

    bool built = false;
    if (with_prefill && with_decode) {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0,
            /*n_tokens=*/n_prefill + decode_bucket,
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
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/n_prefill,
            segments.data(), (int)segments.size(),
            /*n_logits_rows=*/gather_rows,
            /*compact_slots=*/true);
        if (!built || !sg.kv_write_rows || !sg.paged_query_seq_ids ||
            !sg.logits_row_indices) {
            fail_pending_build();
        }
    } else if (with_prefill) {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0, /*n_tokens=*/n_prefill,
            /*with_mask=*/false, /*capture=*/false,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0,
            /*logits_tail_rows=*/0,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/1,
            /*seq_slot=*/0,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/n_prefill,
            segments.data(), (int)segments.size(),
            /*n_logits_rows=*/gather_rows);
        if (!built || !sg.kv_write_rows || !sg.paged_query_seq_ids ||
            !sg.logits_row_indices) {
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
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/0,
            /*prefill_segments=*/nullptr, 0,
            /*n_logits_rows=*/0,
            /*compact_slots=*/true);
        if (!built || !sg.kv_write_rows) {
            std::fprintf(stderr, "[parallel] decode build failed\n");
            for (StepOutput & out : outputs) out.failed = true;
            return false;
        }
    }
    if (!with_prefill && !with_decode) return true;

    // Token axis: [prompt chunks in selected order | compact decode rows].
    const int n_decode  = with_decode ? decode_bucket : 0;
    const int n_total   = n_prefill + n_decode;

    std::vector<float> embed_buf((size_t)hidden * n_total);
    for (size_t i = 0; i < selected.size(); ++i) {
        const SelectedPrefill & sel = selected[i];
        const int seg_off = segments[i].token_offset;
        const PendingPrefill & pending = *pending_prefills_[(size_t)sel.slot];
        if (!w.embedder.embed(
                pending.prompt.data() + sel.kv_pos, sel.chunk,
                embed_buf.data() + (size_t)hidden * seg_off)) {
            for (StepOutput & out : outputs) out.failed = true;
            return false;
        }
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
    for (size_t i = 0; i < selected.size(); ++i) {
        const SelectedPrefill & sel = selected[i];
        const int seg_off = segments[i].token_offset;
        fill_qwen35_mrope_positions(
            pos_buf.data() + (size_t)4 * seg_off, sel.kv_pos, sel.chunk);
    }
    if (n_decode > 0) {
        std::copy(dec_pos.begin(), dec_pos.end(),
                  pos_buf.begin() + (size_t)4 * n_prefill);
    }
    ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                            sizeof(int32_t) * pos_buf.size());

    std::vector<int64_t> rows((size_t)n_total * n_head_kv, scratch_row_);
    for (int h = 0; h < n_head_kv; h++) {
        for (size_t si = 0; si < selected.size(); ++si) {
            const SelectedPrefill & sel = selected[si];
            const int seg_off = segments[si].token_offset;
            for (int i = 0; i < sel.chunk; i++) {
                rows[(size_t)h * n_total + seg_off + i] =
                    sel.rows[(size_t)i];
            }
        }
        for (int s = 0; s < n_decode; s++) {
            rows[(size_t)h * n_total + n_prefill + s] =
                dec_rows[(size_t)h * decode_bucket + s];
        }
    }
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());
    if (n_prefill > 0) {
        // Ragged read metadata: chunk rows carry their slot and their own
        // inclusive positions; decode rows carry their slot and appended
        // position; bucket padding stays -1. The logits gather takes the
        // committing chunks' last rows (selected order), then decode rows.
        std::vector<int32_t> query_slots((size_t)n_total, -1);
        std::vector<int32_t> query_positions((size_t)n_total, -1);
        std::vector<int32_t> logits_rows;
        logits_rows.reserve((size_t)gather_rows);
        for (size_t si = 0; si < selected.size(); ++si) {
            const SelectedPrefill & sel = selected[si];
            const int seg_off = segments[si].token_offset;
            for (int i = 0; i < sel.chunk; ++i) {
                query_slots[(size_t)(seg_off + i)] = sel.slot;
                query_positions[(size_t)(seg_off + i)] = sel.kv_pos + i;
            }
            if (sel.commit) {
                logits_rows.push_back(seg_off + sel.chunk - 1);
            }
        }
        for (int row = 0; row < live_count; ++row) {
            query_slots[(size_t)(n_prefill + row)] =
                live_slot_ids[(size_t)row];
            query_positions[(size_t)(n_prefill + row)] =
                live_positions[(size_t)row];
        }
        for (int row = 0; with_decode && row < decode_bucket; ++row) {
            logits_rows.push_back(n_prefill + row);
        }
        if (logits_rows.empty()) {
            // Non-committing pure prefill: one dummy, unsampled row.
            logits_rows.push_back(n_total - 1);
        }
        ggml_backend_tensor_set(
            sg.paged_query_seq_ids, query_slots.data(), 0,
            sizeof(int32_t) * query_slots.size());
        ggml_backend_tensor_set(
            sg.paged_query_positions, query_positions.data(), 0,
            sizeof(int32_t) * query_positions.size());
        ggml_backend_tensor_set(
            sg.logits_row_indices, logits_rows.data(), 0,
            sizeof(int32_t) * logits_rows.size());
    }
    if (n_decode > 0) {
        ggml_backend_tensor_set(
            sg.active_slot_ids, active_slot_ids.data(), 0,
            sizeof(int32_t) * active_slot_ids.size());
        ggml_backend_tensor_set(
            sg.state_slot_ids, state_slot_ids.data(), 0,
            sizeof(int32_t) * state_slot_ids.size());
    }
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens.data(), 0,
        sizeof(int32_t) * seq_lens.size());

    const auto st =
        ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[parallel] step compute failed (prefill=%d live=%d)\n",
            n_prefill, with_decode ? 1 : 0);
        for (StepOutput & out : outputs) out.failed = true;
        return false;
    }

    // Prefill steps gather [commit rows | decode rows]; decode-only steps
    // keep the whole bucket, starting at row zero.
    const int dec_row0 = with_prefill ? n_commits : 0;
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

    int commit_row = 0;
    for (const SelectedPrefill & sel : selected) {
        auto & pending = pending_prefills_[(size_t)sel.slot];
        if (!pending) continue;
        pending->progress += sel.chunk;
        if (sel.commit) {
            StepOutput out;
            out.slot = sel.slot;
            out.prefill_done = true;
            // Committing rows lead the logits gather in selected order.
            out.token = sample_graph_row(sel.slot, commit_row++);
            slots_.commit_prefill(sel.slot, (int)pending->prompt.size());
            outputs.push_back(out);
            pending.reset();
        }
    }
    return true;
}

void Qwen35SeqEngine::retire(int slot) {
    if (slot >= 0 && slot < (int)pending_prefills_.size()) {
        pending_prefills_[(size_t)slot].reset();
    }
    if (!slots_.is_active(slot)) return;
    slots_.retire(slot);
}

}  // namespace dflash::common
