// Concurrent slot engine for the paged Qwen3.5/3.6 backend (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, the step graph, and the device metadata uploads —
// single-writer by construction, no locks. See qwen35_seq_engine.h for the
// layering.

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

    // PR #1 deliberately drains this prefill before decoding resumes, but
    // admission itself is claim-only. Keeping the prompt pending here gives
    // the follow-up fused scheduler the final ownership/API shape without
    // making this baseline non-blocking.
    reset_slot_recurrent_state(b_.cache_, slot);
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

bool Qwen35SeqEngine::step_prefill(std::vector<StepOutput> & outputs) {
    PendingPrefill & pp = *pending_prefill_;
    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int prompt_len = (int)pp.prompt.size();
    const int kv_pos = pp.progress;

    static const int prefill_ubatch = qwen35_prefill_ubatch(512);
    const int chunk = std::min(prefill_ubatch, prompt_len - kv_pos);
    const bool commit = kv_pos + chunk >= prompt_len;

    auto fail = [&](const char * log_message, const char * client_message) {
        std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                     log_message, pp.slot);
        StepOutput out;
        out.slot = pp.slot;
        out.failed = true;
        out.error = client_message;
        out.prefill_done = true;
        outputs.push_back(std::move(out));
        pending_prefill_.reset();
        return true;
    };

    SeqSlotManager::PrefillChunk pc =
        slots_.append_prefill(pp.slot, chunk);
    if (!pc.ok || pc.rows.size() != (size_t)chunk) {
        return fail(pc.busy ? "KV pool exhausted during blocking prefill"
                            : "prefill K/V allocation failed",
                    pc.busy ? "paged KV pool exhausted during prefill"
                            : "prefill K/V allocation failed");
    }
    if (!pc.new_blocks.empty()) {
        if (!upload_block_table_delta(
                pp.slot, pc.first_new_block, pc.new_blocks.data(),
                pc.new_blocks.size())) {
            return fail("prefill block-table delta exceeds device capacity",
                        "prefill block-table update failed");
        }
    }

    const bool with_mask =
        (b_.cfg_.kq_stride_pad > KQ_MASK_PAD) || (chunk > 1);
    if (!build_target_step(sg, w, b_.cache_, b_.target_backend_,
                           /*kv_start=*/kv_pos, /*n_tokens=*/chunk,
                           with_mask, /*capture=*/false,
                           /*capture_delta_intermediate=*/false,
                           /*fa_window=*/0,
                           /*last_token_logits_only=*/!commit,
                           b_.cfg_.kq_stride_pad,
                           /*capture_moe_router=*/false,
                           /*kvflash_mask=*/false,
                           /*capture_qk=*/false,
                           /*paged_attention=*/false,
                           /*n_seqs=*/1,
                           /*seq_slot=*/pp.slot,
                           /*paged_prefill=*/true) ||
        !sg.kv_write_rows) {
        return fail("prefill graph build failed",
                    "prefill graph build failed");
    }

    std::vector<int64_t> rows((size_t)chunk * w.n_head_kv);
    for (int h = 0; h < w.n_head_kv; h++) {
        for (int i = 0; i < chunk; i++) {
            rows[(size_t)h * chunk + i] = pc.rows[(size_t)i];
        }
    }
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());

    std::vector<float> embed_buf((size_t)w.n_embd * chunk);
    if (!w.embedder.embed(pp.prompt.data() + kv_pos, chunk,
                          embed_buf.data())) {
        return fail("prefill embed failed", "prefill embedding failed");
    }
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                            sizeof(float) * embed_buf.size());

    std::vector<int32_t> pos_buf((size_t)4 * chunk, 0);
    fill_qwen35_mrope_positions(pos_buf.data(), kv_pos, chunk);
    ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                            sizeof(int32_t) * pos_buf.size());

    upload_qwen35_causal_mask(
        sg.attn_mask, kv_pos, chunk, b_.cfg_.kq_stride_pad);

    if (ggml_backend_graph_compute(b_.target_backend_, sg.gf) !=
        GGML_STATUS_SUCCESS) {
        return fail("prefill compute failed", "prefill compute failed");
    }

    pp.progress += chunk;
    if (commit) {
        StepOutput out;
        out.slot = pp.slot;
        out.prefill_done = true;
        out.token = sample_graph_row(pp.slot, chunk - 1);
        slots_.commit_prefill(pp.slot, prompt_len);
        outputs.push_back(out);
        pending_prefill_.reset();
    }
    return true;
}

bool Qwen35SeqEngine::step(const std::vector<StepInput> & inputs,
                                  std::vector<StepOutput> & outputs) {
    outputs.clear();
    if (pending_prefill_) {
        if (!inputs.empty()) {
            std::fprintf(stderr,
                "[parallel] blocking prefill step received decode inputs\n");
            return false;
        }
        return step_prefill(outputs);
    }
    if (inputs.empty()) return slots_.decoding_count() == 0;

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int n_slots   = slots_.slot_count();
    const int hidden    = w.n_embd;
    const int n_head_kv = w.n_head_kv;

    // Every decoding slot must appear. Admission prefill is drained before
    // the scheduler resumes this path.
    if ((int)inputs.size() != slots_.decoding_count()) {
        std::fprintf(stderr,
            "[parallel] step got %zu inputs for %d decoding slots\n",
            inputs.size(), slots_.decoding_count());
        return false;
    }

    outputs.reserve(inputs.size());
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
                ? "paged KV pool exhausted during decode"
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
    if (live_count == 0) return true;
    const int bucket = decode_bucket_width(live_count);

    // Pack live sequences first and pad only to the next graph bucket. Paged
    // attention and GDN consume active_slot_ids to reach the persistent
    // physical metadata/state; -1 padding rows are computation-only.
    std::vector<int32_t> tokens((size_t)bucket, 0);
    std::vector<int32_t> pos_buf((size_t)4 * bucket, 0);
    std::vector<int64_t> rows((size_t)bucket * n_head_kv, scratch_row_);
    std::vector<int32_t> active_slot_ids((size_t)bucket, -1);
    std::vector<int32_t> state_slot_ids((size_t)bucket, 0);
    // Device kv lengths, rebuilt per step from the live rows. The kernel
    // reads this only through active_slot_ids, so entries for idle,
    // prefilling, or retired slots are never consumed and stay 0. The fed
    // token's row is written and attended in this same step: position + 1.
    std::vector<int32_t> seq_lens((size_t)n_slots, 0);
    for (int row = 0; row < live_count; ++row) {
        tokens[(size_t)row] = live_tokens[(size_t)row];
        const int pos = live_positions[(size_t)row];
        pos_buf[(size_t)4 * row + 0] = pos;
        pos_buf[(size_t)4 * row + 1] = pos;
        pos_buf[(size_t)4 * row + 2] = pos;
        active_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        state_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        seq_lens[(size_t)live_slot_ids[(size_t)row]] = pos + 1;
        for (int h = 0; h < n_head_kv; ++h) {
            rows[(size_t)h * bucket + row] =
                live_physical_rows[(size_t)row];
        }
    }

    if (!build_target_step(sg, w, b_.cache_, b_.target_backend_,
                           /*kv_start=*/0, /*n_tokens=*/bucket,
                           /*with_mask=*/false, /*capture=*/false,
                           /*capture_delta_intermediate=*/false,
                           /*fa_window=*/0,
                           /*last_token_logits_only=*/false,
                           b_.cfg_.kq_stride_pad,
                           /*capture_moe_router=*/false,
                           /*kvflash_mask=*/false,
                           /*capture_qk=*/false,
                           /*paged_attention=*/true,
                           /*n_seqs=*/bucket,
                           /*seq_slot=*/0,
                           /*paged_prefill=*/false,
                           /*paged_max_kv_len=*/max_kv_len,
                           /*compact_slots=*/true)) {
        for (StepOutput & o : outputs) o.failed = true;
        return false;
    }

    std::vector<float> embed_buf((size_t)hidden * bucket);
    if (!w.embedder.embed(tokens.data(), bucket, embed_buf.data())) {
        for (StepOutput & o : outputs) o.failed = true;
        return false;
    }
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                            sizeof(float) * (size_t)hidden * bucket);
    ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                            sizeof(int32_t) * pos_buf.size());
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());
    ggml_backend_tensor_set(sg.active_slot_ids, active_slot_ids.data(), 0,
                            sizeof(int32_t) * active_slot_ids.size());
    ggml_backend_tensor_set(sg.state_slot_ids, state_slot_ids.data(), 0,
                            sizeof(int32_t) * state_slot_ids.size());
    ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens,
                            seq_lens.data(), 0,
                            sizeof(int32_t) * seq_lens.size());

    auto st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        for (StepOutput & o : outputs) o.failed = true;
        return false;
    }

    std::vector<int32_t> argmax((size_t)bucket, -1);
    ggml_backend_tensor_get(sg.argmax_tokens, argmax.data(), 0,
                            sizeof(int32_t) * (size_t)bucket);
    std::vector<float> logits_buf;
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        StepOutput & out = outputs[oi];
        if (out.failed) continue;
        const int row = output_rows[oi];
        if (row < 0 || row >= live_count) {
            out.failed = true;
            continue;
        }
        slots_.commit_step(out.slot);
        out.token = sample_graph_row(
            out.slot, row, &argmax[(size_t)row], &logits_buf);
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
