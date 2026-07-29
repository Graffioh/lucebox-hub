// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "graph_builders.h"
#include "attn_masks.h"
#include "common/sampler.h"
#include "internal.h"

#include <algorithm>
#include <chrono>
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

int decode_bucket_graph_key(int width) {
    int key = 1; // zero is reserved for non-bucketed prefill/verify graphs
    while (width > 1) {
        width >>= 1;
        ++key;
    }
    return key;
}

} // namespace

bool Qwen35SeqEngine::token_is_eos(int32_t token) const {
    return b_.token_is_eos(token);
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        int n_gen) {
    AdmitResult r;
    if (pending_prefill_) {
        r.busy = true;
        r.error = "another admission is still prefilling";
        return r;
    }

    SeqSlotManager::AdmitOutcome ao =
        slots_.admit(request_id, (int)prompt.size(), n_gen, sampler);
    if (!ao.ok) {
        r.busy = ao.busy;
        r.error = std::move(ao.error);
        return r;
    }
    const int slot = ao.slot;

    // The staging K/V and recurrent slabs carry this sequence across its
    // incremental prefill chunks. The slot's device length remains zero
    // until the final chunk commits.
    reset_prefill_staging(b_.cache_);

    PendingPrefill pp;
    pp.slot = slot;
    pp.prompt = prompt;
    pp.admitted_at = std::chrono::steady_clock::now();
    pending_prefill_ = std::move(pp);

    r.ok = true;
    r.slot = slot;
    r.n_gen_cap = ao.n_gen_cap;
    return r;
}

int32_t Qwen35SeqEngine::sample_prefill_first_token(int slot) {
    // The committing graph leaves the prompt's final logits at tail row 0.
    const int vocab = b_.w_.n_vocab;
    SeqSlot & seq = slots_.slot(slot);
    int32_t first = -1;
    if (seq.sampler.needs_logit_processing()) {
        std::vector<float> logits_buf((size_t)vocab);
        ggml_backend_tensor_get(b_.sg_.logits, logits_buf.data(), 0,
                                sizeof(float) * (size_t)vocab);
        first = sample_logits(logits_buf.data(), vocab, seq.sampler,
                              seq.sample_history, seq.rng);
    } else {
        ggml_backend_tensor_get(b_.sg_.argmax_tokens, &first, 0,
                                sizeof(int32_t));
    }
    first = b_.apply_min_tokens_floor(first, /*generated=*/0,
                                      /*logits_row_offset=*/0);
    return first;
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
    PendingPrefill * pp = pending_prefill_ ? &*pending_prefill_ : nullptr;
    if (inputs.empty() && !pp) return true;

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden    = w.n_embd;
    const int vocab     = w.n_vocab;
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
            if (app.new_block_index < 0 ||
                app.new_block_index >= b_.cache_.paged_block_table->ne[0]) {
                out.error = "decode block-table entry exceeds device capacity";
                outputs.push_back(out);
                output_rows.push_back(compact_row);
                continue;
            }
            ggml_backend_tensor_set(
                b_.cache_.paged_block_table, &app.new_block,
                (size_t)in.slot * b_.cache_.paged_block_table->nb[1] +
                    (size_t)app.new_block_index * sizeof(int32_t),
                sizeof(int32_t));
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
    if (!with_decode && !pp) return true;
    const int decode_bucket = with_decode ? decode_bucket_width(live_count) : 0;

    // Pack live sequences first and pad only to the next graph bucket. Paged
    // attention and GDN use these ids to reach physical metadata/state.
    std::vector<int32_t> dec_tokens((size_t)decode_bucket, 0);
    std::vector<int32_t> dec_pos((size_t)4 * decode_bucket, 0);
    std::vector<int64_t> dec_rows(
        (size_t)decode_bucket * n_head_kv, scratch_row_);
    std::vector<int32_t> active_slot_ids((size_t)decode_bucket, -1);
    std::vector<int32_t> state_slot_ids((size_t)decode_bucket, 0);
    for (int row = 0; row < live_count; ++row) {
        dec_tokens[(size_t)row] = live_tokens[(size_t)row];
        const int pos = live_positions[(size_t)row];
        dec_pos[(size_t)4 * row + 0] = pos;
        dec_pos[(size_t)4 * row + 1] = pos;
        dec_pos[(size_t)4 * row + 2] = pos;
        active_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        state_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        for (int h = 0; h < n_head_kv; ++h) {
            dec_rows[(size_t)h * decode_bucket + row] =
                live_physical_rows[(size_t)row];
        }
    }

    static const int prefill_ubatch = [] {
        if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
            const int v = std::atoi(s);
            if (v >= 1) return v;
        }
        return 512;
    }();
    int chunk = 0;
    bool commit = false;
    int kv_pos = 0;
    bool with_prefill = false;
    std::vector<int64_t> pf_rows;
    std::vector<float> pf_embed;

    auto fail_pending = [&](const char * log_message,
                            const char * client_message) {
        if (!pp) return;
        std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                     log_message, pp->slot);
        StepOutput out;
        out.slot = pp->slot;
        out.failed = true;
        out.error = client_message;
        out.prefill_done = true;
        outputs.push_back(std::move(out));
        pending_prefill_.reset();
        pp = nullptr;
        chunk = 0;
        commit = false;
        with_prefill = false;
        pf_rows.clear();
        pf_embed.clear();
    };

    if (pp) {
        const int prompt_len = (int)pp->prompt.size();
        kv_pos = pp->progress;
        chunk = std::min(prefill_ubatch, prompt_len - kv_pos);
        commit = (kv_pos + chunk >= prompt_len);

        SeqSlotManager::PrefillChunk pc =
            slots_.append_prefill(pp->slot, chunk);
        if (pc.busy) {
            // Preserve the partial prefill and let live decode advance. A
            // later retirement may free the blocks this chunk needs.
            chunk = 0;
            commit = false;
        } else if (!pc.ok || pc.rows.size() != (size_t)chunk) {
            fail_pending("prefill K/V allocation failed",
                         "prefill K/V allocation failed");
        } else {
            if (!pc.new_blocks.empty()) {
                const int64_t end =
                    (int64_t)pc.first_new_block + pc.new_blocks.size();
                if (pc.first_new_block < 0 ||
                    end > b_.cache_.paged_block_table->ne[0]) {
                    fail_pending(
                        "prefill block-table delta exceeds device capacity",
                        "prefill block-table update failed");
                } else {
                    ggml_backend_tensor_set(
                        b_.cache_.paged_block_table, pc.new_blocks.data(),
                        (size_t)pp->slot *
                                b_.cache_.paged_block_table->nb[1] +
                            (size_t)pc.first_new_block * sizeof(int32_t),
                        pc.new_blocks.size() * sizeof(int32_t));
                }
            }
            if (pp) {
                pf_rows = std::move(pc.rows);
                with_prefill = true;
                pf_embed.resize((size_t)hidden * chunk);
                if (!w.embedder.embed(pp->prompt.data() + kv_pos, chunk,
                                      pf_embed.data())) {
                    fail_pending("prefill embed failed",
                                 "prefill embedding failed");
                }
            }
        }
    }

    // A prefill graph build failure fails only the admission and falls back
    // to the established decode graph for existing streams.
    auto fail_pending_build = [&]() {
        fail_pending("prefill step build failed",
                     "prefill graph build failed");
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
            /*last_token_logits_only=*/false,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/decode_bucket,
            /*seq_slot=*/pp->slot,
            /*paged_prefill=*/true,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/chunk,
            /*prefill_commit=*/commit,
            /*logits_tail_rows=*/decode_bucket + (commit ? 1 : 0),
            /*compact_slots=*/true,
            /*graph_key_slot=*/decode_bucket_graph_key(decode_bucket));
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
            /*last_token_logits_only=*/false,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/false,
            /*n_seqs=*/1,
            /*seq_slot=*/pp->slot,
            /*paged_prefill=*/true,
            /*paged_max_kv_len=*/0,
            /*n_prefill_tokens=*/0,
            /*prefill_commit=*/commit,
            /*logits_tail_rows=*/1);
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
            /*last_token_logits_only=*/false,
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
            /*logits_tail_rows=*/0,
            /*compact_slots=*/true,
            /*graph_key_slot=*/decode_bucket_graph_key(decode_bucket));
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
        std::copy(pf_embed.begin(), pf_embed.end(), embed_buf.begin());
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
    for (int i = 0; i < n_prefill; i++) {
        const int p = kv_pos + i;
        pos_buf[4 * i + 0] = p;
        pos_buf[4 * i + 1] = p;
        pos_buf[4 * i + 2] = p;
    }
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
                pf_rows[(size_t)i];
        }
        for (int s = 0; s < n_decode; s++) {
            rows[(size_t)h * n_total + n_prefill + s] =
                dec_rows[(size_t)h * decode_bucket + s];
        }
    }
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());

    if (n_prefill > 0 && sg.attn_mask) {
        std::vector<uint16_t> mask_buf;
        build_causal_mask(mask_buf, kv_pos + n_prefill, n_prefill, kv_pos,
                          b_.cfg_.kq_stride_pad, /*win_start=*/0,
                          (int)sg.attn_mask->ne[0]);
        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }
    if (n_decode > 0) {
        ggml_backend_tensor_set(
            sg.active_slot_ids, active_slot_ids.data(), 0,
            sizeof(int32_t) * active_slot_ids.size());
        ggml_backend_tensor_set(
            sg.state_slot_ids, state_slot_ids.data(), 0,
            sizeof(int32_t) * state_slot_ids.size());
        ggml_backend_tensor_set(
            b_.cache_.paged_kv_seq_lens, slots_.lens_host().data(), 0,
            sizeof(int32_t) * slots_.lens_host().size());
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
            if (oi >= output_rows.size() || output_rows[oi] < 0 ||
                output_rows[oi] >= live_count) {
                out.failed = true;
                continue;
            }
            SeqSlot & seq = slots_.slot(out.slot);
            slots_.commit_step(out.slot);
            const size_t row =
                (size_t)(dec_row0 + output_rows[oi]);
            int32_t next;
            if (seq.sampler.needs_logit_processing()) {
                if (logits_buf.empty()) logits_buf.resize((size_t)vocab);
                ggml_backend_tensor_get(
                    sg.logits, logits_buf.data(),
                    row * (size_t)vocab * sizeof(float),
                    sizeof(float) * (size_t)vocab);
                next = sample_logits(logits_buf.data(), vocab, seq.sampler,
                                     seq.sample_history, seq.rng);
            } else {
                next = argmax[row];
            }
            next = b_.apply_min_tokens_floor(
                next, (int)seq.sample_history.size(),
                row * (size_t)vocab * sizeof(float));
            out.token = next;
        }
    }

    if (with_prefill) {
        pp->progress += chunk;
        if (commit) {
            StepOutput out;
            out.slot = pp->slot;
            out.prefill_done = true;
            out.token = sample_prefill_first_token(pp->slot);
            out.prefill_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pp->admitted_at).count();
            slots_.commit_prefill(pp->slot, (int)pp->prompt.size());
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
    if (b_.cache_.paged_kv_seq_lens) {
        const int32_t zero = 0;
        ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens, &zero,
                                (size_t)slot * sizeof(int32_t),
                                sizeof(int32_t));
    }
}

}  // namespace dflash::common
