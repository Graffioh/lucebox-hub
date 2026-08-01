// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "graph_builders.h"
#include "common/paged_attention_config.h"
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

int concurrent_prefill_ubatch() {
    static const int value = [] {
        if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
            const int v = std::atoi(s);
            if (v >= 1) return v;
        }
        return 512;
    }();
    return value;
}

}  // namespace

Qwen35SeqEngine::Qwen35SeqEngine(
        Qwen35Backend & backend, PagedKvPool & pool,
        int max_ctx, int64_t scratch_row)
    : b_(backend),
      slots_(pool, max_ctx, paged_kv_admission_watermark_blocks(
          pool.physical_block_count(), concurrent_prefill_ubatch())),
      scratch_row_(scratch_row) {
    pending_prefills_.resize((size_t)slots_.slot_count());
}

bool Qwen35SeqEngine::prefill_pending() const {
    return std::any_of(pending_prefills_.begin(), pending_prefills_.end(),
                       [](const auto & p) { return p.has_value(); });
}

bool Qwen35SeqEngine::token_is_eos(int32_t token) const {
    return b_.token_is_eos(token);
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        int n_gen,
        const ResumeState * resume) {
    AdmitResult r;
    SeqSlotManager::AdmitOutcome ao =
        slots_.admit(request_id, (int)prompt.size(), n_gen, sampler,
                     resume ? &resume->sample_history : nullptr,
                     resume ? &resume->rng : nullptr);
    if (!ao.ok) {
        r.busy = ao.busy;
        r.error = std::move(ao.error);
        return r;
    }
    const int slot = ao.slot;

    // Reused slots may contain a retired request's DeltaNet recurrence. KV
    // rows need no clear because the new block table cannot name stale rows.
    reset_recurrent_slot(b_.cache_, slot);

    PendingPrefill pp;
    pp.slot = slot;
    pp.prompt = prompt;
    pp.admitted_at = std::chrono::steady_clock::now();
    pending_prefills_[(size_t)slot] = std::move(pp);

    r.ok = true;
    r.slot = slot;
    r.n_gen_cap = ao.n_gen_cap;
    return r;
}

bool Qwen35SeqEngine::capture_resume_state(
        int slot, ResumeState & out) const {
    return slots_.capture_sampling_state(
        slot, out.sample_history, out.rng);
}

int32_t Qwen35SeqEngine::sample_prefill_first_token(
        int slot, int logits_row) {
    const int vocab = b_.w_.n_vocab;
    SeqSlot & seq = slots_.slot(slot);
    int32_t first = -1;
    if (seq.sampler.needs_logit_processing()) {
        std::vector<float> logits_buf((size_t)vocab);
        ggml_backend_tensor_get(b_.sg_.logits, logits_buf.data(),
                                (size_t)logits_row * vocab * sizeof(float),
                                sizeof(float) * (size_t)vocab);
        first = sample_logits(logits_buf.data(), vocab, seq.sampler,
                              seq.sample_history, seq.rng);
    } else {
        ggml_backend_tensor_get(b_.sg_.argmax_tokens, &first,
                                (size_t)logits_row * sizeof(int32_t),
                                sizeof(int32_t));
    }
    first = b_.apply_min_tokens_floor(
        first, (int)seq.sample_history.size(),
        (size_t)logits_row * vocab * sizeof(float));
    return first;
}

bool Qwen35SeqEngine::step(
        const std::vector<StepInput> & inputs,
        std::vector<StepOutput> & outputs) {
    outputs.clear();
    const int n_slots = slots_.slot_count();

    // Every decoding slot must appear. Prefilling slots are represented by
    // the engine's ragged prompt segments instead.
    if ((int)inputs.size() != slots_.decoding_count()) {
        std::fprintf(stderr,
            "[parallel] step got %zu inputs for %d decoding slots\n",
            inputs.size(), slots_.decoding_count());
        return false;
    }
    if (inputs.empty() && !prefill_pending()) return true;

    // Preflight every boundary allocation before mutating any slot. Without
    // this transaction boundary, early inputs could append successfully and a
    // later input could discover exhaustion before the graph executes.
    std::vector<int> decode_slots;
    decode_slots.reserve(inputs.size());
    for (const StepInput & in : inputs) decode_slots.push_back(in.slot);
    const int pressure_slot = slots_.decode_pressure_slot(decode_slots);
    if (pressure_slot >= 0) {
        StepOutput out;
        out.slot = pressure_slot;
        out.pool_exhausted = true;
        outputs.push_back(std::move(out));
        return true;
    }

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden    = w.n_embd;
    const int vocab     = w.n_vocab;
    const int n_head_kv = w.n_head_kv;

    outputs.reserve(inputs.size() + pending_prefills_.size());
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
    std::vector<int> decode_write_slots;
    decode_write_slots.reserve(inputs.size());
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
        decode_write_slots.push_back(in.slot);
        max_kv_len = std::max(max_kv_len, app.position + 1);
        out.failed = false;
        outputs.push_back(out);
        output_rows.push_back(compact_row);
    }
    const size_t decode_output_count = outputs.size();
    const int live_count = (int)live_tokens.size();
    const bool with_decode = live_count > 0;
    const int decode_bucket = with_decode ? decode_bucket_width(live_count) : 0;

    // Pack live sequences first and pad only to the next graph bucket. Paged
    // attention and GDN use these ids to reach physical metadata/state.
    std::vector<int32_t> dec_tokens((size_t)decode_bucket, 0);
    std::vector<int32_t> dec_pos((size_t)4 * decode_bucket, 0);
    std::vector<int64_t> dec_rows(
        (size_t)decode_bucket * n_head_kv, scratch_row_);
    std::vector<int32_t> dec_query_pos((size_t)decode_bucket, -1);
    std::vector<int32_t> active_slot_ids((size_t)decode_bucket, -1);
    std::vector<int32_t> state_slot_ids((size_t)decode_bucket, 0);
    for (int row = 0; row < live_count; ++row) {
        dec_tokens[(size_t)row] = live_tokens[(size_t)row];
        const int pos = live_positions[(size_t)row];
        dec_pos[(size_t)4 * row + 0] = pos;
        dec_pos[(size_t)4 * row + 1] = pos;
        dec_pos[(size_t)4 * row + 2] = pos;
        dec_query_pos[(size_t)row] = pos;
        active_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        state_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        for (int h = 0; h < n_head_kv; ++h) {
            dec_rows[(size_t)h * decode_bucket + row] =
                live_physical_rows[(size_t)row];
        }
    }

    struct SelectedPrefill {
        int slot = -1;
        int kv_pos = 0;
        int chunk = 0;
        bool commit = false;
        std::vector<int64_t> rows;
    };
    std::vector<SelectedPrefill> selected;

    auto fail_pending = [&](int slot, const char * log_message,
                            const char * client_message) {
        if (slot < 0 || slot >= n_slots ||
            !pending_prefills_[(size_t)slot]) return;
        std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                     log_message, slot);
        StepOutput out;
        out.slot = slot;
        out.failed = true;
        out.error = client_message;
        out.prefill_done = true;
        outputs.push_back(std::move(out));
        pending_prefills_[(size_t)slot].reset();
    };

    // Divide one global token budget fairly among the pending prompts. The
    // starting slot rotates each step, so a short prompt behind long prompts
    // cannot be perpetually last.
    int n_pending = 0;
    for (const auto & p : pending_prefills_) n_pending += p ? 1 : 0;
    int budget = concurrent_prefill_ubatch();
    int left = n_pending;
    const int start = prefill_cursor_ % std::max(1, n_slots);
    for (int scan = 0; scan < n_slots && budget > 0; ++scan) {
        const int slot = (start + scan) % n_slots;
        auto & pending = pending_prefills_[(size_t)slot];
        if (!pending) continue;
        PendingPrefill & pp = *pending;
        const int prompt_len = (int)pp.prompt.size();
        const int share = std::max(1, budget / std::max(1, left));
        const int chunk = std::min(share, prompt_len - pp.progress);
        --left;
        if (chunk <= 0) {
            fail_pending(slot, "invalid prefill progress",
                         "invalid prefill progress");
            continue;
        }
        SeqSlotManager::PrefillChunk pc = slots_.append_prefill(slot, chunk);
        if (pc.busy) {
            continue;
        } else if (!pc.ok || pc.rows.size() != (size_t)chunk) {
            fail_pending(slot, "prefill K/V allocation failed",
                         "prefill K/V allocation failed");
        } else {
            if (!pc.new_blocks.empty()) {
                const int64_t end =
                    (int64_t)pc.first_new_block + pc.new_blocks.size();
                if (pc.first_new_block < 0 ||
                    end > b_.cache_.paged_block_table->ne[0]) {
                    fail_pending(slot,
                        "prefill block-table delta exceeds device capacity",
                        "prefill block-table update failed");
                } else {
                    ggml_backend_tensor_set(
                        b_.cache_.paged_block_table, pc.new_blocks.data(),
                        (size_t)slot *
                                b_.cache_.paged_block_table->nb[1] +
                            (size_t)pc.first_new_block * sizeof(int32_t),
                        pc.new_blocks.size() * sizeof(int32_t));
                }
            }
            if (pending) {
                SelectedPrefill pf;
                pf.slot = slot;
                pf.kv_pos = pp.progress;
                pf.chunk = chunk;
                pf.commit = pp.progress + chunk >= prompt_len;
                pf.rows = std::move(pc.rows);
                selected.push_back(std::move(pf));
                budget -= chunk;
                max_kv_len = std::max(max_kv_len, pp.progress + chunk);
            }
        }
    }
    prefill_cursor_ = (start + 1) % std::max(1, n_slots);

    if (!with_decode && selected.empty()) return true;

    // Token axis: [ragged prompt chunks | compact bucketed decode rows].
    int n_prefill = 0;
    for (const SelectedPrefill & pf : selected) n_prefill += pf.chunk;
    const int n_decode  = with_decode ? decode_bucket : 0;
    const int n_total   = n_prefill + n_decode;

    std::vector<float> embed_buf((size_t)hidden * n_total);
    int token_off = 0;
    for (const SelectedPrefill & pf : selected) {
        const PendingPrefill & pp = *pending_prefills_[(size_t)pf.slot];
        if (!w.embedder.embed(pp.prompt.data() + pf.kv_pos, pf.chunk,
                              embed_buf.data() + (size_t)hidden * token_off)) {
            fail_pending(pf.slot, "prefill embed failed",
                         "prefill embedding failed");
            return false;
        }
        token_off += pf.chunk;
    }
    if (n_decode > 0 &&
        !w.embedder.embed(dec_tokens.data(), n_decode,
                          embed_buf.data() + (size_t)hidden * n_prefill)) {
        for (StepOutput & out : outputs) out.failed = true;
        return false;
    }
    std::vector<int32_t> pos_buf((size_t)4 * n_total, 0);
    token_off = 0;
    for (const SelectedPrefill & pf : selected) {
        for (int i = 0; i < pf.chunk; ++i) {
            const int p = pf.kv_pos + i;
            pos_buf[4 * (token_off + i) + 0] = p;
            pos_buf[4 * (token_off + i) + 1] = p;
            pos_buf[4 * (token_off + i) + 2] = p;
        }
        token_off += pf.chunk;
    }
    if (n_decode > 0) {
        std::copy(dec_pos.begin(), dec_pos.end(),
                  pos_buf.begin() + (size_t)4 * n_prefill);
    }
    std::vector<int64_t> rows((size_t)n_total * n_head_kv, scratch_row_);
    for (int h = 0; h < n_head_kv; h++) {
        token_off = 0;
        for (const SelectedPrefill & pf : selected) {
            for (int i = 0; i < pf.chunk; ++i) {
                rows[(size_t)h * n_total + token_off + i] =
                    pf.rows[(size_t)i];
            }
            token_off += pf.chunk;
        }
        for (int s = 0; s < n_decode; s++) {
            rows[(size_t)h * n_total + n_prefill + s] =
                dec_rows[(size_t)h * decode_bucket + s];
        }
    }
    PagedStepPlan plan;
    plan.with_decode = with_decode;
    plan.state_write_slots = decode_write_slots;
    std::vector<int32_t> query_slots((size_t)n_total, -1);
    std::vector<int32_t> query_positions((size_t)n_total, -1);
    token_off = 0;
    int n_commits = 0;
    for (const SelectedPrefill & pf : selected) {
        plan.prefills.push_back({token_off, pf.chunk, pf.slot});
        for (int i = 0; i < pf.chunk; ++i) {
            query_slots[(size_t)token_off + i] = pf.slot;
            query_positions[(size_t)token_off + i] = pf.kv_pos + i;
        }
        if (pf.commit) {
            plan.logits_rows.push_back(token_off + pf.chunk - 1);
            ++n_commits;
        }
        token_off += pf.chunk;
    }
    if (with_decode) {
        for (int row = 0; row < decode_bucket; ++row) {
            query_slots[(size_t)n_prefill + row] =
                active_slot_ids[(size_t)row];
            query_positions[(size_t)n_prefill + row] =
                dec_query_pos[(size_t)row];
            plan.logits_rows.push_back(n_prefill + row);
        }
    }
    if (plan.logits_rows.empty()) {
        // Keep the graph's output contract small on non-committing pure
        // prefill steps; this row is not sampled.
        plan.logits_rows.push_back(n_total - 1);
    }

    const int graph_n_seqs = with_decode ? decode_bucket : n_slots;
    const bool built = build_target_step(
        sg, w, b_.cache_, b_.target_backend_,
        /*kv_start=*/0, n_total,
        /*with_mask=*/false, /*capture=*/false,
        /*capture_delta_intermediate=*/false,
        /*fa_window=*/0, /*last_token_logits_only=*/false,
        b_.cfg_.kq_stride_pad,
        /*capture_moe_router=*/false,
        /*kvflash_mask=*/false,
        /*capture_qk=*/false,
        /*paged_attention=*/true,
        /*n_seqs=*/graph_n_seqs,
        /*seq_slot=*/0,
        max_kv_len,
        /*logits_tail_rows=*/0,
        &plan,
        /*compact_slots=*/with_decode,
        /*graph_key_slot=*/with_decode
            ? decode_bucket_graph_key(decode_bucket) : 0);
    if (!built || !sg.kv_write_rows || !sg.paged_query_seq_ids ||
        !sg.paged_query_positions || !sg.logits_row_indices ||
        (with_decode && (!sg.active_slot_ids || !sg.state_slot_ids))) {
        std::fprintf(stderr, "[parallel] ragged step build failed\n");
        return false;
    }

    // The graph rebuild above creates the input tensors, so upload all
    // step-specific metadata after it.
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                            sizeof(float) * (size_t)hidden * n_total);
    ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                            sizeof(int32_t) * pos_buf.size());
    ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                            sizeof(int64_t) * rows.size());
    ggml_backend_tensor_set(sg.paged_query_seq_ids, query_slots.data(), 0,
                            sizeof(int32_t) * query_slots.size());
    ggml_backend_tensor_set(sg.paged_query_positions,
                            query_positions.data(), 0,
                            sizeof(int32_t) * query_positions.size());
    ggml_backend_tensor_set(sg.logits_row_indices, plan.logits_rows.data(), 0,
                            sizeof(int32_t) * plan.logits_rows.size());
    if (with_decode) {
        ggml_backend_tensor_set(
            sg.active_slot_ids, active_slot_ids.data(), 0,
            sizeof(int32_t) * active_slot_ids.size());
        ggml_backend_tensor_set(
            sg.state_slot_ids, state_slot_ids.data(), 0,
            sizeof(int32_t) * state_slot_ids.size());
    }
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, slots_.lens_host().data(), 0,
        sizeof(int32_t) * slots_.lens_host().size());

    const auto st =
        ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[parallel] ragged step compute failed (prefill=%d live=%d)\n",
            n_prefill, with_decode ? 1 : 0);
        for (StepOutput & out : outputs) out.failed = true;
        return false;
    }

    const int dec_row0 = n_commits;
    if (n_decode > 0) {
        std::vector<int32_t> argmax(
            (size_t)(dec_row0 + decode_bucket), -1);
        ggml_backend_tensor_get(sg.argmax_tokens, argmax.data(), 0,
                                sizeof(int32_t) * argmax.size());
        std::vector<float> logits_buf;
        for (size_t oi = 0; oi < decode_output_count; ++oi) {
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

    int commit_row = 0;
    for (const SelectedPrefill & pf : selected) {
        auto & pending = pending_prefills_[(size_t)pf.slot];
        if (!pending) continue;
        pending->progress += pf.chunk;
        if (pf.commit) {
            StepOutput out;
            out.slot = pf.slot;
            out.prefill_done = true;
            out.token = sample_prefill_first_token(pf.slot, commit_row++);
            out.prefill_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pending->admitted_at).count();
            slots_.commit_prefill(pf.slot, (int)pending->prompt.size());
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
    if (b_.cache_.paged_kv_seq_lens) {
        const int32_t zero = 0;
        ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens, &zero,
                                (size_t)slot * sizeof(int32_t),
                                sizeof(int32_t));
    }
}

}  // namespace dflash::common
