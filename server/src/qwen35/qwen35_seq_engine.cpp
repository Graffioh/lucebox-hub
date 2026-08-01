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
#include "common/sampler.h"
#include "internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
    const int prompt_len = (int)prompt.size();

    // Slot choice, pool-handle lifecycle, worst-case reservation, and the
    // per-slot sampler/RNG/history state are the manager's; the engine only
    // runs the prefill and pushes the returned metadata to the device.
    SeqSlotManager::AdmitOutcome ao =
        slots_.admit(request_id, prompt_len, n_gen, sampler);
    if (!ao.ok) {
        r.busy = ao.busy;
        r.error = std::move(ao.error);
        return r;
    }
    const int slot = ao.slot;
    if (ao.table_column.size() > (size_t)b_.cache_.paged_block_table->ne[0]) {
        slots_.retire(slot);
        r.error = "block table exceeds the device column capacity";
        return r;
    }

    // Fresh recurrent state for this slot only; staging gets the zeroed
    // buffer the dense prefill path expects.
    reset_slot_recurrent_state(b_.cache_, slot);
    reset_prefill_staging(b_.cache_);

    const auto t0 = std::chrono::steady_clock::now();
    int32_t first_tok = -1;
    const int committed =
        prefill_slot(slot, prompt, ao.prompt_rows, &first_tok);
    if (committed != prompt_len) {
        slots_.retire(slot);
        r.error = "slot prefill failed";
        return r;
    }
    slots_.commit_prefill(slot, committed);
    r.prefill_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Upload this slot's full (reserved) block-table column and its length.
    // reserve() fixed the whole table at admission, so this single upload
    // covers every block the sequence will ever touch.
    ggml_backend_tensor_set(
        b_.cache_.paged_block_table, ao.table_column.data(),
        (size_t)slot * b_.cache_.paged_block_table->nb[1],
        ao.table_column.size() * sizeof(int32_t));
    ggml_backend_tensor_set(b_.cache_.paged_kv_seq_lens,
                            &slots_.lens_host()[(size_t)slot],
                            (size_t)slot * sizeof(int32_t), sizeof(int32_t));

    r.ok = true;
    r.slot = slot;
    r.first_token = first_tok;
    return r;
}

int Qwen35SeqEngine::prefill_slot(int slot,
                                         const std::vector<int32_t> & tokens,
                                         const std::vector<int64_t> & phys_rows,
                                         int32_t * first_token_out) {
    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden = w.n_embd;
    const int vocab  = w.n_vocab;
    int prefill_ubatch = 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        prefill_ubatch = std::max(1, std::atoi(s));
    }
    const int prompt_len = (int)tokens.size();
    if ((size_t)prompt_len != phys_rows.size()) return -1;

    SeqSlot & seq = slots_.slot(slot);
    std::vector<float> embed_buf((size_t)hidden * prefill_ubatch);
    std::vector<int64_t> rows;
    int committed = 0;
    for (int start = 0; start < prompt_len;) {
        const int kv_pos = start;
        const int n_tokens = std::min(prefill_ubatch, prompt_len - start);
        const bool with_mask =
            (b_.cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        if (!build_target_step(sg, w, b_.cache_, b_.target_backend_,
                               /*kv_start=*/kv_pos, /*n_tokens=*/n_tokens,
                               with_mask, /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/(start + n_tokens < prompt_len),
                               b_.cfg_.kq_stride_pad,
                               /*capture_moe_router=*/false,
                               /*kvflash_mask=*/false,
                               /*capture_qk=*/false,
                               /*paged_attention=*/false,
                               /*n_seqs=*/1,
                               /*seq_slot=*/slot,
                               /*paged_prefill=*/true)) {
            std::fprintf(stderr, "[parallel] prefill build @%d failed\n", kv_pos);
            return -1;
        }
        if (!sg.kv_write_rows) {
            std::fprintf(stderr,
                "[parallel] prefill requires the set_rows path\n");
            return -1;
        }
        // [n_tokens, n_head_kv] ne0-major (see do_prefill's pooled branch).
        rows.assign((size_t)n_tokens * w.n_head_kv, 0);
        for (int h = 0; h < w.n_head_kv; h++) {
            for (int i = 0; i < n_tokens; i++) {
                rows[(size_t)h * n_tokens + i] = phys_rows[(size_t)(kv_pos + i)];
            }
        }
        ggml_backend_tensor_set(sg.kv_write_rows, rows.data(), 0,
                                sizeof(int64_t) * rows.size());

        if (!w.embedder.embed(tokens.data() + start, n_tokens,
                              embed_buf.data())) {
            return -1;
        }
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * (size_t)hidden * n_tokens);

        std::vector<int32_t> pos_buf((size_t)4 * n_tokens, 0);
        for (int i = 0; i < n_tokens; i++) {
            const int p = kv_pos + i;
            pos_buf[4 * i + 0] = p;
            pos_buf[4 * i + 1] = p;
            pos_buf[4 * i + 2] = p;
            pos_buf[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                                sizeof(int32_t) * pos_buf.size());

        if (sg.attn_mask) {
            const int kv_len = kv_pos + n_tokens;
            std::vector<uint16_t> mask_buf;
            const int kv_pad_override = (int)sg.attn_mask->ne[0];
            build_causal_mask(mask_buf, kv_len, n_tokens, kv_pos,
                              b_.cfg_.kq_stride_pad, /*win_start=*/0,
                              kv_pad_override);
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        auto st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[parallel] prefill compute @%d failed\n", kv_pos);
            return -1;
        }

        const bool is_final = (start + n_tokens >= prompt_len);
        if (is_final && first_token_out) {
            if (seq.sampler.needs_logit_processing()) {
                std::vector<float> logits_buf((size_t)vocab);
                ggml_backend_tensor_get(sg.logits, logits_buf.data(),
                                        (size_t)(n_tokens - 1) * vocab * sizeof(float),
                                        sizeof(float) * vocab);
                *first_token_out = sample_logits(logits_buf.data(), vocab,
                                                 seq.sampler,
                                                 seq.sample_history, seq.rng);
            } else {
                int32_t tok = -1;
                ggml_backend_tensor_get(sg.argmax_tokens, &tok,
                                        sizeof(int32_t) * (size_t)(n_tokens - 1),
                                        sizeof(int32_t));
                *first_token_out = tok;
            }
            *first_token_out = b_.apply_min_tokens_floor(
                *first_token_out, /*generated=*/0,
                (size_t)(n_tokens - 1) * (size_t)vocab * sizeof(float));
        }
        committed = kv_pos + n_tokens;
        start += n_tokens;
    }
    return committed;
}

bool Qwen35SeqEngine::step(const std::vector<StepInput> & inputs,
                                  std::vector<StepOutput> & outputs) {
    outputs.clear();
    if (inputs.empty()) return true;

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int n_slots   = slots_.slot_count();
    const int hidden    = w.n_embd;
    const int vocab     = w.n_vocab;
    const int n_head_kv = w.n_head_kv;

    // Every ACTIVE slot must appear in `inputs`: the batched forward updates
    // every slot's recurrent state, so a live slot left out would have its
    // DeltaNet state advanced by the dummy token. Enforce the contract.
    if ((int)inputs.size() != slots_.active_count()) {
        std::fprintf(stderr,
            "[parallel] step got %zu inputs for %d active slots\n",
            inputs.size(), slots_.active_count());
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
            outputs.push_back(out);
            output_rows.push_back(compact_row);
            continue;
        }
        // Cache-row allocation, the context guard, the kv-length mirror,
        // and the penalty history all live in the manager.
        const SeqSlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
        if (!app.ok) {
            outputs.push_back(out);
            output_rows.push_back(compact_row);
            continue;
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
    for (int row = 0; row < live_count; ++row) {
        tokens[(size_t)row] = live_tokens[(size_t)row];
        const int pos = live_positions[(size_t)row];
        pos_buf[(size_t)4 * row + 0] = pos;
        pos_buf[(size_t)4 * row + 1] = pos;
        pos_buf[(size_t)4 * row + 2] = pos;
        active_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
        state_slot_ids[(size_t)row] = live_slot_ids[(size_t)row];
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
                           /*compact_slots=*/true,
                           /*graph_key_slot=*/decode_bucket_graph_key(bucket))) {
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
                            slots_.lens_host().data(), 0,
                            sizeof(int32_t) * slots_.lens_host().size());

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
        SeqSlot & s = slots_.slot(out.slot);
        slots_.commit_step(out.slot);
        int32_t next;
        if (s.sampler.needs_logit_processing()) {
            if (logits_buf.empty()) logits_buf.resize((size_t)vocab);
            ggml_backend_tensor_get(sg.logits, logits_buf.data(),
                                    (size_t)row * (size_t)vocab * sizeof(float),
                                    sizeof(float) * (size_t)vocab);
            next = sample_logits(logits_buf.data(), vocab, s.sampler,
                                 s.sample_history, s.rng);
        } else {
            next = argmax[(size_t)row];
        }
        next = b_.apply_min_tokens_floor(next, (int)s.sample_history.size(),
                                         (size_t)row * (size_t)vocab *
                                             sizeof(float));
        out.token = next;
    }
    return true;
}

void Qwen35SeqEngine::retire(int slot) {
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
