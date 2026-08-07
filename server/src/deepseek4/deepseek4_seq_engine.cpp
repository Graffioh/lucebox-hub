#include "deepseek4_seq_engine.h"

#include "deepseek4_backend.h"
#include "common/sampler.h"

#include <algorithm>
#include <cstdio>
#include <unordered_set>

namespace dflash::common {

DeepSeek4SeqEngine::DeepSeek4SeqEngine(
        DeepSeek4Backend & backend, PagedKvPool & pool, int max_ctx,
        uint32_t table_stride)
    : b_(backend), slots_(pool, max_ctx), stride_(table_stride),
      host_tables_((size_t)pool.max_sequences() * table_stride, -1) {}

bool DeepSeek4SeqEngine::token_is_eos(int32_t token) const {
    return deepseek4_is_eos_tok(token, b_.w_);
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    if (pending_) return {false, true, -1, "another admission is still prefilling"};
    AdmitResult result = slots_.admit(request_id, (int)prompt.size(), sampler);
    if (!result.ok) return result;
    std::fill_n(host_tables_.data() + (size_t)result.slot * stride_, stride_, -1);
    pending_ = PendingPrefill{result.slot, 0, prompt};
    return result;
}

bool DeepSeek4SeqEngine::set_block(int slot, int logical, int32_t physical) {
    if (slot < 0 || slot >= slots_.slot_count() || logical < 0 ||
        (uint32_t)logical >= stride_ || !b_.paged_cache_.block_table) return false;
    host_tables_[(size_t)slot * stride_ + (size_t)logical] = physical;
    ggml_tensor * table = b_.paged_cache_.block_table;
    ggml_backend_tensor_set(table, &physical,
        (size_t)slot * table->nb[1] + (size_t)logical * sizeof(int32_t),
        sizeof(physical));
    return true;
}

void DeepSeek4SeqEngine::fail_prefill(
        std::vector<StepOutput> & outputs, const std::string & error) {
    if (!pending_) return;
    std::fprintf(stderr, "[deepseek4-parallel] reference prefill slot %d: %s\n",
                 pending_->slot, error.c_str());
    outputs.push_back({pending_->slot, -1, true, error, true});
    pending_.reset();
}

bool DeepSeek4SeqEngine::step(
        const std::vector<StepInput> & inputs,
        std::vector<StepOutput> & outputs) {
    outputs.clear();
    if ((int)inputs.size() != slots_.decoding_count()) return false;

    // Validate the complete decoder set before append_token mutates allocation
    // and fed-token history. Every active decoder must occur exactly once.
    std::unordered_set<int> supplied;
    for (const StepInput & in : inputs) {
        if (!slots_.is_active(in.slot) || slots_.is_prefilling(in.slot) ||
            !supplied.insert(in.slot).second) return false;
    }
    for (int slot = 0; slot < slots_.slot_count(); ++slot)
        if (slots_.is_active(slot) && !slots_.is_prefilling(slot) &&
            !supplied.count(slot)) return false;
    if (inputs.empty() && !pending_) return true;

    std::vector<int32_t> lane_tokens;
    std::vector<int64_t> lane_positions;
    std::vector<int32_t> lane_slots;
    std::vector<int> output_lanes;
    outputs.reserve(inputs.size() + 1);
    for (const StepInput & in : inputs) {
        StepOutput out; out.slot = in.slot;
        const auto app = slots_.append_token(in.slot, in.token);
        if (!app.ok) {
            out.failed = true;
            out.error = app.busy ? "paged KV pool exhausted during decode; raise --kv-pool-tokens"
                                 : "decode K/V append failed";
            outputs.push_back(std::move(out)); output_lanes.push_back(-1);
            continue;
        }
        if (app.new_block >= 0 &&
            !set_block(in.slot, app.new_block_index, app.new_block)) {
            out.failed = true; out.error = "decode block-table update failed";
            outputs.push_back(std::move(out)); output_lanes.push_back(-1);
            continue;
        }
        output_lanes.push_back((int)lane_tokens.size());
        lane_tokens.push_back(in.token);
        lane_positions.push_back(app.position);
        lane_slots.push_back(in.slot);
        outputs.push_back(std::move(out));
    }

    bool prefill_commit = false;
    int prefill_lane = -1;
    if (pending_) {
        auto chunk = slots_.append_prefill(pending_->slot, 1);
        if (chunk.busy) {
            // Admission does not reserve the prompt's future blocks. A live
            // decoder may temporarily claim the last free block; retry after
            // another sequence retires rather than failing the admission.
        } else if (!chunk.ok || chunk.rows.size() != 1) {
            fail_prefill(outputs, "reference prefill K/V append failed");
        } else if (!chunk.new_blocks.empty() &&
                   !set_block(pending_->slot, chunk.first_new_block,
                              chunk.new_blocks.front())) {
            fail_prefill(outputs, "reference prefill block-table update failed");
        } else {
            prefill_lane = (int)lane_tokens.size();
            lane_tokens.push_back(pending_->prompt[pending_->progress]);
            lane_positions.push_back((int64_t)pending_->progress);
            lane_slots.push_back(pending_->slot);
            prefill_commit = pending_->progress + 1 == pending_->prompt.size();
        }
    }
    if (lane_tokens.empty()) return true;

    std::vector<float> embeddings((size_t)b_.w_.n_embd * lane_tokens.size());
    if (!b_.w_.embedder.embed(lane_tokens.data(), (int)lane_tokens.size(),
                              embeddings.data())) {
        for (StepOutput & out : outputs) if (!out.failed) {
            out.failed = true; out.error = "token embedding failed";
        }
        if (prefill_lane >= 0) fail_prefill(outputs, "reference prefill embedding failed");
        return false;
    }
    std::vector<int32_t> compact_tables(lane_tokens.size() * stride_, -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane)
        std::copy_n(host_tables_.data() + (size_t)lane_slots[lane] * stride_,
                    stride_, compact_tables.data() + lane * stride_);
    // Keep the persistent metadata authoritative as well. The gathered
    // reference currently consumes the exact compact host tables above, but
    // future device-side addressing must observe the same step atomically.
    std::vector<int32_t> lengths((size_t)slots_.slot_count(), 0);
    std::vector<int32_t> active((size_t)slots_.slot_count(), -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane) {
        lengths[(size_t)lane_slots[lane]] = (int32_t)lane_positions[lane] + 1;
        active[lane] = lane_slots[lane];
    }
    ggml_backend_tensor_set(b_.paged_cache_.sequence_lengths, lengths.data(), 0,
                            lengths.size() * sizeof(int32_t));
    ggml_backend_tensor_set(b_.paged_cache_.active_slot_ids, active.data(), 0,
                            active.size() * sizeof(int32_t));
    std::vector<float> logits;
    std::vector<int32_t> argmax;
    if (!deepseek4_paged_gathered_step(
            b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
            embeddings.data(), lane_tokens.data(), lane_positions.data(),
            lane_slots.data(), (uint32_t)lane_tokens.size(), compact_tables.data(),
            stride_, logits, argmax, b_.moe_hybrid_.get(), b_.routing_stats_.get())) {
        for (StepOutput & out : outputs) if (!out.failed) {
            out.failed = true; out.error = "DeepSeek4 gathered paged graph failed";
        }
        if (prefill_lane >= 0) fail_prefill(outputs, "gathered paged graph failed");
        return false;
    }

    auto sample_lane = [&](int slot, int lane) {
        SeqSlot & seq = slots_.slot(slot);
        if (!seq.sampler.needs_logit_processing()) return argmax[(size_t)lane];
        return sample_logits(logits.data() + (size_t)lane * b_.w_.n_vocab,
                             b_.w_.n_vocab, seq.sampler,
                             seq.sample_history, seq.rng);
    };
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (output_lanes[i] < 0) continue;
        slots_.commit_step(inputs[i].slot);
        outputs[i].token = sample_lane(inputs[i].slot, output_lanes[i]);
    }
    if (prefill_lane >= 0) {
        ++pending_->progress;
        if (prefill_commit) {
            StepOutput out; out.slot = pending_->slot; out.prefill_done = true;
            SeqSlot & seq = slots_.slot(out.slot);
            if (seq.sampler.needs_logit_processing()) {
                // Match single-request DS4: first-token penalties see the
                // prompt, then append_token records each token actually fed.
                seq.sample_history = pending_->prompt;
            }
            out.token = sample_lane(out.slot, prefill_lane);
            slots_.commit_prefill(out.slot, (int)pending_->prompt.size());
            outputs.push_back(std::move(out)); pending_.reset();
        }
    }
    return true;
}

void DeepSeek4SeqEngine::retire(int slot) {
    if (pending_ && pending_->slot == slot) pending_.reset();
    slots_.retire(slot);
    if (slot >= 0 && slot < slots_.slot_count())
        std::fill_n(host_tables_.data() + (size_t)slot * stride_, stride_, -1);
}

} // namespace dflash::common
