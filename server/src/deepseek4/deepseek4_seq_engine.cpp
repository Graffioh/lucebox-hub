#include "deepseek4_seq_engine.h"

#include "deepseek4_backend.h"
#include "common/sampler.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

DeepSeek4SeqEngine::DeepSeek4SeqEngine(
        DeepSeek4Backend & backend, PagedKvPool & pool, int max_ctx,
        uint32_t table_stride)
    : b_(backend), slots_(pool, max_ctx), stride_(table_stride),
      host_tables_((size_t)pool.max_sequences() * table_stride, -1) {}

bool DeepSeek4SeqEngine::token_is_eos(int32_t token) const {
    return deepseek4_is_eos_tok(token, b_.w_);
}

StepPlanLimits DeepSeek4SeqEngine::step_plan_limits(
        int decode_rows) const {
    // The gathered graph accepts at most sixteen independent lanes and does
    // not permit two rows from the same sequence. A prompt therefore advances
    // by one token while every live decoder still shares the same weight pass.
    const int available = std::max(0, 16 - decode_rows);
    return {available, 1, available, 1};
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    using AdmitStatus = AdmitResult::Status;
    AdmitResult result = slots_.admit(
        request_id, prompt, sampler);
    if (result.status != AdmitStatus::admitted) return result;
    if (result.slot < 0 || result.slot >= slots_.slot_count()) {
        result.status = AdmitStatus::failed;
        result.error = "invalid DeepSeek4 serving slot";
        return result;
    }
    std::fill_n(host_tables_.data() + (size_t)result.slot * stride_,
                stride_, -1);
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)result.slot);
    return result;
}

bool DeepSeek4SeqEngine::set_block(int slot, int logical, int32_t physical) {
    if (slot < 0 || slot >= slots_.slot_count() || logical < 0 ||
        (uint32_t)logical >= stride_ || !b_.paged_cache_.block_table) {
        return false;
    }
    host_tables_[(size_t)slot * stride_ + (size_t)logical] = physical;
    ggml_tensor * table = b_.paged_cache_.block_table;
    ggml_backend_tensor_set(
        table, &physical,
        (size_t)slot * table->nb[1] +
            (size_t)logical * sizeof(int32_t),
        sizeof(physical));
    return true;
}

void DeepSeek4SeqEngine::fail_prefill(
        int slot, std::vector<PrefillOutput> & outputs,
        const std::string & error) {
    std::fprintf(stderr, "[deepseek4-parallel] prefill slot %d: %s\n",
                 slot, error.c_str());
    PrefillOutput out;
    out.slot = slot;
    out.status = PrefillOutput::Status::failed;
    out.error = error;
    outputs.push_back(std::move(out));
}

SeqEngine::StepResult DeepSeek4SeqEngine::step(const StepPlan & plan) {
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
        return fail_step("decode plan does not cover every live DeepSeek4 slot");
    }
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & input : inputs) {
        if (input.slot < 0 || input.slot >= n_slots || input.token < 0 ||
            decode_seen[(size_t)input.slot] ||
            !slots_.is_active(input.slot) ||
            slots_.is_prefilling(input.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 decode row");
        }
        decode_seen[(size_t)input.slot] = 1;
    }

    const StepPlanLimits limits = step_plan_limits((int)inputs.size());
    if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
        return fail_step("DeepSeek4 step exceeds the sixteen-lane graph");
    }
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens != 1 || prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 prefill row");
        }
        prefill_seen[(size_t)slice.slot] = 1;
    }
    if (inputs.empty() && plan.prefills.empty()) return result;

    std::vector<int32_t> lane_tokens;
    std::vector<int64_t> lane_positions;
    std::vector<int32_t> lane_slots;
    std::vector<int> decode_lanes;
    lane_tokens.reserve(inputs.size() + plan.prefills.size());
    lane_positions.reserve(inputs.size() + plan.prefills.size());
    lane_slots.reserve(inputs.size() + plan.prefills.size());
    decode_lanes.reserve(inputs.size());
    result.decode.reserve(inputs.size());
    result.prefills.reserve(plan.prefills.size());

    for (const StepInput & input : inputs) {
        DecodeOutput out;
        out.slot = input.slot;
        const SeqSlotManager::StepAppend append =
            slots_.append_token(input.slot, input.token);
        if (!append.ok) {
            out.failed = true;
            out.error = append.busy
                ? "paged KV pool exhausted during DeepSeek4 decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "DeepSeek4 decode K/V append failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        if (append.new_block >= 0 &&
            !set_block(input.slot, append.new_block_index,
                       append.new_block)) {
            out.failed = true;
            out.error = "DeepSeek4 decode block-table update failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        decode_lanes.push_back((int)lane_tokens.size());
        lane_tokens.push_back(input.token);
        lane_positions.push_back(append.position);
        lane_slots.push_back(input.slot);
        result.decode.push_back(std::move(out));
    }

    struct PrefillLane {
        int slot = -1;
        int lane = -1;
        bool commit = false;
    };
    std::vector<PrefillLane> prefill_lanes;
    prefill_lanes.reserve(plan.prefills.size());
    for (const PrefillSlice & slice : plan.prefills) {
        SeqSlotManager::PrefillChunk chunk =
            slots_.append_prefill(slice.slot, 1);
        if (!chunk.ok || chunk.rows.size() != 1) {
            fail_prefill(slice.slot, result.prefills,
                         "DeepSeek4 prefill K/V append failed");
            continue;
        }
        if (!chunk.new_blocks.empty() &&
            !set_block(slice.slot, chunk.first_new_block,
                       chunk.new_blocks.front())) {
            fail_prefill(slice.slot, result.prefills,
                         "DeepSeek4 prefill block-table update failed");
            continue;
        }
        const SeqSlot & slot = slots_.slot(slice.slot);
        const bool commit = slot.cur_pos == (int)slot.prompt.size();
        prefill_lanes.push_back(
            {slice.slot, (int)lane_tokens.size(), commit});
        lane_tokens.push_back(slot.prompt[(size_t)slot.cur_pos - 1]);
        lane_positions.push_back(slot.cur_pos - 1);
        lane_slots.push_back(slice.slot);
    }

    if (lane_tokens.empty()) return result;
    if (lane_tokens.size() > 16) {
        return fail_step("DeepSeek4 gathered step exceeds sixteen lanes");
    }

    std::vector<float> embeddings(
        (size_t)b_.w_.n_embd * lane_tokens.size());
    if (!b_.w_.embedder.embed(lane_tokens.data(), (int)lane_tokens.size(),
                              embeddings.data())) {
        return fail_step("DeepSeek4 token embedding failed");
    }

    std::vector<int32_t> compact_tables(
        lane_tokens.size() * stride_, -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane) {
        std::copy_n(
            host_tables_.data() + (size_t)lane_slots[lane] * stride_,
            stride_, compact_tables.data() + lane * stride_);
    }

    std::vector<int32_t> lengths((size_t)n_slots, 0);
    std::vector<int32_t> active((size_t)n_slots, -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane) {
        lengths[(size_t)lane_slots[lane]] =
            (int32_t)lane_positions[lane] + 1;
        active[lane] = lane_slots[lane];
    }
    ggml_backend_tensor_set(
        b_.paged_cache_.sequence_lengths, lengths.data(), 0,
        lengths.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        b_.paged_cache_.active_slot_ids, active.data(), 0,
        active.size() * sizeof(int32_t));

    std::vector<float> logits;
    std::vector<int32_t> argmax;
    if (!deepseek4_paged_gathered_step(
            b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
            embeddings.data(), lane_tokens.data(), lane_positions.data(),
            lane_slots.data(), (uint32_t)lane_tokens.size(),
            compact_tables.data(), stride_, logits, argmax,
            b_.moe_hybrid_.get(), b_.routing_stats_.get())) {
        return fail_step("DeepSeek4 gathered paged graph failed");
    }

    auto sample_lane = [&](int slot_id, int lane) {
        SeqSlot & slot = slots_.slot(slot_id);
        if (!slot.sampler.needs_logit_processing()) {
            return argmax[(size_t)lane];
        }
        return sample_logits(
            logits.data() + (size_t)lane * b_.w_.n_vocab,
            b_.w_.n_vocab, slot.sampler, slot.sample_history, slot.rng);
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane < 0) continue;
        DecodeOutput & out = result.decode[i];
        slots_.commit_step(out.slot);
        out.token = sample_lane(out.slot, lane);
    }
    for (const PrefillLane & prefill : prefill_lanes) {
        PrefillOutput out;
        out.slot = prefill.slot;
        if (prefill.commit) {
            out.status = PrefillOutput::Status::completed;
            out.token = sample_lane(prefill.slot, prefill.lane);
            slots_.commit_prefill(prefill.slot);
        }
        result.prefills.push_back(std::move(out));
    }
    return result;
}

void DeepSeek4SeqEngine::retire(int slot) {
    if (!slots_.is_active(slot)) return;
    slots_.retire(slot);
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)slot);
    std::fill_n(host_tables_.data() + (size_t)slot * stride_, stride_, -1);
}

} // namespace dflash::common
