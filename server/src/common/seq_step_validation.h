// Model-neutral validation of the SeqEngine step protocol.
//
// Engines own state and graph lowering, but the common scheduler must never
// trust malformed row ownership: re-feeding a decode token after an omitted
// output silently corrupts that sequence. Keep the validation GPU-free so
// every backend gets the same fail-stop boundary and host tests.

#pragma once

#include "seq_engine.h"

#include <string>
#include <vector>

namespace dflash::common {

inline std::string validate_step_result(
        const SeqEngine::StepPlan & plan,
        const SeqEngine::StepResult & result,
        int slot_count) {
    using Status = SeqEngine::StepResult::Status;
    const bool payload_empty =
        result.outputs.empty() && result.prefill_progress.empty();

    if (slot_count < 1) return "slot count must be positive";
    if (result.status == Status::failed) {
        if (result.error.empty()) return "failed result has no diagnostic";
        if (!payload_empty) return "failed result exposes partial payload";
        return {};
    }
    if (result.status == Status::resource_blocked) {
        if (!payload_empty) {
            return "resource_blocked result exposes partial progress";
        }
        return {};
    }
    if (result.status == Status::idle) {
        if (!plan.decode.empty() || !plan.prefills.empty()) {
            return "idle result returned for a nonempty plan";
        }
        if (!payload_empty || !result.error.empty()) {
            return "idle result carries payload or an error";
        }
        return {};
    }
    if (!result.error.empty()) {
        return "progressed result carries a whole-step error";
    }

    std::vector<uint8_t> decode_planned((size_t)slot_count, 0);
    std::vector<int> prefill_limit((size_t)slot_count, 0);
    for (const SeqEngine::StepInput & input : plan.decode) {
        if (input.slot < 0 || input.slot >= slot_count || input.token < 0) {
            return "decode plan contains an invalid row";
        }
        if (decode_planned[(size_t)input.slot]) {
            return "decode plan contains a duplicate slot";
        }
        decode_planned[(size_t)input.slot] = 1;
    }
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= slot_count ||
            slice.max_tokens <= 0) {
            return "prefill plan contains an invalid slice";
        }
        if (decode_planned[(size_t)slice.slot] ||
            prefill_limit[(size_t)slice.slot] != 0) {
            return "step plan assigns a slot more than once";
        }
        prefill_limit[(size_t)slice.slot] = slice.max_tokens;
    }

    std::vector<uint8_t> output_seen((size_t)slot_count, 0);
    std::vector<uint8_t> prefill_failed((size_t)slot_count, 0);
    std::vector<uint8_t> prefill_completed((size_t)slot_count, 0);
    for (const SeqEngine::StepOutput & output : result.outputs) {
        if (output.slot < 0 || output.slot >= slot_count) {
            return "output names an unknown slot";
        }
        if (output_seen[(size_t)output.slot]) {
            return "step returned duplicate outputs for one slot";
        }
        if (output.failed && output.error.empty()) {
            return "failed row has no diagnostic";
        }
        if (!output.failed && output.token < 0) {
            return "successful row has no sampled token";
        }

        if (output.prefill_done) {
            if (prefill_limit[(size_t)output.slot] == 0) {
                return "prefill output names an unselected slot";
            }
            prefill_completed[(size_t)output.slot] = 1;
            prefill_failed[(size_t)output.slot] = output.failed ? 1 : 0;
        } else if (!decode_planned[(size_t)output.slot]) {
            return "decode output names an unplanned slot";
        }
        output_seen[(size_t)output.slot] = 1;
    }

    std::vector<uint8_t> progress_seen((size_t)slot_count, 0);
    for (const SeqEngine::PrefillProgress & progress :
         result.prefill_progress) {
        if (progress.slot < 0 || progress.slot >= slot_count ||
            prefill_limit[(size_t)progress.slot] == 0) {
            return "prefill progress names an unselected slot";
        }
        if (progress_seen[(size_t)progress.slot]) {
            return "step returned duplicate prefill progress";
        }
        if (progress.tokens <= 0 ||
            progress.tokens > prefill_limit[(size_t)progress.slot]) {
            return "prefill progress exceeds its selected slice";
        }
        progress_seen[(size_t)progress.slot] = 1;
    }

    for (const SeqEngine::StepInput & input : plan.decode) {
        if (!output_seen[(size_t)input.slot]) {
            return "step omitted an output for a decode slot";
        }
    }
    for (const PrefillSlice & slice : plan.prefills) {
        const size_t slot = (size_t)slice.slot;
        if (!progress_seen[slot] && !prefill_failed[slot]) {
            return "step omitted progress for a selected prefill";
        }
        if (prefill_completed[slot] && !prefill_failed[slot] &&
            !progress_seen[slot]) {
            return "successful prefill completion has no progress";
        }
    }
    if (payload_empty) return "progressed result contains no work";
    return {};
}

}  // namespace dflash::common
