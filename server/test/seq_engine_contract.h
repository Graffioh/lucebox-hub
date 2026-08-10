// Host-side executable contract for common/seq_engine.h.
//
// The checker deliberately knows only logical slots, decode inputs, and
// scheduler-selected prefill slices. It exercises the largest useful cohort
// up to two sequences allowed by the advertised limits, mixed
// decode/prefill, validation failures, retryable blocking, and slot reuse
// without importing a model graph or cache representation.

#pragma once

#include "common/seq_engine.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

inline std::vector<std::string> check_seq_engine_contract(SeqEngine & engine) {
    using Status = SeqEngine::StepResult::Status;

    std::vector<std::string> violations;
    auto require = [&violations](bool ok, const char * message) {
        if (!ok) violations.emplace_back(message);
    };

    const int n_slots = engine.slot_count();
    require(n_slots >= 2,
            "slot_count() must be at least 2 for the concurrency contract");
    if (n_slots < 2) return violations;

    const StepPlanLimits idle_limits = engine.step_plan_limits(0);
    const StepPlanLimits mixed_limits = engine.step_plan_limits(1);
    const auto supports_one_prefill = [](const StepPlanLimits & limits) {
        return limits.max_prefill_sequences >= 1 &&
               limits.max_prefill_tokens_per_sequence >= 1 &&
               limits.max_prefill_tokens_total >= 1;
    };
    require(supports_one_prefill(idle_limits),
            "idle step_plan_limits() must permit one prefill token");
    require(supports_one_prefill(mixed_limits),
            "mixed step_plan_limits() must permit one prefill token");
    require(engine.max_context() >= 3,
            "max_context() must fit the contract-check prompts");
    if (!supports_one_prefill(idle_limits) ||
        !supports_one_prefill(mixed_limits) || engine.max_context() < 3) {
        return violations;
    }

    // Exercise at most two concurrent prefills, but never infer K=2 support
    // from slot_count(): sequence count and the total-token cap are separate
    // engine capabilities.
    const int idle_cohort_size = std::min({
        2,
        n_slots,
        idle_limits.max_prefill_sequences,
        idle_limits.max_prefill_tokens_total,
    });

    const SamplerCfg greedy{};
    SamplerCfg seeded{};
    seeded.temp = 0.7f;
    seeded.seed = 20260809;

    std::vector<bool> active((size_t)n_slots, false);
    std::vector<bool> decoding((size_t)n_slots, false);
    std::vector<int> remaining((size_t)n_slots, 0);
    std::vector<int32_t> next_token((size_t)n_slots, -1);

    auto retire_all = [&]() {
        for (int slot = 0; slot < n_slots; ++slot) {
            if (active[(size_t)slot]) engine.retire(slot);
            active[(size_t)slot] = false;
            decoding[(size_t)slot] = false;
            remaining[(size_t)slot] = 0;
        }
    };

    auto record_admit = [&](uint64_t request_id,
                            const std::vector<int32_t> & prompt,
                            const SamplerCfg & sampler) {
        const SeqEngine::AdmitResult result =
            engine.admit(request_id, prompt, sampler);
        require(result.ok, "admit() must succeed while a slot is free");
        if (!result.ok) return -1;
        require(result.slot >= 0 && result.slot < n_slots,
                "admit() returned an unknown slot");
        if (result.slot < 0 || result.slot >= n_slots) return -1;
        require(!active[(size_t)result.slot],
                "admit() reused a live slot");
        if (active[(size_t)result.slot]) return -1;
        active[(size_t)result.slot] = true;
        decoding[(size_t)result.slot] = false;
        remaining[(size_t)result.slot] = (int)prompt.size();
        return result.slot;
    };

    auto validate_status_helpers = [&](const SeqEngine::StepResult & result) {
        require(result.ok() == (result.status != Status::failed),
                "StepResult::ok() disagrees with status");
        require(result.made_progress() ==
                    (result.status == Status::progressed),
                "StepResult::made_progress() disagrees with status");
    };

    // A blocked step is atomic and may be retried unchanged. A backend must
    // not report retryable no-progress forever: that would deadlock the only
    // worker able to retire sequences and release capacity.
    auto run_bounded = [&](const SeqEngine::StepPlan & plan,
                           SeqEngine::StepResult & result) {
        constexpr int kMaxBlockedRetries = 3;
        for (int attempt = 0; attempt < kMaxBlockedRetries; ++attempt) {
            result = engine.step(plan);
            validate_status_helpers(result);
            if (result.status != Status::resource_blocked) return true;
            require(result.outputs.empty() && result.prefill_progress.empty(),
                    "resource_blocked must not report partial progress");
        }
        require(false,
                "resource_blocked no-progress must be bounded");
        return false;
    };

    auto slice_limit = [&](const SeqEngine::StepPlan & plan) {
        return engine.step_plan_limits((int)plan.decode.size())
            .max_prefill_tokens_per_sequence;
    };

    // Validate and apply one successful logical step to the checker's mirror.
    auto apply_progress = [&](const SeqEngine::StepPlan & plan,
                              const SeqEngine::StepResult & result) {
        require(result.status == Status::progressed,
                "valid planned work must return progressed");
        require(result.ok() && result.made_progress(),
                "progressed result helpers must report progress");
        require(result.error.empty(),
                "a successful step must not carry a whole-step error");
        if (result.status != Status::progressed) return false;

        std::vector<bool> decode_answered((size_t)n_slots, false);
        std::vector<bool> prefill_answered((size_t)n_slots, false);
        std::vector<bool> completed((size_t)n_slots, false);

        for (const SeqEngine::StepOutput & output : result.outputs) {
            require(output.slot >= 0 && output.slot < n_slots,
                    "step() returned an output for an unknown slot");
            if (output.slot < 0 || output.slot >= n_slots) continue;
            require(!output.failed,
                    "contract-check work must not fail a request");
            if (!output.failed) {
                require(output.token >= 0,
                        "successful output must carry a token");
            }

            if (output.prefill_done) {
                const bool selected = std::any_of(
                    plan.prefills.begin(), plan.prefills.end(),
                    [&](const PrefillSlice & slice) {
                        return slice.slot == output.slot;
                    });
                require(selected,
                        "prefill_done must name a selected prefill slot");
                require(!completed[(size_t)output.slot],
                        "prefill completion was reported twice");
                completed[(size_t)output.slot] = true;
                next_token[(size_t)output.slot] = output.token;
                continue;
            }

            const bool planned = std::any_of(
                plan.decode.begin(), plan.decode.end(),
                [&](const SeqEngine::StepInput & input) {
                    return input.slot == output.slot;
                });
            require(planned,
                    "step() returned an output for an unplanned decode slot");
            require(!decode_answered[(size_t)output.slot],
                    "step() returned two decode outputs for one slot");
            decode_answered[(size_t)output.slot] = true;
            next_token[(size_t)output.slot] = output.token;
        }

        for (const SeqEngine::PrefillProgress & progress :
             result.prefill_progress) {
            require(progress.slot >= 0 && progress.slot < n_slots,
                    "prefill progress named an unknown slot");
            if (progress.slot < 0 || progress.slot >= n_slots) continue;
            auto selected = std::find_if(
                plan.prefills.begin(), plan.prefills.end(),
                [&](const PrefillSlice & slice) {
                    return slice.slot == progress.slot;
                });
            require(selected != plan.prefills.end(),
                    "prefill progress named an unselected slot");
            if (selected == plan.prefills.end()) continue;
            require(!prefill_answered[(size_t)progress.slot],
                    "step() returned two progress records for one prefill");
            prefill_answered[(size_t)progress.slot] = true;
            require(progress.tokens > 0 &&
                        progress.tokens <= selected->max_tokens,
                    "prefill progress exceeded its selected slice");
            require(progress.tokens <= remaining[(size_t)progress.slot],
                    "prefill progress exceeded the prompt remainder");
            remaining[(size_t)progress.slot] -= progress.tokens;
        }

        for (const SeqEngine::StepInput & input : plan.decode) {
            if (input.slot >= 0 && input.slot < n_slots) {
                require(decode_answered[(size_t)input.slot],
                        "step() left a decoding slot without an output");
            }
        }
        for (const PrefillSlice & slice : plan.prefills) {
            require(slice.max_tokens > 0 &&
                        slice.max_tokens <= slice_limit(plan),
                    "StepPlan prefill slice exceeded engine limits");
            if (slice.slot < 0 || slice.slot >= n_slots) continue;
            require(prefill_answered[(size_t)slice.slot],
                    "step() left a selected prefill without progress");
            const bool done = remaining[(size_t)slice.slot] == 0;
            require(completed[(size_t)slice.slot] == done,
                    "prefill_done disagrees with reported token progress");
            if (done) decoding[(size_t)slice.slot] = true;
        }
        return true;
    };

    auto execute = [&](const SeqEngine::StepPlan & plan) {
        SeqEngine::StepResult result;
        if (!run_bounded(plan, result)) return false;
        return apply_progress(plan, result);
    };

    auto decode_inputs = [&]() {
        std::vector<SeqEngine::StepInput> inputs;
        for (int slot = 0; slot < n_slots; ++slot) {
            if (active[(size_t)slot] && decoding[(size_t)slot]) {
                inputs.push_back({slot, next_token[(size_t)slot]});
            }
        }
        return inputs;
    };

    // Admit the long prompt in the same idle cohort only when the engine can
    // actually execute two slices. K=1 engines admit it after the short
    // prompt releases its staging resource, then exercise the same mixed path.
    const int short_slot = record_admit(1, {11}, greedy);
    int long_slot = -1;
    if (idle_cohort_size >= 2) {
        long_slot = record_admit(2, {21, 22, 23}, seeded);
    }
    if (short_slot < 0 || (idle_cohort_size >= 2 && long_slot < 0)) {
        retire_all();
        return violations;
    }
    if (long_slot >= 0) {
        require(short_slot != long_slot,
                "two pending admissions must own distinct slots");
    }
    require(engine.prefill_pending(),
            "successful admissions must leave prefill work pending");

    SeqEngine::StepPlan first_plan;
    first_plan.prefills.push_back({short_slot, 1});
    if (long_slot >= 0) {
        const int long_grant = std::min({
            2,
            idle_limits.max_prefill_tokens_per_sequence,
            idle_limits.max_prefill_tokens_total - 1,
        });
        first_plan.prefills.push_back({long_slot, long_grant});
    }
    if (!execute(first_plan)) {
        retire_all();
        return violations;
    }
    require(decoding[(size_t)short_slot],
            "the short prefill must complete in its selected slice");

    if (long_slot < 0) {
        long_slot = record_admit(2, {21, 22, 23}, seeded);
        if (long_slot < 0) {
            retire_all();
            return violations;
        }
    }
    require(!decoding[(size_t)long_slot] &&
                remaining[(size_t)long_slot] > 0,
            "long member must remain pending after the short member completes");
    require(engine.prefill_pending(),
            "prefill_pending() must report the unfinished long prompt");

    for (int iteration = 0;
         remaining[(size_t)long_slot] > 0 && iteration < 8;
         ++iteration) {
        SeqEngine::StepPlan mixed;
        mixed.decode = decode_inputs();
        const StepPlanLimits limits =
            engine.step_plan_limits((int)mixed.decode.size());
        require(supports_one_prefill(limits),
                "mixed step_plan_limits() must permit continued prefill");
        if (!supports_one_prefill(limits)) break;
        mixed.prefills.push_back({
            long_slot,
            std::min({
                remaining[(size_t)long_slot],
                limits.max_prefill_tokens_per_sequence,
                limits.max_prefill_tokens_total,
            }),
        });
        if (!execute(mixed)) break;
    }
    require(remaining[(size_t)long_slot] == 0,
            "mixed prefill did not complete within bounded progress steps");
    require(!engine.prefill_pending(),
            "prefill_pending() stayed true after the cohort completed");

    // Full decode coverage, including the token handoff back into the engine.
    SeqEngine::StepPlan decode_plan;
    decode_plan.decode = decode_inputs();
    require(decode_plan.decode.size() == 2,
            "both completed admissions must enter decode");
    execute(decode_plan);

    // A full engine is retryable admission pressure, not a request error.
    if (n_slots == 2) {
        const SeqEngine::AdmitResult full = engine.admit(3, {31}, greedy);
        require(!full.ok && full.busy && full.slot < 0,
                "a full engine must report busy=true without claiming a slot");
    }

    // Omitting a decoder or attaching prefill work to a decoding slot is a
    // terminal plan-validation failure and must not partially advance state.
    auto require_failed = [&](const SeqEngine::StepPlan & invalid,
                              const char * message) {
        const SeqEngine::StepResult result = engine.step(invalid);
        validate_status_helpers(result);
        require(result.status == Status::failed, message);
        require(!result.ok() && !result.made_progress(),
                "failed result helpers must reject the step");
        require(!result.error.empty(),
                "failed step must explain the validation error");
        require(result.outputs.empty() && result.prefill_progress.empty(),
                "failed step must not report partial progress");
    };

    SeqEngine::StepPlan omitted;
    omitted.decode.push_back(decode_plan.decode.front());
    require_failed(omitted,
                   "step() must reject a plan that omits a decoding slot");

    SeqEngine::StepPlan invalid_prefill = decode_plan;
    invalid_prefill.prefills.push_back({short_slot, 1});
    require_failed(invalid_prefill,
                   "step() must reject prefill work for a decoding slot");

    // Start fresh and complete the advertised idle cohort in one step. When
    // K=2 is available, this still catches engines that accidentally retain a
    // scalar completion/output path.
    retire_all();
    const int simultaneous_a = record_admit(10, {41}, greedy);
    int simultaneous_b = -1;
    if (idle_cohort_size >= 2) {
        simultaneous_b = record_admit(11, {51}, seeded);
    }
    if (simultaneous_a >= 0 &&
        (idle_cohort_size < 2 || simultaneous_b >= 0)) {
        SeqEngine::StepPlan simultaneous;
        simultaneous.prefills.push_back({simultaneous_a, 1});
        if (simultaneous_b >= 0) {
            simultaneous.prefills.push_back({simultaneous_b, 1});
        }
        execute(simultaneous);
        require(decoding[(size_t)simultaneous_a],
                "selected prefill must report completion");
        if (simultaneous_b >= 0) {
            require(decoding[(size_t)simultaneous_b],
                    "one K=2 step must report simultaneous completions");
        }
    }

    // Retire is idempotent, frees capacity, and cancels unfinished prefill.
    const int freed = simultaneous_a;
    if (freed >= 0) {
        engine.retire(freed);
        engine.retire(freed);
        active[(size_t)freed] = false;
        decoding[(size_t)freed] = false;
        const int replacement = record_admit(12, {61, 62}, greedy);
        require(replacement >= 0,
                "admit() must reuse capacity after retire()");
        if (replacement >= 0) {
            engine.retire(replacement);
            engine.retire(replacement);
            active[(size_t)replacement] = false;
            decoding[(size_t)replacement] = false;
            remaining[(size_t)replacement] = 0;
            require(!engine.prefill_pending(),
                    "retiring the only pending prefill must clear it");
        }
    }

    retire_all();
    const SeqEngine::StepResult idle = engine.step({});
    validate_status_helpers(idle);
    require(idle.status == Status::idle && idle.ok() &&
                !idle.made_progress(),
            "step() with no work must return idle");
    require(idle.outputs.empty() && idle.prefill_progress.empty() &&
                idle.error.empty(),
            "idle result must not retain work or errors");

    const int reused = record_admit(13, {71}, greedy);
    require(reused >= 0,
            "an engine must admit again after every slot retires");
    retire_all();
    require(!engine.prefill_pending(),
            "fully retired engine must have no pending prefill");
    return violations;
}

}  // namespace dflash::common
