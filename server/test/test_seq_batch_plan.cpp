#include "common/seq_batch_plan.h"
#include "common/seq_step_validation.h"
#include "host_check.h"

#include <cstdio>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    const std::vector<PrefillCandidate> pending{
        {7, 30}, {2, 10}, {5, 20}, {1, 40},
    };

    StepPlanPolicy policy;
    StepPlanLimits idle_limits{/*max sequences=*/2,
                               /*per sequence=*/2048,
                               /*total=*/4096};
    StepPlanLimits mixed_limits{/*max sequences=*/2,
                                /*per sequence=*/512,
                                /*total=*/1024};

    auto idle = plan_prefill_slices(
        pending, false, idle_limits, policy);
    CHECK(idle.size() == 2);
    CHECK(idle[0].slot == 2);
    CHECK(idle[1].slot == 5);
    CHECK(idle[0].max_tokens == 2048);
    CHECK(idle[1].max_tokens == 2048);

    auto mixed = plan_prefill_slices(
        pending, true, mixed_limits, policy);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].max_tokens == 512);
    CHECK(mixed[1].max_tokens == 512);

    mixed_limits.max_prefill_tokens_per_sequence = 256;
    mixed_limits.max_prefill_tokens_total = 512;
    mixed = plan_prefill_slices(pending, true, mixed_limits, policy);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].max_tokens == 256);
    CHECK(mixed[1].max_tokens == 256);

    policy.mixed_prefill_token_budget = 300;
    mixed = plan_prefill_slices(pending, true, mixed_limits, policy);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].slot == 2);
    CHECK(mixed[0].max_tokens == 256);
    CHECK(mixed[1].slot == 5);
    CHECK(mixed[1].max_tokens == 44);

    auto rotated = plan_prefill_slices(
        pending, true, mixed_limits, policy, /*round_robin_start=*/1);
    CHECK(rotated.size() == 2);
    CHECK(rotated[0].slot == 2 && rotated[0].max_tokens == 44);
    CHECK(rotated[1].slot == 5 && rotated[1].max_tokens == 256);

    // Engine hard totals clamp custom policy and the rotating cursor prevents
    // a sub-quantum budget from choosing the oldest lane forever.
    mixed_limits.max_prefill_tokens_total = 200;
    auto clamped0 = plan_prefill_slices(
        pending, true, mixed_limits, policy, 0);
    auto clamped1 = plan_prefill_slices(
        pending, true, mixed_limits, policy, 1);
    CHECK(clamped0.size() == 1 && clamped0[0].slot == 2 &&
          clamped0[0].max_tokens == 200);
    CHECK(clamped1.size() == 1 && clamped1[0].slot == 5 &&
          clamped1[0].max_tokens == 200);

    mixed_limits.max_prefill_tokens_per_sequence = 0;
    CHECK(plan_prefill_slices(
        pending, true, mixed_limits, policy).empty());

    policy.prefill_quantum = 0;
    CHECK(plan_prefill_slices(
        pending, false, idle_limits, policy).empty());
    policy.prefill_quantum = 512;
    idle_limits.max_prefill_sequences = 0;
    CHECK(plan_prefill_slices(
        pending, false, idle_limits, policy).empty());
    CHECK(plan_prefill_slices({}, false, idle_limits, policy).empty());

    // The same model-neutral layer validates engine row ownership before the
    // scheduler mutates socket/request state.
    SeqEngine::StepPlan work;
    work.decode = {{0, 7}};
    work.prefills = {{1, 4}};

    SeqEngine::StepResult good;
    good.status = SeqEngine::StepResult::Status::progressed;
    good.outputs.push_back({0, 11, false, {}, false});
    good.prefill_progress.push_back({1, 4});
    CHECK(validate_step_result(work, good, 2).empty());

    SeqEngine::StepResult complete = good;
    complete.outputs.push_back({1, 12, false, {}, true});
    CHECK(validate_step_result(work, complete, 2).empty());

    SeqEngine::StepResult missing_decode = good;
    missing_decode.outputs.clear();
    CHECK(!validate_step_result(work, missing_decode, 2).empty());

    SeqEngine::StepResult duplicate_decode = good;
    duplicate_decode.outputs.push_back({0, 12, false, {}, false});
    CHECK(!validate_step_result(work, duplicate_decode, 2).empty());

    SeqEngine::StepResult missing_prefill = good;
    missing_prefill.prefill_progress.clear();
    CHECK(!validate_step_result(work, missing_prefill, 2).empty());

    SeqEngine::StepResult over_prefill = good;
    over_prefill.prefill_progress[0].tokens = 5;
    CHECK(!validate_step_result(work, over_prefill, 2).empty());

    // A selected prefill may terminate with a per-request error instead of a
    // progress record; the scheduler will retire that slot.
    SeqEngine::StepResult prefill_failure = good;
    prefill_failure.prefill_progress.clear();
    prefill_failure.outputs.push_back(
        {1, -1, true, "prefill failed", true});
    CHECK(validate_step_result(work, prefill_failure, 2).empty());

    SeqEngine::StepResult bad_row_failure = prefill_failure;
    bad_row_failure.outputs.back().error.clear();
    CHECK(!validate_step_result(work, bad_row_failure, 2).empty());

    SeqEngine::StepResult failed;
    failed.status = SeqEngine::StepResult::Status::failed;
    failed.error = "device compute failed";
    CHECK(validate_step_result(work, failed, 2).empty());
    failed.outputs.push_back({0, -1, true, "partial", false});
    CHECK(!validate_step_result(work, failed, 2).empty());

    SeqEngine::StepResult blocked;
    blocked.status = SeqEngine::StepResult::Status::resource_blocked;
    CHECK(validate_step_result(work, blocked, 2).empty());
    blocked.prefill_progress.push_back({1, 1});
    CHECK(!validate_step_result(work, blocked, 2).empty());

    SeqEngine::StepResult idle_result;
    idle_result.status = SeqEngine::StepResult::Status::idle;
    CHECK(validate_step_result({}, idle_result, 2).empty());
    CHECK(!validate_step_result(work, idle_result, 2).empty());

    SeqEngine::StepPlan duplicate_plan = work;
    duplicate_plan.decode.push_back({0, 8});
    CHECK(!validate_step_result(duplicate_plan, good, 2).empty());

    std::printf("test_seq_batch_plan: %d checks passed\n", g_checks);
    return 0;
}
