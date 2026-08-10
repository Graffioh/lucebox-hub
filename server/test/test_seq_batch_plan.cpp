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

    StepPlanLimits idle_limits{/*max sequences=*/2,
                               /*per sequence=*/2048,
                               /*total=*/4096,
                               /*allocation quantum=*/512};
    StepPlanLimits mixed_limits{/*max sequences=*/2,
                                /*per sequence=*/512,
                                /*total=*/1024,
                                /*allocation quantum=*/512};

    auto idle = plan_prefill_slices(
        pending, idle_limits);
    CHECK(idle.size() == 2);
    CHECK(idle[0].slot == 2);
    CHECK(idle[1].slot == 5);
    CHECK(idle[0].max_tokens == 2048);
    CHECK(idle[1].max_tokens == 2048);

    auto mixed = plan_prefill_slices(
        pending, mixed_limits);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].max_tokens == 512);
    CHECK(mixed[1].max_tokens == 512);

    mixed_limits.max_prefill_tokens_per_sequence = 256;
    mixed_limits.max_prefill_tokens_total = 512;
    mixed = plan_prefill_slices(pending, mixed_limits);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].max_tokens == 256);
    CHECK(mixed[1].max_tokens == 256);

    mixed_limits.max_prefill_tokens_total = 301;
    mixed = plan_prefill_slices(pending, mixed_limits);
    CHECK(mixed.size() == 2);
    CHECK(mixed[0].slot == 2);
    CHECK(mixed[0].max_tokens == 256);
    CHECK(mixed[1].slot == 5);
    CHECK(mixed[1].max_tokens == 45);

    auto rotated = plan_prefill_slices(
        pending, mixed_limits, /*round_robin_start=*/1);
    CHECK(rotated.size() == 2);
    CHECK(rotated[0].slot == 2 && rotated[0].max_tokens == 45);
    CHECK(rotated[1].slot == 5 && rotated[1].max_tokens == 256);

    // A budget smaller than one quantum advances one lane; rotation prevents
    // the oldest lane from winning every step.
    mixed_limits.max_prefill_tokens_total = 200;
    auto clamped0 = plan_prefill_slices(
        pending, mixed_limits, 0);
    auto clamped1 = plan_prefill_slices(
        pending, mixed_limits, 1);
    CHECK(clamped0.size() == 1 && clamped0[0].slot == 2 &&
          clamped0[0].max_tokens == 200);
    CHECK(clamped1.size() == 1 && clamped1[0].slot == 5 &&
          clamped1[0].max_tokens == 200);

    // The packed Qwen policy fills all eight idle lanes with one 512-token
    // segment, while a mixed step rotates four such segments fairly.
    const std::vector<PrefillCandidate> packed{
        {0, 0}, {1, 1}, {2, 2}, {3, 3},
        {4, 4}, {5, 5}, {6, 6}, {7, 7},
    };
    StepPlanLimits packed_mixed{/*max sequences=*/8,
                                /*per sequence=*/512,
                                /*total=*/2048,
                                /*allocation quantum=*/512};
    auto packed0 = plan_prefill_slices(packed, packed_mixed, 0);
    CHECK(packed0.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(packed0[(size_t)i].slot == i);
        CHECK(packed0[(size_t)i].max_tokens == 512);
    }
    auto packed4 = plan_prefill_slices(packed, packed_mixed, 4);
    CHECK(packed4.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CHECK(packed4[(size_t)i].slot == i + 4);
        CHECK(packed4[(size_t)i].max_tokens == 512);
    }

    packed_mixed.prefill_allocation_quantum = 0;
    CHECK(plan_prefill_slices(packed, packed_mixed).empty());

    mixed_limits.max_prefill_tokens_per_sequence = 0;
    CHECK(plan_prefill_slices(
        pending, mixed_limits).empty());

    idle_limits.max_prefill_sequences = 0;
    CHECK(plan_prefill_slices(
        pending, idle_limits).empty());
    CHECK(plan_prefill_slices({}, idle_limits).empty());

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

    SeqEngine::StepResult failed_with_progress = prefill_failure;
    failed_with_progress.prefill_progress.push_back({1, 1});
    CHECK(!validate_step_result(work, failed_with_progress, 2).empty());

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

    SeqEngine::StepResult unknown_status = good;
    unknown_status.status = static_cast<SeqEngine::StepResult::Status>(-1);
    CHECK(!validate_step_result(work, unknown_status, 2).empty());

    SeqEngine::StepPlan duplicate_plan = work;
    duplicate_plan.decode.push_back({0, 8});
    CHECK(!validate_step_result(duplicate_plan, good, 2).empty());

    std::printf("test_seq_batch_plan: %d checks passed\n", g_checks);
    return 0;
}
