#include "CppUnitTestFramework.hpp"
#include "async_shadow_batch.h"

#include <cstdint>
#include <limits>
#include <vector>

using dflash::common::match_async_shadow_candidate;
using dflash::common::plan_async_shadow_batch;

namespace {
struct AsyncShadowBatchFixture {};
}

TEST_CASE(AsyncShadowBatchFixture, branches_at_lowest_confidence_eligible_depth) {
    constexpr int rows = 6;
    constexpr int top_k = 3;
    std::vector<float> lp = {
        -0.1f, -1.1f, -2.1f,
        -0.1f, -0.2f, -2.0f,
        -0.1f, -0.8f, -1.0f,
        -0.1f, -0.4f, -0.7f,
        -0.1f, -0.15f, -0.9f,
        -0.1f, -0.6f, -0.8f,
    };
    std::vector<int32_t> ids;
    for (int row = 0; row < rows; ++row) {
        ids.push_back(100 + row);
        ids.push_back(200 + row);
        ids.push_back(300 + row);
    }

    const auto plan = plan_async_shadow_batch(
        lp, ids, rows, top_k, /*branches=*/2,
        /*min_depth=*/2, /*max_depth=*/5, /*committed=*/1000);
    CHECK(plan.branch_depth == 4);
    CHECK(plan.confidence_margin > 0.049f);
    CHECK(plan.confidence_margin < 0.051f);
    REQUIRE(plan.candidates.size() == 2);
    CHECK(plan.candidates[0].endpoint_pos == 1004);
    CHECK(plan.candidates[0].pending_token == 204);
    CHECK(plan.candidates[0].draft_rank == 1);
    CHECK(plan.candidates[1].pending_token == 304);
    CHECK(plan.candidates[1].draft_rank == 2);
}

TEST_CASE(AsyncShadowBatchFixture, endpoint_match_requires_position_and_token) {
    std::vector<float> lp = {
        -0.1f, -1.0f, -2.0f,
        -0.1f, -0.2f, -0.3f,
        -0.1f, -0.4f, -0.5f,
    };
    std::vector<int32_t> ids = {
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
    };
    const auto plan = plan_async_shadow_batch(
        lp, ids, /*rows=*/3, /*top_k=*/3, /*branches=*/2,
        /*min_depth=*/1, /*max_depth=*/1, /*committed=*/50);
    REQUIRE(plan.candidates.size() == 2);
    CHECK(match_async_shadow_candidate(plan, 51, 21, true, 1) == 0);
    CHECK(match_async_shadow_candidate(plan, 51, 22, true, 1) == 1);
    CHECK(match_async_shadow_candidate(plan, 52, 21, true, 1) == -1);
    CHECK(match_async_shadow_candidate(plan, 51, 20, true, 1) == -1);
    CHECK(match_async_shadow_candidate(plan, 51, 21, false, 1) == -1);
    CHECK(match_async_shadow_candidate(plan, 51, 21, true, 2) == -1);
}

TEST_CASE(AsyncShadowBatchFixture, all_accepted_endpoint_requires_unavailable_next_row) {
    std::vector<float> lp = {
        -0.1f, -1.0f,
        -0.1f, -0.8f,
        -0.1f, -0.11f,
    };
    std::vector<int32_t> ids = {
        10, 11,
        20, 21,
        30, 31,
    };

    // Three input rows can all be accepted, but row 3 (the logits needed to
    // predict the following target bonus token) is not in this draft block.
    CHECK(plan_async_shadow_batch(
        lp, ids, /*rows=*/3, /*top_k=*/2, /*branches=*/1,
        /*min_depth=*/3, /*max_depth=*/3, /*committed=*/50).candidates.empty());
}

TEST_CASE(AsyncShadowBatchFixture, invalid_or_incomplete_plans_fail_closed) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> lp = {
        -0.1f, -1.0f, -2.0f,
        nan, -0.2f, -0.3f,
    };
    std::vector<int32_t> duplicate_ids = {
        10, 11, 12,
        20, 21, 21,
    };
    CHECK(plan_async_shadow_batch(
        lp, duplicate_ids, 2, 3, 2, 2, 2, 0).candidates.empty());
    CHECK(plan_async_shadow_batch(
        lp, duplicate_ids, 2, 3, 2, 2, 1, 0).candidates.empty());
    CHECK(plan_async_shadow_batch(
        lp, duplicate_ids, 2, 2, 2, 1, 1, 0).candidates.empty());
}
