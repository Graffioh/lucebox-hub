#include "CppUnitTestFramework.hpp"
#include "qwen35/qwen35_shadow_drafter.h"

using dflash::common::qwen35_shadow_outcome_matches;
using dflash::common::qwen35_shadow_accepted_tokens_match;

namespace {
struct AsyncShadowOutcomeFixture {};
}

TEST_CASE(AsyncShadowOutcomeFixture, exact_fast_rollback_outcome_matches) {
    CHECK(qwen35_shadow_outcome_matches(
        /*source_committed=*/100,
        /*predicted_depth=*/7,
        /*predicted_endpoint=*/107,
        /*predicted_pending=*/42,
        /*actual_endpoint=*/107,
        /*actual_pending=*/42,
        /*used_fast_rollback=*/true,
        /*actual_commit_count=*/7));
}

TEST_CASE(AsyncShadowOutcomeFixture, mismatched_key_fails_closed) {
    CHECK(!qwen35_shadow_outcome_matches(100, 7, 107, 42,
                                         108, 42, true, 7));
    CHECK(!qwen35_shadow_outcome_matches(100, 7, 107, 42,
                                         107, 43, true, 7));
    CHECK(!qwen35_shadow_outcome_matches(100, 7, 107, 42,
                                         107, 42, true, 6));
    CHECK(!qwen35_shadow_outcome_matches(100, 7, 108, 42,
                                         108, 42, true, 7));
}

TEST_CASE(AsyncShadowOutcomeFixture, replay_and_invalid_predictions_never_match) {
    CHECK(!qwen35_shadow_outcome_matches(100, 7, 107, 42,
                                         107, 42, false, 7));
    CHECK(!qwen35_shadow_outcome_matches(-1, 7, 6, 42,
                                         6, 42, true, 7));
    CHECK(!qwen35_shadow_outcome_matches(100, 0, 100, 42,
                                         100, 42, true, 0));
}

TEST_CASE(AsyncShadowOutcomeFixture, accepted_token_content_must_match) {
    const std::vector<int32_t> proposal = {10, 11, 12, 13};
    CHECK(qwen35_shadow_accepted_tokens_match(proposal, 3, {10, 11, 12}));
    CHECK(!qwen35_shadow_accepted_tokens_match(proposal, 3, {10, 99, 12}));
    CHECK(!qwen35_shadow_accepted_tokens_match(proposal, 3, {10, 11}));
    CHECK(!qwen35_shadow_accepted_tokens_match(proposal, 0, {}));
}
