// Unit tests for should_reject_oversized and fit_reply_to_context — pure, GPU-free.

#include "CppUnitTestFramework.hpp"
#include "server/admission.h"

namespace {
struct AdmissionFixture {};
}

TEST_CASE(AdmissionFixture, test_small_prompt_no_compression_accepts) {
    CHECK(!should_reject_oversized(100, 100, 1024, false));
}

TEST_CASE(AdmissionFixture, test_oversized_no_compression_rejects) {
    CHECK(should_reject_oversized(900, 200, 1024, false));
}

TEST_CASE(AdmissionFixture, test_reply_over_limit_with_compression_accepts) {
    CHECK(!should_reject_oversized(65000, 2048, 65536, true));
}

TEST_CASE(AdmissionFixture, test_exactly_at_limit_accepts) {
    CHECK(!should_reject_oversized(1024, 0, 1024, false));
    CHECK(!should_reject_oversized(512, 512, 1024, false));
}

TEST_CASE(AdmissionFixture, test_one_over_limit_no_compression_rejects) {
    CHECK(should_reject_oversized(1025, 0, 1024, false));
}

TEST_CASE(AdmissionFixture, test_one_over_limit_with_compression_rejects) {
    CHECK(should_reject_oversized(1025, 0, 1024, true));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_compressed_within_budget) {
    CHECK(!effective_prompt_overflows(5000, 0, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_full_cache_hit_uses_served_size) {
    CHECK(!effective_prompt_overflows(70000, 800, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_post_compress_genuinely_oversized) {
    CHECK(effective_prompt_overflows(60000, -1, 10000, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_verbatim_within_budget) {
    CHECK(!effective_prompt_overflows(1000, -1, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_zero_length_hit_is_a_hit) {
    CHECK(!effective_prompt_overflows(70000, 0, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_full_cache_hit_still_too_large) {
    CHECK(effective_prompt_overflows(200000, 60000, 10000, 65536));
}

TEST_CASE(AdmissionFixture, test_reply_fits_what_the_prompt_leaves) {
    // Claude Code asks for 32000 whatever the prompt.
    CHECK(fit_reply_to_context(110000, 32000, 131072) == 21072);
    CHECK(fit_reply_to_context(1000, 32000, 131072) == 32000);
    CHECK(fit_reply_to_context(131072, 32000, 131072) == 0);
    CHECK(fit_reply_to_context(140000, 32000, 131072) == 0);
    // Trimmed, the request is no longer oversized; a prompt that fills the
    // context still is.
    CHECK(!should_reject_oversized(110000, fit_reply_to_context(110000, 32000, 131072),
                                   131072, false));
    CHECK(should_reject_oversized(140000, fit_reply_to_context(140000, 32000, 131072),
                                  131072, false));
}

// With compression on, --max-ctx is the longest raw prompt PFlash takes (the drafter scores it whole): a prompt
// that fills the context passes even when its reply would not fit, since compression makes room; a longer one is
// refused.
TEST_CASE(AdmissionFixture, test_oversized_with_compression_respects_max_ctx) {
    CHECK(!should_reject_oversized(204800, 1, 204800, true));
    CHECK(!should_reject_oversized(200000, 32000, 204800, true));
    CHECK(should_reject_oversized(204801, 0, 204800, true));
    CHECK(should_reject_oversized(200000, 1, 139264, true));
    CHECK(should_reject_oversized(204800, 1, 204800, false));
    CHECK(!should_reject_oversized(1000, 1, 139264, false));
}
