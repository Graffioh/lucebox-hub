// Unit tests for the skip-park policy — pure, GPU-free.
//
// Covers: the hardware crash guard (skip_park_allowed), the auto|on|off mode
// parse, the startup estimator (skip_park_required_bytes), and the precedence
// contract of resolve_skip_park — explicit off/on beat the estimate, the
// guard beats an explicit on, and auto follows the footprint vs. free VRAM.

#include "CppUnitTestFramework.hpp"
#include "placement/skip_park_guard.h"

namespace {
struct SkipParkGuardFixture {};
}

using namespace luce::common;

static constexpr size_t GiB = 1024ull * 1024 * 1024;

TEST_CASE(SkipParkGuardFixture, T1_not_requested_stays_off) {
    CHECK(!skip_park_allowed(false, 24 * GiB, 32768));
}

TEST_CASE(SkipParkGuardFixture, T2_big_card_any_ctx) {
    CHECK(skip_park_allowed(true, 32 * GiB, 131072));
}

TEST_CASE(SkipParkGuardFixture, T3_small_card_small_ctx_allowed) {
    CHECK(skip_park_allowed(true, 24 * GiB, 65536));
}

TEST_CASE(SkipParkGuardFixture, T4_small_card_big_ctx_downgraded) {
    CHECK(!skip_park_allowed(true, 24 * GiB, 131072));
}

TEST_CASE(SkipParkGuardFixture, T5_boundary_ctx_one_over) {
    CHECK(!skip_park_allowed(true, 24 * GiB, 65537));
}

TEST_CASE(SkipParkGuardFixture, T6_boundary_vram_just_under_32g) {
    CHECK(!skip_park_allowed(true, 32 * GiB - 1, 131072));
}

// ── Mode parse ────────────────────────────────────────────────────────────

TEST_CASE(SkipParkGuardFixture, T7_mode_parse) {
    SkipParkMode m = SkipParkMode::Auto;
    CHECK(parse_skip_park_mode("auto", m) && m == SkipParkMode::Auto);
    CHECK(parse_skip_park_mode("on", m) && m == SkipParkMode::On);
    CHECK(parse_skip_park_mode("off", m) && m == SkipParkMode::Off);
    CHECK(!parse_skip_park_mode("yes", m));
    CHECK(!parse_skip_park_mode("", m));
    CHECK(std::string(skip_park_mode_name(SkipParkMode::Auto)) == "auto");
    CHECK(std::string(skip_park_mode_name(SkipParkMode::On)) == "on");
    CHECK(std::string(skip_park_mode_name(SkipParkMode::Off)) == "off");
}

// ── Startup resolution ────────────────────────────────────────────────────

// Qwen3.5-0.8B-ish footprint: ~1.6 GiB weights, ~15 KB/token hybrid runtime.
static SkipParkDrafterInfo qwen35_like_info() {
    SkipParkDrafterInfo i;
    i.recognized = true;
    i.weights_bytes = int64_t(1668 * 1024 * 1024);
    i.runtime_bytes_per_token = 15000;
    i.fixed_bytes = 384ll * 1024 * 1024;
    i.context_length = 262144;
    return i;
}

TEST_CASE(SkipParkGuardFixture, T8_explicit_off_always_wins) {
    const auto d = resolve_skip_park(SkipParkMode::Off, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/64 * GiB, /*total=*/64 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T9_explicit_on_bypasses_estimate) {
    // Free VRAM far too small for the estimate — explicit on wins anyway.
    const auto d = resolve_skip_park(SkipParkMode::On, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/1, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T10_explicit_on_blocked_by_guard) {
    // 24 GiB card with ctx > 64K: the crash guard beats even force-on.
    const auto d = resolve_skip_park(SkipParkMode::On, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/16 * GiB, /*total=*/24 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
    // Same card at 64K ctx is under the guard — force-on honored.
    const auto ok = resolve_skip_park(SkipParkMode::On, /*drafter=*/true,
                                      qwen35_like_info(),
                                      /*free=*/16 * GiB, /*total=*/24 * GiB,
                                      /*max_ctx=*/65536);
    CHECK(ok.enabled);
}

TEST_CASE(SkipParkGuardFixture, T11_auto_fits_with_margin) {
    // required = (1.63 GiB + 0.375 GiB + 15000·131072) × 1.25 ≈ 4.9 GiB.
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/8 * GiB, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(d.enabled);
    CHECK(d.window_tokens == 131072);
    CHECK(d.required_bytes > 0);
}

TEST_CASE(SkipParkGuardFixture, T12_auto_exceeds_free) {
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/1 * GiB, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T13_auto_unknown_footprint_off) {
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     SkipParkDrafterInfo{},
                                     /*free=*/32 * GiB, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T14_no_drafter_off_in_all_modes) {
    CHECK(!resolve_skip_park(SkipParkMode::Auto, false, qwen35_like_info(),
                           32 * GiB, 32 * GiB, 131072).enabled);
    CHECK(!resolve_skip_park(SkipParkMode::On, false, qwen35_like_info(),
                           32 * GiB, 32 * GiB, 131072).enabled);
}

TEST_CASE(SkipParkGuardFixture, T15_window_capped_by_drafter_ctx) {
    // max_ctx above the drafter's native window must not inflate the estimate.
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/64 * GiB, /*total=*/80 * GiB,
                                     /*max_ctx=*/1048576);
    CHECK(d.window_tokens == 262144);
}

TEST_CASE(SkipParkGuardFixture, T16_estimate_margin_and_guard_on_auto) {
    const auto info = qwen35_like_info();
    const int64_t raw = info.weights_bytes + info.fixed_bytes +
                        info.runtime_bytes_per_token * 65536;
    CHECK(skip_park_required_bytes(info, 65536) ==
          raw + raw * kSkipParkSafetyMarginPercent / 100);
    // Guard still applies under auto: <32 GiB with ctx>64K resolves off even
    // when the footprint would fit.
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     info, /*free=*/20 * GiB,
                                     /*total=*/24 * GiB, /*max_ctx=*/131072);
    CHECK(!d.enabled);
}
