#include "rocmfp2_mix_dispatch.h"

#include <cstdio>

namespace {

bool expect(bool condition, const char * message) {
    if (condition) return true;
    std::fprintf(stderr, "test_rocmfp2_mix_dispatch: %s\n", message);
    return false;
}

bool expect_policy(
        bool gfx1151, int n_tokens, bool row4_opt_in,
        int rows_per_warp, int warps_per_block,
        const char * message) {
    const auto policy = rocmfp2_mix_moe_launch_policy(
        gfx1151, n_tokens, row4_opt_in);
    return expect(
        policy.rows_per_warp == rows_per_warp &&
            policy.warps_per_block == warps_per_block,
        message);
}

} // namespace

int main() {
    bool ok = true;

    ok &= expect(!rocmfp2_mix_row4_env_enabled(nullptr),
                 "an unset opt-in was enabled");
    ok &= expect(!rocmfp2_mix_row4_env_enabled(""),
                 "an empty opt-in was enabled");
    ok &= expect(!rocmfp2_mix_row4_env_enabled("0"),
                 "value 0 enabled row4");
    ok &= expect(!rocmfp2_mix_row4_env_enabled("true"),
                 "a non-numeric value enabled row4");
    ok &= expect(!rocmfp2_mix_row4_env_enabled("01"),
                 "a zero-prefixed value enabled row4");
    ok &= expect(!rocmfp2_mix_row4_env_enabled("10"),
                 "a value beginning with 1 enabled row4");
    ok &= expect(!rocmfp2_mix_row4_env_enabled("1 "),
                 "a suffixed value enabled row4");
    ok &= expect(rocmfp2_mix_row4_env_enabled("1"),
                 "the exact opt-in value did not enable row4");

    ok &= expect_policy(true, 3, true, 4, 2,
                        "gfx1151 q3 did not select row4");
    ok &= expect_policy(true, 4, true, 4, 2,
                        "gfx1151 q4 did not select row4");
    ok &= expect_policy(true, 1, true, 2, 8,
                        "gfx1151 q1 launch policy changed");
    ok &= expect_policy(true, 2, true, 2, 8,
                        "gfx1151 q2 launch policy changed");
    ok &= expect_policy(true, 5, true, 2, 4,
                        "gfx1151 q5 launch policy changed");
    ok &= expect_policy(true, 6, true, 2, 4,
                        "gfx1151 q6 launch policy changed");
    ok &= expect_policy(true, 3, false, 2, 4,
                        "disabled gfx1151 q3 did not use row2");
    ok &= expect_policy(true, 4, false, 2, 4,
                        "disabled gfx1151 q4 did not use row2");
    ok &= expect_policy(false, 3, true, 2, 2,
                        "non-gfx1151 q3 inherited row4");
    ok &= expect_policy(false, 4, true, 2, 2,
                        "gfx1201/non-gfx1151 q4 launch policy changed");

    const auto enabled_q4 = rocmfp2_mix_moe_launch_policy(true, 4, true);
    const auto disabled_q4 = rocmfp2_mix_moe_launch_policy(true, 4, false);
    ok &= expect(
        enabled_q4.rows_per_warp * enabled_q4.warps_per_block ==
            disabled_q4.rows_per_warp * disabled_q4.warps_per_block,
        "gfx1151 q4 changed rows covered per block");

    return ok ? 0 : 1;
}
