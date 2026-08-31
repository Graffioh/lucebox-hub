#pragma once

struct Rocmfp2MixMoeLaunchPolicy {
    int rows_per_warp;
    int warps_per_block;
};

inline bool rocmfp2_mix_row4_env_enabled(const char * value) {
    return value == nullptr || value[0] != '0' || value[1] != '\0';
}

inline Rocmfp2MixMoeLaunchPolicy rocmfp2_mix_moe_launch_policy(
        bool gfx1151, int n_tokens, bool row4_opt_in) {
    const bool row4 = row4_opt_in && gfx1151 && n_tokens > 2;
    return {
        row4 ? 4 : 2,
        row4 ? 2 : (gfx1151 ? (n_tokens <= 2 ? 8 : 4) : 2),
    };
}
