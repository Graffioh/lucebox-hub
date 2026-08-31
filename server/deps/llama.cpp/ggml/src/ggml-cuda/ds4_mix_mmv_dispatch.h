#pragma once

#include <cstdint>

inline bool ds4_mix_mmv_q6_env_enabled(const char * value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline int64_t ds4_mix_mmv_max_tokens(bool q6_opt_in) {
    return q6_opt_in ? 6 : 5;
}

inline bool ds4_mix_mmv_supported_tokens(
        int64_t n_tokens, bool q6_opt_in) {
    return n_tokens >= 1 &&
           n_tokens <= ds4_mix_mmv_max_tokens(q6_opt_in);
}
