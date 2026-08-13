// Qwen3.5 text M-RoPE position helpers.
//
// ggml_rope_multi consumes four axis-major rows:
//   [axis0 token positions][axis1 ...][axis2 ...][axis3 ...]
// Keeping this layout in one helper prevents multi-token prefill/verify paths
// from accidentally using token-major groups, a mistake that one-token AR
// decode cannot expose.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::common {

inline std::vector<int32_t> qwen35_text_mrope_positions(int base_pos,
                                                        int n_tokens) {
    if (n_tokens <= 0) return {};

    std::vector<int32_t> positions((size_t)4 * (size_t)n_tokens, 0);
    for (int i = 0; i < n_tokens; ++i) {
        const int32_t pos = (int32_t)(base_pos + i);
        positions[(size_t)0 * n_tokens + i] = pos;
        positions[(size_t)1 * n_tokens + i] = pos;
        positions[(size_t)2 * n_tokens + i] = pos;
        // Axis 3 remains zero for plain text.
    }
    return positions;
}

}  // namespace dflash::common
