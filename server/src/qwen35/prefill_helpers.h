// Shared prefill helpers for Qwen3.5/3.6.

#pragma once

#include "attn_masks.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace dflash::common {

inline int qwen35_prefill_ubatch(int fallback) {
    const char * value = std::getenv("DFLASH27B_PREFILL_UBATCH");
    return value ? std::max(1, std::atoi(value)) : fallback;
}

inline void fill_qwen35_mrope_positions(int32_t * positions,
                                        int base_pos, int n_tokens) {
    for (int i = 0; i < n_tokens; ++i) {
        const int p = base_pos + i;
        positions[4 * i + 0] = p;
        positions[4 * i + 1] = p;
        positions[4 * i + 2] = p;
        positions[4 * i + 3] = 0;
    }
}

inline void upload_qwen35_causal_mask(ggml_tensor * mask, int kv_start,
                                       int n_tokens, int kq_stride_pad) {
    if (!mask) return;
    std::vector<uint16_t> data;
    build_causal_mask(data, kv_start + n_tokens, n_tokens, kv_start,
                      kq_stride_pad, /*win_start=*/0, (int)mask->ne[0]);
    ggml_backend_tensor_set(mask, data.data(), 0,
                            sizeof(uint16_t) * data.size());
}

}  // namespace dflash::common
