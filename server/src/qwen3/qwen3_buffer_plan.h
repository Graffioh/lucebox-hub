#pragma once

#include <cstddef>

namespace dflash::common {

struct Qwen3DrafterBufferPlan {
    std::size_t rope_k_buffers;
    std::size_t value_buffers;
    std::size_t rope_q_tail_buffers;
};

inline Qwen3DrafterBufferPlan qwen3_drafter_buffer_plan(
        bool nope_tail, int n_layer) {
    const std::size_t layers = n_layer > 0 ? (std::size_t)n_layer : 0u;
    return {
        nope_tail ? (layers > 0 ? 1u : 0u) : layers,
        layers > 0 ? 1u : 0u,
        nope_tail ? 0u : layers,
    };
}

} // namespace dflash::common
