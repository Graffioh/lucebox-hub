#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::common {

// Pure host-side allocation plan. Byte counts describe tensor payloads (ggml
// alignment/padding is deliberately excluded).
struct DeepSeek4PagedCachePlan {
    uint32_t slots = 0;
    uint32_t max_ctx = 0;
    uint32_t physical_blocks = 0;
    uint32_t max_blocks_per_sequence = 0;
    uint64_t metadata_bytes = 0;
    uint64_t raw_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t total_persistent_bytes = 0;
    std::vector<uint32_t> ratios;
    std::vector<uint64_t> physical_rows;
};

// Ratios must contain only 0, 4, or 128. A ratio-zero layer has a raw ring
// but no compressed storage or compressor state.
bool plan_deepseek4_paged_cache(uint32_t head_dim,
                                uint32_t indexer_head_dim,
                                uint32_t slots,
                                uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<uint32_t> & ratios,
                                DeepSeek4PagedCachePlan & out);

} // namespace dflash::common
