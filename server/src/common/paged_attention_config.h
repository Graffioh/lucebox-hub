// Paged K/V block geometry.
//
// Compatibility rules for --paged-attention live in feature_gate.cpp and the
// capability table in model_capabilities.h; only the sizing arithmetic that
// the allocator, the backend, and the gate all need is here. The one rule
// that cannot move is the 256-wide K/V head check in Qwen35Backend::init():
// it needs loaded tensor dimensions, which GgufModelInfo does not carry.

#pragma once

#include <algorithm>
#include <cstdint>

namespace dflash::common {

constexpr int PAGED_BLOCK_SIZE = 16;

constexpr int paged_block_count(int max_ctx) {
    return (max_ctx + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;
}

constexpr int paged_token_capacity(int max_ctx) {
    return paged_block_count(max_ctx) * PAGED_BLOCK_SIZE;
}

// Inputs for sizing a concurrent paged pool from memory that remains after
// model weights have loaded. `fixed_cache_bytes` covers cache allocations that
// do not shrink with the pool (per-slot recurrent state and metadata);
// `reserve_bytes` leaves room for runtime graph buffers. Prefill K/V shares
// the same on-demand pool as decode K/V.
struct PagedKvAutoBudget {
    int64_t free_bytes        = 0;
    int64_t fixed_cache_bytes = 0;
    int64_t reserve_bytes     = 0;
    int64_t bytes_per_token   = 0;
};

// Return a whole-block physical capacity, capped by the old
// n_slots * max_ctx policy and by the pool's signed-int tensor address space.
// Zero means the supplied memory budget cannot hold even one block.
inline int64_t paged_kv_auto_pool_tokens(int max_ctx, int n_slots,
                                         const PagedKvAutoBudget & budget) {
    if (max_ctx < 1 || n_slots < 1 || budget.free_bytes <= 0 ||
        budget.bytes_per_token <= 0) {
        return 0;
    }
    const int64_t usable = std::max<int64_t>(
        0, budget.free_bytes - budget.fixed_cache_bytes -
               budget.reserve_bytes);
    int64_t tokens = usable / budget.bytes_per_token;
    tokens = (tokens / PAGED_BLOCK_SIZE) * PAGED_BLOCK_SIZE;
    const int64_t logical_cap =
        (int64_t)n_slots * (int64_t)paged_token_capacity(max_ctx);
    const int64_t address_cap =
        ((int64_t)INT32_MAX - PAGED_BLOCK_SIZE) / PAGED_BLOCK_SIZE *
        PAGED_BLOCK_SIZE;
    return std::max<int64_t>(
        0, std::min(tokens, std::min(logical_cap, address_cap)));
}

// Admission retains enough headroom for one full prefill chunk, or one per
// cent of the pool for large pools. Running prefill/decode slots may consume
// this watermark; it only prevents a new request from creating immediate
// pressure.
inline uint32_t paged_kv_admission_watermark_blocks(
        uint32_t pool_blocks, int prefill_chunk_tokens) {
    const uint32_t chunk_blocks =
        (uint32_t)std::max(1, paged_block_count(
            std::max(1, prefill_chunk_tokens)));
    const uint32_t percent_blocks =
        std::max<uint32_t>(1, (pool_blocks + 99u) / 100u);
    return std::max(chunk_blocks, percent_blocks);
}

}  // namespace dflash::common
