// Multi-sequence host residency for PagedKvPool full-attention K/V blocks.
//
// The policy and bookkeeping are backend-neutral. The Qwen engine supplies a
// pinned allocator plus async full-block D2H/H2D callbacks that capture its
// dedicated copy stream and know how to concatenate every full-attention K/V
// tensor into one host block. The manager batches transfers and synchronizes
// once before returning any operation that permits device K/V to be read or a
// recycled physical block to be written.

#pragma once

#include "paged_kv_pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace dflash::common {

enum class PagedKvResidencyStatus : uint8_t {
    Ok = 0,
    InvalidArgument,
    SequenceNotRegistered,
    StaleHandle,
    PoolExhausted,
    NoEvictableBlock,
    HostAllocationFailed,
    TransferFailed,
    HostCopyMissing,
    InconsistentPoolState,
};

const char * paged_kv_residency_status_string(PagedKvResidencyStatus status);

struct PagedKvResidencyConfig {
    // Bytes copied for one physical pool block across all full-attention K/V
    // tensors. Recurrent/DeltaNet state is intentionally outside this pager.
    size_t block_bytes = 0;

    // Hard bound for materialized logical blocks. Zero uses the complete
    // physical PagedKvPool. Reserved-but-unappended pool blocks still consume
    // physical capacity and may temporarily make the effective bound smaller.
    uint32_t resident_budget_blocks = 0;

    // Attention sinks and the trailing local window are never automatic
    // eviction victims. Explicit evict_block(..., allow_protected=true) is
    // available for teardown/tests only.
    uint32_t sink_blocks = 1;
    uint32_t tail_blocks = 4;
};

struct PagedKvResidencyTransferOps {
    // Production implementations should use cudaMallocHost/hipHostMalloc (or
    // the equivalent backend pinned allocator). Every allocation is exactly
    // config.block_bytes and lives until forget_sequence()/reset().
    std::function<void *(size_t)> allocate_pinned;
    std::function<void(void *)> free_pinned;

    // Queue a complete physical-block copy on one dedicated copy stream.
    // handle/logical_block are supplied for diagnostics only; physical_block
    // identifies the source/destination in the Qwen paged K/V tensors.
    std::function<bool(PagedKvSequenceHandle,
                       uint32_t /*logical_block*/,
                       uint32_t /*physical_block*/,
                       void * /*host_dst*/,
                       size_t)> copy_out_async;
    std::function<bool(PagedKvSequenceHandle,
                       uint32_t /*logical_block*/,
                       uint32_t /*physical_block*/,
                       const void * /*host_src*/,
                       size_t)> copy_in_async;

    // Synchronize that copy stream. The manager calls it once after a batch
    // and before attention reads or append writes can observe remapped blocks.
    std::function<bool()> synchronize;
};

struct PagedKvResidencyStats {
    uint64_t page_ins = 0;
    uint64_t page_outs = 0;
    uint64_t resident_blocks = 0;
    uint64_t reselects = 0;
    uint64_t host_bytes = 0;
    uint64_t moved_bytes = 0;
};

struct PagedKvResidentAppendResult {
    PagedKvResidencyStatus status = PagedKvResidencyStatus::Ok;
    PagedKvAppendResult pool_result;

    explicit operator bool() const {
        return status == PagedKvResidencyStatus::Ok &&
               pool_result.status == PagedKvStatus::Ok;
    }
};

class PagedKvResidencyManager {
public:
    // Throws std::invalid_argument for an empty transfer callback, zero block
    // bytes, or a resident budget larger than the physical pool.
    PagedKvResidencyManager(PagedKvPool & pool,
                            PagedKvResidencyConfig config,
                            PagedKvResidencyTransferOps transfers);
    ~PagedKvResidencyManager();

    PagedKvResidencyManager(const PagedKvResidencyManager &) = delete;
    PagedKvResidencyManager & operator=(const PagedKvResidencyManager &) = delete;

    // Call immediately after PagedKvPool::acquire[_reserved](). Registration
    // rejects pre-existing cold entries because their host backing is unknown.
    PagedKvResidencyStatus register_sequence(PagedKvSequenceHandle handle);

    // Free this sequence's pinned backing. Call before pool.release, and do
    // not release the pool handle unless this returns Ok: a failed transfer
    // barrier keeps its physical pages and pinned buffers quarantined until
    // forget_sequence is retried. The handle must still match the manager's
    // registered generation.
    PagedKvResidencyStatus forget_sequence(PagedKvSequenceHandle handle);
    void reset();

    // Recommended append integration: restores a cold partial append head,
    // fairly evicts enough blocks, synchronizes the copy stream, appends in
    // PagedKvPool, and marks every touched block pending until the engine has
    // finished the target graph that writes the returned physical rows.
    PagedKvResidentAppendResult append(PagedKvSequenceHandle handle,
                                       uint32_t token_count,
                                       bool only_first_last_slots = false);

    // Split integration for callers whose slot manager owns pool.append():
    //   prepare_append(handle, n); pool.append(handle, n); observe_append(...)
    // prepare_append always synchronizes before returning Ok. observe_append
    // must be called before the next residency operation.
    PagedKvResidencyStatus prepare_append(PagedKvSequenceHandle handle,
                                          uint32_t token_count);
    PagedKvResidencyStatus observe_append(
        PagedKvSequenceHandle handle, const PagedKvAppendResult & result);

    // Clear all blocks staged by append()/observe_append() for this sequence.
    // Call only after synchronizing the target compute that wrote every returned
    // physical row. Pending blocks cannot be evicted or deselected, preventing a
    // later slot staged in the same packed step from recycling an unwritten page.
    PagedKvResidencyStatus commit_pending_writes(
        PagedKvSequenceHandle handle);

    // Restore the requested host-backed logical blocks as one transfer batch.
    // The operation synchronizes before returning Ok, so attention may read
    // the returned pool block table immediately.
    PagedKvResidencyStatus ensure_resident(
        PagedKvSequenceHandle handle,
        const std::vector<uint32_t> & logical_blocks);

    // Explicit single-block eviction. Automatic policy never evicts sink/tail
    // blocks; protected eviction requires an explicit opt-in. Pending target
    // writes are never evictable, including with allow_protected=true.
    PagedKvResidencyStatus evict_block(PagedKvSequenceHandle handle,
                                       uint32_t logical_block,
                                       bool allow_protected = false);

    // Update recency after a block participates in attention.
    PagedKvResidencyStatus touch(PagedKvSequenceHandle handle,
                                 uint32_t logical_block);

    // Optional relevance array, one score per materialized logical block.
    // Higher values are retained. Without scores, reselection uses LRU.
    PagedKvResidencyStatus set_scores(PagedKvSequenceHandle handle,
                                      const std::vector<float> & scores);
    PagedKvResidencyStatus reselect(PagedKvSequenceHandle handle);

    // Evict unprotected blocks above each active sequence's deterministic
    // fair share. Spare capacity remains borrowable; it is reclaimed first
    // from borrowers when another sequence needs a block.
    PagedKvResidencyStatus rebalance();

    // Explicit barrier for engine paths that directly issue operations and
    // then read the paged K/V tensors. Normally append/ensure/reselect/evict
    // already provide this barrier.
    PagedKvResidencyStatus synchronize_before_read();

    uint32_t fair_quota(PagedKvSequenceHandle handle) const;
    bool is_resident(PagedKvSequenceHandle handle,
                     uint32_t logical_block) const;
    PagedKvResidencyStats stats() const;

private:
    struct BlockState {
        void * host = nullptr;
        bool host_valid = false;
        bool write_pending = false;
        bool page_out_pending = false;
        bool page_in_pending = false;
        bool reservation_pending = false;
        bool score_valid = false;
        float score = 0.0f;
        uint64_t last_use = 0;
    };

    struct SequenceState {
        bool active = false;
        uint64_t generation = 0;
        std::vector<BlockState> blocks;
    };

    struct Victim {
        bool found = false;
        PagedKvSequenceHandle handle;
        uint32_t logical_block = 0;
        int class_rank = 0;
        float score = 0.0f;
        uint64_t last_use = 0;
    };

    struct PendingTransfer {
        PagedKvSequenceHandle handle;
        uint32_t logical_block = 0;
        uint32_t physical_block = PAGED_KV_COLD_BLOCK;
    };

    PagedKvResidencyStatus validate_registered(
        PagedKvSequenceHandle handle) const;
    PagedKvResidencyStatus refresh_sequence(
        PagedKvSequenceHandle handle, bool appended_rows = false,
        uint32_t append_first = 0, uint32_t append_count = 0);
    PagedKvResidencyStatus finish_transfers(
        PagedKvResidencyStatus status);

    bool is_protected(PagedKvSequenceHandle handle,
                      uint32_t logical_block) const;
    uint32_t sequence_resident_count(PagedKvSequenceHandle handle) const;
    uint32_t sequence_pending_page_out_count(
        PagedKvSequenceHandle handle) const;
    uint32_t sequence_protected_count(PagedKvSequenceHandle handle) const;
    uint32_t total_resident_count() const;
    uint32_t quota_for_slot(uint32_t slot) const;

    Victim choose_victim(PagedKvSequenceHandle requester) const;
    PagedKvResidencyStatus make_room(PagedKvSequenceHandle requester,
                                     uint32_t pool_blocks_needed,
                                     uint32_t future_resident_blocks);
    PagedKvResidencyStatus evict_block_async(
        PagedKvSequenceHandle handle, uint32_t logical_block,
        bool allow_protected);
    PagedKvResidencyStatus restore_block_async(
        PagedKvSequenceHandle handle, uint32_t logical_block);

    void free_sequence_buffers(SequenceState & state) noexcept;

    PagedKvPool & pool_;
    PagedKvResidencyConfig config_;
    PagedKvResidencyTransferOps transfers_;
    std::vector<SequenceState> sequences_;
    PagedKvResidencyStats stats_;
    uint64_t clock_ = 0;
    bool transfers_pending_ = false;
    bool transfer_barrier_failed_ = false;
    std::vector<PendingTransfer> pending_page_outs_;
    std::vector<PendingTransfer> pending_page_ins_;
};

}  // namespace dflash::common
