#include "paged_kv_residency.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dflash::common {

namespace {

PagedKvResidencyStatus from_pool_status(PagedKvStatus status) {
    switch (status) {
        case PagedKvStatus::Ok:
            return PagedKvResidencyStatus::Ok;
        case PagedKvStatus::StaleHandle:
            return PagedKvResidencyStatus::StaleHandle;
        case PagedKvStatus::BlocksExhausted:
            return PagedKvResidencyStatus::PoolExhausted;
        case PagedKvStatus::InvalidArgument:
        case PagedKvStatus::LogicalBlockOutOfRange:
        case PagedKvStatus::BlockNotResident:
        case PagedKvStatus::BlockAlreadyResident:
            return PagedKvResidencyStatus::InvalidArgument;
        case PagedKvStatus::DuplicateRequest:
        case PagedKvStatus::SequenceSlotsExhausted:
            return PagedKvResidencyStatus::InconsistentPoolState;
    }
    return PagedKvResidencyStatus::InconsistentPoolState;
}

}  // namespace

const char * paged_kv_residency_status_string(
        PagedKvResidencyStatus status) {
    switch (status) {
        case PagedKvResidencyStatus::Ok:
            return "ok";
        case PagedKvResidencyStatus::InvalidArgument:
            return "invalid argument";
        case PagedKvResidencyStatus::SequenceNotRegistered:
            return "sequence not registered";
        case PagedKvResidencyStatus::StaleHandle:
            return "stale sequence handle";
        case PagedKvResidencyStatus::PoolExhausted:
            return "physical pool exhausted";
        case PagedKvResidencyStatus::NoEvictableBlock:
            return "no evictable block";
        case PagedKvResidencyStatus::HostAllocationFailed:
            return "pinned host allocation failed";
        case PagedKvResidencyStatus::TransferFailed:
            return "K/V transfer failed";
        case PagedKvResidencyStatus::HostCopyMissing:
            return "cold block has no valid host copy";
        case PagedKvResidencyStatus::InconsistentPoolState:
            return "residency state disagrees with paged pool";
    }
    return "unknown paged K/V residency status";
}

PagedKvResidencyManager::PagedKvResidencyManager(
        PagedKvPool & pool, PagedKvResidencyConfig config,
        PagedKvResidencyTransferOps transfers)
    : pool_(pool), config_(config), transfers_(std::move(transfers)) {
    if (config_.block_bytes == 0 || !transfers_.allocate_pinned ||
        !transfers_.free_pinned || !transfers_.copy_out_async ||
        !transfers_.copy_in_async || !transfers_.synchronize) {
        throw std::invalid_argument("invalid paged K/V residency callbacks");
    }
    if (config_.resident_budget_blocks == 0) {
        config_.resident_budget_blocks = pool_.physical_block_count();
    }
    if (config_.resident_budget_blocks > pool_.physical_block_count()) {
        throw std::invalid_argument("resident budget exceeds paged K/V pool");
    }
    sequences_.resize(pool_.max_sequences());
}

PagedKvResidencyManager::~PagedKvResidencyManager() {
    try {
        reset();
    } catch (...) {
        // Destructors must not propagate callback failures. Production pinned
        // allocators/free functions are non-throwing; this catch protects test
        // and plugin callbacks from terminating teardown.
    }
}

PagedKvResidencyStatus PagedKvResidencyManager::validate_registered(
        PagedKvSequenceHandle handle) const {
    if (transfer_barrier_failed_) {
        return PagedKvResidencyStatus::TransferFailed;
    }
    if (handle.slot >= sequences_.size()) {
        return PagedKvResidencyStatus::StaleHandle;
    }
    const SequenceState & state = sequences_[handle.slot];
    if (!state.active) {
        return PagedKvResidencyStatus::SequenceNotRegistered;
    }
    if (state.generation != handle.generation) {
        return PagedKvResidencyStatus::StaleHandle;
    }
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::register_sequence(
        PagedKvSequenceHandle handle) {
    if (transfer_barrier_failed_) {
        return PagedKvResidencyStatus::TransferFailed;
    }
    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) {
        return from_pool_status(pool_status);
    }
    if (handle.slot >= sequences_.size()) {
        return PagedKvResidencyStatus::StaleHandle;
    }
    if (std::find(snapshot.block_table.begin(), snapshot.block_table.end(),
                  PAGED_KV_COLD_BLOCK) != snapshot.block_table.end()) {
        return PagedKvResidencyStatus::InconsistentPoolState;
    }

    SequenceState & state = sequences_[handle.slot];
    if (state.active && state.generation == handle.generation) {
        return PagedKvResidencyStatus::Ok;
    }
    if (state.active) {
        const auto synced = synchronize_before_read();
        if (synced != PagedKvResidencyStatus::Ok) return synced;
        free_sequence_buffers(state);
    }
    state.active = true;
    state.generation = handle.generation;
    state.blocks.resize(snapshot.block_table.size());
    for (BlockState & block : state.blocks) block.last_use = ++clock_;
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::forget_sequence(
        PagedKvSequenceHandle handle) {
    // Teardown is also the recovery path after a failed copy-stream barrier.
    // validate_registered() deliberately rejects ordinary operations while a
    // transfer is quarantined, so validate the generation directly here and
    // allow synchronize_before_read() to retry the barrier.
    if (handle.slot >= sequences_.size()) {
        return PagedKvResidencyStatus::StaleHandle;
    }
    const SequenceState & state = sequences_[handle.slot];
    if (!state.active) {
        return PagedKvResidencyStatus::SequenceNotRegistered;
    }
    if (state.generation != handle.generation) {
        return PagedKvResidencyStatus::StaleHandle;
    }
    const auto synced = synchronize_before_read();
    if (synced != PagedKvResidencyStatus::Ok) return synced;
    free_sequence_buffers(sequences_[handle.slot]);
    return PagedKvResidencyStatus::Ok;
}

void PagedKvResidencyManager::reset() {
    // A failed barrier should not let teardown free memory still referenced by
    // an async transfer. Leak those buffers rather than creating a use-after-
    // free; the process/backend is already unhealthy in this case.
    if (synchronize_before_read() != PagedKvResidencyStatus::Ok) return;
    for (SequenceState & state : sequences_) free_sequence_buffers(state);
    stats_ = {};
    clock_ = 0;
}

void PagedKvResidencyManager::free_sequence_buffers(
        SequenceState & state) noexcept {
    for (BlockState & block : state.blocks) {
        if (!block.host) continue;
        try {
            transfers_.free_pinned(block.host);
        } catch (...) {
            // The callback contract is non-throwing. Continue freeing other
            // pages if a third-party implementation violates it.
        }
        block.host = nullptr;
        if (stats_.host_bytes >= config_.block_bytes) {
            stats_.host_bytes -= config_.block_bytes;
        }
    }
    state = {};
}

PagedKvResidencyStatus PagedKvResidencyManager::refresh_sequence(
        PagedKvSequenceHandle handle, bool appended_rows,
        uint32_t append_first, uint32_t append_count) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);

    SequenceState & state = sequences_[handle.slot];
    if (snapshot.block_table.size() < state.blocks.size()) {
        return PagedKvResidencyStatus::InconsistentPoolState;
    }
    state.blocks.resize(snapshot.block_table.size());
    for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
        if (snapshot.block_table[logical] == PAGED_KV_COLD_BLOCK &&
            !state.blocks[logical].host_valid) {
            return PagedKvResidencyStatus::HostCopyMissing;
        }
    }

    if (appended_rows && append_count > 0) {
        const uint64_t last = static_cast<uint64_t>(append_first) +
                              append_count - 1;
        const uint32_t first_block = append_first / pool_.block_size();
        const uint32_t last_block =
            static_cast<uint32_t>(last / pool_.block_size());
        if (last_block >= state.blocks.size()) {
            return PagedKvResidencyStatus::InconsistentPoolState;
        }
        for (uint32_t logical = first_block; logical <= last_block; ++logical) {
            // The target graph has not written the returned rows yet. Retain a
            // restored partial page's old host image until commit, but make the
            // physical page ineligible for eviction in every policy path.
            state.blocks[logical].write_pending = true;
            state.blocks[logical].last_use = ++clock_;
        }
    }
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::prepare_append(
        PagedKvSequenceHandle handle, uint32_t token_count) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;

    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);
    if (token_count > std::numeric_limits<uint32_t>::max() -
                          snapshot.kv_seq_len) {
        return PagedKvResidencyStatus::InvalidArgument;
    }
    if (token_count == 0) return finish_transfers(PagedKvResidencyStatus::Ok);

    BlockState * append_head = nullptr;
    bool restore_partial_head = false;
    uint32_t partial_head = 0;
    if (snapshot.kv_seq_len % pool_.block_size() != 0) {
        partial_head = snapshot.kv_seq_len / pool_.block_size();
        append_head = &sequences_[handle.slot].blocks[partial_head];
        append_head->reservation_pending = true;
        restore_partial_head =
            snapshot.block_table[partial_head] == PAGED_KV_COLD_BLOCK;
    }

    const uint32_t new_length = snapshot.kv_seq_len + token_count;
    const uint32_t required_blocks =
        1 + (new_length - 1) / pool_.block_size();
    const uint32_t additional_blocks = required_blocks -
        static_cast<uint32_t>(snapshot.block_table.size());
    const uint32_t globally_needed = additional_blocks >
            snapshot.reserved_block_count
        ? additional_blocks - snapshot.reserved_block_count : 0;
    // Reserve the partial-head restoration and every new physical page as one
    // transaction. Restoring first and making room later can otherwise select
    // the just-restored append head as the next eviction victim.
    const auto room = make_room(
        handle, globally_needed + (restore_partial_head ? 1u : 0u),
        additional_blocks + (restore_partial_head ? 1u : 0u));
    if (room != PagedKvResidencyStatus::Ok) {
        if (append_head) append_head->reservation_pending = false;
        return room;
    }
    if (restore_partial_head) {
        const auto restored = restore_block_async(
            handle, partial_head);
        if (restored != PagedKvResidencyStatus::Ok) {
            const auto finished = finish_transfers(restored);
            if (append_head) append_head->reservation_pending = false;
            return finished;
        }
    }
    const auto finished = finish_transfers(PagedKvResidencyStatus::Ok);
    if (append_head) append_head->reservation_pending = false;
    return finished;
}

PagedKvResidentAppendResult PagedKvResidencyManager::append(
        PagedKvSequenceHandle handle, uint32_t token_count,
        bool only_first_last_slots) {
    PagedKvResidentAppendResult result;
    result.status = prepare_append(handle, token_count);
    if (result.status != PagedKvResidencyStatus::Ok) return result;

    result.pool_result =
        pool_.append(handle, token_count, only_first_last_slots);
    if (result.pool_result.status != PagedKvStatus::Ok) {
        result.status = from_pool_status(result.pool_result.status);
        return result;
    }
    result.status = observe_append(handle, result.pool_result);
    return result;
}

PagedKvResidencyStatus PagedKvResidencyManager::observe_append(
        PagedKvSequenceHandle handle, const PagedKvAppendResult & result) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    if (result.status != PagedKvStatus::Ok) return from_pool_status(result.status);

    SequenceState & state = sequences_[handle.slot];
    for (const PagedKvBlockRemap & remap : result.remapped_cold_blocks) {
        if (remap.logical_block >= state.blocks.size() ||
            !state.blocks[remap.logical_block].host_valid) {
            return finish_transfers(PagedKvResidencyStatus::HostCopyMissing);
        }
        BlockState & block = state.blocks[remap.logical_block];
        // A callback may reject only after queuing a prefix of a multi-tensor
        // copy. Quarantine the destination before invoking it either way.
        transfers_pending_ = true;
        block.page_in_pending = true;
        bool queued = false;
        try {
            queued = transfers_.copy_in_async(
                handle, remap.logical_block, remap.physical_block,
                block.host, config_.block_bytes);
        } catch (...) {
            queued = false;
        }
        const PendingTransfer transfer{
            handle, remap.logical_block, remap.physical_block};
        if (!queued) {
            pending_page_in_rollbacks_.push_back(transfer);
            return finish_transfers(PagedKvResidencyStatus::TransferFailed);
        }
        pending_page_ins_.push_back(transfer);
    }
    const auto synced = finish_transfers(PagedKvResidencyStatus::Ok);
    if (synced != PagedKvResidencyStatus::Ok) return synced;

    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);
    if (result.token_count > snapshot.kv_seq_len) {
        return PagedKvResidencyStatus::InconsistentPoolState;
    }
    return refresh_sequence(
        handle, /*appended_rows=*/true,
        snapshot.kv_seq_len - result.token_count, result.token_count);
}

PagedKvResidencyStatus PagedKvResidencyManager::commit_pending_writes(
        PagedKvSequenceHandle handle) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;

    // The caller supplies the target-compute -> pager dependency by
    // synchronizing the target backend before this call. Any old host image is
    // stale only now, after the device rows have actually been overwritten.
    SequenceState & state = sequences_[handle.slot];
    for (BlockState & block : state.blocks) {
        if (!block.write_pending) continue;
        block.write_pending = false;
        block.host_valid = false;
        block.last_use = ++clock_;
    }
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::finish_transfers(
        PagedKvResidencyStatus status) {
    const auto synced = synchronize_before_read();
    return synced == PagedKvResidencyStatus::Ok ? status : synced;
}

PagedKvResidencyStatus PagedKvResidencyManager::synchronize_before_read() {
    if (!transfers_pending_) {
        return transfer_barrier_failed_
            ? PagedKvResidencyStatus::TransferFailed
            : PagedKvResidencyStatus::Ok;
    }
    bool ok = false;
    try {
        ok = transfers_.synchronize();
    } catch (...) {
        ok = false;
    }
    if (!ok) {
        // A failed barrier does not prove that the stream stopped using its
        // source and destination blocks. Keep every transfer pending and every
        // H2D destination mapped (therefore quarantined from the free list),
        // and reject all manager operations until an explicit retry confirms
        // that the stream has drained.
        transfer_barrier_failed_ = true;
        return PagedKvResidencyStatus::TransferFailed;
    }

    transfers_pending_ = false;
    transfer_barrier_failed_ = false;

    PagedKvResidencyStatus result = PagedKvResidencyStatus::Ok;
    for (const PendingTransfer & transfer : pending_page_outs_) {
        if (validate_registered(transfer.handle) !=
            PagedKvResidencyStatus::Ok) {
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        BlockState & block =
            sequences_[transfer.handle.slot].blocks[transfer.logical_block];
        block.page_out_pending = false;
        PagedKvSequenceSnapshot snapshot;
        if (pool_.sequence(transfer.handle, snapshot) != PagedKvStatus::Ok ||
            transfer.logical_block >= snapshot.block_table.size() ||
            snapshot.block_table[transfer.logical_block] !=
                transfer.physical_block) {
            block.host_valid = false;
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        uint32_t released = PAGED_KV_COLD_BLOCK;
        if (pool_.page_out_block(
                transfer.handle, transfer.logical_block, released) !=
                PagedKvStatus::Ok || released != transfer.physical_block) {
            block.host_valid = false;
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        block.host_valid = true;
        stats_.page_outs++;
        stats_.moved_bytes += config_.block_bytes;
    }
    for (const PendingTransfer & transfer : pending_page_ins_) {
        if (validate_registered(transfer.handle) !=
            PagedKvResidencyStatus::Ok) {
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        BlockState & block =
            sequences_[transfer.handle.slot].blocks[transfer.logical_block];
        block.page_in_pending = false;
        stats_.page_ins++;
        stats_.moved_bytes += config_.block_bytes;
    }
    for (const PendingTransfer & transfer : pending_page_in_rollbacks_) {
        if (validate_registered(transfer.handle) !=
            PagedKvResidencyStatus::Ok) {
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        BlockState & block =
            sequences_[transfer.handle.slot].blocks[transfer.logical_block];
        PagedKvSequenceSnapshot snapshot;
        if (pool_.sequence(transfer.handle, snapshot) != PagedKvStatus::Ok ||
            transfer.logical_block >= snapshot.block_table.size() ||
            snapshot.block_table[transfer.logical_block] !=
                transfer.physical_block) {
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        uint32_t released = PAGED_KV_COLD_BLOCK;
        if (pool_.page_out_block(
                transfer.handle, transfer.logical_block, released) !=
                PagedKvStatus::Ok || released != transfer.physical_block) {
            result = PagedKvResidencyStatus::InconsistentPoolState;
            continue;
        }
        block.page_in_pending = false;
    }
    pending_page_outs_.clear();
    pending_page_ins_.clear();
    pending_page_in_rollbacks_.clear();
    if (result != PagedKvResidencyStatus::Ok) return result;
    return PagedKvResidencyStatus::Ok;
}

bool PagedKvResidencyManager::is_protected(
        PagedKvSequenceHandle handle, uint32_t logical_block) const {
    PagedKvSequenceSnapshot snapshot;
    if (pool_.sequence(handle, snapshot) != PagedKvStatus::Ok ||
        logical_block >= snapshot.block_table.size()) {
        return true;
    }
    if (logical_block < config_.sink_blocks) return true;
    const uint32_t tail_begin = snapshot.block_table.size() > config_.tail_blocks
        ? static_cast<uint32_t>(snapshot.block_table.size()) -
              config_.tail_blocks
        : 0;
    return logical_block >= tail_begin;
}

uint32_t PagedKvResidencyManager::sequence_resident_count(
        PagedKvSequenceHandle handle) const {
    uint32_t count = 0;
    return pool_.resident_block_count(handle, count) == PagedKvStatus::Ok
        ? count : 0;
}

uint32_t PagedKvResidencyManager::sequence_pending_page_out_count(
        PagedKvSequenceHandle handle) const {
    if (handle.slot >= sequences_.size()) return 0;
    const SequenceState & state = sequences_[handle.slot];
    if (!state.active || state.generation != handle.generation) return 0;
    return static_cast<uint32_t>(std::count_if(
        state.blocks.begin(), state.blocks.end(),
        [](const BlockState & block) { return block.page_out_pending; }));
}

uint32_t PagedKvResidencyManager::sequence_protected_count(
        PagedKvSequenceHandle handle) const {
    PagedKvSequenceSnapshot snapshot;
    if (pool_.sequence(handle, snapshot) != PagedKvStatus::Ok) return 0;
    uint32_t count = 0;
    for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
        if (snapshot.block_table[logical] != PAGED_KV_COLD_BLOCK &&
            is_protected(handle, logical)) {
            ++count;
        }
    }
    return count;
}

uint32_t PagedKvResidencyManager::total_resident_count() const {
    uint32_t total = 0;
    for (uint32_t slot = 0; slot < sequences_.size(); ++slot) {
        const SequenceState & state = sequences_[slot];
        if (!state.active) continue;
        total += sequence_resident_count({slot, state.generation});
    }
    return total;
}

uint32_t PagedKvResidencyManager::quota_for_slot(uint32_t slot) const {
    uint32_t active = 0;
    uint32_t rank = 0;
    for (uint32_t i = 0; i < sequences_.size(); ++i) {
        if (!sequences_[i].active) continue;
        if (i < slot) ++rank;
        ++active;
    }
    if (active == 0 || slot >= sequences_.size() ||
        !sequences_[slot].active) {
        return 0;
    }
    const uint32_t base = config_.resident_budget_blocks / active;
    const uint32_t remainder = config_.resident_budget_blocks % active;
    const uint32_t fair = base + (rank < remainder ? 1u : 0u);
    const PagedKvSequenceHandle handle{slot, sequences_[slot].generation};
    return std::max(fair, sequence_protected_count(handle));
}

uint32_t PagedKvResidencyManager::fair_quota(
        PagedKvSequenceHandle handle) const {
    return validate_registered(handle) == PagedKvResidencyStatus::Ok
        ? quota_for_slot(handle.slot) : 0;
}

PagedKvResidencyManager::Victim PagedKvResidencyManager::choose_victim(
        PagedKvSequenceHandle requester) const {
    Victim best;
    for (uint32_t slot = 0; slot < sequences_.size(); ++slot) {
        const SequenceState & state = sequences_[slot];
        if (!state.active) continue;
        const PagedKvSequenceHandle owner{slot, state.generation};
        PagedKvSequenceSnapshot snapshot;
        if (pool_.sequence(owner, snapshot) != PagedKvStatus::Ok) continue;
        const uint32_t resident = sequence_resident_count(owner);
        const uint32_t pending_page_outs =
            sequence_pending_page_out_count(owner);
        const uint32_t resident_after_pending =
            resident > pending_page_outs ? resident - pending_page_outs : 0;
        const uint32_t quota = quota_for_slot(slot);

        int class_rank = 2;
        if (resident_after_pending > quota) class_rank = 0;
        else if (slot == requester.slot &&
                 state.generation == requester.generation) class_rank = 1;

        for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
            if (snapshot.block_table[logical] == PAGED_KV_COLD_BLOCK ||
                state.blocks[logical].write_pending ||
                state.blocks[logical].page_out_pending ||
                state.blocks[logical].page_in_pending ||
                state.blocks[logical].reservation_pending ||
                is_protected(owner, logical)) {
                continue;
            }
            const BlockState & block = state.blocks[logical];
            // An unscored page is the first eviction candidate once relevance
            // scoring is active for only part of a sequence.
            const float score = block.score_valid
                ? block.score : -std::numeric_limits<float>::infinity();
            const bool better = !best.found ||
                class_rank < best.class_rank ||
                (class_rank == best.class_rank && score < best.score) ||
                (class_rank == best.class_rank && score == best.score &&
                 block.last_use < best.last_use) ||
                (class_rank == best.class_rank && score == best.score &&
                 block.last_use == best.last_use &&
                 (slot < best.handle.slot ||
                  (slot == best.handle.slot &&
                   logical < best.logical_block)));
            if (better) {
                best.found = true;
                best.handle = owner;
                best.logical_block = logical;
                best.class_rank = class_rank;
                best.score = score;
                best.last_use = block.last_use;
            }
        }
    }
    return best;
}

PagedKvResidencyStatus PagedKvResidencyManager::make_room(
        PagedKvSequenceHandle requester, uint32_t pool_blocks_needed,
        uint32_t future_resident_blocks) {
    const uint32_t free = pool_.free_block_count();
    const uint32_t need_for_pool = pool_blocks_needed > free
        ? pool_blocks_needed - free : 0;
    const uint32_t resident = total_resident_count();
    const uint64_t projected =
        static_cast<uint64_t>(resident) + future_resident_blocks;
    const uint32_t need_for_budget =
        projected > config_.resident_budget_blocks
        ? static_cast<uint32_t>(projected - config_.resident_budget_blocks)
        : 0;
    const uint32_t evictions = std::max(need_for_pool, need_for_budget);
    for (uint32_t i = 0; i < evictions; ++i) {
        const Victim victim = choose_victim(requester);
        if (!victim.found) {
            return finish_transfers(PagedKvResidencyStatus::NoEvictableBlock);
        }
        const auto status = evict_block_async(
            victim.handle, victim.logical_block,
            /*allow_protected=*/false);
        if (status != PagedKvResidencyStatus::Ok) return finish_transfers(status);
    }
    // A page is recyclable only after the complete D2H batch succeeds.
    return finish_transfers(PagedKvResidencyStatus::Ok);
}

PagedKvResidencyStatus PagedKvResidencyManager::evict_block_async(
        PagedKvSequenceHandle handle, uint32_t logical_block,
        bool allow_protected) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);
    if (logical_block >= snapshot.block_table.size()) {
        return PagedKvResidencyStatus::InvalidArgument;
    }
    if (snapshot.block_table[logical_block] == PAGED_KV_COLD_BLOCK) {
        return PagedKvResidencyStatus::InvalidArgument;
    }

    BlockState & block = sequences_[handle.slot].blocks[logical_block];
    if (block.write_pending || block.page_out_pending ||
        block.page_in_pending ||
        (!allow_protected && is_protected(handle, logical_block))) {
        return PagedKvResidencyStatus::NoEvictableBlock;
    }
    if (!block.host) {
        try {
            block.host = transfers_.allocate_pinned(config_.block_bytes);
        } catch (...) {
            block.host = nullptr;
        }
        if (!block.host) return PagedKvResidencyStatus::HostAllocationFailed;
        stats_.host_bytes += config_.block_bytes;
    }

    const uint32_t physical = snapshot.block_table[logical_block];
    bool queued = false;
    // False may mean a multi-tensor callback queued only a prefix. Invalidate
    // the old host image and require a barrier before any later operation.
    transfers_pending_ = true;
    block.host_valid = false;
    try {
        queued = transfers_.copy_out_async(
            handle, logical_block, physical, block.host,
            config_.block_bytes);
    } catch (...) {
        queued = false;
    }
    if (!queued) return PagedKvResidencyStatus::TransferFailed;
    block.page_out_pending = true;
    pending_page_outs_.push_back({handle, logical_block, physical});
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::restore_block_async(
        PagedKvSequenceHandle handle, uint32_t logical_block) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);
    if (logical_block >= snapshot.block_table.size()) {
        return PagedKvResidencyStatus::InvalidArgument;
    }
    if (snapshot.block_table[logical_block] != PAGED_KV_COLD_BLOCK) {
        sequences_[handle.slot].blocks[logical_block].last_use = ++clock_;
        return PagedKvResidencyStatus::Ok;
    }
    BlockState & block = sequences_[handle.slot].blocks[logical_block];
    if (block.page_out_pending || block.page_in_pending) {
        return PagedKvResidencyStatus::InconsistentPoolState;
    }
    if (!block.host || !block.host_valid) {
        return PagedKvResidencyStatus::HostCopyMissing;
    }

    uint32_t physical = PAGED_KV_COLD_BLOCK;
    const PagedKvStatus page_status =
        pool_.page_in_block(handle, logical_block, physical);
    if (page_status != PagedKvStatus::Ok) return from_pool_status(page_status);

    bool queued = false;
    transfers_pending_ = true;
    block.page_in_pending = true;
    try {
        queued = transfers_.copy_in_async(
            handle, logical_block, physical, block.host,
            config_.block_bytes);
    } catch (...) {
        queued = false;
    }
    const PendingTransfer transfer{handle, logical_block, physical};
    if (!queued) {
        // The callback may have queued a prefix. Keep the physical mapping
        // quarantined until finish_transfers() proves the stream is drained.
        pending_page_in_rollbacks_.push_back(transfer);
        return PagedKvResidencyStatus::TransferFailed;
    }
    pending_page_ins_.push_back(transfer);
    block.last_use = ++clock_;
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::ensure_resident(
        PagedKvSequenceHandle handle,
        const std::vector<uint32_t> & logical_blocks) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;

    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);
    std::vector<uint32_t> cold;
    std::vector<uint8_t> seen(snapshot.block_table.size(), 0);
    // Validate the complete request before reserving any member. Otherwise a
    // later bad index leaves earlier blocks permanently ineligible as victims.
    if (std::any_of(logical_blocks.begin(), logical_blocks.end(),
                    [&](uint32_t logical) {
                        return logical >= snapshot.block_table.size();
                    })) {
        return PagedKvResidencyStatus::InvalidArgument;
    }
    for (uint32_t logical : logical_blocks) {
        if (seen[logical]) continue;
        seen[logical] = 1;
        sequences_[handle.slot].blocks[logical].reservation_pending = true;
        if (snapshot.block_table[logical] == PAGED_KV_COLD_BLOCK) {
            cold.push_back(logical);
        } else {
            sequences_[handle.slot].blocks[logical].last_use = ++clock_;
        }
    }
    const auto clear_reservations = [&] {
        for (uint32_t logical = 0; logical < seen.size(); ++logical) {
            if (seen[logical]) {
                sequences_[handle.slot].blocks[logical].reservation_pending = false;
            }
        }
    };
    const auto room = make_room(
        handle, static_cast<uint32_t>(cold.size()),
        static_cast<uint32_t>(cold.size()));
    if (room != PagedKvResidencyStatus::Ok) {
        clear_reservations();
        return room;
    }

    // Restore the requested set from one reserved capacity pool. No member can
    // become the victim of a later member in this same operation.
    for (uint32_t logical : cold) {
        const auto status = restore_block_async(
            handle, logical);
        if (status != PagedKvResidencyStatus::Ok) {
            const auto finished = finish_transfers(status);
            clear_reservations();
            return finished;
        }
    }
    const auto finished = finish_transfers(PagedKvResidencyStatus::Ok);
    clear_reservations();
    return finished;
}

PagedKvResidencyStatus PagedKvResidencyManager::evict_block(
        PagedKvSequenceHandle handle, uint32_t logical_block,
        bool allow_protected) {
    return finish_transfers(
        evict_block_async(handle, logical_block, allow_protected));
}

PagedKvResidencyStatus PagedKvResidencyManager::touch(
        PagedKvSequenceHandle handle, uint32_t logical_block) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    if (logical_block >= sequences_[handle.slot].blocks.size()) {
        return PagedKvResidencyStatus::InvalidArgument;
    }
    sequences_[handle.slot].blocks[logical_block].last_use = ++clock_;
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::set_scores(
        PagedKvSequenceHandle handle, const std::vector<float> & scores) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    SequenceState & state = sequences_[handle.slot];
    if (scores.empty()) {
        for (BlockState & block : state.blocks) block.score_valid = false;
        return PagedKvResidencyStatus::Ok;
    }
    if (scores.size() != state.blocks.size() ||
        std::any_of(scores.begin(), scores.end(),
                    [](float score) { return !std::isfinite(score); })) {
        return PagedKvResidencyStatus::InvalidArgument;
    }
    for (uint32_t i = 0; i < scores.size(); ++i) {
        state.blocks[i].score = scores[i];
        state.blocks[i].score_valid = true;
    }
    return PagedKvResidencyStatus::Ok;
}

PagedKvResidencyStatus PagedKvResidencyManager::reselect(
        PagedKvSequenceHandle handle) {
    const auto registered = validate_registered(handle);
    if (registered != PagedKvResidencyStatus::Ok) return registered;
    stats_.reselects++;

    PagedKvSequenceSnapshot snapshot;
    const PagedKvStatus pool_status = pool_.sequence(handle, snapshot);
    if (pool_status != PagedKvStatus::Ok) return from_pool_status(pool_status);
    SequenceState & state = sequences_[handle.slot];

    std::vector<uint8_t> wanted(snapshot.block_table.size(), 0);
    uint32_t wanted_count = 0;
    for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
        if (state.blocks[logical].write_pending ||
            is_protected(handle, logical)) {
            wanted[logical] = 1;
            ++wanted_count;
        }
    }
    const uint32_t target = std::max(
        wanted_count,
        std::min<uint32_t>(quota_for_slot(handle.slot),
                           snapshot.block_table.size()));

    std::vector<uint32_t> candidates;
    for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
        if (!wanted[logical]) candidates.push_back(logical);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](uint32_t a, uint32_t b) {
        const BlockState & lhs = state.blocks[a];
        const BlockState & rhs = state.blocks[b];
        if (lhs.score_valid != rhs.score_valid) return lhs.score_valid;
        if (lhs.score_valid && lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.last_use != rhs.last_use) return lhs.last_use > rhs.last_use;
        return a < b;
    });
    for (uint32_t logical : candidates) {
        if (wanted_count >= target) break;
        wanted[logical] = 1;
        ++wanted_count;
    }

    for (uint32_t logical = 0; logical < wanted.size(); ++logical) {
        if (wanted[logical]) state.blocks[logical].reservation_pending = true;
    }
    const auto clear_reservations = [&] {
        for (uint32_t logical = 0; logical < wanted.size(); ++logical) {
            if (wanted[logical]) state.blocks[logical].reservation_pending = false;
        }
    };

    // Out first, making one batch of free physical pages before recalls.
    for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
        if (!wanted[logical] &&
            snapshot.block_table[logical] != PAGED_KV_COLD_BLOCK) {
            const auto status = evict_block_async(
                handle, logical, /*allow_protected=*/false);
            if (status != PagedKvResidencyStatus::Ok) {
                const auto finished = finish_transfers(status);
                clear_reservations();
                return finished;
            }
        }
    }
    auto finished = finish_transfers(PagedKvResidencyStatus::Ok);
    if (finished != PagedKvResidencyStatus::Ok) {
        clear_reservations();
        return finished;
    }

    std::vector<uint32_t> cold_wanted;
    for (uint32_t logical = 0; logical < snapshot.block_table.size(); ++logical) {
        if (wanted[logical] && !is_resident(handle, logical)) {
            cold_wanted.push_back(logical);
        }
    }
    const auto room = make_room(
        handle, static_cast<uint32_t>(cold_wanted.size()),
        static_cast<uint32_t>(cold_wanted.size()));
    if (room != PagedKvResidencyStatus::Ok) {
        clear_reservations();
        return room;
    }
    for (uint32_t logical : cold_wanted) {
        const auto status = restore_block_async(
            handle, logical);
        if (status != PagedKvResidencyStatus::Ok) {
            finished = finish_transfers(status);
            clear_reservations();
            return finished;
        }
    }
    finished = finish_transfers(PagedKvResidencyStatus::Ok);
    clear_reservations();
    return finished;
}

PagedKvResidencyStatus PagedKvResidencyManager::rebalance() {
    while (true) {
        // The dummy requester prevents any real sequence from getting the
        // requester-swap class. A class-0 victim is necessarily over quota.
        const PagedKvSequenceHandle none{
            std::numeric_limits<uint32_t>::max(), 0};
        const Victim victim = choose_victim(none);
        if (!victim.found || victim.class_rank != 0) break;
        const auto status = evict_block_async(
            victim.handle, victim.logical_block,
            /*allow_protected=*/false);
        if (status != PagedKvResidencyStatus::Ok) {
            return finish_transfers(status);
        }
    }
    return finish_transfers(PagedKvResidencyStatus::Ok);
}

bool PagedKvResidencyManager::is_resident(
        PagedKvSequenceHandle handle, uint32_t logical_block) const {
    if (validate_registered(handle) != PagedKvResidencyStatus::Ok) return false;
    PagedKvSequenceSnapshot snapshot;
    return pool_.sequence(handle, snapshot) == PagedKvStatus::Ok &&
           logical_block < snapshot.block_table.size() &&
           snapshot.block_table[logical_block] != PAGED_KV_COLD_BLOCK;
}

PagedKvResidencyStats PagedKvResidencyManager::stats() const {
    PagedKvResidencyStats out = stats_;
    out.resident_blocks = total_resident_count();
    return out;
}

}  // namespace dflash::common
