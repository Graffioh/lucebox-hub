
#define GENERATE_UNIT_TEST_MAIN
#include "CppUnitTestFramework.hpp"
#include "common/concurrency/paged_kv_residency.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dflash::common;

namespace {
struct PagedKvResidencyFixture {};

PagedKvSequenceHandle acquire(PagedKvPool & pool, uint64_t request) {
    PagedKvSequenceHandle handle;
    if (pool.acquire(request, handle) != PagedKvStatus::Ok) {
        throw std::runtime_error("acquire failed");
    }
    return handle;
}

PagedKvSequenceSnapshot snapshot(PagedKvPool & pool,
                                 PagedKvSequenceHandle handle) {
    PagedKvSequenceSnapshot out;
    if (pool.sequence(handle, out) != PagedKvStatus::Ok) {
        throw std::runtime_error("sequence failed");
    }
    return out;
}

struct MockTransfers {
    struct Pending {
        std::function<void()> apply;
    };

    explicit MockTransfers(uint32_t blocks, size_t bytes)
        : block_bytes(bytes), device((size_t)blocks * bytes, 0) {}

    PagedKvResidencyTransferOps callbacks() {
        return {
            [this](size_t bytes) -> void * {
                if (fail_alloc || bytes != block_bytes) return nullptr;
                allocations++;
                return new uint8_t[bytes];
            },
            [this](void * ptr) {
                frees++;
                delete[] static_cast<uint8_t *>(ptr);
            },
            [this](PagedKvSequenceHandle, uint32_t, uint32_t physical,
                   void * host, size_t bytes) {
                if (fail_copy_out || bytes != block_bytes) return false;
                pending.push_back({[this, physical, host, bytes] {
                    std::memcpy(host, &device[(size_t)physical * block_bytes], bytes);
                }});
                return !fail_copy_out_after_queue;
            },
            [this](PagedKvSequenceHandle, uint32_t, uint32_t physical,
                   const void * host, size_t bytes) {
                if (fail_copy_in || bytes != block_bytes) return false;
                pending.push_back({[this, physical, host, bytes] {
                    std::memcpy(&device[(size_t)physical * block_bytes], host, bytes);
                }});
                return !fail_copy_in_after_queue;
            },
            [this] {
                syncs++;
                if (fail_sync) return false;
                for (Pending & op : pending) op.apply();
                pending.clear();
                return true;
            },
        };
    }

    void fill(uint32_t physical, uint8_t value) {
        std::fill_n(&device[(size_t)physical * block_bytes], block_bytes, value);
    }

    bool block_is(uint32_t physical, uint8_t value) const {
        const auto begin = device.begin() + (size_t)physical * block_bytes;
        return std::all_of(begin, begin + block_bytes,
                           [value](uint8_t byte) { return byte == value; });
    }

    size_t block_bytes;
    std::vector<uint8_t> device;
    std::vector<Pending> pending;
    int allocations = 0;
    int frees = 0;
    int syncs = 0;
    bool fail_alloc = false;
    bool fail_copy_out = false;
    bool fail_copy_in = false;
    bool fail_copy_out_after_queue = false;
    bool fail_copy_in_after_queue = false;
    bool fail_sync = false;
};

PagedKvResidencyConfig config(size_t block_bytes, uint32_t budget,
                              uint32_t sink = 0, uint32_t tail = 0) {
    return {block_bytes, budget, sink, tail};
}

}  // namespace

TEST_CASE(PagedKvResidencyFixture, page_roundtrip_is_bit_exact_and_barriered) {
    PagedKvPool pool(/*physical blocks=*/3, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(3, 32);
    PagedKvResidencyManager pager(pool, config(32, 3), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 12));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    const auto table = snapshot(pool, handle).block_table;
    io.fill(table[0], 0x11);
    io.fill(table[1], 0x22);
    io.fill(table[2], 0x33);

    CHECK(pager.evict_block(handle, 1) == PagedKvResidencyStatus::Ok);
    CHECK(io.syncs == 1);
    CHECK(snapshot(pool, handle).block_table[1] == PAGED_KV_COLD_BLOCK);
    io.fill(table[1], 0xEE);  // recycled device bytes must not affect backing

    CHECK(pager.ensure_resident(handle, {1}) == PagedKvResidencyStatus::Ok);
    CHECK(io.syncs == 2);
    const uint32_t restored = snapshot(pool, handle).block_table[1];
    CHECK(io.block_is(restored, 0x22));
    CHECK(pager.stats().page_outs == 1);
    CHECK(pager.stats().page_ins == 1);
    CHECK(pager.stats().resident_blocks == 3);
    CHECK(pager.stats().host_bytes == 32);
    CHECK(pager.stats().moved_bytes == 64);
}

TEST_CASE(PagedKvResidencyFixture, fair_share_reclaims_borrowed_pages) {
    PagedKvPool pool(/*physical blocks=*/6, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(6, 16);
    PagedKvResidencyManager pager(pool, config(16, 6), io.callbacks());
    const auto first = acquire(pool, 1);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 20));  // borrows five of six pages while alone
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.fair_quota(first) == 6);

    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.fair_quota(first) == 3);
    CHECK(pager.fair_quota(second) == 3);
    CHECK(pager.append(second, 12));
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.stats().page_outs == 2);
    CHECK(pager.stats().resident_blocks == 6);

    uint32_t first_resident = 0;
    uint32_t second_resident = 0;
    CHECK(pool.resident_block_count(first, first_resident) == PagedKvStatus::Ok);
    CHECK(pool.resident_block_count(second, second_resident) == PagedKvStatus::Ok);
    CHECK(first_resident == 3);
    CHECK(second_resident == 3);
}

TEST_CASE(PagedKvResidencyFixture, sink_and_tail_are_never_auto_evicted) {
    PagedKvPool pool(/*physical blocks=*/5, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(5, 16);
    PagedKvResidencyManager pager(
        pool, config(16, 5, /*sink=*/1, /*tail=*/1), io.callbacks());
    const auto first = acquire(pool, 1);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 16));  // logical blocks 0..3
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(second, 8));  // needs two pages; only one was free
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);

    const auto first_table = snapshot(pool, first).block_table;
    CHECK(first_table[0] != PAGED_KV_COLD_BLOCK);  // sink
    CHECK(first_table[3] != PAGED_KV_COLD_BLOCK);  // tail
    CHECK(first_table[1] == PAGED_KV_COLD_BLOCK ||
          first_table[2] == PAGED_KV_COLD_BLOCK);
    CHECK(pager.evict_block(first, 0) ==
          PagedKvResidencyStatus::NoEvictableBlock);
}

TEST_CASE(PagedKvResidencyFixture, append_restores_cold_partial_head) {
    PagedKvPool pool(/*physical blocks=*/2, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(2, 16);
    PagedKvResidencyManager pager(pool, config(16, 2), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 2));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    const uint32_t head = snapshot(pool, handle).block_table[0];
    io.fill(head, 0xA5);
    CHECK(pager.evict_block(handle, 0) == PagedKvResidencyStatus::Ok);
    io.fill(head, 0x00);

    const auto append = pager.append(handle, 1);
    CHECK(append);
    CHECK(append.pool_result.remapped_cold_blocks.empty());
    CHECK(append.pool_result.write_slots[0].logical_position == 2);
    CHECK(io.block_is(append.pool_result.write_slots[0].physical_block, 0xA5));
    CHECK(pager.stats().page_ins == 1);
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
}

TEST_CASE(PagedKvResidencyFixture, scores_drive_reselection) {
    PagedKvPool pool(/*physical blocks=*/3, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(3, 16);
    PagedKvResidencyManager pager(pool, config(16, 3), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 12));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(handle, 2) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(handle, 0) == PagedKvResidencyStatus::Ok);
    CHECK(pager.stats().resident_blocks == 1);
    const auto peer = acquire(pool, 2);
    CHECK(pager.register_sequence(peer) == PagedKvResidencyStatus::Ok);
    CHECK(pager.fair_quota(handle) == 2);
    CHECK(pager.set_scores(handle, {1.0f, 2.0f, 9.0f}) ==
          PagedKvResidencyStatus::Ok);
    CHECK(pager.reselect(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.stats().reselects == 1);
    CHECK(pager.is_resident(handle, 2));
    CHECK(pager.is_resident(handle, 1));
    CHECK(!pager.is_resident(handle, 0));
}

TEST_CASE(PagedKvResidencyFixture, allocation_and_copy_failures_are_explicit) {
    PagedKvPool pool(/*physical blocks=*/2, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(2, 16);
    PagedKvResidencyManager pager(pool, config(16, 2), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 8));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);

    io.fail_alloc = true;
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::HostAllocationFailed);
    CHECK(pager.is_resident(handle, 0));
    io.fail_alloc = false;
    io.fail_copy_out = true;
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(pager.is_resident(handle, 0));
}

TEST_CASE(PagedKvResidencyFixture,
          failed_copy_out_prefix_is_barriered_before_retry) {
    PagedKvPool pool(/*physical blocks=*/1, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(1, 16);
    PagedKvResidencyManager pager(pool, config(16, 1), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 4));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);

    io.fail_copy_out_after_queue = true;
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(io.pending.empty());
    CHECK(pager.is_resident(handle, 0));
    CHECK(pager.stats().page_outs == 0);

    io.fail_copy_out_after_queue = false;
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);
    CHECK(!pager.is_resident(handle, 0));
}

TEST_CASE(PagedKvResidencyFixture,
          rejected_append_remap_is_rolled_back_after_barrier) {
    PagedKvPool pool(/*physical blocks=*/1, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(1, 16);
    PagedKvResidencyManager pager(pool, config(16, 1), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 2));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);

    io.fail_copy_in = true;
    const auto append = pool.append(handle, 1);
    CHECK(append.status == PagedKvStatus::Ok);
    CHECK(append.remapped_cold_blocks.size() == 1);
    CHECK(pager.observe_append(handle, append) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(snapshot(pool, handle).block_table[0] == PAGED_KV_COLD_BLOCK);
    CHECK(pool.free_block_count() == 1);
    CHECK(pager.stats().page_ins == 0);
}

TEST_CASE(PagedKvResidencyFixture,
          failed_copy_in_prefix_quarantines_mapping_until_barrier) {
    PagedKvPool pool(/*physical blocks=*/1, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(1, 16);
    PagedKvResidencyManager pager(pool, config(16, 1), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 4));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);

    io.fail_copy_in_after_queue = true;
    io.fail_sync = true;
    CHECK(pager.ensure_resident(handle, {0}) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(snapshot(pool, handle).block_table[0] != PAGED_KV_COLD_BLOCK);
    CHECK(pool.free_block_count() == 0);
    CHECK(io.pending.size() == 1);

    io.fail_sync = false;
    CHECK(pager.synchronize_before_read() == PagedKvResidencyStatus::Ok);
    CHECK(snapshot(pool, handle).block_table[0] == PAGED_KV_COLD_BLOCK);
    CHECK(pool.free_block_count() == 1);
    CHECK(pager.stats().page_ins == 0);
}

TEST_CASE(PagedKvResidencyFixture, forget_frees_host_backing_and_stale_is_rejected) {
    PagedKvPool pool(/*physical blocks=*/2, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(2, 16);
    PagedKvResidencyManager pager(pool, config(16, 2), io.callbacks());
    const auto old_handle = acquire(pool, 1);
    CHECK(pager.register_sequence(old_handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(old_handle, 8));
    CHECK(pager.commit_pending_writes(old_handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(old_handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);
    CHECK(io.allocations == 1);
    CHECK(pager.forget_sequence(old_handle) == PagedKvResidencyStatus::Ok);
    CHECK(io.frees == 1);
    CHECK(pool.release(old_handle) == PagedKvStatus::Ok);

    const auto replacement = acquire(pool, 2);
    CHECK(replacement.generation != old_handle.generation);
    CHECK(pager.register_sequence(replacement) == PagedKvResidencyStatus::Ok);
    CHECK(pager.touch(old_handle, 0) == PagedKvResidencyStatus::StaleHandle);
}

TEST_CASE(PagedKvResidencyFixture,
          forget_retries_failed_barrier_before_releasing_backing) {
    PagedKvPool pool(/*physical blocks=*/1, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(1, 16);
    PagedKvResidencyManager pager(pool, config(16, 1), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 4));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);

    io.fail_sync = true;
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(pool.free_block_count() == 0);
    CHECK(io.frees == 0);
    CHECK(pager.forget_sequence(handle) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(pool.free_block_count() == 0);
    CHECK(io.frees == 0);

    io.fail_sync = false;
    CHECK(pager.forget_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(io.frees == 1);
    CHECK(pool.release(handle) == PagedKvStatus::Ok);
}

TEST_CASE(PagedKvResidencyFixture,
          invalid_restore_request_does_not_leak_reservations) {
    PagedKvPool pool(/*physical blocks=*/1, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(1, 16);
    PagedKvResidencyManager pager(pool, config(16, 1), io.callbacks());
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 4));
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);

    CHECK(pager.ensure_resident(first, {0, 1}) ==
          PagedKvResidencyStatus::InvalidArgument);

    CHECK(pager.append(second, 4));
    CHECK(snapshot(pool, first).block_table[0] == PAGED_KV_COLD_BLOCK);
}

TEST_CASE(PagedKvResidencyFixture,
          staged_writes_are_never_recycled_across_slots) {
    PagedKvPool pool(/*physical blocks=*/3, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(3, 16);
    PagedKvResidencyManager pager(pool, config(16, 3), io.callbacks());
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);

    CHECK(pager.append(first, 8));
    CHECK(pager.append(second, 4));
    const auto first_staged = snapshot(pool, first).block_table;
    const auto second_staged = snapshot(pool, second).block_table;
    CHECK(first_staged.size() == 2);
    CHECK(second_staged.size() == 1);

    const auto blocked = pager.append(first, 4);
    CHECK(blocked.status == PagedKvResidencyStatus::NoEvictableBlock);
    CHECK(pager.stats().page_outs == 0);
    CHECK(pager.evict_block(second, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::NoEvictableBlock);
    CHECK(pager.reselect(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.stats().page_outs == 0);
    CHECK(snapshot(pool, first).block_table == first_staged);
    CHECK(snapshot(pool, second).block_table == second_staged);

    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    const auto grown = pager.append(first, 4);
    CHECK(grown);
    CHECK(pager.stats().page_outs == 1);
    CHECK(snapshot(pool, second).block_table == second_staged);
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
}

TEST_CASE(PagedKvResidencyFixture,
          cross_slot_eviction_is_visible_in_full_victim_snapshot) {
    PagedKvPool pool(/*physical blocks=*/4, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(4, 16);
    PagedKvResidencyManager pager(pool, config(16, 4), io.callbacks());
    const auto first = acquire(pool, 1);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 12));
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    const auto before = snapshot(pool, first).block_table;

    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(second, 8));
    const auto victim = snapshot(pool, first).block_table;
    const auto requester = snapshot(pool, second).block_table;
    CHECK(victim.size() == before.size());
    CHECK(requester.size() == 2);

    size_t cold = victim.size();
    for (size_t logical = 0; logical < victim.size(); ++logical) {
        if (victim[logical] == PAGED_KV_COLD_BLOCK) {
            CHECK(cold == victim.size());
            cold = logical;
        }
    }
    CHECK(cold < victim.size());
    CHECK(std::find(requester.begin(), requester.end(), before[cold]) !=
          requester.end());
    CHECK(pager.stats().page_outs == 1);
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);
}

TEST_CASE(PagedKvResidencyFixture,
          sixteen_slots_progress_with_adaptive_fair_quotas) {
    constexpr uint32_t kSlots = 16;
    constexpr uint32_t kPoolBlocks = 96;
    PagedKvPool pool(kPoolBlocks, kSlots, /*block size=*/4);
    MockTransfers io(kPoolBlocks, 16);
    PagedKvResidencyManager pager(
        pool, config(16, kPoolBlocks, /*sink=*/1, /*tail=*/4),
        io.callbacks());

    std::vector<PagedKvSequenceHandle> handles;
    handles.reserve(kSlots);
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
        handles.push_back(acquire(pool, slot + 1));
        CHECK(pager.register_sequence(handles.back()) ==
              PagedKvResidencyStatus::Ok);
    }
    for (const auto handle : handles) {
        CHECK(pager.append(handle, 4));
        CHECK(pager.commit_pending_writes(handle) ==
              PagedKvResidencyStatus::Ok);
    }
    for (const auto handle : handles) {
        CHECK(pager.append(handle, 28));  // eight logical blocks total
        CHECK(pager.commit_pending_writes(handle) ==
              PagedKvResidencyStatus::Ok);
    }

    CHECK(pager.stats().resident_blocks == kPoolBlocks);
    CHECK(pager.stats().page_outs == kSlots * 2);
    for (const auto handle : handles) {
        CHECK(pager.fair_quota(handle) == kPoolBlocks / kSlots);
        const auto table = snapshot(pool, handle).block_table;
        CHECK(table.size() == 8);
        CHECK(table[0] != PAGED_KV_COLD_BLOCK);
        for (size_t logical = 4; logical < 8; ++logical) {
            CHECK(table[logical] != PAGED_KV_COLD_BLOCK);
        }
    }
}

TEST_CASE(PagedKvResidencyFixture,
          failed_copy_out_barrier_retains_the_device_mapping) {
    PagedKvPool pool(/*physical blocks=*/2, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(2, 16);
    PagedKvResidencyManager pager(pool, config(16, 2), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 8));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    const auto before = snapshot(pool, handle).block_table;

    io.fail_sync = true;
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(snapshot(pool, handle).block_table == before);
    CHECK(pool.free_block_count() == 0);
    CHECK(pager.stats().page_outs == 0);
}

TEST_CASE(PagedKvResidencyFixture,
          failed_copy_in_barrier_quarantines_mapping_until_stream_drains) {
    PagedKvPool pool(/*physical blocks=*/1, /*sequences=*/1, /*block size=*/4);
    MockTransfers io(1, 16);
    PagedKvResidencyManager pager(pool, config(16, 1), io.callbacks());
    const auto handle = acquire(pool, 1);
    CHECK(pager.register_sequence(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(handle, 4));
    CHECK(pager.commit_pending_writes(handle) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);

    io.fail_sync = true;
    CHECK(pager.ensure_resident(handle, {0}) ==
          PagedKvResidencyStatus::TransferFailed);
    const uint32_t quarantined = snapshot(pool, handle).block_table[0];
    CHECK(quarantined != PAGED_KV_COLD_BLOCK);
    CHECK(pool.free_block_count() == 0);
    CHECK(pager.stats().page_ins == 0);

    CHECK(pager.evict_block(handle, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::TransferFailed);
    CHECK(io.pending.size() == 1);

    io.fail_sync = false;
    CHECK(pager.synchronize_before_read() == PagedKvResidencyStatus::Ok);
    CHECK(snapshot(pool, handle).block_table[0] == quarantined);
    CHECK(pool.free_block_count() == 0);
    CHECK(pager.stats().page_ins == 1);
}

TEST_CASE(PagedKvResidencyFixture,
          rebalance_evicts_only_blocks_above_fair_quota) {
    PagedKvPool pool(/*physical blocks=*/6, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(6, 16);
    PagedKvResidencyManager pager(pool, config(16, 6), io.callbacks());
    const auto borrower = acquire(pool, 1);
    CHECK(pager.register_sequence(borrower) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(borrower, 20));
    CHECK(pager.commit_pending_writes(borrower) ==
          PagedKvResidencyStatus::Ok);

    const auto peer = acquire(pool, 2);
    CHECK(pager.register_sequence(peer) == PagedKvResidencyStatus::Ok);
    CHECK(pager.fair_quota(borrower) == 3);
    CHECK(pager.rebalance() == PagedKvResidencyStatus::Ok);

    uint32_t borrower_resident = 0;
    CHECK(pool.resident_block_count(borrower, borrower_resident) ==
          PagedKvStatus::Ok);
    CHECK(borrower_resident == 3);
    CHECK(pager.stats().page_outs == 2);
}

TEST_CASE(PagedKvResidencyFixture,
          append_reserves_partial_head_and_new_block_together) {
    PagedKvPool pool(/*physical blocks=*/2, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(2, 16);
    PagedKvResidencyManager pager(pool, config(16, 2), io.callbacks());
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 2));
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(first, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);
    CHECK(pager.append(second, 4));
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);

    const auto grown = pager.append(first, 3);
    CHECK(grown);
    const auto table = snapshot(pool, first).block_table;
    CHECK(table.size() == 2);
    CHECK(table[0] != PAGED_KV_COLD_BLOCK);
    CHECK(table[1] != PAGED_KV_COLD_BLOCK);
    CHECK(snapshot(pool, second).block_table[0] == PAGED_KV_COLD_BLOCK);
}

TEST_CASE(PagedKvResidencyFixture,
          resident_partial_head_stays_within_budget_during_growth) {
    PagedKvPool pool(/*physical blocks=*/2, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(2, 16);
    PagedKvResidencyManager pager(
        pool, config(16, /*budget=*/2, /*sink=*/0, /*tail=*/0),
        io.callbacks());
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 2));
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    const uint32_t append_head = snapshot(pool, first).block_table[0];
    CHECK(pager.append(second, 4));
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);

    const auto grown = pager.append(first, 3);
    CHECK(grown);
    const auto table = snapshot(pool, first).block_table;
    CHECK(table.size() == 2);
    CHECK(table[0] == append_head);
    CHECK(table[1] != PAGED_KV_COLD_BLOCK);
    CHECK(snapshot(pool, second).block_table[0] == PAGED_KV_COLD_BLOCK);
    CHECK(pager.stats().resident_blocks == 2);
}
TEST_CASE(PagedKvResidencyFixture,
          requested_restore_set_is_protected_as_one_batch) {
    PagedKvPool pool(/*physical blocks=*/3, /*sequences=*/2, /*block size=*/4);
    MockTransfers io(3, 16);
    PagedKvResidencyManager pager(pool, config(16, 3), io.callbacks());
    const auto first = acquire(pool, 1);
    const auto second = acquire(pool, 2);
    CHECK(pager.register_sequence(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.register_sequence(second) == PagedKvResidencyStatus::Ok);
    CHECK(pager.append(first, 8));
    CHECK(pager.commit_pending_writes(first) == PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(first, 0, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);
    CHECK(pager.evict_block(first, 1, /*allow_protected=*/true) ==
          PagedKvResidencyStatus::Ok);
    CHECK(pager.append(second, 8));
    CHECK(pager.commit_pending_writes(second) == PagedKvResidencyStatus::Ok);

    CHECK(pager.ensure_resident(first, {0, 1}) ==
          PagedKvResidencyStatus::Ok);
    CHECK(pager.is_resident(first, 0));
    CHECK(pager.is_resident(first, 1));
}
