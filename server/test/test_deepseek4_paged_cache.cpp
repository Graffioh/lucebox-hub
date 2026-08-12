#include "deepseek4/deepseek4_paged_cache.h"
#include "host_check.h"
#include <cstdio>
#include <limits>
using namespace dflash::common;
static int g_checks = 0;
int main() {
    DeepSeek4PagedCachePlan p, twice;
    CHECK(plan_deepseek4_paged_cache(512, 128, 3, 4096, 40, {0, 4, 128}, p));
    CHECK(p.max_blocks_per_sequence == 32 && p.physical_rows[0] == 0);
    CHECK(p.physical_rows[1] == 1280 && p.physical_rows[2] == 40);
    CHECK(p.raw_bytes == uint64_t(3) * 512 * 128 * 3 * 2);
    CHECK(p.metadata_bytes == uint64_t(32 * 3 + 3 + 3) * 4);
    CHECK(plan_deepseek4_paged_cache(512, 128, 6, 4096, 40, {0, 4, 128}, twice));
    // Paged rows are shared; only raw rings, metadata, and compressor state scale by slots.
    CHECK(twice.compressed_bytes == p.compressed_bytes);
    CHECK(twice.raw_bytes == p.raw_bytes * 2 && twice.state_bytes == p.state_bytes * 2);
    DeepSeek4PagedCachePlan sixteen;
    CHECK(plan_deepseek4_paged_cache(512, 128, 16, 4096, 40,
                                     {0, 4, 128}, sixteen));
    CHECK(sixteen.slots == 16 && sixteen.max_blocks_per_sequence == 32);
    CHECK(sixteen.compressed_bytes == p.compressed_bytes);
    CHECK(sixteen.raw_bytes == p.raw_bytes / 3 * 16);
    CHECK(!plan_deepseek4_paged_cache(512, 128, 1, 4096, 40, {4, 16}, twice));
    CHECK(!plan_deepseek4_paged_cache(512, 128, 1, 4096,
          std::numeric_limits<uint32_t>::max(), {4}, twice));

    const int32_t slots[] = {2, 5, -1};
    const int64_t positions[] = {3, 259, 999};
    const int32_t tables[] = {4, 3, 2, 6, 1, 7, 0, 0, 0};
    std::vector<DeepSeek4GatheredLaneRows> rows;
    CHECK(prepare_deepseek4_gathered_lane_rows(
        slots, positions, 3, tables, 3, 8, 4, rows));
    CHECK(rows.size() == 3);
    CHECK(rows[0].raw_history == std::vector<int64_t>({256, 257, 258}));
    CHECK(rows[0].raw_scatter == 259);
    CHECK(rows[0].compressed_emitted && rows[0].compressed_scatter == 4 * 32);
    CHECK(rows[0].compressed_history.empty());
    CHECK(rows[1].raw_history.size() == 127);
    CHECK(rows[1].raw_history.front() == 5 * 128 + 4);
    CHECK(rows[1].raw_history.back() == 5 * 128 + 2);
    CHECK(rows[1].compressed_history.size() == 64);
    CHECK(rows[1].compressed_history.front() == 6 * 32);
    CHECK(rows[1].compressed_history[31] == 6 * 32 + 31);
    CHECK(rows[1].compressed_history[32] == 1 * 32);
    CHECK(rows[1].compressed_history.back() == 1 * 32 + 31);
    CHECK(rows[1].compressed_emitted && rows[1].compressed_scatter == 7 * 32);
    CHECK(rows[2].raw_history.empty() && rows[2].compressed_history.empty());
    CHECK(rows[2].raw_scatter == -1 && rows[2].compressed_scatter == -1);
    CHECK(rows[2].position == 0);

    std::vector<int32_t> sixteen_slots(16);
    std::vector<int64_t> sixteen_positions(16, 0);
    std::vector<int32_t> sixteen_tables(16);
    for (int i = 0; i < 16; ++i) {
        sixteen_slots[(size_t) i] = i;
        sixteen_tables[(size_t) i] = i;
    }
    CHECK(prepare_deepseek4_gathered_lane_rows(
        sixteen_slots.data(), sixteen_positions.data(), 16,
        sixteen_tables.data(), 1, 16, 4, rows));
    CHECK(rows.size() == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(rows[(size_t) i].slot == i);
        CHECK(rows[(size_t) i].raw_history.empty());
        CHECK(rows[(size_t) i].raw_scatter == int64_t(i * 128));
    }

    const int32_t boundary_slot[] = {1};
    const int32_t boundary_table[] = {0, 1};
    for (int64_t pos : {127LL, 128LL, 129LL}) {
        CHECK(prepare_deepseek4_gathered_lane_rows(
            boundary_slot, &pos, 1, boundary_table, 2, 2, 0, rows));
        CHECK(rows[0].raw_history.size() == (pos == 127 ? 127u : 127u));
        CHECK(rows[0].raw_history.front() == 128 + (pos == 127 ? 0 : pos - 127));
        CHECK(rows[0].raw_history.back() == 128 + ((pos - 1) % 128));
    }

    const int64_t ratio128_pos[] = {255};
    CHECK(prepare_deepseek4_gathered_lane_rows(
        slots, ratio128_pos, 1, tables, 3, 8, 128, rows));
    CHECK(rows[0].compressed_history.size() == 1);
    CHECK(rows[0].compressed_history[0] == 4);
    CHECK(rows[0].compressed_emitted && rows[0].compressed_scatter == 3);
    std::printf("OK test_deepseek4_paged_cache (%d checks)\n", g_checks);
    return 0;
}
