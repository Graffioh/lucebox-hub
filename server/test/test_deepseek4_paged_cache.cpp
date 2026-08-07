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
    CHECK(!plan_deepseek4_paged_cache(512, 128, 1, 4096, 40, {4, 16}, twice));
    CHECK(!plan_deepseek4_paged_cache(512, 128, 1, 4096,
          std::numeric_limits<uint32_t>::max(), {4}, twice));
    std::printf("OK test_deepseek4_paged_cache (%d checks)\n", g_checks);
    return 0;
}
