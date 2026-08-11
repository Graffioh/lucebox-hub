#include "deepseek4/deepseek4_page_layout.h"
#include "host_check.h"

#include <cstdint>
#include <cstdio>
#include <limits>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    CHECK(ds4_raw_ring_row(0) == 0);
    CHECK(ds4_raw_ring_row(127) == 127);
    CHECK(ds4_raw_ring_row(128) == 0);
    CHECK(ds4_raw_ring_row(255) == 127);

    uint64_t row = 999;
    bool emitted = true;
    CHECK(ds4_compressed_page_row(3, 7, 4, row, emitted));
    CHECK(emitted && row == 7 * 32);
    CHECK(ds4_compressed_page_row(4, 7, 4, row, emitted));
    CHECK(!emitted && row == 7 * 32);
    CHECK(ds4_compressed_page_row(127, 7, 4, row, emitted));
    CHECK(emitted && row == 7 * 32 + 31);
    CHECK(ds4_compressed_page_row(128, 42, 4, row, emitted));
    CHECK(!emitted);
    CHECK(ds4_compressed_page_row(131, 42, 4, row, emitted));
    CHECK(emitted && row == 42 * 32);
    CHECK(ds4_compressed_page_row(255, 3, 4, row, emitted));
    CHECK(emitted && row == 3 * 32 + 31);

    CHECK(ds4_compressed_page_row(127, 91, 128, row, emitted));
    CHECK(emitted && row == 91);
    CHECK(ds4_compressed_page_row(128, 2, 128, row, emitted));
    CHECK(!emitted);
    CHECK(ds4_compressed_page_row(255, 2, 128, row, emitted));
    CHECK(emitted && row == 2);

    uint64_t capacity = 0;
    CHECK(ds4_compressed_page_capacity(5, 4, capacity) && capacity == 160);
    CHECK(ds4_compressed_page_capacity(5, 128, capacity) && capacity == 5);
    CHECK(!ds4_compressed_page_capacity(5, 0, capacity));
    CHECK(!ds4_compressed_page_capacity(5, 16, capacity));
    CHECK(!ds4_compressed_page_capacity(
        std::numeric_limits<uint64_t>::max(), 4, capacity));
    CHECK(!ds4_compressed_page_row(3,
        std::numeric_limits<uint64_t>::max(), 4, row, emitted));
    CHECK(!ds4_compressed_page_row(3, 0, 16, row, emitted));

    std::printf("OK test_deepseek4_page_layout (%d checks)\n", g_checks);
    return 0;
}
