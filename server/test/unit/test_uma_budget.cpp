#include "CppUnitTestFramework.hpp"
#include "../../deps/llama.cpp/ggml/src/ggml-cuda/uma-budget.h"

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <filesystem>
#include <fstream>
#include <unistd.h>
#endif

namespace {
struct UmaBudget {};
constexpr size_t GiB = size_t(1) << 30;
// A 125 GiB integrated-GPU box, as /proc/meminfo reports MemTotal.
constexpr size_t kStrixRam = 125 * GiB + GiB / 10;
}

// The overcommit this check exists for: a 103.9 GiB managed buffer while
// MemAvailable says ~92 GiB. It must fail up front instead of ending in the
// OOM killer.
TEST_CASE(UmaBudget, rejects_allocation_larger_than_available_ram) {
    const size_t size = 103 * GiB + 9 * GiB / 10;
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(size, 92 * GiB, kStrixRam));
}

// The case the managed path is for: a large model on a box whose RAM is free.
TEST_CASE(UmaBudget, admits_large_model_when_ram_is_free) {
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(86 * GiB, 118 * GiB, kStrixRam));
}

// The reserve kept free is 1/16 of RAM, at least 1 GiB and at most 8 GiB:
// 1, 4 and 8 GiB on 8, 64 and 256 GiB boxes with half their RAM available.
TEST_CASE(UmaBudget, keeps_reserve_free) {
    struct Box { size_t ram; size_t reserve; };
    for (const Box box : {Box{8 * GiB, 1 * GiB}, Box{64 * GiB, 4 * GiB}, Box{256 * GiB, 8 * GiB}}) {
        const size_t available = box.ram / 2;
        CHECK_TRUE(ggml_cuda_uma_alloc_fits(available - box.reserve, available, box.ram));
        CHECK_FALSE(ggml_cuda_uma_alloc_fits(available - box.reserve + 1, available, box.ram));
    }
}

// Less available than the reserve, including none at all, rejects any
// allocation without the subtraction wrapping around; a huge size does not
// overflow either.
TEST_CASE(UmaBudget, near_empty_ram_and_huge_sizes_reject) {
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(1, GiB / 4, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(1, 0, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(SIZE_MAX, 100 * GiB, kStrixRam));
}

// After a large run the GPU driver keeps the freed pages in its pool, which
// MemAvailable leaves out: 58 GiB available, 61 GiB held by the driver. An
// 86 GiB model still fits, because the kernel hands the pool back under
// pressure.
TEST_CASE(UmaBudget, counts_the_driver_page_pool_after_a_run) {
    // 125 GiB total; 64 GiB attributed (free, cache, anon, slab), so 61 GiB
    // belong to nothing /proc/meminfo lists.
    const char * meminfo =
        "MemTotal:       131072000 kB\n"
        "MemFree:         52428800 kB\n"
        "MemAvailable:    60817408 kB\n"
        "Cached:           8388608 kB\n"
        "AnonPages:        5242880 kB\n"
        "Slab:             1048576 kB\n"
        "HugePages_Total:        0\n"
        "Hugepagesize:        2048 kB\n";
    const uint64_t no_live_gpu_buffers = 0;
    uint64_t available = 0;
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, &no_live_gpu_buffers, nullptr, &available));
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(86 * GiB, (size_t) available, kStrixRam));

    // Without knowing which GPU buffers are live, the pool cannot be told
    // apart from them and is not counted.
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, nullptr, nullptr, &available));
    CHECK_TRUE(available == 60817408ull * 1024);
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(86 * GiB, (size_t) available, kStrixRam));
}

// A missing MemAvailable is unknown (the caller keeps the old behaviour and
// admits); a MemAvailable of zero is a reading, and with no driver pool to
// add, nothing fits in it.
TEST_CASE(UmaBudget, unknown_ram_differs_from_no_ram) {
    uint64_t available = 1;
    CHECK_FALSE(ggml_cuda_uma_available_bytes("MemTotal: 131072000 kB\n", nullptr, nullptr, &available));
    CHECK_FALSE(ggml_cuda_uma_available_bytes(nullptr, nullptr, nullptr, &available));
    CHECK_TRUE(ggml_cuda_uma_available_bytes("MemTotal: 131072000 kB\nMemAvailable: 0 kB\n", nullptr, nullptr, &available));
    CHECK_TRUE(available == 0);
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(GiB, (size_t) available, kStrixRam));
}

// A container sees the host's MemAvailable, but the pages are charged to its
// cgroup as they are mapped: a 60 GiB managed buffer in a 4 GiB cgroup on a
// free 125 GiB box was OOM-killed during the allocation. The room under the
// limit bounds the budget.
TEST_CASE(UmaBudget, cgroup_limit_bounds_the_budget) {
    const char * meminfo =
        "MemTotal:       131072000 kB\n"
        "MemFree:        110000000 kB\n"
        "MemAvailable:   123731968 kB\n";
    const uint64_t no_live_gpu_buffers = 0;
    const uint64_t room = 4 * (uint64_t) GiB;
    uint64_t available = 0;
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, &no_live_gpu_buffers, &room, &available));
    CHECK_TRUE(available == room);
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(60 * GiB, (size_t) available, kStrixRam));

    const uint64_t roomy = 64 * (uint64_t) GiB;
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, &no_live_gpu_buffers, &roomy, &available));
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(40 * GiB, (size_t) available, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(60 * GiB, (size_t) available, kStrixRam));
}

#if defined(__linux__)
// The walk from the process's group to the root takes the smallest room. A
// group whose memory.current cannot be read keeps its limit as the room
// instead of dropping it: there the child's 4 GiB limit binds, not the
// parent's 6 GiB left.
TEST_CASE(UmaBudget, cgroup_limit_without_usage_still_bounds_the_room) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("luce_test_cgroup_" + std::to_string((long long) getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "app" / "job");
    const auto write = [](const fs::path & path, const char * text) { std::ofstream(path) << text; };
    write(root / "memory.max", "max\n");
    write(root / "app" / "memory.max", "8589934592\n");      // 8 GiB
    write(root / "app" / "memory.current", "2147483648\n");  // 2 GiB used
    write(root / "app" / "job" / "memory.max", "4294967296\n");  // 4 GiB, no memory.current

    uint64_t room = 0;
    CHECK_TRUE(ggml_cuda_uma_cgroup_room_bytes_at(root.string(), "/app/job", &room));
    CHECK_TRUE(room == 4 * (uint64_t) GiB);
    fs::remove_all(root);
}
#endif // defined(__linux__)
