#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <fstream>
#endif

// RAM check for managed (unified) allocations on integrated GPUs (Linux), shared
// by ggml-cuda.cu, server/src/common (gpu_page_pool, platform_io) and the unit
// tests.
//
// cudaMallocManaged/hipMallocManaged can succeed for more memory than is free:
// the pages are taken as they are mapped (on Strix Halo, without XNACK, during
// the call itself), and a shortfall ends in reclaim and the OOM killer instead
// of an allocation error. So a managed allocation must fit in the RAM that is
// available now (MemAvailable plus the GPU driver's reclaimable page pool,
// swap excluded: GPU pages pushed to swap stall the device; at most the room
// under this process's cgroup memory limits) minus a reserve for the kernel,
// the driver and the rest of the process: 1/16 of RAM, between 1 GiB and
// 8 GiB (7.8 GiB on a 125 GiB box).

static inline size_t ggml_cuda_uma_reserve_bytes(size_t total_ram) {
    constexpr size_t min_reserve = (size_t) 1 << 30;
    constexpr size_t max_reserve = (size_t) 8 << 30;
    const size_t reserve = total_ram / 16;
    return reserve < min_reserve ? min_reserve : reserve > max_reserve ? max_reserve : reserve;
}

static inline bool ggml_cuda_uma_alloc_fits(size_t size, size_t available, size_t total_ram) {
    const size_t reserve = ggml_cuda_uma_reserve_bytes(total_ram);
    return available > reserve && size <= available - reserve;
}

// Memory the GPU driver is holding for reuse (see gpu_page_pool.h). Freed GPU
// buffers in system RAM stay in the kernel's TTM page pool, which serves the
// next GPU allocation first and shrinks under memory pressure, but
// MemAvailable does not count it: after one large run it can read tens of GiB
// low. The pool is only visible to root (debugfs), so this estimates it as the
// RAM /proc/meminfo attributes to nothing, minus the GPU buffers live now
// (amdgpu sysfs) and a 1 GiB margin for pages owned by other drivers.
static inline uint64_t ggml_cuda_uma_page_pool_bytes(const char * meminfo_text, uint64_t live_gpu_host_bytes) {
    // Fields that together describe every page /proc/meminfo can attribute.
    static const char * const accounted_fields[] = {
        "MemFree", "Buffers", "Cached", "SwapCached", "AnonPages", "Slab",
        "KernelStack", "PageTables", "SecPageTables", "Percpu", "Bounce",
    };
    constexpr uint64_t other_drivers_margin = 1ULL << 30;
    if (!meminfo_text) return 0;
    uint64_t total = 0, accounted = 0, huge_pages = 0, huge_page_kb = 0, hugetlb_kb = 0;
    bool has_hugetlb = false;
    std::istringstream lines(meminfo_text);
    std::string line;
    while (std::getline(lines, line)) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = line.substr(0, colon);
        const uint64_t value = std::strtoull(line.c_str() + colon + 1, nullptr, 10);
        if (key == "MemTotal") total = value;
        else if (key == "HugePages_Total") huge_pages = value;
        else if (key == "Hugepagesize") huge_page_kb = value;
        else if (key == "Hugetlb") { hugetlb_kb = value; has_hugetlb = true; }
        else for (const char * field : accounted_fields) if (key == field) accounted += value;
    }
    // Hugetlb covers pools of every page size; older kernels only report the
    // default pool's page count.
    accounted += has_hugetlb ? hugetlb_kb : huge_pages * huge_page_kb;
    if (total <= accounted) return 0;
    const uint64_t unattributed = (total - accounted) * 1024;
    const uint64_t in_use = live_gpu_host_bytes + other_drivers_margin;
    return unattributed > in_use ? unattributed - in_use : 0;
}

// RAM a managed allocation can take now: MemAvailable plus the driver's page
// pool, which is credited only when the live GPU buffers are known
// (live_gpu_host_bytes not null), at most the room under the cgroup limits
// (cgroup_room, null without a limit). A MemAvailable of 0 is a reading like
// any other. False when MemAvailable is missing: the caller cannot tell, which
// is not the same as no RAM.
static inline bool ggml_cuda_uma_available_bytes(const char * meminfo_text, const uint64_t * live_gpu_host_bytes,
                                                 const uint64_t * cgroup_room, uint64_t * available) {
    if (!meminfo_text) return false;
    std::istringstream lines(meminfo_text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.compare(0, 13, "MemAvailable:") != 0) continue;
        char * end = nullptr;
        const uint64_t kb = std::strtoull(line.c_str() + 13, &end, 10);
        if (end == line.c_str() + 13) return false;
        *available = kb * 1024;
        if (live_gpu_host_bytes) *available += ggml_cuda_uma_page_pool_bytes(meminfo_text, *live_gpu_host_bytes);
        if (cgroup_room && *cgroup_room < *available) *available = *cgroup_room;
        return true;
    }
    return false;
}

#if defined(__linux__)
static inline bool ggml_cuda_uma_read_meminfo(std::string * text) {
    std::ifstream input("/proc/meminfo");
    if (!input) return false;
    std::stringstream buffer;
    buffer << input.rdbuf();
    *text = buffer.str();
    return !text->empty();
}

// GTT bytes in use on every amdgpu device. False when none reports it, or when
// one has the counter but it cannot be read: the pool cannot then be told
// apart from pages that are in use.
static inline bool ggml_cuda_uma_live_gpu_host_bytes(uint64_t * total) {
    *total = 0;
    bool found = false;
    DIR * dir = opendir("/sys/class/drm");
    if (!dir) return false;
    while (const dirent * entry = readdir(dir)) {
        const std::string name = entry->d_name;
        // card0, card1, ... but not the connector entries such as card0-DP-1.
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
        std::ifstream used("/sys/class/drm/" + name + "/device/mem_info_gtt_used");
        if (!used.is_open()) continue;  // not an amdgpu device
        uint64_t bytes = 0;
        if (!(used >> bytes)) { closedir(dir); return false; }
        *total += bytes;
        found = true;
    }
    closedir(dir);
    return found;
}

// Room left under the cgroup v2 memory limits from `group` (a path such as
// "/user.slice/app.scope") up to the root of the hierarchy mounted at `root`.
// False when no group on the way has a limit.
static inline bool ggml_cuda_uma_cgroup_room_bytes_at(const std::string & root, std::string group,
                                                      uint64_t * room) {
    const auto read_u64 = [](const std::string & path, uint64_t * out) {
        std::ifstream file(path);
        unsigned long long value = 0;
        if (!(file >> value)) return false; // also memory.max "max": no limit
        *out = (uint64_t) value;
        return true;
    };
    bool limited = false;
    // The walk ends at the mount root "/" (checked once): a container with
    // its own cgroup namespace sees its group as "/", its limit at the root.
    while (!group.empty()) {
        uint64_t max = 0, cur = 0;
        const std::string dir = root + (group == "/" ? std::string() : group);
        if (read_u64(dir + "/memory.max", &max)) {
            // A limit whose usage cannot be read still bounds the room.
            if (!read_u64(dir + "/memory.current", &cur)) cur = 0;
            const uint64_t left = max > cur ? max - cur : 0;
            if (!limited || left < *room) *room = left;
            limited = true;
        }
        if (group == "/") break;
        group.resize(group.rfind('/') == 0 ? 1 : group.rfind('/'));
    }
    return limited;
}

// The same for this process's group. A container sees the host's
// MemAvailable, but a managed buffer's pages are charged to its group as they
// are allocated.
static inline bool ggml_cuda_uma_cgroup_room_bytes(uint64_t * room) {
    std::string group;
    std::ifstream cgroup("/proc/self/cgroup");
    std::string line;
    while (std::getline(cgroup, line)) {
        if (line.compare(0, 3, "0::") == 0) {
            group = line.substr(3);
            while (!group.empty() && (group.back() == '\n' || group.back() == '\r')) group.pop_back();
            break;
        }
    }
    return ggml_cuda_uma_cgroup_room_bytes_at("/sys/fs/cgroup", group, room);
}
#endif // defined(__linux__)
