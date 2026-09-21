#include "gpu_page_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#endif

namespace luce::common {

namespace {

constexpr uint64_t OTHER_DRIVERS_MARGIN = 1ULL << 30;

// Fields that together describe every page /proc/meminfo can attribute.
constexpr const char * ACCOUNTED[] = {
    "MemFree", "Buffers", "Cached", "SwapCached", "AnonPages", "Slab",
    "KernelStack", "PageTables", "SecPageTables", "Percpu", "Bounce",
};

#if defined(__linux__)
// False when no amdgpu device reports its GTT use: the pool cannot then be told
// apart from pages owned by other drivers.
bool live_gpu_host_bytes(uint64_t & total) {
    total = 0;
    bool found = false;
    DIR * dir = opendir("/sys/class/drm");
    if (!dir) return false;
    while (const dirent * entry = readdir(dir)) {
        const std::string name = entry->d_name;
        // card0, card1, ... but not the connector entries such as card0-DP-1.
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
        std::ifstream used("/sys/class/drm/" + name + "/device/mem_info_gtt_used");
        uint64_t bytes = 0;
        if (used >> bytes) { total += bytes; found = true; }
    }
    closedir(dir);
    return found;
}
#endif

} // namespace

uint64_t reclaimable_gpu_page_pool_bytes(const char * meminfo_text, uint64_t live_gpu_host_bytes) {
    if (!meminfo_text) return 0;
    uint64_t total = 0, accounted = 0, huge_pages = 0, huge_page_kb = 0;
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
        else for (const char * field : ACCOUNTED) if (key == field) accounted += value;
    }
    accounted += huge_pages * huge_page_kb;
    if (total <= accounted) return 0;
    const uint64_t unattributed = (total - accounted) * 1024;
    const uint64_t in_use = live_gpu_host_bytes + OTHER_DRIVERS_MARGIN;
    return unattributed > in_use ? unattributed - in_use : 0;
}

uint64_t reclaimable_gpu_page_pool_bytes() {
#if defined(__linux__)
    std::ifstream input("/proc/meminfo");
    if (!input) return 0;
    uint64_t live = 0;
    if (!live_gpu_host_bytes(live)) return 0;
    std::stringstream text;
    text << input.rdbuf();
    return reclaimable_gpu_page_pool_bytes(text.str().c_str(), live);
#else
    return 0;
#endif
}

} // namespace luce::common
