#pragma once

#include <cstdint>

namespace dflash::common {

struct Qwen35RoctxMetadata {
    uint64_t round_id = 0;
    const char * path = nullptr;
    int spec_tree_width = -1;
    int live_slots = -1;
    int decode_bucket = -1;
    int prefill_tokens = -1;
    int prefill_segments = -1;
    int target_rows = -1;
    int max_kv_len = -1;
};

struct Qwen35RoctxCallbacks {
    int (*push)(const char * message) = nullptr;
    int (*pop)() = nullptr;
};

bool qwen35_roctx_env_enabled(const char * value);

class Qwen35RoctxRange {
public:
    Qwen35RoctxRange(const char * scope, const Qwen35RoctxMetadata & metadata);
    Qwen35RoctxRange(const char * scope, const Qwen35RoctxMetadata & metadata,
                     bool enabled, Qwen35RoctxCallbacks callbacks);
    ~Qwen35RoctxRange();

    Qwen35RoctxRange(const Qwen35RoctxRange &) = delete;
    Qwen35RoctxRange & operator=(const Qwen35RoctxRange &) = delete;

private:
    int (*pop_)() = nullptr;
    bool pushed_ = false;
};

} // namespace dflash::common
