// oflash_props.h — the OFlash stats snapshot served under /props ("oflash"
// section; additive, no props_schema bump). Split from oflash_runtime.h so
// model_backend.h can expose it without pulling in threads/shm machinery.

#pragma once

#include <cstdint>
#include <string>

namespace dflash::common::oflash {

struct OFlashPropsSnapshot {
    bool enabled = false;
    std::string profile;
    uint64_t adapter_generation = 0;
    uint64_t swaps = 0, promotes = 0, rollbacks = 0;
    double rolling_al = 0.0;      // guard baseline window mean
    double probation_al = 0.0;    // 0 when not in probation
    bool training_disabled = false;
    bool trainer_alive = false;
    uint64_t records_written = 0, records_dropped = 0;
    uint64_t ring_backlog_bytes = 0;
    uint64_t steps = 0;
};

}  // namespace dflash::common::oflash
