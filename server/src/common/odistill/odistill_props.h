// odistill_props.h — the ODistill stats snapshot served under /props ("odistill"
// section; additive, no props_schema bump). Split from odistill_runtime.h so
// model_backend.h can expose it without pulling in threads/shm machinery.

#pragma once

#include <cstdint>
#include <string>

namespace dflash::common::odistill {

struct ODistillPropsSnapshot {
    bool enabled = false;
    std::string profile;
    uint64_t adapter_generation = 0;
    uint64_t swaps = 0, promotes = 0, rollbacks = 0;
    double rolling_al = 0.0;      // guard baseline window mean
    double probation_al = 0.0;
    bool in_probation = false;    // controls null vs numeric /props value
    bool training_disabled = false;
    bool trainer_alive = false;
    uint64_t records_written = 0, records_dropped = 0;
    uint64_t ring_backlog_bytes = 0;
    uint64_t steps = 0;
};

}  // namespace dflash::common::odistill
