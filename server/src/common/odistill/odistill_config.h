// odistill_config.h — operator-facing ODistill configuration (ODISTILL.md §6.4).
//
// Parsed in server_main.cpp, carried through BackendArgs into Qwen35Config.
// Everything defaults to off/conservative; --odistill is the master switch.

#pragma once

#include <string>

namespace dflash::common::odistill {

struct ODistillConfig {
    bool enabled = false;          // --odistill

    // Trainer placement: "cpu" or a HIP/CUDA ordinal string ("1"). Passed
    // to the sidecar (which scopes itself via HIP_VISIBLE_DEVICES before
    // importing torch); the engine does not interpret it beyond validation.
    std::string device = "1";      // --odistill-device (box: 1 = Strix iGPU)
    std::string dtype  = "auto";   // --odistill-dtype (auto|fp16|bf16|fp32)

    std::string profile = "default";  // --odistill-profile
    int         lora_rank = 16;       // --odistill-lora-rank
    float       lora_alpha = 32.0f;   // --odistill-alpha
    std::string dir;                  // --odistill-dir (default ~/.lucebox/odistill)

    // Capture ring sizing + label detail.
    int ring_mb = 512;             // --odistill-ring-mb (shm, unified memory)
    int topk    = 8;               // --odistill-topk (0 = skip top-K capture)
    int backfill_rows = 128;       // context rows backfilled per request

    // Sidecar launch. Empty = capture-only (M0 telemetry mode): the ring is
    // exposed and a trainer may be attached by hand.
    std::string trainer_bin;       // --odistill-trainer-bin
};

}  // namespace dflash::common::odistill
