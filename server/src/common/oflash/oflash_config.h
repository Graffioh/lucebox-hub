// oflash_config.h — operator-facing OFlash configuration (OFLASH.md §6.4).
//
// Parsed in server_main.cpp, carried through BackendArgs into Qwen35Config.
// Everything defaults to off/conservative; --oflash is the master switch.

#pragma once

#include <string>

namespace dflash::common::oflash {

struct OFlashConfig {
    bool enabled = false;          // --oflash

    // Trainer placement: "cpu" or a HIP/CUDA ordinal string ("1"). Passed
    // to the sidecar (which scopes itself via HIP_VISIBLE_DEVICES before
    // importing torch); the engine does not interpret it beyond validation.
    std::string device = "1";      // --oflash-device (box: 1 = Strix iGPU)

    std::string profile = "default";  // --oflash-profile
    int         lora_rank = 16;       // --oflash-lora-rank
    float       lora_alpha = 32.0f;   // --oflash-alpha
    std::string dir;                  // --oflash-dir (default ~/.lucebox/oflash)

    // Capture ring sizing + label detail.
    int ring_mb = 2048;            // --oflash-ring-mb (shm, unified memory)
    int topk    = 32;              // --oflash-topk (0 = skip top-K capture)
    int backfill_rows = 512;       // context rows backfilled per request

    // Sidecar launch. Empty = capture-only (M0 telemetry mode): the ring is
    // exposed and a trainer may be attached by hand.
    std::string trainer_bin;       // --oflash-trainer-bin
};

}  // namespace dflash::common::oflash
