// Qwen3.6 BTFlash persistent recurrent-state bank and winner compaction.

#pragma once

#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

struct Qwen35BTFlashStateBank {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int width = 0;
    std::vector<ggml_tensor *> ssm_state;
    std::vector<ggml_tensor *> conv_state;
    std::vector<ggml_tensor *> ssm_slots;
    std::vector<ggml_tensor *> conv_slots;

    Qwen35BTFlashStateBank() = default;
    ~Qwen35BTFlashStateBank();
    Qwen35BTFlashStateBank(const Qwen35BTFlashStateBank &) = delete;
    Qwen35BTFlashStateBank & operator=(const Qwen35BTFlashStateBank &) = delete;

    bool allocate(const TargetWeights & weights, ggml_backend_t backend,
                  int branch_width);
    bool capture(const TargetCache & cache);
    bool restore(TargetCache & cache, int winner) const;
    void clear();
};

// Compact the interleaved winner suffix onto the serial cache spine. The
// recurrent winner is restored separately from Qwen35BTFlashStateBank.
bool compact_qwen35_btflash_winner(TargetCache & cache,
                                   int branch_start,
                                   int width,
                                   int fed_steps,
                                   int winner);

}  // namespace dflash::common
