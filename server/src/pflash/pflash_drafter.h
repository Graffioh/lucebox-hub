// In-process PFlash drafter for speculative prefill.
//
// The drafter is the Qwen3.5-0.8B scorer (qwen35_drafter.cpp +
// qwen35_loader.cpp): it runs the model's first fifteen blocks and scores
// the context with block 15's NoPE Q/K attention-mass head, with the
// all-layer running-max scorer kept as an opt-in alternative
// (PFLASH_QWEN35_LEGACY_SCORER=1 or the PFLASH scorer config).
//
// Hosted in the SAME process / SAME ggml allocator as the dflash target, so
// we never pay the cross-process VRAM contention that broke the Python
// subprocess integration.
//
// Public entry point: drafter_score_and_compress() takes raw input token IDs,
// runs the full pflash compression pipeline in C++, returns the surviving
// token IDs (drafter vocab).

#pragma once

#include "common/pflash_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash::common {

struct Qwen35DrafterState;

struct DrafterContext {
    ggml_backend_t        backend = nullptr;   // owned (created in load_drafter)
    Qwen35DrafterState *  state   = nullptr;   // owned scorer weights + heads
    int                   gpu     = -1;
    bool                  loaded  = false;
};

// Load the drafter GGUF (a Qwen3.5-0.8B GGUF).
// Creates a fresh GPU backend if `backend` is null. Otherwise uses the
// caller-provided backend (so the drafter shares the daemon's allocator).
//
// `gpu_layers` is accepted for API compat but ignored — every layer goes on
// the GPU since the drafter weights are only ~1.5 GB.
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  DrafterContext & out);
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  int gpu, DrafterContext & out);

void free_drafter(DrafterContext & ctx);

// Free only model weights, keeping the backend alive for reuse.
// Avoids repeated ggml backend create/destroy during daemon reuse.
void free_drafter_weights(DrafterContext & ctx);

// Score the context with the block-15 scoring head, then run strict budget
// selection (or the configured selection mode). Returns surviving token IDs
// (drafter vocab).
//
//   ids          input token IDs of length S
//   keep_ratio   fraction of the token budget to keep
//   chunk_size   span granularity (default 32)
//   n_lookahead  Q tokens used for scorer attention (default 8)
//   pool_kernel  AvgPool kernel for score smoothing (default 13)
//   score_query_end  exclusive end of the scorer query window in ids;
//                    required (negative values are rejected)
//
// On failure returns empty vector + sets last_error.
std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float  keep_ratio,
    int    chunk_size  = 32,
    int    n_lookahead = 8,
    int    pool_kernel = 13,
    int    score_query_end = -1,
    const std::vector<PFlashTokenSpan> &
        required_instruction_spans = {});

} // namespace dflash::common
