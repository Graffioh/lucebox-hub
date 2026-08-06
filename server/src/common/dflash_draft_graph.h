// Generic DFlash draft-step graph builder.
//
// The DFlash draft model is the same single Qwen3-style network for every
// target architecture (see dflash_target.h). This wrapper sets up a
// StepGraph, optionally views into a feature mirror, calls the universal
// build_draft_graph (src/draft/draft_graph.h), and reserves the gallocr
// buffer.

#pragma once

#include "step_graph.h"
#include "dflash_feature_ring.h"
#include "internal.h"  // DraftWeights

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Host-side inputs for one packed drafter batch. Independent draft blocks are
// flattened along the query dimension, while the masks make attention
// block-diagonal over the noise rows. Every block can still read the same
// committed prefix.
struct DraftPackedBatchLayout {
    int query_count = 0;
    int full_kv_stride = 0;
    int swa_kv_stride = 0;
    std::vector<int32_t> positions_q;
    std::vector<int32_t> positions_k;
    std::vector<uint16_t> mask_full;
    std::vector<uint16_t> mask_swa;
};

bool make_draft_packed_batch_layout(
    int ctx_len,
    int block_size,
    int batch_size,
    int swa_window,
    DraftPackedBatchLayout & out);

// Draft forward: speculative next-token prediction using target features.
//   lm_head: optional target lm_head tensor for fused projection. When
//   nullptr, the draft graph emits hidden states only and the caller is
//   responsible for projection (e.g. via build_lm_head_step on the target).
//   ctx_len_max: upper bound on ctx_len across all future calls (used to
//   pre-size allocations so gallocr never needs to reallocate).
bool build_draft_step(
    StepGraph & sg,
    const DraftWeights & dw,
    ggml_tensor * lm_head,
    ggml_backend_t backend,
    int ctx_len,
    const DraftFeatureMirror * mirror = nullptr,
    int committed = 0,
    int ctx_len_max = 0,
    bool pad_ctx = false);

// Build one real, packed draft batch. The graph contains `batch_size`
// independent block_size-wide alternatives and performs one set of wide
// matmuls over all rows. Context feature fusion is shared across the batch.
// Positions and isolation masks are uploaded by this function; the caller
// only uploads `inp_embed` and, when a mirror view is unavailable, the shared
// target feature prefix.
bool build_draft_packed_batch_step(
    StepGraph & sg,
    const DraftWeights & dw,
    ggml_backend_t backend,
    int ctx_len,
    int batch_size,
    const DraftFeatureMirror * mirror = nullptr,
    int committed = 0);

}  // namespace dflash::common
