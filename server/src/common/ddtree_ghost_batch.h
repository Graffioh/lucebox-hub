// Uncertainty-gated ghost batching for exact DDTree verification.
//
// This policy never scores or selects a path for quality. It preserves the
// normal complete DFlash chain and adds short, fully conditioned alternatives
// only when the draft's first continuation is uncertain. The target still
// chooses the accepted path by exact speculative matching.

#pragma once

#include "ddtree.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

constexpr int DDTREE_GHOST_BASELINE_NODES = 15;
constexpr int DDTREE_GHOST_BRANCHES = 3;
constexpr int DDTREE_GHOST_BRANCH_NODES = 5;
constexpr int DDTREE_GHOST_ALLOC_ROWS = 32;

struct DDTreeConfidenceDecision {
    bool uncertain = false;
    float margin = 0.0f;
};

// The metric is the top-1/top-2 log-probability margin for the first draft
// continuation. A round launches a ghost batch only when margin < threshold;
// equality stays on the existing chain fast path.
DDTreeConfidenceDecision ddtree_confidence_gate(
    const float * top_log_probs,
    int positions,
    int top_k,
    float threshold);

// Builds the fixed 31-real-row shape:
//   root + complete 15-node baseline + 3 independent 5-node ghost branches.
// Ghost paths are ghost-major. Returns a root-only tree on invalid input.
DDTree build_ghost_batch_ddtree(
    const int32_t * baseline_tokens,
    int baseline_nodes,
    const int32_t * ghost_tokens,
    int ghost_count,
    int ghost_nodes);

// Real rows rounded to a target verify tile.
int ddtree_ghost_batch_required_rows(
    int baseline_nodes = DDTREE_GHOST_BASELINE_NODES,
    int ghost_count = DDTREE_GHOST_BRANCHES,
    int ghost_nodes = DDTREE_GHOST_BRANCH_NODES,
    int tile = DDTREE_GHOST_ALLOC_ROWS);

// Extract rank-1..rank-N alternatives for the first continuation row.
bool ddtree_ghost_batch_seed_tokens(
    const int32_t * top_token_ids,
    int positions,
    int top_k,
    int ghost_count,
    std::vector<int32_t> & seeds);

}  // namespace dflash::common
