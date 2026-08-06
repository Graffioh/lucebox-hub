// QGBatching (Quality Ghost Batching) — quality-oriented ghost batching for
// one generation.
//
// QGBatching reuses the existing DDTree verify/rollback machinery, but builds a
// deliberately simple shape: K independent short chains sharing one root.
// Each hypothetical chain is a ghost branch. One packed drafter execution
// containing multiple ghost branches is a ghost batch. The target scores every
// complete ghost branch and keeps the normal top-1 path unless another clears
// a conservative margin.

// Self-contained: depends only on DDTree and the standard library.

#pragma once

#include "ddtree.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

struct QGBatchingTree {
    DDTree tree;

    // Flat-tree indices for each ghost branch, excluding the shared root.
    // Ghost branch 0 is the drafter's ordinary top-1 continuation.
    std::vector<std::vector<int>> ghost_branches;
};

struct QGBatchingSelection {
    int ghost_branch = 0;
    float score = 0.0f;       // mean target log-probability, nats/token
    float main_score = 0.0f;
    std::vector<int> accepted; // shared root followed by selected ghost branch
};

// Build `ghost_branch_count` disjoint chains of `horizon` tokens. The first
// token of ghost branch b uses rank b from draft row 0; descendants use the
// top-1 token from subsequent rows. This is intentionally compatible with the
// existing single spine-conditioned draft block.
QGBatchingTree build_qgbatching_tree(const int32_t * top_token_ids,
                                     int draft_rows,
                                     int top_k,
                                     int ghost_branch_count,
                                     int horizon);

// Build the same tree from fully conditioned, ghost-branch-major paths:
//   ghost_branch_tokens[ghost_branch * horizon + depth].
// This is the Seed-and-Expand entry point used by QGBatching.
QGBatchingTree build_qgbatching_tree_from_paths(
    const int32_t * ghost_branch_tokens,
    int ghost_branch_count,
    int horizon);

// Number of target rows needed by QGBatching, rounded to the target's verify tile.
int qgbatching_required_rows(int ghost_branch_count, int horizon, int tile = 32);

// Score complete ghost branches under target logits. Logits row 0 predicts the
// first token; the row for each ghost-branch node predicts its child. Ghost
// branch 0 wins ties and alternatives must exceed it by `margin` nats/token.
QGBatchingSelection select_qgbatching_ghost_branch(
    const QGBatchingTree & qtree,
    const float * logits,
    int logits_rows,
    int vocab,
    float margin);

}  // namespace dflash::common
