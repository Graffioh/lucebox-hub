// GQBatching (Ghost Quality Batching) — quality-oriented branching for one
// generation.
//
// GQBatching reuses the existing DDTree verify/rollback machinery, but builds a
// deliberately simple shape: K independent short chains sharing one root.
// The target scores every complete chain and keeps the normal top-1 chain
// unless another candidate clears a conservative margin.

// Self-contained: depends only on DDTree and the standard library.

#pragma once

#include "ddtree.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

struct GQBatchingTree {
    DDTree tree;

    // Flat-tree indices for each candidate, excluding the shared root.
    // Branch 0 is the drafter's ordinary top-1 continuation.
    std::vector<std::vector<int>> branches;
};

struct GQBatchingSelection {
    int branch = 0;
    float score = 0.0f;       // mean target log-probability, nats/token
    float main_score = 0.0f;
    std::vector<int> accepted; // shared root followed by selected branch
};

// Build `branch_count` disjoint chains of `horizon` tokens. The first token
// of branch b uses rank b from draft row 0; descendants use the top-1 token
// from subsequent rows. This is intentionally compatible with the existing
// single spine-conditioned draft block.
GQBatchingTree build_gqbatching_tree(const int32_t * top_token_ids,
                             int draft_rows,
                             int top_k,
                             int branch_count,
                             int horizon);

// Build the same tree from fully conditioned, branch-major paths:
//   branch_tokens[branch * horizon + depth].
// This is the Seed-and-Expand entry point used by GQBatching.
GQBatchingTree build_gqbatching_tree_from_paths(const int32_t * branch_tokens,
                                              int branch_count,
                                              int horizon);

// Number of target rows needed by GQBatching, rounded to the target's verify tile.
int gqbatching_required_rows(int branch_count, int horizon, int tile = 32);

// Score complete branches under target logits. Logits row 0 predicts the
// first branch token; the row for each branch node predicts its child.
// Branch 0 wins ties and alternatives must exceed it by `margin` nats/token.
GQBatchingSelection select_gqbatching_branch(const GQBatchingTree & qtree,
                                     const float * logits,
                                     int logits_rows,
                                     int vocab,
                                     float margin);

}  // namespace dflash::common
