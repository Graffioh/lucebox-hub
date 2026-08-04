// SpecLA confidence-guided draft-tree pruning (arXiv:2607.16673 §6.1).
//
// build_ddtree's best-first expansion pops candidates in descending
// cumulative path log-probability q(v), so the tau_tree window
// (keep q(v) >= q* - tau) is a single early-stop comparison. These tests pin
// the contract: the margin prunes exactly the out-of-window candidates, the
// retained set stays ancestor-closed, the budget still caps the tree, and the
// default (infinite tau) reproduces the unpruned tree.

#include "CppUnitTestFramework.hpp"
#include "ddtree.h"

#include <cmath>
#include <limits>
#include <vector>

using dflash::common::build_ddtree;
using dflash::common::DDTree;

namespace {
struct DdtreeTauFixture {};

// L=3 positions, K=2 ranks. Rank-0 chain is strong; rank-1 siblings weak.
//   depth 1: {-0.1, -3.0}
//   depth 2: {-0.2, -3.5}
//   depth 3: {-0.3, -4.0}
// Cumulative scores: top chain -0.1/-0.3/-0.6; the best sibling is -3.0.
const float kLp[6]  = { -0.1f, -3.0f, -0.2f, -3.5f, -0.3f, -4.0f };
const int32_t kIds[6] = { 10, 11, 20, 21, 30, 31 };

bool ancestor_closed(const DDTree & t) {
    for (int i = 1; i <= t.n_nodes; i++) {
        const int p = t.parents[i];
        if (p < 0 || p > t.n_nodes || p >= i) return false;
    }
    return true;
}
}  // namespace

TEST_CASE(DdtreeTauFixture, infinite_tau_matches_unpruned_tree) {
    DDTree base = build_ddtree(kLp, kIds, 3, 2, 8, /*chain_seed=*/false);
    DDTree inf  = build_ddtree(kLp, kIds, 3, 2, 8, /*chain_seed=*/false,
                               std::numeric_limits<float>::infinity());
    CHECK(base.n_nodes == inf.n_nodes);
    CHECK(base.token_ids == inf.token_ids);
    CHECK(base.parents == inf.parents);
}

TEST_CASE(DdtreeTauFixture, margin_prunes_low_confidence_branches) {
    // Window of 1.0 below q* = -0.1 keeps the top chain (-0.1, -0.3, -0.6)
    // and prunes every sibling (best sibling -3.0).
    DDTree t = build_ddtree(kLp, kIds, 3, 2, 8, /*chain_seed=*/false, 1.0f);
    CHECK(t.n_nodes == 3);
    CHECK(t.token_ids == (std::vector<int32_t>{10, 20, 30}));
    CHECK(ancestor_closed(t));

    // A wide window admits the siblings again and expansion runs to the
    // budget cap (the candidate space under the siblings exceeds it).
    DDTree wide = build_ddtree(kLp, kIds, 3, 2, 8, /*chain_seed=*/false, 10.0f);
    CHECK(wide.n_nodes == 8);
    CHECK(ancestor_closed(wide));
}

TEST_CASE(DdtreeTauFixture, budget_still_caps_within_window) {
    DDTree t = build_ddtree(kLp, kIds, 3, 2, 2, /*chain_seed=*/false, 10.0f);
    CHECK(t.n_nodes == 2);
    CHECK(ancestor_closed(t));
}

TEST_CASE(DdtreeTauFixture, chain_seed_exempt_from_margin) {
    // With the seed on, the full top-1 chain is inserted even where the
    // cumulative score falls out of a tiny window; only heap candidates
    // (the siblings) are pruned.
    DDTree t = build_ddtree(kLp, kIds, 3, 2, 8, /*chain_seed=*/true, 0.05f);
    CHECK(t.n_nodes == 3);
    CHECK(t.token_ids == (std::vector<int32_t>{10, 20, 30}));
    CHECK(ancestor_closed(t));
}
