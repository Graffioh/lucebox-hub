#include "CppUnitTestFramework.hpp"
#include "ddtree_ghost_batch.h"
#include "dflash_draft_graph.h"

#include <vector>

using namespace dflash::common;

namespace {
struct DDTreeGhostBatchFixture {};
}

TEST_CASE(DDTreeGhostBatchFixture, confidence_gate_has_stable_boundary) {
    const float log_probs[] = {-0.20f, -0.55f, -1.2f, -2.0f};

    auto uncertain = ddtree_confidence_gate(
        log_probs, /*positions=*/1, /*top_k=*/4, /*threshold=*/0.36f);
    CHECK(uncertain.uncertain);
    CHECK_CLOSE(uncertain.margin, 0.35f, 1e-6f);

    const auto boundary = ddtree_confidence_gate(
        log_probs, 1, 4, /*threshold=*/0.35f);
    CHECK(!boundary.uncertain);

    const auto confident = ddtree_confidence_gate(
        log_probs, 1, 4, /*threshold=*/0.20f);
    CHECK(!confident.uncertain);
}

TEST_CASE(DDTreeGhostBatchFixture, extracts_three_non_top1_seeds) {
    const int32_t ids[] = {10, 11, 12, 13};
    std::vector<int32_t> seeds;
    CHECK(ddtree_ghost_batch_seed_tokens(ids, 1, 4, 3, seeds));
    CHECK(seeds == (std::vector<int32_t>{11, 12, 13}));
}

TEST_CASE(DDTreeGhostBatchFixture, ghost_tree_keeps_complete_baseline_under_32_rows) {
    std::vector<int32_t> baseline(DDTREE_GHOST_BASELINE_NODES);
    for (int i = 0; i < (int)baseline.size(); ++i) baseline[i] = 100 + i;

    std::vector<int32_t> ghosts(
        DDTREE_GHOST_BRANCHES * DDTREE_GHOST_BRANCH_NODES);
    for (int ghost = 0; ghost < DDTREE_GHOST_BRANCHES; ++ghost) {
        for (int depth = 0; depth < DDTREE_GHOST_BRANCH_NODES; ++depth) {
            ghosts[(size_t)ghost * DDTREE_GHOST_BRANCH_NODES + depth] =
                200 + ghost * 10 + depth;
        }
    }

    const DDTree tree = build_ghost_batch_ddtree(
        baseline.data(), baseline.size(), ghosts.data(),
        DDTREE_GHOST_BRANCHES, DDTREE_GHOST_BRANCH_NODES);

    CHECK_EQUAL(tree.n_nodes, 30);
    CHECK_EQUAL(tree.n_nodes + 1, 31);
    CHECK_EQUAL(ddtree_ghost_batch_required_rows(), 32);

    int node = 0;
    for (int depth = 0; depth < (int)baseline.size(); ++depth) {
        const auto it = tree.child_maps[node].find(baseline[depth]);
        CHECK(it != tree.child_maps[node].end());
        node = it->second;
        CHECK_EQUAL(tree.depths[node - 1], depth + 1);
    }
}

TEST_CASE(DDTreeGhostBatchFixture, ghost_branch_visibility_is_local) {
    const int32_t baseline[] = {10, 11, 12};
    const int32_t ghosts[] = {
        20, 21,
        30, 31,
    };
    const DDTree tree = build_ghost_batch_ddtree(
        baseline, 3, ghosts, 2, 2);

    CHECK_EQUAL(tree.n_nodes, 7);
    const int rows = tree.n_nodes + 1;
    const int ghost0_seed = tree.child_maps[0].at(20);
    const int ghost0_leaf = tree.child_maps[ghost0_seed].at(21);
    const int ghost1_seed = tree.child_maps[0].at(30);
    CHECK_EQUAL(tree.visibility[(size_t)ghost0_leaf * rows], 1);
    CHECK_EQUAL(tree.visibility[(size_t)ghost0_leaf * rows + ghost0_seed], 1);
    CHECK_EQUAL(tree.visibility[(size_t)ghost0_leaf * rows + ghost1_seed], 0);
    CHECK_EQUAL(tree.visibility[(size_t)ghost0_leaf * rows + 1], 0);
}

TEST_CASE(DDTreeGhostBatchFixture, exact_following_accepts_available_alternative) {
    const int32_t baseline[] = {10, 11, 12};
    const int32_t ghosts[] = {
        20, 21,
        30, 31,
    };
    const DDTree tree = build_ghost_batch_ddtree(
        baseline, 3, ghosts, 2, 2);
    std::vector<int32_t> posterior(tree.n_nodes + 1, 999);
    const int seed = tree.child_maps[0].at(20);
    const int leaf = tree.child_maps[seed].at(21);
    posterior[0] = 20;
    posterior[seed] = 21;
    posterior[leaf] = 77;

    int bonus = -1;
    const std::vector<int> accepted =
        follow_verified_tree(tree, posterior.data(), bonus);
    CHECK(accepted == (std::vector<int>{0, seed, leaf}));
    CHECK_EQUAL(bonus, 77);
}

TEST_CASE(DDTreeGhostBatchFixture, ghost_batch_layout_isolates_full_and_swa_paths) {
    DraftPackedBatchLayout layout;
    CHECK(make_draft_packed_batch_layout(
        /*ctx_len=*/3, /*block_size=*/2, /*batch_size=*/3,
        /*swa_window=*/2, layout));

    CHECK(layout.positions_q ==
          (std::vector<int32_t>{3, 4, 3, 4, 3, 4}));
    static constexpr uint16_t ZERO = 0x0000;
    static constexpr uint16_t NEG_INF = 0xFC00;
    const int row = 2;  // ghost 1, depth 0
    const uint16_t * full =
        layout.mask_full.data() + (size_t)row * layout.full_kv_stride;
    CHECK_EQUAL(full[0], ZERO);
    CHECK_EQUAL(full[3], NEG_INF);
    CHECK_EQUAL(full[5], ZERO);
    CHECK_EQUAL(full[6], ZERO);
    CHECK_EQUAL(full[7], NEG_INF);

    const uint16_t * swa =
        layout.mask_swa.data() + (size_t)row * layout.swa_kv_stride;
    CHECK_EQUAL(swa[0], ZERO);
    CHECK_EQUAL(swa[2], NEG_INF);
    CHECK_EQUAL(swa[4], ZERO);
    CHECK_EQUAL(swa[5], NEG_INF);
    CHECK_EQUAL(swa[6], NEG_INF);
}
