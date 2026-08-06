#include "CppUnitTestFramework.hpp"
#include "dflash_draft_graph.h"
#include "gbatching.h"

#include <vector>

using namespace dflash::common;

namespace {
struct GBatchingFixture {};

std::vector<float> neutral_logits(int rows, int vocab) {
    return std::vector<float>((size_t)rows * vocab, 0.0f);
}
}  // namespace

TEST_CASE(GBatchingFixture, builds_disjoint_short_branches) {
    // Three draft rows, top-3 each. Only the first row fans out.
    const int32_t ids[] = {
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
    };
    const GBatchingTree qtree = build_gbatching_tree(ids, 3, 3, 3, 3);

    CHECK_EQUAL(qtree.tree.n_nodes, 9);
    CHECK_EQUAL(qtree.branches.size(), (size_t)3);
    CHECK_EQUAL(qtree.tree.token_ids[qtree.branches[0][0] - 1], 10);
    CHECK_EQUAL(qtree.tree.token_ids[qtree.branches[1][0] - 1], 11);
    CHECK_EQUAL(qtree.tree.token_ids[qtree.branches[2][0] - 1], 12);
    CHECK_EQUAL(qtree.tree.token_ids[qtree.branches[2][1] - 1], 20);
    CHECK_EQUAL(qtree.tree.parents[qtree.branches[2][0]], 0);
    CHECK_EQUAL(qtree.tree.parents[qtree.branches[2][1]], qtree.branches[2][0]);

    const int n = qtree.tree.n_nodes + 1;
    CHECK_EQUAL(qtree.tree.visibility[(size_t)qtree.branches[2][2] * n], 1);
    CHECK_EQUAL(qtree.tree.visibility[(size_t)qtree.branches[2][2] * n +
                                      qtree.branches[0][0]], 0);
}

TEST_CASE(GBatchingFixture, builds_fully_conditioned_branch_paths) {
    const int32_t paths[] = {
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
    };
    const GBatchingTree tree =
        build_gbatching_tree_from_paths(paths, 3, 3);

    CHECK_EQUAL(tree.tree.n_nodes, 9);
    CHECK_EQUAL(tree.branches.size(), 3u);
    for (int branch = 0; branch < 3; ++branch) {
        CHECK_EQUAL(tree.branches[branch].size(), 3u);
        for (int depth = 0; depth < 3; ++depth) {
            const int node = tree.branches[branch][depth];
            CHECK_EQUAL(tree.tree.token_ids[node - 1],
                        paths[branch * 3 + depth]);
            CHECK_EQUAL(tree.tree.depths[node - 1], depth + 1);
        }
    }
}

TEST_CASE(GBatchingFixture, rows_round_to_verify_tile) {
    CHECK_EQUAL(gbatching_required_rows(4, 7), 32);
    CHECK_EQUAL(gbatching_required_rows(2, 15), 32);
    CHECK_EQUAL(gbatching_required_rows(4, 8), 64);
}

TEST_CASE(GBatchingFixture, packed_draft_layout_isolates_branches) {
    DraftPackedBatchLayout layout;
    CHECK(make_draft_packed_batch_layout(
        /*ctx_len=*/3, /*block_size=*/2, /*batch_size=*/3,
        /*swa_window=*/2, layout));

    CHECK_EQUAL(layout.query_count, 6);
    CHECK(layout.positions_q ==
          (std::vector<int32_t>{3, 4, 3, 4, 3, 4}));
    CHECK(layout.positions_k ==
          (std::vector<int32_t>{0, 1, 2, 3, 4, 3, 4, 3, 4}));

    static constexpr uint16_t ZERO = 0x0000;
    static constexpr uint16_t NEG_INF = 0xFC00;
    const int row = 2; // branch 1, depth 0
    const uint16_t * full = layout.mask_full.data() +
        (size_t)row * layout.full_kv_stride;
    CHECK_EQUAL(full[0], ZERO); // shared prefix
    CHECK_EQUAL(full[3], NEG_INF); // branch 0 noise is isolated
    CHECK_EQUAL(full[5], ZERO); // own branch, depth 0
    CHECK_EQUAL(full[6], ZERO); // full attention sees own complete block
    CHECK_EQUAL(full[7], NEG_INF); // branch 2 noise is isolated

    const uint16_t * swa = layout.mask_swa.data() +
        (size_t)row * layout.swa_kv_stride;
    CHECK_EQUAL(swa[0], ZERO); // shared SWA prefix
    CHECK_EQUAL(swa[2], NEG_INF); // branch 0 noise is isolated
    CHECK_EQUAL(swa[4], ZERO); // own causal key at depth 0
    CHECK_EQUAL(swa[5], NEG_INF); // own future key is hidden
    CHECK_EQUAL(swa[6], NEG_INF); // branch 2 noise is isolated
}

TEST_CASE(GBatchingFixture, main_branch_wins_without_clear_improvement) {
    const int32_t ids[] = {1, 2, 3, 1};
    const GBatchingTree qtree = build_gbatching_tree(ids, 2, 2, 2, 2);
    auto logits = neutral_logits(qtree.tree.n_nodes + 1, 4);

    // Slightly prefer branch 1 at the root, but not by the configured margin.
    logits[2] = 0.05f;
    const auto selected = select_gbatching_branch(
        qtree, logits.data(), qtree.tree.n_nodes + 1, 4, 0.10f);
    CHECK_EQUAL(selected.branch, 0);
    CHECK_EQUAL(selected.accepted.size(), (size_t)3);
    CHECK_EQUAL(selected.accepted[0], 0);
    CHECK_EQUAL(selected.accepted[1], 1);
    CHECK_EQUAL(selected.accepted[2], 2);
}

TEST_CASE(GBatchingFixture, stronger_target_branch_is_selected) {
    const int32_t ids[] = {1, 2, 3, 1};
    const GBatchingTree qtree = build_gbatching_tree(ids, 2, 2, 2, 2);
    auto logits = neutral_logits(qtree.tree.n_nodes + 1, 4);

    // Root strongly prefers branch 1's first token. Its second token is also
    // preferred from that branch's parent row.
    logits[2] = 2.0f;
    const int branch1_parent = qtree.branches[1][0];
    logits[(size_t)branch1_parent * 4 + 3] = 2.0f;
    const auto selected = select_gbatching_branch(
        qtree, logits.data(), qtree.tree.n_nodes + 1, 4, 0.10f);
    CHECK_EQUAL(selected.branch, 1);
    CHECK(selected.score > selected.main_score + 0.10f);
    CHECK_EQUAL(selected.accepted.size(), (size_t)3);
    CHECK_EQUAL(selected.accepted[0], 0);
    CHECK_EQUAL(selected.accepted[1], qtree.branches[1][0]);
    CHECK_EQUAL(selected.accepted[2], qtree.branches[1][1]);
}

TEST_CASE(GBatchingFixture, rejects_impossible_shape) {
    const int32_t ids[] = {1, 2, 3, 4};
    const GBatchingTree qtree = build_gbatching_tree(ids, 2, 2, 3, 2);
    CHECK_EQUAL(qtree.tree.n_nodes, 0);
    CHECK(qtree.branches.empty());
}
