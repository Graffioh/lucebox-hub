#include "CppUnitTestFramework.hpp"
#include "qflash.h"

#include <vector>

using namespace dflash::common;

namespace {
struct QFlashFixture {};

std::vector<float> neutral_logits(int rows, int vocab) {
    return std::vector<float>((size_t)rows * vocab, 0.0f);
}
}  // namespace

TEST_CASE(QFlashFixture, builds_disjoint_short_branches) {
    // Three draft rows, top-3 each. Only the first row fans out.
    const int32_t ids[] = {
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
    };
    const QFlashTree qtree = build_qflash_tree(ids, 3, 3, 3, 3);

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

TEST_CASE(QFlashFixture, rows_round_to_verify_tile) {
    CHECK_EQUAL(qflash_required_rows(4, 7), 32);
    CHECK_EQUAL(qflash_required_rows(2, 15), 32);
    CHECK_EQUAL(qflash_required_rows(4, 8), 64);
}

TEST_CASE(QFlashFixture, main_branch_wins_without_clear_improvement) {
    const int32_t ids[] = {1, 2, 3, 1};
    const QFlashTree qtree = build_qflash_tree(ids, 2, 2, 2, 2);
    auto logits = neutral_logits(qtree.tree.n_nodes + 1, 4);

    // Slightly prefer branch 1 at the root, but not by the configured margin.
    logits[2] = 0.05f;
    const auto selected = select_qflash_branch(
        qtree, logits.data(), qtree.tree.n_nodes + 1, 4, 0.10f);
    CHECK_EQUAL(selected.branch, 0);
    CHECK_EQUAL(selected.accepted.size(), (size_t)3);
    CHECK_EQUAL(selected.accepted[0], 0);
    CHECK_EQUAL(selected.accepted[1], 1);
    CHECK_EQUAL(selected.accepted[2], 2);
}

TEST_CASE(QFlashFixture, stronger_target_branch_is_selected) {
    const int32_t ids[] = {1, 2, 3, 1};
    const QFlashTree qtree = build_qflash_tree(ids, 2, 2, 2, 2);
    auto logits = neutral_logits(qtree.tree.n_nodes + 1, 4);

    // Root strongly prefers branch 1's first token. Its second token is also
    // preferred from that branch's parent row.
    logits[2] = 2.0f;
    const int branch1_parent = qtree.branches[1][0];
    logits[(size_t)branch1_parent * 4 + 3] = 2.0f;
    const auto selected = select_qflash_branch(
        qtree, logits.data(), qtree.tree.n_nodes + 1, 4, 0.10f);
    CHECK_EQUAL(selected.branch, 1);
    CHECK(selected.score > selected.main_score + 0.10f);
    CHECK_EQUAL(selected.accepted.size(), (size_t)3);
    CHECK_EQUAL(selected.accepted[0], 0);
    CHECK_EQUAL(selected.accepted[1], qtree.branches[1][0]);
    CHECK_EQUAL(selected.accepted[2], qtree.branches[1][1]);
}

TEST_CASE(QFlashFixture, rejects_impossible_shape) {
    const int32_t ids[] = {1, 2, 3, 4};
    const QFlashTree qtree = build_qflash_tree(ids, 2, 2, 3, 2);
    CHECK_EQUAL(qtree.tree.n_nodes, 0);
    CHECK(qtree.branches.empty());
}
