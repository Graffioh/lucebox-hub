#include "qgbatching.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dflash::common {

namespace {

float token_log_probability(const float * row, int vocab, int32_t token) {
    if (!row || token < 0 || token >= vocab || vocab <= 0) {
        return -std::numeric_limits<float>::infinity();
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; ++i) max_logit = std::max(max_logit, row[i]);

    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        sum += std::exp(static_cast<double>(row[i] - max_logit));
    }
    return row[token] - max_logit - static_cast<float>(std::log(sum));
}

}  // namespace

QGBatchingTree build_qgbatching_tree(const int32_t * top_token_ids,
                                     int draft_rows,
                                     int top_k,
                                     int ghost_branch_count,
                                     int horizon) {
    if (!top_token_ids || draft_rows <= 0 || top_k <= 0 ||
        ghost_branch_count <= 0 || horizon <= 0 ||
        ghost_branch_count > top_k || horizon > draft_rows) {
        QGBatchingTree out;
        out.tree.parents.push_back(-1);
        out.tree.child_maps.emplace_back();
        out.tree.visibility.assign(1, 1);
        return out;
    }

    std::vector<int32_t> ghost_branch_tokens(
        (size_t)ghost_branch_count * horizon);
    for (int ghost_branch = 0; ghost_branch < ghost_branch_count; ++ghost_branch) {
        for (int depth = 0; depth < horizon; ++depth) {
            const int rank = depth == 0 ? ghost_branch : 0;
            ghost_branch_tokens[(size_t)ghost_branch * horizon + depth] =
                top_token_ids[(size_t)depth * top_k + rank];
        }
    }
    return build_qgbatching_tree_from_paths(
        ghost_branch_tokens.data(), ghost_branch_count, horizon);
}

QGBatchingTree build_qgbatching_tree_from_paths(
    const int32_t * ghost_branch_tokens,
    int ghost_branch_count,
    int horizon) {
    QGBatchingTree out;
    out.tree.parents.push_back(-1);
    out.tree.child_maps.emplace_back();

    if (!ghost_branch_tokens || ghost_branch_count <= 0 || horizon <= 0) {
        out.tree.visibility.assign(1, 1);
        return out;
    }

    const int nodes = ghost_branch_count * horizon;
    out.tree.token_ids.reserve(nodes);
    out.tree.depths.reserve(nodes);
    out.tree.parents.reserve(nodes + 1);
    out.tree.child_maps.reserve(nodes + 1);
    out.ghost_branches.resize(ghost_branch_count);

    // Ghost-branch-major order is DFS order for this tree shape.
    for (int ghost_branch = 0; ghost_branch < ghost_branch_count; ++ghost_branch) {
        int parent = 0;
        auto & path = out.ghost_branches[ghost_branch];
        path.reserve(horizon);
        for (int depth = 0; depth < horizon; ++depth) {
            const int32_t token =
                ghost_branch_tokens[(size_t)ghost_branch * horizon + depth];
            const int node = ++out.tree.n_nodes;

            out.tree.token_ids.push_back(token);
            out.tree.depths.push_back(depth + 1);
            out.tree.parents.push_back(parent);
            out.tree.child_maps.emplace_back();
            out.tree.child_maps[parent][token] = node;
            path.push_back(node);
            parent = node;
        }
    }

    const int n = out.tree.n_nodes + 1;
    out.tree.visibility.assign((size_t)n * n, 0);
    out.tree.visibility[0] = 1;
    for (int node = 1; node < n; ++node) {
        const int parent = out.tree.parents[node];
        for (int col = 0; col < node; ++col) {
            out.tree.visibility[(size_t)node * n + col] =
                out.tree.visibility[(size_t)parent * n + col];
        }
        out.tree.visibility[(size_t)node * n + node] = 1;
    }
    return out;
}

int qgbatching_required_rows(int ghost_branch_count, int horizon, int tile) {
    if (ghost_branch_count <= 0 || horizon <= 0) return 0;
    const int rows = 1 + ghost_branch_count * horizon;
    if (tile <= 1) return rows;
    return ((rows + tile - 1) / tile) * tile;
}

QGBatchingSelection select_qgbatching_ghost_branch(
    const QGBatchingTree & qtree,
    const float * logits,
    int logits_rows,
    int vocab,
    float margin) {
    QGBatchingSelection out;
    out.accepted.push_back(0);
    if (!logits || vocab <= 0 || logits_rows < qtree.tree.n_nodes + 1 ||
        qtree.ghost_branches.empty()) {
        out.score = out.main_score = -std::numeric_limits<float>::infinity();
        return out;
    }

    std::vector<float> scores(qtree.ghost_branches.size(),
                              -std::numeric_limits<float>::infinity());
    for (size_t ghost_branch = 0;
         ghost_branch < qtree.ghost_branches.size();
         ++ghost_branch) {
        const auto & path = qtree.ghost_branches[ghost_branch];
        if (path.empty()) continue;

        float total = 0.0f;
        int parent = 0;
        bool valid = true;
        for (int node : path) {
            if (node <= 0 || node > qtree.tree.n_nodes ||
                qtree.tree.parents[node] != parent) {
                valid = false;
                break;
            }
            const int32_t token = qtree.tree.token_ids[node - 1];
            const float lp = token_log_probability(
                logits + (size_t)parent * vocab, vocab, token);
            if (!std::isfinite(lp)) {
                valid = false;
                break;
            }
            total += lp;
            parent = node;
        }
        if (valid) {
            scores[ghost_branch] = total / static_cast<float>(path.size());
        }
    }

    out.main_score = scores[0];
    out.score = scores[0];
    const float required = scores[0] + std::max(0.0f, margin);
    for (size_t ghost_branch = 1; ghost_branch < scores.size(); ++ghost_branch) {
        if (scores[ghost_branch] > required && scores[ghost_branch] > out.score) {
            out.ghost_branch = static_cast<int>(ghost_branch);
            out.score = scores[ghost_branch];
        }
    }
    out.accepted.insert(out.accepted.end(),
                        qtree.ghost_branches[out.ghost_branch].begin(),
                        qtree.ghost_branches[out.ghost_branch].end());
    return out;
}

}  // namespace dflash::common
