#include "ddtree_adaptive.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace dflash::common {

namespace {

DDTree root_only_tree() {
    DDTree tree;
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();
    tree.visibility.assign(1, 1);
    return tree;
}

void finish_visibility(DDTree & tree) {
    const int rows = tree.n_nodes + 1;
    tree.visibility.assign((size_t)rows * rows, 0);
    tree.visibility[0] = 1;
    for (int node = 1; node < rows; ++node) {
        const int parent = tree.parents[node];
        for (int col = 0; col < node; ++col) {
            tree.visibility[(size_t)node * rows + col] =
                tree.visibility[(size_t)parent * rows + col];
        }
        tree.visibility[(size_t)node * rows + node] = 1;
    }
}

bool checked_shape(int baseline_nodes, int branch_count, int branch_nodes) {
    if (baseline_nodes <= 0 || branch_count <= 0 || branch_nodes <= 0) {
        return false;
    }
    const int max = std::numeric_limits<int>::max();
    return branch_count <= (max - baseline_nodes) / branch_nodes;
}

}  // namespace

DDTreeConfidenceDecision ddtree_confidence_gate(
    const float * top_log_probs,
    int positions,
    int top_k,
    float threshold) {
    DDTreeConfidenceDecision out;
    if (!top_log_probs || positions <= 0 || top_k < 2 ||
        !std::isfinite(threshold) || threshold < 0.0f) {
        return out;
    }
    const float top1 = top_log_probs[0];
    const float top2 = top_log_probs[1];
    if (!std::isfinite(top1) || !std::isfinite(top2)) return out;
    out.margin = std::max(0.0f, top1 - top2);
    out.uncertain = out.margin < threshold;
    return out;
}

DDTree build_adaptive_ddtree(
    const int32_t * baseline_tokens,
    int baseline_nodes,
    const int32_t * branch_tokens,
    int branch_count,
    int branch_nodes) {
    if (!baseline_tokens || !branch_tokens ||
        !checked_shape(baseline_nodes, branch_count, branch_nodes)) {
        return root_only_tree();
    }

    DDTree tree;
    const int nodes = baseline_nodes + branch_count * branch_nodes;
    tree.token_ids.reserve(nodes);
    tree.depths.reserve(nodes);
    tree.parents.reserve(nodes + 1);
    tree.child_maps.reserve(nodes + 1);
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    auto append = [&](int32_t token, int depth, int parent) {
        const int node = ++tree.n_nodes;
        tree.token_ids.push_back(token);
        tree.depths.push_back(depth);
        tree.parents.push_back(parent);
        tree.child_maps.emplace_back();
        tree.child_maps[parent][token] = node;
        return node;
    };

    int parent = 0;
    for (int depth = 0; depth < baseline_nodes; ++depth) {
        parent = append(baseline_tokens[depth], depth + 1, parent);
    }

    // Root children must be distinct or child_maps would silently make one
    // path unreachable during exact target following.
    std::unordered_map<int32_t, bool> root_tokens;
    root_tokens.emplace(baseline_tokens[0], true);
    for (int branch = 0; branch < branch_count; ++branch) {
        const int32_t seed = branch_tokens[(size_t)branch * branch_nodes];
        if (!root_tokens.emplace(seed, true).second) return root_only_tree();
        parent = 0;
        for (int depth = 0; depth < branch_nodes; ++depth) {
            const int32_t token =
                branch_tokens[(size_t)branch * branch_nodes + depth];
            parent = append(token, depth + 1, parent);
        }
    }

    finish_visibility(tree);
    return tree;
}

int ddtree_adaptive_required_rows(
    int baseline_nodes,
    int branch_count,
    int branch_nodes,
    int tile) {
    if (!checked_shape(baseline_nodes, branch_count, branch_nodes)) return 0;
    const int rows = 1 + baseline_nodes + branch_count * branch_nodes;
    if (tile <= 1) return rows;
    if (rows > std::numeric_limits<int>::max() - (tile - 1)) return 0;
    return ((rows + tile - 1) / tile) * tile;
}

bool ddtree_adaptive_seed_tokens(
    const int32_t * top_token_ids,
    int positions,
    int top_k,
    int branch_count,
    std::vector<int32_t> & seeds) {
    seeds.clear();
    if (!top_token_ids || positions <= 0 || top_k <= branch_count ||
        branch_count <= 0) {
        return false;
    }
    seeds.reserve(branch_count);
    for (int branch = 0; branch < branch_count; ++branch) {
        seeds.push_back(top_token_ids[branch + 1]);
    }
    return true;
}

}  // namespace dflash::common
