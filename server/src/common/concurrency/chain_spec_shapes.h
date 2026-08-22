// Pure host-side shape helpers for path-shaped DSpark verification.

#pragma once

#include "common/ddtree.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

inline int chain_decode_bucket_width(int lanes) {
    static constexpr int buckets[] = {
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    };
    if (lanes <= 0) return 0;
    for (int bucket : buckets) {
        if (bucket >= lanes) return bucket;
    }
    return 64;
}

// draft_tokens[0] is the already-pending root; positions 1.. form the
// proposal. DDTree's flat indices then coincide with chain depth.
inline DDTree make_chain_verify_tree(
        const std::vector<int32_t> & draft_tokens) {
    DDTree tree;
    if (draft_tokens.empty()) return tree;

    tree.n_nodes = static_cast<int>(draft_tokens.size()) - 1;
    tree.token_ids.assign(draft_tokens.begin() + 1, draft_tokens.end());
    tree.depths.resize(static_cast<size_t>(tree.n_nodes));
    tree.parents.resize(static_cast<size_t>(tree.n_nodes) + 1);
    tree.child_maps.resize(static_cast<size_t>(tree.n_nodes) + 1);
    tree.parents[0] = -1;
    for (int node = 1; node <= tree.n_nodes; ++node) {
        tree.depths[static_cast<size_t>(node) - 1] = node;
        tree.parents[static_cast<size_t>(node)] = node - 1;
        tree.child_maps[static_cast<size_t>(node) - 1]
                       [tree.token_ids[static_cast<size_t>(node) - 1]] = node;
    }

    const int width = tree.n_nodes + 1;
    tree.visibility.assign(static_cast<size_t>(width) * width, 0);
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col <= row; ++col) {
            tree.visibility[static_cast<size_t>(row) * width + col] = 1;
        }
    }
    return tree;
}

// A chain verify always includes the pending root. Depth 1 would therefore
// be an AR-equivalent target step, which is forbidden once a request has
// committed to sticky speculation. `requested == 0` means use the configured
// drafter maximum; any other invalid value fails closed.
inline int resolve_chain_verify_depth(int requested, int maximum) {
    if (maximum < 2) return 0;
    if (requested == 0) return maximum;
    return requested >= 2 && requested <= maximum ? requested : 0;
}

// Keep proposal generation at its configured maximum while allowing one
// common verify depth to be selected per round. Failure leaves the proposal
// untouched, which makes malformed controller/config output non-destructive.
inline bool truncate_chain_proposal(
        std::vector<int32_t> & draft_tokens, int verify_depth) {
    if (verify_depth < 2 ||
        verify_depth > static_cast<int>(draft_tokens.size())) {
        return false;
    }
    draft_tokens.resize(static_cast<size_t>(verify_depth));
    return true;
}

struct ChainLaunchShape {
    int spec_lanes = 0;
    int tree_bucket = 0;
    int tree_rows = 0;
    int ar_lanes = 0;
    int ar_bucket = 0;
    int accepted_rows = 0;
    int commit_rows = 0;
};

inline ChainLaunchShape chain_launch_shape(
        const std::vector<uint8_t> & admitted,
        const std::vector<int> & accepted_lengths,
        int tree_width) {
    ChainLaunchShape shape;
    const size_t count = admitted.size();
    for (size_t i = 0; i < count; ++i) {
        if (admitted[i]) {
            ++shape.spec_lanes;
            if (i < accepted_lengths.size()) {
                shape.accepted_rows += std::max(0, accepted_lengths[i]);
            }
        }
    }
    shape.ar_lanes = static_cast<int>(count) - shape.spec_lanes;
    shape.tree_bucket = chain_decode_bucket_width(shape.spec_lanes);
    shape.tree_rows = shape.tree_bucket * std::max(0, tree_width);
    shape.ar_bucket = chain_decode_bucket_width(shape.ar_lanes);
    shape.commit_rows = shape.accepted_rows + shape.ar_bucket;
    return shape;
}

// Proposal preparation is independent per request. A requested speculative
// lane whose proposal failed must be removed from both executor cohorts: it is
// a lane-local failure, never an AR fallback for that decode step.
enum class ChainLaneDisposition : uint8_t {
    AR,
    Speculation,
    Failed,
};

inline ChainLaneDisposition chain_lane_disposition(
        bool requested_speculation, bool proposal_failed) {
    if (proposal_failed) return ChainLaneDisposition::Failed;
    return requested_speculation
        ? ChainLaneDisposition::Speculation
        : ChainLaneDisposition::AR;
}

inline bool chain_lane_executes(ChainLaneDisposition disposition) {
    return disposition != ChainLaneDisposition::Failed;
}

// The pending root at path[0] was sampled by the preceding target step, so
// the ordinary sampler has already applied the min-token EOS floor to it.
// Accepted children would bypass that sampler. Stop before an EOS that is
// still below the floor so replay samples a replacement from the kept
// tip's exact logits. Once the floor is met, keep the EOS itself but discard
// deeper accepted tokens that the scheduler would hide after retirement.
template <typename IsEos>
inline size_t chain_min_tokens_safe_prefix(
        const std::vector<int32_t> & path,
        int generated_tokens_before_root,
        int min_tokens,
        IsEos is_eos) {
    const int generated = std::max(0, generated_tokens_before_root);
    for (size_t child = 1; child < path.size(); ++child) {
        if (!is_eos(path[child])) continue;
        return generated + static_cast<int>(child) < min_tokens
            ? child : child + 1;
    }
    return path.size();
}

}  // namespace dflash::common
