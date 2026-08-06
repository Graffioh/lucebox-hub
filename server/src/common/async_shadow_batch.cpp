#include "async_shadow_batch.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dflash::common {

AsyncShadowPlan plan_async_shadow_batch(
    const std::vector<float> & top_log_probs,
    const std::vector<int32_t> & top_token_ids,
    int draft_rows,
    int top_k,
    int branch_count,
    int min_depth,
    int max_depth,
    int committed) {
    AsyncShadowPlan out;
    if (draft_rows <= 1 || top_k < 2 || branch_count <= 0 ||
        branch_count >= top_k || committed < 0 ||
        top_log_probs.size() != (size_t)draft_rows * top_k ||
        top_token_ids.size() != top_log_probs.size()) {
        return out;
    }
    out.source_committed = committed;

    const int lo = std::max(1, min_depth);
    const int hi = std::min(draft_rows - 1, max_depth);
    float best_margin = std::numeric_limits<float>::infinity();
    int recovery_row = -1;
    for (int accepted = lo; accepted <= hi; ++accepted) {
        // draft row `accepted` produced the token rejected after exactly
        // `accepted` input rows were committed. Row accepted-1 predicts the
        // preceding token and cannot predict this recovery token.
        const int row = accepted;
        const float top1 = top_log_probs[(size_t)row * top_k];
        const float top2 = top_log_probs[(size_t)row * top_k + 1];
        if (!std::isfinite(top1) || !std::isfinite(top2)) continue;
        const float margin = top1 - top2;
        if (margin < best_margin) {
            best_margin = margin;
            out.branch_depth = accepted;
            recovery_row = row;
        }
    }
    if (out.branch_depth < 0 || recovery_row < 0) return out;

    out.confidence_margin = best_margin;
    const size_t row = (size_t)recovery_row * top_k;
    const int32_t top1_token = top_token_ids[row];
    for (int rank = 1; rank < top_k &&
         (int)out.candidates.size() < branch_count; ++rank) {
        const int32_t token = top_token_ids[row + rank];
        if (token < 0 || token == top1_token) continue;
        const bool duplicate = std::any_of(
            out.candidates.begin(), out.candidates.end(),
            [&](const AsyncShadowCandidate & candidate) {
                return candidate.pending_token == token;
            });
        if (duplicate) continue;
        out.candidates.push_back({
            committed + out.branch_depth,
            token,
            rank,
        });
    }
    if ((int)out.candidates.size() != branch_count) {
        out = {};
    }
    return out;
}

int match_async_shadow_candidate(
    const AsyncShadowPlan & plan,
    int endpoint_pos,
    int32_t pending_token,
    bool used_fast_rollback,
    int commit_count) {
    if (!used_fast_rollback || plan.source_committed < 0 ||
        commit_count != plan.branch_depth ||
        endpoint_pos != plan.source_committed + commit_count) {
        return -1;
    }
    for (size_t i = 0; i < plan.candidates.size(); ++i) {
        if (plan.candidates[i].endpoint_pos == endpoint_pos &&
            plan.candidates[i].pending_token == pending_token) {
            return (int)i;
        }
    }
    return -1;
}

}  // namespace dflash::common
