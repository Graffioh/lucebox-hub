// Async shadow batching policy helpers.
//
// A shadow is only a speculative proposal cache entry. The target's ordinary
// acceptance path remains authoritative: a result may be reused only when its
// assumed future (committed position, pending token) exactly matches the state
// produced by target verification.

#pragma once

#include <cstdint>
#include <vector>

namespace dflash::common {

struct AsyncShadowCandidate {
    int endpoint_pos = -1;
    int32_t pending_token = -1;
    int draft_rank = -1;
};

struct AsyncShadowPlan {
    int source_committed = -1;
    int branch_depth = -1;
    float confidence_margin = 0.0f;
    std::vector<AsyncShadowCandidate> candidates;
};

// Pick the lowest-confidence eligible rejection position, then use ranks
// 1..branch_count as likely recovery tokens. With the DFlash chain layout,
// accepting k input rows rejects draft_tok[k] against target row k-1, so the
// recovery candidates come from projected draft row k (not k-1). The cache
// key endpoint is committed+k. The all-accepted endpoint is not eligible:
// this block has no draft logits for its following target bonus token.
AsyncShadowPlan plan_async_shadow_batch(
    const std::vector<float> & top_log_probs,
    const std::vector<int32_t> & top_token_ids,
    int draft_rows,
    int top_k,
    int branch_count,
    int min_depth,
    int max_depth,
    int committed);

// Return the packed branch index only for an exact endpoint match.
// A shadow is valid only for the exact fast-rollback transition it modeled.
// Legacy replay may coincidentally produce the same position/token pair but
// has different target feature state and must never match.
int match_async_shadow_candidate(
    const AsyncShadowPlan & plan,
    int endpoint_pos,
    int32_t pending_token,
    bool used_fast_rollback,
    int commit_count);

}  // namespace dflash::common
