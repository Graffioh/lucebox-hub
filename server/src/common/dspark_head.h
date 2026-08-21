#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

bool dspark_markov_correct_greedy_chain(const DraftWeights & dw,
                                        ggml_backend_t backend,
                                        DFlashTarget & target,
                                        const float * local_hidden,
                                        int q_len,
                                        int32_t last_tok,
                                        float confidence_threshold,
                                        std::vector<int32_t> & draft_tok);

// Fused variant: base logits (one lm_head matmul over all candidates) +
// unrolled Markov correction chain + in-graph argmax feeding the next
// step's get_rows, all in ONE graph on the draft backend. No host logits
// round-trip. When confidence_out is non-null and the checkpoint has a
// compatible confidence head, returns one score per candidate from the same
// graph and host synchronization as the token ids. `confidence_hidden`, when
// non-null, has the same padded layout as `local_hidden` and supplies the
// pre-output-norm state expected by the confidence head. Callers without a
// separate state retain the legacy behavior by leaving it null.
bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out = nullptr,
                                              const float * confidence_hidden = nullptr);

// Outputs embedded in a caller-owned graph for a lane-batched DSpark chain.
// Each depth tensor is shaped [n_lanes], with confidence in [1, n_lanes].
struct DSparkBatchedChainOutputs {
    int n_lanes = 0;
    int q_len = 0;
    std::vector<ggml_tensor *> tokens;
    std::vector<ggml_tensor *> confidence;
};

// Append a depth-major, multi-lane Markov chain to an existing draft graph.
// The lane backbones remain independent; their hidden tensors stay on-device.
// One lm_head matmul covers every (depth, lane), then each depth performs a
// batched Markov lookup, correction, argmax, and calibrated confidence head.
bool build_dspark_markov_batched_chain(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const DraftWeights & dw,
    ggml_tensor * lm_head,
    const std::vector<ggml_tensor *> & hidden_by_lane,
    const std::vector<ggml_tensor *> & prenorm_by_lane,
    ggml_tensor * seed_tokens,
    int q_len,
    bool want_confidence,
    DSparkBatchedChainOutputs & out);

// DDTree candidate generation with the Markov correction: base logits for
// all n_tokens positions in ONE lm_head matmul; rows 1..n-1 get the low-rank
// previous-token bias chained along the main (argmax) path; top-K extracted
// on host via extract_draft_topk. Output contract matches
// DFlashTarget::project_hidden_to_topk (row 0 = seed position, uncorrected).
bool dspark_markov_project_topk(const DraftWeights & dw,
                                ggml_backend_t backend,
                                ggml_tensor * lm_head,
                                const float * hidden,
                                int n_tokens, int K, float temperature,
                                int32_t last_tok,
                                std::vector<float> & top_log_probs,
                                std::vector<int32_t> & top_token_ids);

}  // namespace dflash::common
