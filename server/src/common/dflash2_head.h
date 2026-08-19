#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// DFlash 2 candidate selector for greedy chain drafting.
//
// For every drafted block position the target lm_head logits are reduced to
// the selector's top-k candidates (log-probs, so per-position constants do
// not matter for the argmax), then one path is traced through them:
//   score(c) = logp(c) + < pred_cb[prev] * hproj(h_pos), succ_cb[c] >
//   prev     = argmax_c score(c)
// starting from the block seed `last_tok`. Runs the projections (hproj GEMV
// and codebook row gathers) in one small graph on `backend`, the k-way path
// search on the host. Fills draft_tok = [last_tok, tok_1 .. tok_{q_len-1}].
bool dflash2_select_chain(const DraftWeights & dw,
                          ggml_backend_t backend,
                          DFlashTarget & target,
                          const float * local_hidden,
                          int q_len,
                          int32_t last_tok,
                          std::vector<int32_t> & draft_tok);

}  // namespace dflash::common
