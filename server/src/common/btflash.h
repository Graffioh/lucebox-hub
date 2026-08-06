// BTFlash prototype policy helpers.
//
// This file deliberately contains no backend or JSON dependencies so the
// request policy, deterministic branch RNG derivation, lineage mask, and
// selector can be exercised by the CPU-only unit target.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct BTFlashConfig {
    int         row_budget  = 0;
    int         k           = 1;
    int         horizon     = 0;
    int         survivors   = 1;
    int         fork_tokens = 8;
    std::string fork        = "fixed";
    std::string select      = "logprob";

    bool enabled() const { return k > 1 && horizon > 0; }
};

// Empty means valid. BT1 intentionally accepts only the two prototype graph
// widths/horizons and a single normalized-logprob winner.
std::string validate_btflash_config(const BTFlashConfig & config);

// SplitMix64-derived independent substream seed. Branch-major sampling calls
// this once per branch after drawing a request/fork seed from the main stream.
uint64_t btflash_branch_seed(uint64_t fork_seed, int branch_id);

// Raw target log-probability for one token. Used as the cheap BT1 selector
// baseline; sampling filters still decide the token itself.
double btflash_token_logprob(const float * logits, int vocab, int token);

// Return the branch with the greatest mean target log-probability. Ties are
// stable and choose the lowest branch id.
int btflash_select_normalized_logprob(const std::vector<double> & logprob_sums,
                                      const std::vector<int> & token_counts);

// Build the F16 attention mask for one K-row lockstep branch forward.
// shared_length is the physical prefix visible to every branch. Each earlier
// branch step is interleaved in branch-major rows after that prefix.
void build_btflash_mask(std::vector<uint16_t> & out,
                        int shared_length,
                        int completed_steps,
                        int width,
                        int kv_pad,
                        int q_pad);

}  // namespace dflash::common
