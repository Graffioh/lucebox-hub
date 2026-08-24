#pragma once

#include "sampler.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Sparse draft law q over the selector's retained top-k candidates.
struct SparseProposalDistribution {
    std::vector<int32_t> token_ids;
    std::vector<float> probabilities;

    bool valid(int vocab) const;
    float probability_of(int32_t token) const;
};

// min(1, p(x) / q(x)) for the draft token x.
float speculative_acceptance_probability(
    const SamplerDistribution & target,
    const SparseProposalDistribution & proposal,
    int32_t proposed_token);

// Sample the rejection residual proportional to max(p - q, 0).
// Returns false only when the inputs are invalid or the residual has zero mass.
bool sample_speculative_residual(
    const SamplerDistribution & target,
    const SparseProposalDistribution & proposal,
    double r_uniform,
    int & out_token);

}  // namespace dflash::common
