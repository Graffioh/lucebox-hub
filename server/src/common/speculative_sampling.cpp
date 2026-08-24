#include "speculative_sampling.h"

#include <algorithm>
#include <cmath>

namespace dflash::common {

bool SparseProposalDistribution::valid(int vocab) const {
    if (vocab <= 0 || token_ids.empty() ||
        token_ids.size() != probabilities.size()) {
        return false;
    }
    double total = 0.0;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (token_ids[i] < 0 || token_ids[i] >= vocab ||
            !std::isfinite(probabilities[i]) ||
            probabilities[i] < 0.0f) {
            return false;
        }
        for (size_t prior = 0; prior < i; ++prior) {
            if (token_ids[prior] == token_ids[i]) return false;
        }
        total += probabilities[i];
    }
    return std::fabs(total - 1.0) <= 1e-4;
}

float SparseProposalDistribution::probability_of(int32_t token) const {
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (token_ids[i] == token) return probabilities[i];
    }
    return 0.0f;
}

float speculative_acceptance_probability(
        const SamplerDistribution & target,
        const SparseProposalDistribution & proposal,
        int32_t proposed_token) {
    if (target.probabilities.empty() ||
        !proposal.valid(static_cast<int>(target.probabilities.size()))) {
        return 0.0f;
    }
    const float draft_probability =
        proposal.probability_of(proposed_token);
    if (!(draft_probability > 0.0f)) return 0.0f;
    return std::min(
        1.0f,
        target.probability_of(proposed_token) / draft_probability);
}

bool sample_speculative_residual(
        const SamplerDistribution & target,
        const SparseProposalDistribution & proposal,
        double r_uniform,
        int & out_token) {
    out_token = -1;
    if (target.probabilities.empty() || target.support.empty() ||
        !proposal.valid(static_cast<int>(target.probabilities.size())) ||
        !std::isfinite(r_uniform) || r_uniform < 0.0 || r_uniform >= 1.0) {
        return false;
    }

    double total = 0.0;
    for (int32_t token : target.support) {
        total += std::max(
            0.0f, target.probability_of(token) -
                      proposal.probability_of(token));
    }
    if (!(total > 0.0)) return false;

    const double target_mass = r_uniform * total;
    double cumulative = 0.0;
    int last_positive = -1;
    for (int32_t token : target.support) {
        const float residual = std::max(
            0.0f, target.probability_of(token) -
                      proposal.probability_of(token));
        if (residual <= 0.0f) continue;
        last_positive = token;
        cumulative += residual;
        if (target_mass <= cumulative) {
            out_token = token;
            return true;
        }
    }
    out_token = last_positive;
    return out_token >= 0;
}

}  // namespace dflash::common
