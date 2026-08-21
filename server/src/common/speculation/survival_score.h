// Reusable scoring contract for conditional acceptance hazards.
#pragma once

#include "common/speculation/speculator.h"

#include <algorithm>
#include <vector>

namespace dflash::common {

// Each hazard is a probability-like, monotone-in-acceptance score for one
// position conditioned on accepting its prefix. The yield includes the root.
inline double hazard_survival_yield(
        const std::vector<double> & hazards, int max_accept) {
    if (max_accept <= 1) return 1.0;
    double expected = 1.0;
    double survival = 1.0;
    const int depth = std::min<int>(
        static_cast<int>(hazards.size()), max_accept - 1);
    for (int i = 0; i < depth; ++i) {
        const double hazard =
            std::clamp(hazards[static_cast<size_t>(i)], 0.0, 1.0);
        survival *= hazard;
        expected += survival;
    }
    return std::clamp(expected, 1.0, static_cast<double>(max_accept));
}

class ConfidenceVectorScorer {
public:
    ActivationEstimate score(
            const std::vector<float> & confidences, int max_accept) const {
        ActivationEstimate estimate;
        const int depth = std::max(
            0, std::min<int>(
                static_cast<int>(confidences.size()), max_accept - 1));
        estimate.conditional_hazards.reserve(static_cast<size_t>(depth));
        for (int i = 0; i < depth; ++i) {
            estimate.conditional_hazards.push_back(
                std::clamp<double>(
                    confidences[static_cast<size_t>(i)], 0.0, 1.0));
        }
        estimate.expected_yield = hazard_survival_yield(
            estimate.conditional_hazards, max_accept);
        return estimate;
    }
};

}  // namespace dflash::common
