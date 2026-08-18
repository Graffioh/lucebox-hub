#pragma once

// Model-neutral confidence value shared by adaptive speculative decoders.
//
// The scheduler never interprets prompt text.  A concrete speculator exposes
// conditional prefix-survival confidence and the ranker converts it to useful
// tokens.  DDTree obtains the values from draft top-1 probabilities; DSpark
// obtains them from its confidence head.  Target-verified accepted tokens are
// deliberately tracked separately and calibrate these estimates online.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dflash::common {

enum class SpeculatorKind : std::uint8_t {
    DDTree,
    DSpark,
};

// Confidence is not necessarily available before routing.  In particular,
// both current adapters must run their drafter; an already-selected proposal
// can expose it for free, while a pre-route DDTree probe costs an extra pass.
enum class SpeculationConfidenceCost : std::uint8_t {
    Unavailable,
    ExtraDraftPass,
    PiggybacksOnProposal,
};

inline double conditional_prefix_survival(
        const float * confidence, int count) {
    double survival = 1.0;
    for (int i = 0; confidence && i < count; ++i) {
        const double conditional = std::clamp(
            std::isfinite(confidence[i])
                ? static_cast<double>(confidence[i]) : 0.0,
            0.0, 1.0);
        survival *= conditional;
    }
    return survival;
}

inline double expected_tokens_from_conditional_confidence(
        const float * confidence, int count) {
    double expected = 1.0;
    double survival = 1.0;
    for (int i = 0; confidence && i < count; ++i) {
        const double conditional = std::clamp(
            std::isfinite(confidence[i])
                ? static_cast<double>(confidence[i]) : 0.0,
            0.0, 1.0);
        survival *= conditional;
        expected += survival;
    }
    return expected;
}

// Non-owning view for a speculator hot path. DSpark can select its verifier
// width without allocating; engines that retain confidence across steps copy
// the same view into SpeculationConfidenceEstimate below.
struct SpeculationConfidenceView {
    SpeculatorKind speculator = SpeculatorKind::DDTree;
    SpeculationConfidenceCost cost =
        SpeculationConfidenceCost::Unavailable;
    bool posthoc_calibrated = false;
    const float * conditional = nullptr;
    int count = 0;

    bool available() const {
        return cost != SpeculationConfidenceCost::Unavailable &&
            conditional && count > 0;
    }

    double prefix_survival(int prefix_tokens) const {
        return conditional_prefix_survival(
            conditional, std::max(0, std::min(count, prefix_tokens)));
    }

    double expected_tokens() const {
        return expected_tokens_from_conditional_confidence(
            conditional, std::max(0, count));
    }
};

inline SpeculationConfidenceView make_speculation_confidence_view(
        SpeculatorKind speculator, const float * confidence, int count,
        SpeculationConfidenceCost cost,
        bool posthoc_calibrated = false) {
    return {
        speculator,
        confidence && count > 0
            ? cost : SpeculationConfidenceCost::Unavailable,
        posthoc_calibrated,
        confidence,
        std::max(0, count),
    };
}

struct SpeculationConfidenceEstimate {
    SpeculatorKind speculator = SpeculatorKind::DDTree;
    SpeculationConfidenceCost cost =
        SpeculationConfidenceCost::Unavailable;
    // True only when the artifact guarantees probability calibration.  Raw
    // DDTree softmax and the current DSpark sigmoid head both leave this false;
    // the ranker still uses them as ordering estimates and corrects them from
    // target outcomes.
    bool posthoc_calibrated = false;
    std::vector<float> conditional;

    bool available() const {
        return cost != SpeculationConfidenceCost::Unavailable &&
            !conditional.empty();
    }

    double maximum_tokens() const {
        return 1.0 + static_cast<double>(conditional.size());
    }

    double expected_tokens() const {
        return expected_tokens_from_conditional_confidence(
            conditional.data(), static_cast<int>(conditional.size()));
    }

    SpeculationConfidenceEstimate limited_to(
            int speculative_tokens) const {
        SpeculationConfidenceEstimate out = *this;
        const std::size_t limit = static_cast<std::size_t>(
            std::max(0, speculative_tokens));
        if (out.conditional.size() > limit) {
            out.conditional.resize(limit);
        }
        return out;
    }
};

inline SpeculationConfidenceEstimate make_speculation_confidence_estimate(
        SpeculatorKind speculator, const float * confidence, int count,
        SpeculationConfidenceCost cost,
        bool posthoc_calibrated = false) {
    const SpeculationConfidenceView view = make_speculation_confidence_view(
        speculator, confidence, count, cost, posthoc_calibrated);
    SpeculationConfidenceEstimate out;
    out.speculator = view.speculator;
    out.cost = view.cost;
    out.posthoc_calibrated = view.posthoc_calibrated;
    if (!view.available()) return out;
    out.conditional.reserve(static_cast<std::size_t>(view.count));
    for (int i = 0; i < view.count; ++i) {
        out.conditional.push_back(static_cast<float>(std::clamp(
            std::isfinite(view.conditional[i])
                ? static_cast<double>(view.conditional[i]) : 0.0,
            0.0, 1.0)));
    }
    return out;
}

struct SpeculationConfidenceProfile {
    SpeculatorKind speculator = SpeculatorKind::DDTree;
    // Half-token bins are coarse enough to transfer target observations
    // between similar requests without pretending raw confidence is exact.
    int expected_half_tokens = 2;

    bool operator<(const SpeculationConfidenceProfile & other) const {
        if (speculator != other.speculator) {
            return static_cast<std::uint8_t>(speculator) <
                static_cast<std::uint8_t>(other.speculator);
        }
        return expected_half_tokens < other.expected_half_tokens;
    }
};

inline SpeculationConfidenceProfile speculation_confidence_profile(
        const SpeculationConfidenceEstimate & estimate) {
    return {
        estimate.speculator,
        static_cast<int>(std::floor(estimate.expected_tokens() * 2.0)),
    };
}

inline SpeculationConfidenceProfile speculation_confidence_profile(
        SpeculatorKind speculator, double expected_tokens) {
    const double sanitized = std::max(
        1.0, std::isfinite(expected_tokens) ? expected_tokens : 1.0);
    return {
        speculator,
        static_cast<int>(std::floor(sanitized * 2.0)),
    };
}

}  // namespace dflash::common
