#pragma once

// Model-neutral, hardware-aware request admission for adaptive verification.
//
// DSpark's scheduler ranks candidate tokens by cumulative survival probability
// and grows the verification batch while expected throughput improves. This
// helper applies the same policy at request granularity. The concrete
// speculator supplies expected useful tokens; the engine supplies observed AR
// and speculative subbatch costs. DDTree can learn value from accepted paths,
// while DSpark can use its calibrated confidence head directly.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace dflash::common {

struct AdaptiveVerificationCandidate {
    int request = -1;
    // Includes the one token that ordinary AR would emit.
    double expected_tokens = 1.0;
    double maximum_tokens = 1.0;
    bool calibrated = false;
};

struct AdaptiveVerificationDecision {
    std::vector<int> requests;
    // The adapter may cheaply calibrate one request (for DDTree, from draft
    // entropy; for DSpark, from its confidence head) before applying requests.
    int calibration_request = -1;
    bool exploring = false;
    double predicted_gain = 1.0;
};

struct AdaptiveVerificationConfig {
    // Preserve margin for timing noise and route-switch overhead.
    double minimum_gain = 1.05;
    double cost_ewma_alpha = 0.35;
};

// Convert conditional survival confidence into expected useful tokens:
// 1 AR token plus the probability of reaching every speculative prefix.
// DSpark supplies calibrated confidence-head values directly.
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

class AdaptiveVerificationRanker {
public:
    AdaptiveVerificationRanker() = default;
    explicit AdaptiveVerificationRanker(AdaptiveVerificationConfig config)
        : config_(sanitize(config)) {}

    void reset() {
        ar_cost_us_.clear();
        spec_cost_us_.clear();
        ar_cost_known_.clear();
        spec_cost_known_.clear();
    }

    void observe_autoregressive(int batch_size, double elapsed_us) {
        observe_cost(ar_cost_us_, ar_cost_known_, batch_size, elapsed_us);
    }

    void observe_speculation(int batch_size, double elapsed_us) {
        observe_cost(spec_cost_us_, spec_cost_known_, batch_size, elapsed_us);
    }

    bool has_autoregressive_cost(int batch_size) const {
        return has_cost(ar_cost_known_, batch_size);
    }

    bool has_speculation_cost(int batch_size) const {
        return has_cost(spec_cost_known_, batch_size);
    }

    double autoregressive_cost_us(int batch_size) const {
        return cost(ar_cost_us_, ar_cost_known_, batch_size);
    }

    double speculation_cost_us(int batch_size) const {
        return cost(spec_cost_us_, spec_cost_known_, batch_size);
    }

    AdaptiveVerificationDecision select(
            int active_requests,
            const std::vector<AdaptiveVerificationCandidate> & candidates)
            const {
        AdaptiveVerificationDecision out;
        if (active_requests <= 0 || candidates.empty() ||
            !has_autoregressive_cost(active_requests)) {
            // First observe the exact all-AR baseline for this occupancy.
            return out;
        }

        std::vector<AdaptiveVerificationCandidate> known;
        std::vector<AdaptiveVerificationCandidate> unknown;
        known.reserve(candidates.size());
        unknown.reserve(candidates.size());
        for (AdaptiveVerificationCandidate candidate : candidates) {
            if (candidate.request < 0) continue;
            candidate.maximum_tokens = std::max(
                1.0, std::isfinite(candidate.maximum_tokens)
                         ? candidate.maximum_tokens : 1.0);
            candidate.expected_tokens = std::clamp(
                std::isfinite(candidate.expected_tokens)
                    ? candidate.expected_tokens : 1.0,
                1.0, candidate.maximum_tokens);
            (candidate.calibrated ? known : unknown).push_back(candidate);
        }
        if (known.empty() && unknown.empty()) return out;

        auto by_value = [](const AdaptiveVerificationCandidate & a,
                           const AdaptiveVerificationCandidate & b) {
            if (a.expected_tokens != b.expected_tokens) {
                return a.expected_tokens > b.expected_tokens;
            }
            return a.request < b.request;
        };
        std::stable_sort(known.begin(), known.end(), by_value);
        std::stable_sort(unknown.begin(), unknown.end(),
            [](const AdaptiveVerificationCandidate & a,
               const AdaptiveVerificationCandidate & b) {
                if (a.maximum_tokens != b.maximum_tokens) {
                    return a.maximum_tokens > b.maximum_tokens;
                }
                return a.request < b.request;
            });
        if (!unknown.empty()) {
            out.calibration_request = unknown.front().request;
        }

        const double baseline =
            static_cast<double>(active_requests) /
            autoregressive_cost_us(active_requests);
        const double required = baseline * config_.minimum_gain;

        int admitted_prefix = 0;
        int missing_cost_prefix = 0;
        double best = baseline;
        double admitted_goodput = baseline;
        double expected_total = static_cast<double>(active_requests);
        for (int k = 1; k <= static_cast<int>(known.size()) &&
                        k <= active_requests; ++k) {
            expected_total += known[(size_t)k - 1].expected_tokens - 1.0;
            double route_us = 0.0;
            if (!combined_cost(active_requests, k, route_us)) {
                missing_cost_prefix = k;
                break;
            }
            const double throughput = expected_total / route_us;
            // DSpark's greedy policy stops at the first non-improving
            // candidate because candidates are already ranked by survival.
            if (throughput <= best) break;
            best = throughput;
            if (throughput >= required) {
                admitted_prefix = k;
                admitted_goodput = throughput;
            }
        }

        // Uncalibrated requests are never sent through an expensive target
        // verification merely to discover their value. calibration_request
        // asks the concrete adapter for its cheap confidence signal instead.

        // A calibrated route shape without a hardware sample gets one bounded
        // probe. Larger prefixes are not explored after a measured decline.
        if (missing_cost_prefix > 0) {
            out.requests.reserve((size_t)missing_cost_prefix);
            for (int i = 0; i < missing_cost_prefix; ++i) {
                out.requests.push_back(known[(size_t)i].request);
            }
            out.exploring = true;
            return out;
        }

        if (admitted_prefix > 0) {
            out.requests.reserve((size_t)admitted_prefix);
            for (int i = 0; i < admitted_prefix; ++i) {
                out.requests.push_back(known[(size_t)i].request);
            }
            out.predicted_gain = admitted_goodput / baseline;
        }
        return out;
    }

private:
    static AdaptiveVerificationConfig sanitize(
            AdaptiveVerificationConfig config) {
        config.minimum_gain = std::max(1.0, config.minimum_gain);
        config.cost_ewma_alpha =
            std::clamp(config.cost_ewma_alpha, 0.0, 1.0);
        return config;
    }

    void observe_cost(std::vector<double> & costs,
                      std::vector<bool> & known,
                      int batch_size, double elapsed_us) {
        if (batch_size <= 0 || !std::isfinite(elapsed_us) ||
            elapsed_us <= 0.0) {
            return;
        }
        const size_t needed = static_cast<size_t>(batch_size) + 1;
        if (costs.size() < needed) costs.resize(needed, 0.0);
        if (known.size() < needed) known.resize(needed, false);
        if (!known[(size_t)batch_size]) {
            costs[(size_t)batch_size] = elapsed_us;
            known[(size_t)batch_size] = true;
            return;
        }
        costs[(size_t)batch_size] =
            config_.cost_ewma_alpha * elapsed_us +
            (1.0 - config_.cost_ewma_alpha) *
                costs[(size_t)batch_size];
    }

    static bool has_cost(const std::vector<bool> & known, int batch_size) {
        return batch_size == 0 ||
            (batch_size > 0 && static_cast<size_t>(batch_size) < known.size() &&
             known[(size_t)batch_size]);
    }

    static double cost(const std::vector<double> & costs,
                       const std::vector<bool> & known, int batch_size) {
        if (batch_size == 0) return 0.0;
        return has_cost(known, batch_size)
            ? costs[(size_t)batch_size]
            : std::numeric_limits<double>::infinity();
    }

    bool combined_cost(int active_requests, int speculative_requests,
                       double & elapsed_us) const {
        const int ar_requests = active_requests - speculative_requests;
        if (speculative_requests <= 0 || ar_requests < 0 ||
            !has_speculation_cost(speculative_requests) ||
            !has_autoregressive_cost(ar_requests)) {
            return false;
        }
        elapsed_us = speculation_cost_us(speculative_requests) +
                     autoregressive_cost_us(ar_requests);
        return std::isfinite(elapsed_us) && elapsed_us > 0.0;
    }

    AdaptiveVerificationConfig config_;
    std::vector<double> ar_cost_us_;
    std::vector<double> spec_cost_us_;
    std::vector<bool> ar_cost_known_;
    std::vector<bool> spec_cost_known_;
};

}  // namespace dflash::common
