#pragma once

// Model-neutral, hardware-aware request admission for adaptive verification.
//
// DSpark's scheduler ranks candidate tokens by cumulative survival probability
// and grows the verification batch while expected throughput improves. This
// helper applies the same policy at request granularity. The concrete
// speculator supplies expected useful tokens; the engine supplies the observed
// cost of each mixed route shape (active requests, speculative requests).
// DDTree can learn value from accepted paths, while DSpark can use its
// calibrated confidence head directly.

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
    // Cheap request prior used only to order otherwise-unknown candidates.
    // Measured expected tokens replace it after the first verification.
    double routing_prior = 0.0;
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
        route_cost_us_.clear();
        route_cost_known_.clear();
    }

    void observe_autoregressive(int batch_size, double elapsed_us) {
        observe_route(batch_size, /*speculative_requests=*/0, elapsed_us);
    }

    // Compatibility helper for engines whose whole batch takes one
    // speculative route. Mixed executors should call observe_route directly.
    void observe_speculation(int batch_size, double elapsed_us) {
        observe_route(batch_size, batch_size, elapsed_us);
    }

    void observe_route(int active_requests, int speculative_requests,
                       double elapsed_us) {
        if (active_requests <= 0 || speculative_requests < 0 ||
            speculative_requests > active_requests ||
            !std::isfinite(elapsed_us) || elapsed_us <= 0.0) {
            return;
        }
        const size_t rows = static_cast<size_t>(active_requests) + 1;
        if (route_cost_us_.size() < rows) route_cost_us_.resize(rows);
        if (route_cost_known_.size() < rows) route_cost_known_.resize(rows);
        std::vector<double> & costs =
            route_cost_us_[(size_t)active_requests];
        std::vector<bool> & known =
            route_cost_known_[(size_t)active_requests];
        const size_t cols = static_cast<size_t>(speculative_requests) + 1;
        if (costs.size() < cols) costs.resize(cols, 0.0);
        if (known.size() < cols) known.resize(cols, false);
        if (!known[(size_t)speculative_requests]) {
            costs[(size_t)speculative_requests] = elapsed_us;
            known[(size_t)speculative_requests] = true;
            return;
        }
        costs[(size_t)speculative_requests] =
            config_.cost_ewma_alpha * elapsed_us +
            (1.0 - config_.cost_ewma_alpha) *
                costs[(size_t)speculative_requests];
    }

    bool has_autoregressive_cost(int batch_size) const {
        return has_route_cost(batch_size, /*speculative_requests=*/0);
    }

    bool has_speculation_cost(int batch_size) const {
        return has_route_cost(batch_size, batch_size);
    }

    double autoregressive_cost_us(int batch_size) const {
        return route_cost_us(batch_size, /*speculative_requests=*/0);
    }

    double speculation_cost_us(int batch_size) const {
        return route_cost_us(batch_size, batch_size);
    }

    bool has_route_cost(int active_requests,
                        int speculative_requests) const {
        return active_requests > 0 && speculative_requests >= 0 &&
            speculative_requests <= active_requests &&
            static_cast<size_t>(active_requests) <
                route_cost_known_.size() &&
            static_cast<size_t>(speculative_requests) <
                route_cost_known_[(size_t)active_requests].size() &&
            route_cost_known_[(size_t)active_requests]
                             [(size_t)speculative_requests];
    }

    double route_cost_us(int active_requests,
                         int speculative_requests) const {
        return has_route_cost(active_requests, speculative_requests)
            ? route_cost_us_[(size_t)active_requests]
                            [(size_t)speculative_requests]
            : std::numeric_limits<double>::infinity();
    }

    AdaptiveVerificationDecision select(
            int active_requests,
            const std::vector<AdaptiveVerificationCandidate> & candidates,
            int max_speculative_requests =
                std::numeric_limits<int>::max(),
            bool probe_uncalibrated_with_verifier = false,
            bool project_cost_from_higher_occupancy = false,
            bool evaluate_nonconvex_prefixes = false) const {
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
                if (a.routing_prior != b.routing_prior) {
                    return a.routing_prior > b.routing_prior;
                }
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
        const int prefix_limit = std::max(0, std::min(
            active_requests, max_speculative_requests));
        auto effective_route_cost = [&](int speculative_requests) {
            if (has_route_cost(active_requests, speculative_requests)) {
                return route_cost_us(
                    active_requests, speculative_requests);
            }
            return project_cost_from_higher_occupancy
                ? projected_route_cost_us(
                      active_requests, speculative_requests)
                : std::numeric_limits<double>::infinity();
        };
        bool measured_route = false;
        for (int k = 1; k <= static_cast<int>(known.size()) &&
                        k <= prefix_limit; ++k) {
            expected_total += known[(size_t)k - 1].expected_tokens - 1.0;
            const double route_us = effective_route_cost(k);
            if (!std::isfinite(route_us)) {
                if (missing_cost_prefix == 0) missing_cost_prefix = k;
                if (!evaluate_nonconvex_prefixes) break;
                continue;
            }
            measured_route = true;
            const double throughput = expected_total / route_us;
            // Request-level route costs can be non-convex: on wide GPUs a
            // k=3 compact bundle may win even when k=1 and k=2 do not.
            if (!evaluate_nonconvex_prefixes && throughput <= best) break;
            if (throughput > best) {
                best = throughput;
                if (throughput >= required) {
                    admitted_prefix = k;
                    admitted_goodput = throughput;
                }
            }
        }

        // By default, calibration_request asks the concrete adapter for a
        // cheap confidence signal. Verifier-side calibration is opt-in below
        // for adapters whose separate confidence pass would duplicate work.

        // A calibrated route shape without a hardware sample gets one bounded
        // probe. Grow a measured winning prefix by one, but do not walk every
        // smaller shape after another measured width has already lost.
        const bool probe_missing = missing_cost_prefix > 0 &&
            (!measured_route ||
             (admitted_prefix > 0 &&
              missing_cost_prefix == admitted_prefix + 1));
        if (probe_missing) {
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
        // A speculator without a cheap calibrated confidence head can turn a
        // bounded verification bundle into useful decoding and calibration.
        // Prefer the executor's widest compact bundle because request-level
        // route costs can be non-convex. Once sampled, its observed accepted
        // yield gates later unknown bundles; perfect-acceptance optimism is
        // used only before a yield sample exists. DDTree uses this at high
        // occupancy, while DSpark can keep using its confidence head.
        if (probe_uncalibrated_with_verifier && !unknown.empty() &&
            !out.exploring &&
            static_cast<int>(out.requests.size()) < prefix_limit) {
            const int probe_count = std::min(
                static_cast<int>(unknown.size()),
                prefix_limit - static_cast<int>(out.requests.size()));
            const int probe_prefix =
                static_cast<int>(out.requests.size()) + probe_count;
            double optimistic_total =
                static_cast<double>(active_requests);
            for (int request : out.requests) {
                const auto candidate = std::find_if(
                    known.begin(), known.end(),
                    [request](const AdaptiveVerificationCandidate & item) {
                        return item.request == request;
                    });
                if (candidate != known.end()) {
                    optimistic_total += candidate->expected_tokens - 1.0;
                }
            }
            for (int i = 0; i < probe_count; ++i) {
                optimistic_total +=
                    unknown[(size_t)i].maximum_tokens - 1.0;
            }
            const double probe_route_us =
                effective_route_cost(probe_prefix);
            const bool unseen_shape = !std::isfinite(probe_route_us);
            const bool can_win = unseen_shape ||
                optimistic_total / probe_route_us >= required;
            if (can_win) {
                for (int i = 0; i < probe_count; ++i) {
                    out.requests.push_back(unknown[(size_t)i].request);
                }
                out.calibration_request = -1;
                out.exploring = true;
            }
        }
        return out;
    }

private:
    double projected_route_cost_us(
            int active_requests, int speculative_requests) const {
        if (!has_autoregressive_cost(active_requests)) {
            return std::numeric_limits<double>::infinity();
        }
        for (int higher = active_requests + 1;
             static_cast<size_t>(higher) < route_cost_known_.size();
             ++higher) {
            if (!has_autoregressive_cost(higher) ||
                !has_route_cost(higher, speculative_requests)) {
                continue;
            }
            const double relative_cost =
                route_cost_us(higher, speculative_requests) /
                autoregressive_cost_us(higher);
            return relative_cost * autoregressive_cost_us(active_requests);
        }
        return std::numeric_limits<double>::infinity();
    }

    static AdaptiveVerificationConfig sanitize(
            AdaptiveVerificationConfig config) {
        config.minimum_gain = std::max(1.0, config.minimum_gain);
        config.cost_ewma_alpha =
            std::clamp(config.cost_ewma_alpha, 0.0, 1.0);
        return config;
    }

    AdaptiveVerificationConfig config_;
    std::vector<std::vector<double>> route_cost_us_;
    std::vector<std::vector<bool>> route_cost_known_;
};

}  // namespace dflash::common
