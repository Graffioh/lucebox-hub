#pragma once

// Model-neutral, hardware-aware request admission for adaptive verification.
//
// DSpark's scheduler ranks candidate tokens by cumulative survival probability
// and grows the verification batch while expected throughput improves. This
// helper applies the same policy at request granularity. The concrete
// speculator supplies expected useful tokens; the engine supplies the observed
// cost of each exact mixed route shape (active requests, speculative requests).
// DDTree can learn value from accepted paths, while DSpark can use its
// calibrated confidence head directly. One ranker instance represents one
// fixed speculator/proposal shape; adapters with ragged verification work must
// keep separate rankers for distinct work buckets.

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace dflash::common {

struct AdaptiveVerificationCandidate {
    int request = -1;
    // Includes the one token that ordinary AR would emit.
    double expected_tokens = 1.0;
    double maximum_tokens = 1.0;
    bool calibrated = false;
    // Cheap request prior used only to order otherwise-unknown candidates.
    // Request-local measurements smoothly replace it during warmup.
    double routing_prior = 0.0;
    // Number of request-local observations supporting the estimate. Shared
    // cohort evidence must never relax AR-peer protection for a new request.
    std::size_t evidence_samples = 0;
    // Generated output already committed for this request. Proven-useful peers
    // with less progress receive compact lanes first to avoid cohort stragglers.
    int progress_tokens = 0;
};

struct AdaptiveVerificationYieldEstimate {
    double expected_tokens = 1.0;
    std::size_t evidence_samples = 0;
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
    // Protect requests that remain AR in a mixed route. A wider slowdown is
    // allowed only after every active request has demonstrated useful
    // speculative yield, so a homogeneous cohort can amortize the work.
    double maximum_ar_peer_slowdown = 1.10;
    double homogeneous_minimum_expected_tokens = 1.5;
    // A slow mixed route is safe only when every request benefits similarly;
    // otherwise scarce verifier lanes create a low-yield AR tail.
    double homogeneous_minimum_relative_yield = 0.60;
    std::size_t homogeneous_minimum_samples = 4;
    std::size_t routing_prior_minimum_samples = 4;
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
        request_yield_.clear();
        routing_prior_yield_.clear();
    }

    void observe_request_yield(std::uint64_t request,
                               double emitted_tokens) {
        if (!valid_yield(emitted_tokens)) return;
        update_yield(request_yield_[request], emitted_tokens);
    }

    void forget_request(std::uint64_t request) {
        request_yield_.erase(request);
    }

    std::optional<double> request_expected_tokens(
            std::uint64_t request) const {
        const auto it = request_yield_.find(request);
        return it == request_yield_.end()
            ? std::nullopt
            : std::optional<double>(it->second.expected_tokens);
    }

    std::size_t request_yield_samples(std::uint64_t request) const {
        const auto it = request_yield_.find(request);
        return it == request_yield_.end() ? 0 : it->second.samples;
    }

    bool has_stable_evidence(std::size_t samples) const {
        return samples >= config_.homogeneous_minimum_samples;
    }

    bool has_useful_yield(double expected_tokens) const {
        return std::isfinite(expected_tokens) &&
            expected_tokens >=
                config_.homogeneous_minimum_expected_tokens;
    }

    bool forms_homogeneous_cohort(
            const std::vector<AdaptiveVerificationCandidate> & candidates,
            bool require_stable_local_evidence = true) const {
        if (candidates.empty()) return false;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = 1.0;
        for (const AdaptiveVerificationCandidate & candidate : candidates) {
            if (!candidate.calibrated ||
                !has_useful_yield(candidate.expected_tokens) ||
                (require_stable_local_evidence &&
                 !has_stable_evidence(candidate.evidence_samples))) {
                return false;
            }
            minimum = std::min(minimum, candidate.expected_tokens);
            maximum = std::max(maximum, candidate.expected_tokens);
        }
        return minimum >=
            config_.homogeneous_minimum_relative_yield * maximum;
    }

    void observe_routing_prior_yield(double routing_prior,
                                     double emitted_tokens) {
        if (!std::isfinite(routing_prior) || !valid_yield(emitted_tokens)) {
            return;
        }
        update_yield(routing_prior_yield_[routing_prior], emitted_tokens);
    }

    std::optional<double> routing_prior_expected_tokens(
            double routing_prior) const {
        if (!std::isfinite(routing_prior)) return std::nullopt;
        const auto it = routing_prior_yield_.find(routing_prior);
        return it == routing_prior_yield_.end() ||
                it->second.samples < config_.routing_prior_minimum_samples
            ? std::nullopt : std::optional<double>(it->second.expected_tokens);
    }

    std::size_t routing_prior_yield_samples(double routing_prior) const {
        if (!std::isfinite(routing_prior)) return 0;
        const auto it = routing_prior_yield_.find(routing_prior);
        return it == routing_prior_yield_.end() ? 0 : it->second.samples;
    }

    std::optional<AdaptiveVerificationYieldEstimate> estimate_request_yield(
            std::uint64_t request, double routing_prior) const {
        const auto request_it = request_yield_.find(request);
        const auto prior_it = std::isfinite(routing_prior)
            ? routing_prior_yield_.find(routing_prior)
            : routing_prior_yield_.end();
        const bool has_request = request_it != request_yield_.end();
        const bool has_prior = prior_it != routing_prior_yield_.end() &&
            prior_it->second.samples >= config_.routing_prior_minimum_samples;
        if (!has_request && !has_prior) return std::nullopt;

        AdaptiveVerificationYieldEstimate out;
        if (!has_request) {
            out.expected_tokens = prior_it->second.expected_tokens;
            out.evidence_samples = 0;
            return out;
        }

        const YieldEstimate & request_estimate = request_it->second;
        out.expected_tokens = request_estimate.expected_tokens;
        out.evidence_samples = request_estimate.samples;
        if (!has_prior) return out;

        const YieldEstimate & prior_estimate = prior_it->second;
        if (request_estimate.samples < config_.homogeneous_minimum_samples) {
            // Shrink the first few noisy request observations toward a stable
            // cohort mean. Once request-local evidence is stable, its measured
            // magnitude fully replaces the prior for goodput decisions.
            const double local_weight = std::min(
                1.0,
                static_cast<double>(request_estimate.samples) /
                    static_cast<double>(
                        config_.homogeneous_minimum_samples));
            out.expected_tokens = prior_estimate.expected_tokens +
                local_weight * (request_estimate.expected_tokens -
                                prior_estimate.expected_tokens);
        }
        return out;
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

    bool has_exact_profile(int active_requests,
                           int max_speculative_requests) const {
        if (!has_autoregressive_cost(active_requests)) return false;
        const int limit = std::max(0, std::min(
            active_requests, max_speculative_requests));
        for (int k = 1; k <= limit; ++k) {
            if (!has_route_cost(active_requests, k)) return false;
        }
        return true;
    }

    bool has_speculative_profile_sample(
            int active_requests, int max_speculative_requests) const {
        const int limit = std::max(0, std::min(
            active_requests, max_speculative_requests));
        for (int k = 1; k <= limit; ++k) {
            if (has_route_cost(active_requests, k)) return true;
        }
        return false;
    }

    AdaptiveVerificationDecision select(
            int active_requests,
            const std::vector<AdaptiveVerificationCandidate> & candidates,
            int max_speculative_requests =
                std::numeric_limits<int>::max(),
            bool probe_uncalibrated_with_verifier = false,
            bool probe_missing_routes = true) const {
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
            if (!std::isfinite(candidate.routing_prior)) {
                candidate.routing_prior = 0.0;
            }
            (candidate.calibrated ? known : unknown).push_back(candidate);
        }
        if (known.empty() && unknown.empty()) return out;

        auto has_stable_useful_yield =
            [&](const AdaptiveVerificationCandidate & candidate) {
                return candidate.expected_tokens >=
                        config_.homogeneous_minimum_expected_tokens &&
                    candidate.evidence_samples >=
                        config_.homogeneous_minimum_samples;
            };
        auto by_value = [&](const AdaptiveVerificationCandidate & a,
                            const AdaptiveVerificationCandidate & b) {
            if (a.expected_tokens != b.expected_tokens) {
                return a.expected_tokens > b.expected_tokens;
            }
            return a.request < b.request;
        };
        const bool homogeneous_speculative_cohort =
            static_cast<int>(known.size()) == active_requests &&
            forms_homogeneous_cohort(
                known, /*require_stable_local_evidence=*/true);
        std::stable_sort(known.begin(), known.end(), by_value);
        if (homogeneous_speculative_cohort) {
            // Once every active request has proven useful, favor the lagging
            // requests so scarce verifier lanes do not create a long AR tail.
            std::stable_sort(
                known.begin(), known.end(),
                [](const AdaptiveVerificationCandidate & a,
                   const AdaptiveVerificationCandidate & b) {
                    if (a.progress_tokens != b.progress_tokens) {
                        return a.progress_tokens < b.progress_tokens;
                    }
                    if (a.expected_tokens != b.expected_tokens) {
                        return a.expected_tokens > b.expected_tokens;
                    }
                    return a.request < b.request;
                });
        } else {
            // For mixed cohorts, rotate only within contiguous half-token
            // request-value buckets. This preserves ordering across materially
            // different expected yields and never uses the raw prompt hint.
            std::map<int, std::vector<std::size_t>> fair_group_positions;
            for (std::size_t i = 0; i < known.size(); ++i) {
                const int yield_bucket = static_cast<int>(
                    std::floor(known[i].expected_tokens * 2.0));
                fair_group_positions[yield_bucket].push_back(i);
            }
            for (const auto & [yield_bucket, positions] :
                    fair_group_positions) {
                (void)yield_bucket;
                if (positions.size() < 2 ||
                    !std::all_of(
                        positions.begin(), positions.end(),
                        [&](std::size_t position) {
                            return has_stable_useful_yield(known[position]);
                        })) {
                    continue;
                }
                std::vector<AdaptiveVerificationCandidate> members;
                members.reserve(positions.size());
                for (std::size_t position : positions) {
                    members.push_back(known[position]);
                }
                std::stable_sort(
                    members.begin(), members.end(),
                    [](const AdaptiveVerificationCandidate & a,
                       const AdaptiveVerificationCandidate & b) {
                        if (a.progress_tokens != b.progress_tokens) {
                            return a.progress_tokens < b.progress_tokens;
                        }
                        if (a.expected_tokens != b.expected_tokens) {
                            return a.expected_tokens > b.expected_tokens;
                        }
                        return a.request < b.request;
                    });
                for (std::size_t i = 0; i < positions.size(); ++i) {
                    known[positions[i]] = members[i];
                }
            }
        }
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
        double best = baseline;
        double admitted_goodput = baseline;
        double expected_total = static_cast<double>(active_requests);
        const int prefix_limit = std::max(0, std::min(
            active_requests, max_speculative_requests));
        // Evaluate every measured width independently. GPU occupancy makes
        // these costs non-convex: k=3 may win even when k=1 and k=2 lose.
        for (int k = 1; k <= static_cast<int>(known.size()) &&
                        k <= prefix_limit; ++k) {
            expected_total += known[(size_t)k - 1].expected_tokens - 1.0;
            if (!has_route_cost(active_requests, k)) continue;
            const double route_us = route_cost_us(active_requests, k);
            const double throughput = expected_total / route_us;
            const bool protects_ar_peers = k == active_requests ||
                homogeneous_speculative_cohort ||
                route_us <= config_.maximum_ar_peer_slowdown *
                    autoregressive_cost_us(active_requests);
            if (!protects_ar_peers) continue;
            if (throughput > best) {
                best = throughput;
                if (throughput >= required) {
                    admitted_prefix = k;
                    admitted_goodput = throughput;
                }
            }
        }

        // Profile one missing exact width per useful decode step. Do not infer
        // C from a neighboring occupancy or stop after a losing smaller width:
        // both assumptions hide real GPU occupancy boundaries. Adapters with
        // no cheap confidence signal may include unknown requests in this
        // bounded prefix and learn their yield from accepted output.
        const int probe_candidates = std::min(
            prefix_limit,
            static_cast<int>(known.size()) +
                (probe_uncalibrated_with_verifier
                    ? static_cast<int>(unknown.size()) : 0));
        int missing_width = 0;
        for (int k = 1; probe_missing_routes &&
                        k <= probe_candidates; ++k) {
            if (has_route_cost(active_requests, k)) continue;
            missing_width = k;
            break;
        }
        if (missing_width > 0) {
            out.requests.reserve((size_t)missing_width);
            const int known_count = std::min(
                missing_width, static_cast<int>(known.size()));
            for (int i = 0; i < known_count; ++i) {
                out.requests.push_back(known[(size_t)i].request);
            }
            for (int i = 0;
                 static_cast<int>(out.requests.size()) < missing_width;
                 ++i) {
                out.requests.push_back(unknown[(size_t)i].request);
            }
            out.calibration_request = -1;
            out.exploring = true;
            return out;
        }

        // Once the hardware widths are known, an adapter without a cheap
        // confidence signal may calibrate one new request inside a useful
        // measured route. The optimistic maximum-token bound prevents probes
        // that cannot possibly clear the normal safety margin.
        if (probe_uncalibrated_with_verifier && !unknown.empty() &&
            prefix_limit > 0) {
            int calibration_width = 0;
            double calibration_goodput = required;
            const int calibration_limit = std::min(
                prefix_limit, static_cast<int>(known.size()) + 1);
            double known_total = static_cast<double>(active_requests);
            for (int k = 1; k <= calibration_limit; ++k) {
                if (k > 1) {
                    known_total +=
                        known[(size_t)k - 2].expected_tokens - 1.0;
                }
                if (!has_route_cost(active_requests, k)) continue;
                const double route_us = route_cost_us(active_requests, k);
                // This is one bounded evidence-gathering step, not a steady
                // route. It may cross the steady AR-peer guard so an otherwise
                // deadlocked homogeneous cohort can prove (or reject) itself.
                const double optimistic_total = known_total +
                    unknown.front().maximum_tokens - 1.0;
                const double throughput = optimistic_total / route_us;
                if (throughput > calibration_goodput) {
                    calibration_goodput = throughput;
                    calibration_width = k;
                }
            }
            if (calibration_width > 0) {
                out.requests.reserve((size_t)calibration_width);
                for (int i = 0; i < calibration_width - 1; ++i) {
                    out.requests.push_back(known[(size_t)i].request);
                }
                out.requests.push_back(unknown.front().request);
                out.calibration_request = -1;
                out.exploring = true;
                return out;
            }
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
        config.maximum_ar_peer_slowdown =
            std::max(1.0, config.maximum_ar_peer_slowdown);
        config.homogeneous_minimum_expected_tokens =
            std::max(1.0, config.homogeneous_minimum_expected_tokens);
        config.homogeneous_minimum_relative_yield = std::clamp(
            config.homogeneous_minimum_relative_yield, 0.0, 1.0);
        config.homogeneous_minimum_samples =
            std::max<std::size_t>(1, config.homogeneous_minimum_samples);
        config.routing_prior_minimum_samples =
            std::max<std::size_t>(1, config.routing_prior_minimum_samples);
        config.cost_ewma_alpha =
            std::clamp(config.cost_ewma_alpha, 0.0, 1.0);
        return config;
    }

    AdaptiveVerificationConfig config_;
    std::vector<std::vector<double>> route_cost_us_;
    std::vector<std::vector<bool>> route_cost_known_;
    struct YieldEstimate {
        double expected_tokens = 1.0;
        std::size_t samples = 0;
    };

    static constexpr std::size_t kMaximumYieldSamples = 64;

    static bool valid_yield(double emitted_tokens) {
        return std::isfinite(emitted_tokens) && emitted_tokens >= 1.0;
    }

    static void update_yield(YieldEstimate & estimate,
                             double emitted_tokens) {
        const std::size_t next_samples =
            std::min(estimate.samples + 1, kMaximumYieldSamples);
        const double alpha = 1.0 / static_cast<double>(next_samples);
        if (estimate.samples == 0) {
            estimate.expected_tokens = emitted_tokens;
        } else {
            estimate.expected_tokens = alpha * emitted_tokens +
                (1.0 - alpha) * estimate.expected_tokens;
        }
        estimate.samples = next_samples;
    }

    // Capped running means represent the complete request/profile while still
    // adapting gradually if a model or speculator changes behavior.
    std::map<std::uint64_t, YieldEstimate> request_yield_;
    std::map<double, YieldEstimate> routing_prior_yield_;
};

}  // namespace dflash::common
