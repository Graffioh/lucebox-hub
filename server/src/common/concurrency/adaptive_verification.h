#pragma once

// Model-neutral, hardware-aware request admission for adaptive verification.
//
// DSpark's scheduler ranks candidate tokens by cumulative survival probability
// and grows the verification batch while expected throughput improves. This
// helper applies the same policy at request granularity. The concrete
// speculator supplies expected useful tokens; the engine supplies the observed
// cost of each exact mixed route shape (active requests, speculative requests).
// DDTree and DSpark expose the same conditional-survival contract. Raw model
// confidence orders cold probes while target-verified accepted paths calibrate
// it online. One ranker instance represents one
// fixed speculator/proposal shape; adapters with ragged verification work must
// keep separate rankers for distinct work buckets.

#include "speculation_confidence.h"
#include "speculation_planning.h"

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
    // True when either the concrete speculator or target outcomes provide an
    // estimate. This does not mean the raw confidence was post-hoc calibrated.
    bool calibrated = false;
    // Raw speculator estimate used to find a target-calibrated confidence
    // profile. Prompt semantics never enter this value.
    double confidence_expected_tokens =
        std::numeric_limits<double>::quiet_NaN();
    // Number of request-local observations supporting the estimate. Shared
    // cohort evidence must never relax AR-peer protection for a new request.
    std::size_t evidence_samples = 0;
    // Generated output already committed for this request. Proven-useful peers
    // with less progress receive compact lanes first to avoid cohort stragglers.
    int progress_tokens = 0;
    // A user-forced request is always present in the selected prefix. Its
    // observed yield still contributes to deciding whether adaptive peers
    // should share the same verifier pass.
    bool required = false;
    SpeculatorKind speculator = SpeculatorKind::DDTree;
};

struct AdaptiveVerificationYieldEstimate {
    double expected_tokens = 1.0;
    std::size_t evidence_samples = 0;
    double confidence_expected_tokens =
        std::numeric_limits<double>::quiet_NaN();
    SpeculatorKind speculator = SpeculatorKind::DDTree;
};

struct AdaptiveVerificationDecision {
    std::vector<int> requests;
    // The adapter may cheaply calibrate one request (for DDTree, from draft
    // entropy; for DSpark, from its confidence head) before applying requests.
    int calibration_request = -1;
    bool exploring = false;
    double predicted_gain = 1.0;
    // Absolute useful-token goodput for a measured steady decision. This is
    // comparable across exact work profiles; zero means the selected action
    // is cold/unmeasured.
    double predicted_goodput = 0.0;
};

enum class AdaptiveVerificationWorkStatus : std::uint8_t {
    Autoregressive,
    Calibration,
    Verification,
    RequiredUnavailable,
    InvalidMenu,
};

struct AdaptiveVerificationWorkDecision {
    AdaptiveVerificationWorkStatus status =
        AdaptiveVerificationWorkStatus::Autoregressive;
    // A work key is returned for calibration and verification actions.
    std::optional<VerifierWorkKey> work;
    AdaptiveVerificationDecision decision;

    bool has_work() const {
        return work.has_value() &&
            (status == AdaptiveVerificationWorkStatus::Calibration ||
             status == AdaptiveVerificationWorkStatus::Verification);
    }
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
    // A complete two-lane verifier profile observes k=1 then k=2, yielding
    // three request outcomes before any steady route is selected.
    std::size_t confidence_profile_minimum_samples = 3;
    // Shared evidence ranks a new request after the minimum above, but its
    // yield magnitude is shrunk toward AR until this many outcomes exist.
    std::size_t confidence_profile_full_weight_samples = 8;
    // A continuous-backlog scheduler may optimize aggregate output after a
    // confidence bucket has accumulated this many target-verified outcomes.
    // This is deliberately stronger than cold ordering, but does not count as
    // request-local proof for closed-cohort AR-peer protection.
    std::size_t confidence_profile_peer_guard_samples = 6;
    double cost_ewma_alpha = 0.35;
};

inline bool adaptive_verification_can_relax_peer_guard(
        int active_requests, int verifier_request_lanes) {
    return active_requests > 0 && verifier_request_lanes > 0 &&
        static_cast<std::int64_t>(active_requests) <=
            2 * static_cast<std::int64_t>(verifier_request_lanes) + 1;
}

inline bool adaptive_verification_can_extend_stable_cohort(
        int active_requests, int verifier_request_lanes) {
    return active_requests > 0 && verifier_request_lanes > 0 &&
        static_cast<std::int64_t>(active_requests) <=
            3 * static_cast<std::int64_t>(verifier_request_lanes);
}

inline bool adaptive_verification_confidence_is_stale(
        int current_progress, int observed_progress,
        int refresh_interval) {
    return refresh_interval > 0 && current_progress >= observed_progress &&
        static_cast<std::int64_t>(current_progress) - observed_progress >=
            refresh_interval;
}

class AdaptiveVerificationRanker {
public:
    AdaptiveVerificationRanker() = default;
    explicit AdaptiveVerificationRanker(AdaptiveVerificationConfig config)
        : config_(sanitize(config)) {}

    void reset() {
        route_cost_us_.clear();
        route_cost_known_.clear();
        route_cost_samples_.clear();
        request_yield_.clear();
        request_confidence_.clear();
        confidence_yield_.clear();
    }

    void observe_request_estimate(
            std::uint64_t request,
            const SpeculationConfidenceEstimate & estimate) {
        if (!estimate.available()) return;
        observe_request_estimate(
            request, estimate.speculator, estimate.expected_tokens());
    }

    void observe_request_estimate(
            std::uint64_t request, SpeculatorKind speculator,
            double expected_tokens) {
        if (!std::isfinite(expected_tokens) || expected_tokens < 1.0) return;
        const auto current = request_confidence_.find(request);
        if (current != request_confidence_.end()) {
            const SpeculationConfidenceProfile old_profile =
                speculation_confidence_profile(
                    current->second.speculator,
                    current->second.expected_tokens);
            const SpeculationConfidenceProfile new_profile =
                speculation_confidence_profile(speculator, expected_tokens);
            if (old_profile.speculator != new_profile.speculator ||
                std::abs(old_profile.expected_half_tokens -
                         new_profile.expected_half_tokens) >= 2) {
                // A materially different drafter regime invalidates the local
                // running target mean. Profile-level target outcomes stay
                // isolated in their original buckets.
                request_yield_.erase(request);
            }
        }
        request_confidence_[request] = {speculator, expected_tokens};
    }

    void observe_request_yield(std::uint64_t request,
                               double emitted_tokens) {
        if (!valid_yield(emitted_tokens)) return;
        update_yield(request_yield_[request], emitted_tokens);
        const auto confidence = request_confidence_.find(request);
        if (confidence != request_confidence_.end()) {
            update_yield(
                confidence_yield_[speculation_confidence_profile(
                    confidence->second.speculator,
                    confidence->second.expected_tokens)], emitted_tokens);
        }
    }

    void forget_request(std::uint64_t request) {
        request_yield_.erase(request);
        request_confidence_.erase(request);
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

    std::optional<double> confidence_profile_expected_tokens(
            SpeculatorKind speculator,
            double confidence_expected_tokens) const {
        if (!std::isfinite(confidence_expected_tokens)) return std::nullopt;
        const auto profile = speculation_confidence_profile(
            speculator, confidence_expected_tokens);
        const auto it = confidence_yield_.find(profile);
        return it == confidence_yield_.end() ||
                it->second.samples <
                    config_.confidence_profile_minimum_samples
            ? std::nullopt
            : std::optional<double>(it->second.expected_tokens);
    }

    std::size_t confidence_profile_yield_samples(
            SpeculatorKind speculator,
            double confidence_expected_tokens) const {
        if (!std::isfinite(confidence_expected_tokens)) return 0;
        const auto it = confidence_yield_.find(
            speculation_confidence_profile(
                speculator, confidence_expected_tokens));
        return it == confidence_yield_.end() ? 0 : it->second.samples;
    }

    bool has_stable_confidence_yield(
            SpeculatorKind speculator,
            double confidence_expected_tokens) const {
        if (!std::isfinite(confidence_expected_tokens)) return false;
        const auto it = confidence_yield_.find(
            speculation_confidence_profile(
                speculator, confidence_expected_tokens));
        return it != confidence_yield_.end() &&
            it->second.samples >=
                config_.confidence_profile_peer_guard_samples &&
            has_useful_yield(it->second.expected_tokens);
    }

    bool forms_stable_confidence_cohort(
            const std::vector<AdaptiveVerificationCandidate> & candidates)
            const {
        if (candidates.empty()) return false;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = 1.0;
        for (const AdaptiveVerificationCandidate & candidate : candidates) {
            if (!has_stable_confidence_yield(
                    candidate.speculator,
                    candidate.confidence_expected_tokens)) {
                return false;
            }
            const auto it = confidence_yield_.find(
                speculation_confidence_profile(
                    candidate.speculator,
                    candidate.confidence_expected_tokens));
            minimum = std::min(minimum, it->second.expected_tokens);
            maximum = std::max(maximum, it->second.expected_tokens);
        }
        return minimum >=
            config_.homogeneous_minimum_relative_yield * maximum;
    }

    std::optional<AdaptiveVerificationYieldEstimate> estimate_request_yield(
            std::uint64_t request,
            bool trust_stable_confidence = false) const {
        const auto request_it = request_yield_.find(request);
        const auto confidence_it = request_confidence_.find(request);
        const bool has_request = request_it != request_yield_.end();
        const bool has_confidence =
            confidence_it != request_confidence_.end();
        auto profile_it = confidence_yield_.end();
        if (has_confidence) {
            profile_it = confidence_yield_.find(
                speculation_confidence_profile(
                    confidence_it->second.speculator,
                    confidence_it->second.expected_tokens));
        }
        const bool has_profile = profile_it != confidence_yield_.end() &&
            profile_it->second.samples >=
                config_.confidence_profile_minimum_samples;
        if (!has_request && !has_confidence && !has_profile) {
            return std::nullopt;
        }

        AdaptiveVerificationYieldEstimate out;
        if (has_confidence) {
            out.confidence_expected_tokens =
                confidence_it->second.expected_tokens;
            out.speculator = confidence_it->second.speculator;
        }
        double trusted_confidence = has_confidence
            ? confidence_it->second.expected_tokens : 1.0;
        if (has_profile) {
            const bool stable_backlog_confidence =
                trust_stable_confidence &&
                profile_it->second.samples >=
                    config_.confidence_profile_peer_guard_samples;
            const double profile_weight = stable_backlog_confidence
                ? 1.0
                : std::min(
                      1.0,
                      static_cast<double>(profile_it->second.samples) /
                          static_cast<double>(
                              config_.confidence_profile_full_weight_samples));
            trusted_confidence += profile_weight *
                (profile_it->second.expected_tokens - trusted_confidence);
        }
        if (!has_request) {
            out.expected_tokens = trusted_confidence;
            out.evidence_samples = 0;
            return out;
        }

        const YieldEstimate & request_estimate = request_it->second;
        out.expected_tokens = request_estimate.expected_tokens;
        out.evidence_samples = request_estimate.samples;
        if (has_confidence &&
            request_estimate.samples <
                config_.homogeneous_minimum_samples) {
            // Blend the first few noisy target observations with the concrete
            // speculator estimate (and any profile calibration). Once local
            // evidence is stable, measured target yield fully replaces it.
            const double local_weight = std::min(
                1.0,
                static_cast<double>(request_estimate.samples) /
                    static_cast<double>(
                        config_.homogeneous_minimum_samples));
            out.expected_tokens = trusted_confidence +
                local_weight * (request_estimate.expected_tokens -
                                trusted_confidence);
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
                       double elapsed_us,
                       bool discard_first_sample = false) {
        if (active_requests <= 0 || speculative_requests < 0 ||
            speculative_requests > active_requests ||
            !std::isfinite(elapsed_us) || elapsed_us <= 0.0) {
            return;
        }
        const size_t rows = static_cast<size_t>(active_requests) + 1;
        if (route_cost_us_.size() < rows) route_cost_us_.resize(rows);
        if (route_cost_known_.size() < rows) route_cost_known_.resize(rows);
        if (route_cost_samples_.size() < rows) route_cost_samples_.resize(rows);
        std::vector<double> & costs =
            route_cost_us_[(size_t)active_requests];
        std::vector<bool> & known =
            route_cost_known_[(size_t)active_requests];
        std::vector<std::size_t> & samples =
            route_cost_samples_[(size_t)active_requests];
        const size_t cols = static_cast<size_t>(speculative_requests) + 1;
        if (costs.size() < cols) costs.resize(cols, 0.0);
        if (known.size() < cols) known.resize(cols, false);
        if (samples.size() < cols) samples.resize(cols, 0);
        if (!known[(size_t)speculative_requests]) {
            costs[(size_t)speculative_requests] = elapsed_us;
            known[(size_t)speculative_requests] = true;
            samples[(size_t)speculative_requests] = 1;
            return;
        }
        if (discard_first_sample &&
            samples[(size_t)speculative_requests] == 1) {
            // The first execution of a new graph shape includes capture and
            // allocator warmup. Replace it with the first replay before using
            // the normal EWMA, rather than permanently teaching the policy
            // that a profitable steady route is cold-start slow.
            costs[(size_t)speculative_requests] = elapsed_us;
            samples[(size_t)speculative_requests] = 2;
            return;
        }
        costs[(size_t)speculative_requests] =
            config_.cost_ewma_alpha * elapsed_us +
            (1.0 - config_.cost_ewma_alpha) *
                costs[(size_t)speculative_requests];
        if (samples[(size_t)speculative_requests] <
            std::numeric_limits<std::size_t>::max()) {
            ++samples[(size_t)speculative_requests];
        }
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

    std::size_t route_cost_samples(int active_requests,
                                   int speculative_requests) const {
        return has_route_cost(active_requests, speculative_requests) &&
                static_cast<std::size_t>(active_requests) <
                    route_cost_samples_.size() &&
                static_cast<std::size_t>(speculative_requests) <
                    route_cost_samples_[(size_t)active_requests].size()
            ? route_cost_samples_[(size_t)active_requests]
                                 [(size_t)speculative_requests]
            : 0;
    }

    bool has_exact_profile(int active_requests,
                           int max_speculative_requests,
                           std::size_t minimum_route_samples = 1,
                           int baseline_speculative_requests = 0) const {
        if (baseline_speculative_requests < 0 ||
            baseline_speculative_requests > active_requests) {
            return false;
        }
        const std::size_t required_samples =
            std::max<std::size_t>(1, minimum_route_samples);
        const std::size_t baseline_samples =
            baseline_speculative_requests == 0 ? 1 : required_samples;
        if (route_cost_samples(
                active_requests, baseline_speculative_requests) <
            baseline_samples) {
            return false;
        }
        const int limit = std::max(
            baseline_speculative_requests,
            std::max(0, std::min(
                active_requests, max_speculative_requests)));
        for (int k = baseline_speculative_requests + 1;
             k <= limit; ++k) {
            if (route_cost_samples(active_requests, k) <
                required_samples) return false;
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
            bool probe_missing_routes = true,
            bool enforce_ar_peer_guard = true,
            std::size_t minimum_speculative_route_samples = 1) const {
        AdaptiveVerificationDecision out;
        if (active_requests <= 0 || candidates.empty()) return out;

        std::vector<AdaptiveVerificationCandidate> required_candidates;
        std::vector<AdaptiveVerificationCandidate> known;
        std::vector<AdaptiveVerificationCandidate> unknown;
        required_candidates.reserve(candidates.size());
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
            if (candidate.required) {
                required_candidates.push_back(candidate);
            } else {
                (candidate.calibrated ? known : unknown).push_back(candidate);
            }
        }
        if (required_candidates.empty() && known.empty() && unknown.empty()) {
            return out;
        }

        std::stable_sort(
            required_candidates.begin(), required_candidates.end(),
            [](const AdaptiveVerificationCandidate & a,
               const AdaptiveVerificationCandidate & b) {
                return a.request < b.request;
            });
        auto select_required = [&]() {
            out.requests.clear();
            out.requests.reserve(required_candidates.size());
            for (const AdaptiveVerificationCandidate & candidate :
                    required_candidates) {
                out.requests.push_back(candidate.request);
            }
        };
        const int required_count =
            static_cast<int>(required_candidates.size());
        const int baseline_width = required_count;
        if (!has_route_cost(active_requests, baseline_width)) {
            // Adaptive-only cohorts first observe all-AR. With user-forced
            // requests, all-AR is unattainable, so their mandatory mixed route
            // is the baseline from which optional adaptive peers are judged.
            select_required();
            return out;
        }

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
            // different expected yields without inspecting prompt semantics.
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
                    !std::any_of(
                        positions.begin(), positions.end(),
                        [&](std::size_t position) {
                            return has_stable_useful_yield(known[position]);
                        })) {
                    continue;
                }
                // One proven-useful member is enough to rotate peers in the
                // same narrow value bucket. Requiring every member to be
                // stable permanently starves raw-confidence peers that never
                // enter the bounded verifier prefix in the first place.
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
                if (a.maximum_tokens != b.maximum_tokens) {
                    return a.maximum_tokens > b.maximum_tokens;
                }
                if (a.progress_tokens != b.progress_tokens) {
                    return a.progress_tokens < b.progress_tokens;
                }
                return a.request < b.request;
            });
        if (!unknown.empty()) {
            out.calibration_request = unknown.front().request;
        }

        std::vector<AdaptiveVerificationCandidate> ordered_known;
        ordered_known.reserve(required_candidates.size() + known.size());
        ordered_known.insert(
            ordered_known.end(), required_candidates.begin(),
            required_candidates.end());
        ordered_known.insert(
            ordered_known.end(), known.begin(), known.end());
        const std::size_t required_route_samples =
            std::max<std::size_t>(
                1, minimum_speculative_route_samples);
        if (route_cost_samples(active_requests, baseline_width) <
            required_route_samples) {
            select_required();
            return out;
        }
        double baseline_expected_total =
            static_cast<double>(active_requests);
        for (const AdaptiveVerificationCandidate & candidate :
                required_candidates) {
            baseline_expected_total += candidate.expected_tokens - 1.0;
        }
        const double baseline_cost =
            route_cost_us(active_requests, baseline_width);
        const double baseline = baseline_expected_total / baseline_cost;
        const double required = baseline * config_.minimum_gain;
        int admitted_prefix = required_count;
        double best = baseline;
        double admitted_goodput = baseline;
        double expected_total = static_cast<double>(active_requests);
        const int prefix_limit = std::max(
            required_count,
            std::max(0, std::min(
                active_requests, max_speculative_requests)));
        // Evaluate every measured width independently. GPU occupancy makes
        // these costs non-convex: k=3 may win even when k=1 and k=2 lose.
        for (int k = 1; k <= static_cast<int>(ordered_known.size()) &&
                        k <= prefix_limit; ++k) {
            expected_total +=
                ordered_known[(size_t)k - 1].expected_tokens - 1.0;
            if (k < required_count) continue;
            if (route_cost_samples(active_requests, k) <
                required_route_samples) continue;
            const double route_us = route_cost_us(active_requests, k);
            const double throughput = expected_total / route_us;
            const bool protects_ar_peers = !enforce_ar_peer_guard ||
                k == active_requests ||
                homogeneous_speculative_cohort ||
                route_us <= config_.maximum_ar_peer_slowdown *
                    baseline_cost;
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
            static_cast<int>(ordered_known.size()) +
                (probe_uncalibrated_with_verifier
                    ? static_cast<int>(unknown.size()) : 0));
        int missing_width = 0;
        for (int k = std::max(1, required_count + 1);
                        probe_missing_routes &&
                        k <= probe_candidates; ++k) {
            if (route_cost_samples(active_requests, k) >=
                required_route_samples) continue;
            missing_width = k;
            break;
        }
        if (missing_width > 0) {
            out.requests.reserve((size_t)missing_width);
            const int known_count = std::min(
                missing_width, static_cast<int>(ordered_known.size()));
            for (int i = 0; i < known_count; ++i) {
                out.requests.push_back(
                    ordered_known[(size_t)i].request);
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
                prefix_limit, static_cast<int>(ordered_known.size()) + 1);
            double known_total = static_cast<double>(active_requests);
            for (int k = 1; k <= calibration_limit; ++k) {
                if (k > 1) {
                    known_total +=
                        ordered_known[(size_t)k - 2].expected_tokens - 1.0;
                }
                if (k <= required_count) continue;
                if (route_cost_samples(active_requests, k) <
                    required_route_samples) continue;
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
                    out.requests.push_back(
                        ordered_known[(size_t)i].request);
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
                out.requests.push_back(
                    ordered_known[(size_t)i].request);
            }
            if (admitted_prefix > required_count) {
                out.predicted_gain = admitted_goodput / baseline;
            }
            out.predicted_goodput = admitted_goodput;
        } else {
            out.predicted_goodput = baseline;
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
        config.confidence_profile_minimum_samples =
            std::max<std::size_t>(
                1, config.confidence_profile_minimum_samples);
        config.confidence_profile_full_weight_samples =
            std::max(config.confidence_profile_minimum_samples,
                     config.confidence_profile_full_weight_samples);
        config.confidence_profile_peer_guard_samples =
            std::max(config.confidence_profile_minimum_samples,
                     config.confidence_profile_peer_guard_samples);
        config.cost_ewma_alpha =
            std::clamp(config.cost_ewma_alpha, 0.0, 1.0);
        return config;
    }

    AdaptiveVerificationConfig config_;
    std::vector<std::vector<double>> route_cost_us_;
    std::vector<std::vector<bool>> route_cost_known_;
    std::vector<std::vector<std::size_t>> route_cost_samples_;
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
    struct StoredConfidenceEstimate {
        SpeculatorKind speculator = SpeculatorKind::DDTree;
        double expected_tokens = 1.0;
    };
    std::map<std::uint64_t, StoredConfidenceEstimate> request_confidence_;
    std::map<SpeculationConfidenceProfile, YieldEstimate> confidence_yield_;
};

// Owns one adaptive ranker per concrete verifier work shape. Speculative route
// timings and confidence calibration remain isolated by VerifierWorkKey, while
// the exact all-AR baseline is shared because it is independent of the
// speculator. Extra confidence-scout time has a separate profile and never
// contaminates either route cost.
class AdaptiveVerificationProfileBank {
public:
    AdaptiveVerificationProfileBank() = default;
    explicit AdaptiveVerificationProfileBank(
            AdaptiveVerificationConfig config)
        : config_(config), cost_ewma_alpha_(std::clamp(
              config.cost_ewma_alpha, 0.0, 1.0)) {}

    AdaptiveVerificationRanker & profile(
            const VerifierWorkKey & work) {
        auto [it, inserted] = profiles_.try_emplace(work, config_);
        if (inserted) {
            // One replay of the current EWMA is sufficient: rankers use the AR
            // sample as a baseline value, not as confidence evidence.
            for (const auto & [active_requests, cost] : ar_cost_us_) {
                it->second.observe_autoregressive(
                    active_requests, cost.elapsed_us);
            }
        }
        return it->second;
    }

    const AdaptiveVerificationRanker * find_profile(
            const VerifierWorkKey & work) const {
        const auto it = profiles_.find(work);
        return it == profiles_.end() ? nullptr : &it->second;
    }

    std::size_t profile_count() const { return profiles_.size(); }

    void observe_autoregressive(int active_requests, double elapsed_us) {
        if (active_requests <= 0 || !valid_cost(elapsed_us)) return;
        update_cost(ar_cost_us_[active_requests], elapsed_us);
        for (auto & [work, ranker] : profiles_) {
            (void)work;
            ranker.observe_autoregressive(active_requests, elapsed_us);
        }
    }

    void observe_route(const VerifierWorkKey & work,
                       int active_requests, int speculative_requests,
                       double elapsed_us,
                       bool discard_first_sample = false) {
        if (speculative_requests == 0) {
            observe_autoregressive(active_requests, elapsed_us);
            return;
        }
        profile(work).observe_route(
            active_requests, speculative_requests, elapsed_us,
            discard_first_sample);
    }

    void observe_request_estimate(
            const VerifierWorkKey & work, std::uint64_t request,
            const SpeculationConfidenceEstimate & estimate) {
        profile(work).observe_request_estimate(request, estimate);
    }

    void observe_request_estimate(
            const VerifierWorkKey & work, std::uint64_t request,
            SpeculatorKind speculator, double expected_tokens) {
        profile(work).observe_request_estimate(
            request, speculator, expected_tokens);
    }

    void observe_request_yield(const VerifierWorkKey & work,
                               std::uint64_t request,
                               double emitted_tokens) {
        profile(work).observe_request_yield(request, emitted_tokens);
    }

    void forget_request(std::uint64_t request) {
        for (auto & [work, ranker] : profiles_) {
            (void)work;
            ranker.forget_request(request);
        }
    }

    // Select one exact verifier work shape from adapter-provided per-request
    // menus for one configured speculator. DDTree, DSpark, and future adapters
    // reuse this contract independently; simultaneous cross-speculator racing
    // requires a separate portfolio policy. A measured profitable steady
    // route wins over any new profiling. Otherwise only the lowest
    // exploration_priority action is returned, so adapters can profile
    // short/cheap shapes before wider alternatives.
    AdaptiveVerificationWorkDecision select_work(
            int active_requests,
            const std::vector<RequestVerifierWorkMenu> & menus,
            bool probe_uncalibrated_with_verifier = false,
            bool enforce_ar_peer_guard = true,
            std::size_t minimum_speculative_route_samples = 1,
            bool trust_stable_confidence = false,
            bool allow_safe_peer_guard_relaxation = false) {
        AdaptiveVerificationWorkDecision out;
        if (active_requests <= 0 ||
            menus.size() != static_cast<std::size_t>(active_requests)) {
            out.status = AdaptiveVerificationWorkStatus::InvalidMenu;
            return out;
        }

        struct WorkOption {
            const SpeculationRequestView * request = nullptr;
            const VerifierWorkPlan * plan = nullptr;
        };
        struct WorkGroup {
            std::vector<WorkOption> options;
            bool traits_initialized = false;
            int max_parallel_requests = std::numeric_limits<int>::max();
            int adaptive_request_limit = std::numeric_limits<int>::max();
            std::uint32_t exploration_priority = 0;
            bool preferred_for_forced_mode = false;
        };
        std::map<VerifierWorkKey, WorkGroup> groups;
        std::map<int, bool> seen_slots;
        std::map<std::uint64_t, bool> seen_requests;
        std::optional<SpeculatorKind> configured_speculator;
        std::size_t required_requests = 0;
        bool required_without_plan = false;

        for (const RequestVerifierWorkMenu & menu : menus) {
            if (menu.request.slot < 0 ||
                !seen_slots.emplace(menu.request.slot, true).second ||
                !seen_requests.emplace(
                    menu.request.request_id, true).second) {
                out.status = AdaptiveVerificationWorkStatus::InvalidMenu;
                return out;
            }
            if (menu.request.required) ++required_requests;
            if (menu.speculative.empty()) {
                if (menu.request.required) required_without_plan = true;
                continue;
            }

            std::map<VerifierWorkKey, bool> menu_work;
            for (const VerifierWorkPlan & plan : menu.speculative) {
                if (!plan.valid() ||
                    (configured_speculator &&
                     *configured_speculator != plan.work.speculator) ||
                    !menu_work.emplace(plan.work, true).second) {
                    out.status = AdaptiveVerificationWorkStatus::InvalidMenu;
                    return out;
                }
                configured_speculator = plan.work.speculator;
                WorkGroup & group = groups[plan.work];
                if (!group.traits_initialized) {
                    group.traits_initialized = true;
                    group.max_parallel_requests =
                        plan.max_parallel_requests;
                    group.adaptive_request_limit =
                        plan.adaptive_request_limit;
                    group.exploration_priority =
                        plan.exploration_priority;
                    group.preferred_for_forced_mode =
                        plan.preferred_for_forced_mode;
                } else if (
                    group.max_parallel_requests !=
                        plan.max_parallel_requests ||
                    group.adaptive_request_limit !=
                        plan.adaptive_request_limit ||
                    group.exploration_priority !=
                        plan.exploration_priority ||
                    group.preferred_for_forced_mode !=
                        plan.preferred_for_forced_mode) {
                    out.status = AdaptiveVerificationWorkStatus::InvalidMenu;
                    return out;
                }
                group.options.push_back({&menu.request, &plan});
            }
        }
        if (required_without_plan) {
            out.status =
                AdaptiveVerificationWorkStatus::RequiredUnavailable;
            return out;
        }
        if (groups.empty()) return out;

        struct Evaluation {
            VerifierWorkKey work;
            AdaptiveVerificationDecision decision;
            std::uint32_t exploration_priority = 0;
            std::size_t required_count = 0;
            bool preferred_for_forced_mode = false;
            bool exact_profile = false;
        };
        std::vector<Evaluation> evaluations;
        evaluations.reserve(groups.size());

        for (const auto & [work, group] : groups) {
            Evaluation evaluation;
            evaluation.work = work;
            evaluation.exploration_priority =
                group.exploration_priority;
            evaluation.preferred_for_forced_mode =
                group.preferred_for_forced_mode;
            std::vector<AdaptiveVerificationCandidate> candidates;
            candidates.reserve(group.options.size());
            for (const WorkOption & option : group.options) {
                const SpeculationRequestView & request = *option.request;
                const VerifierWorkPlan & plan = *option.plan;

                AdaptiveVerificationRanker & ranker = profile(work);
                const double confidence =
                    plan.bounded_confidence_expected_tokens();
                if (plan.has_confidence()) {
                    ranker.observe_request_estimate(
                        request.request_id, work.speculator,
                        confidence);
                }

                AdaptiveVerificationCandidate candidate;
                candidate.request = request.slot;
                candidate.maximum_tokens = plan.maximum_emitted_tokens;
                candidate.confidence_expected_tokens = confidence;
                candidate.progress_tokens = request.progress_tokens;
                candidate.required = request.required;
                candidate.speculator = work.speculator;

                if (plan.has_confidence()) {
                    const auto estimate = ranker.estimate_request_yield(
                        request.request_id, trust_stable_confidence);
                    candidate.expected_tokens = std::clamp(
                        estimate ? estimate->expected_tokens : confidence,
                        1.0, plan.maximum_emitted_tokens);
                    candidate.evidence_samples = estimate
                        ? estimate->evidence_samples : 0;
                    candidate.calibrated = true;
                } else {
                    const auto target =
                        ranker.request_expected_tokens(request.request_id);
                    if (target) {
                        candidate.expected_tokens = std::clamp(
                            *target, 1.0,
                            plan.maximum_emitted_tokens);
                        candidate.evidence_samples =
                            ranker.request_yield_samples(request.request_id);
                        candidate.calibrated = true;
                    }
                }
                candidates.push_back(candidate);

                if (request.required) {
                    ++evaluation.required_count;
                }
            }
            // One executor shape must be able to carry every Always request.
            if (evaluation.required_count != required_requests ||
                evaluation.required_count >
                    static_cast<std::size_t>(
                        group.max_parallel_requests) ||
                candidates.empty()) {
                continue;
            }

            AdaptiveVerificationRanker & ranker = profile(work);
            const int profile_limit = std::max(
                static_cast<int>(evaluation.required_count),
                std::min({
                    active_requests,
                    group.adaptive_request_limit,
                    static_cast<int>(candidates.size())}));
            evaluation.exact_profile = ranker.has_exact_profile(
                active_requests, profile_limit,
                minimum_speculative_route_samples,
                static_cast<int>(evaluation.required_count));
            const int executable_lanes = std::max(
                static_cast<int>(evaluation.required_count),
                std::min(
                    group.adaptive_request_limit,
                    static_cast<int>(candidates.size())));
            const bool broad_verifier_coverage =
                adaptive_verification_can_relax_peer_guard(
                    active_requests, executable_lanes);
            const bool stable_bounded_cohort =
                evaluation.exact_profile &&
                candidates.size() ==
                    static_cast<std::size_t>(active_requests) &&
                adaptive_verification_can_extend_stable_cohort(
                    active_requests, executable_lanes) &&
                ranker.forms_stable_confidence_cohort(candidates);
            const bool relax_peer_guard =
                allow_safe_peer_guard_relaxation &&
                (broad_verifier_coverage || stable_bounded_cohort);
            evaluation.decision = ranker.select(
                active_requests, candidates,
                group.adaptive_request_limit,
                probe_uncalibrated_with_verifier,
                /*probe_missing_routes=*/true,
                enforce_ar_peer_guard && !relax_peer_guard,
                minimum_speculative_route_samples);
            evaluations.push_back(std::move(evaluation));
        }
        if (evaluations.empty()) {
            if (required_requests > 0) {
                out.status =
                    AdaptiveVerificationWorkStatus::RequiredUnavailable;
            }
            return out;
        }

        auto choose = [&](const Evaluation & evaluation) {
            out.work = evaluation.work;
            out.decision = evaluation.decision;
            out.status = evaluation.decision.requests.empty()
                ? AdaptiveVerificationWorkStatus::Calibration
                : AdaptiveVerificationWorkStatus::Verification;
        };

        // Compare measured work alternatives by absolute useful-token
        // goodput. Relative gain is profile-local and cannot rank two shapes
        // whose required baselines have different costs.
        const Evaluation * measured = nullptr;
        for (const Evaluation & evaluation : evaluations) {
            if (evaluation.decision.exploring ||
                evaluation.decision.requests.empty() ||
                evaluation.decision.predicted_goodput <= 0.0) {
                continue;
            }
            if (!measured ||
                evaluation.decision.predicted_goodput >
                    measured->decision.predicted_goodput ||
                (evaluation.decision.predicted_goodput ==
                     measured->decision.predicted_goodput &&
                 evaluation.work < measured->work)) {
                measured = &evaluation;
            }
        }
        if (measured) {
            choose(*measured);
            return out;
        }

        // Forced requests cannot fall back to AR. Adapter preference and
        // exploration priority are used only while every executable shape is
        // cold/unmeasured.
        if (required_requests > 0) {
            const Evaluation * forced = nullptr;
            for (const Evaluation & evaluation : evaluations) {
                if (evaluation.decision.requests.size() < required_requests) {
                    continue;
                }
                if (!forced ||
                    (evaluation.preferred_for_forced_mode &&
                     !forced->preferred_for_forced_mode) ||
                    (evaluation.preferred_for_forced_mode ==
                         forced->preferred_for_forced_mode &&
                     evaluation.exploration_priority <
                         forced->exploration_priority) ||
                    (evaluation.preferred_for_forced_mode ==
                         forced->preferred_for_forced_mode &&
                     evaluation.exploration_priority ==
                         forced->exploration_priority &&
                     evaluation.work < forced->work)) {
                    forced = &evaluation;
                }
            }
            if (forced) choose(*forced);
            return out;
        }

        // Missing verifier widths and confidence needed to begin an incomplete
        // profile are primary bounded exploration. A pure calibration request
        // on a complete losing profile is deferred until every other work
        // option has had its exploration opportunity.
        const Evaluation * exploration = nullptr;
        for (const Evaluation & evaluation : evaluations) {
            const bool calibration =
                evaluation.decision.requests.empty() &&
                evaluation.decision.calibration_request >= 0;
            if (!evaluation.decision.exploring &&
                (!calibration || evaluation.exact_profile)) {
                continue;
            }
            if (!exploration ||
                evaluation.exploration_priority <
                    exploration->exploration_priority ||
                (evaluation.exploration_priority ==
                     exploration->exploration_priority &&
                 evaluation.work < exploration->work)) {
                exploration = &evaluation;
            }
        }
        if (exploration) {
            choose(*exploration);
            return out;
        }

        const Evaluation * fallback_calibration = nullptr;
        for (const Evaluation & evaluation : evaluations) {
            if (!evaluation.decision.requests.empty() ||
                evaluation.decision.calibration_request < 0) {
                continue;
            }
            if (!fallback_calibration ||
                evaluation.exploration_priority <
                    fallback_calibration->exploration_priority ||
                (evaluation.exploration_priority ==
                     fallback_calibration->exploration_priority &&
                 evaluation.work < fallback_calibration->work)) {
                fallback_calibration = &evaluation;
            }
        }
        if (fallback_calibration) choose(*fallback_calibration);
        return out;
    }

    bool has_autoregressive_cost(int active_requests) const {
        return ar_cost_us_.find(active_requests) != ar_cost_us_.end();
    }

    double autoregressive_cost_us(int active_requests) const {
        const auto it = ar_cost_us_.find(active_requests);
        return it == ar_cost_us_.end()
            ? std::numeric_limits<double>::infinity()
            : it->second.elapsed_us;
    }

    std::size_t autoregressive_cost_samples(int active_requests) const {
        const auto it = ar_cost_us_.find(active_requests);
        return it == ar_cost_us_.end() ? 0 : it->second.samples;
    }

    void observe_scout(const ConfidenceScoutWorkKey & work,
                       int request_count, double elapsed_us) {
        if (request_count <= 0 || !valid_cost(elapsed_us)) return;
        update_cost(scout_cost_us_[{work, request_count}], elapsed_us);
    }

    bool has_scout_cost(const ConfidenceScoutWorkKey & work,
                        int request_count) const {
        return scout_cost_us_.find({work, request_count}) !=
            scout_cost_us_.end();
    }

    double scout_cost_us(const ConfidenceScoutWorkKey & work,
                         int request_count) const {
        const auto it = scout_cost_us_.find({work, request_count});
        return it == scout_cost_us_.end()
            ? std::numeric_limits<double>::infinity()
            : it->second.elapsed_us;
    }

    std::size_t scout_cost_samples(const ConfidenceScoutWorkKey & work,
                                   int request_count) const {
        const auto it = scout_cost_us_.find({work, request_count});
        return it == scout_cost_us_.end() ? 0 : it->second.samples;
    }

    // Compatibility for adapters not yet assigning stable scout shape IDs.
    // New integrations should use ConfidenceScoutWorkKey so depths and
    // executor paths cannot contaminate one another.
    void observe_scout(SpeculatorKind speculator, int request_count,
                       double elapsed_us) {
        observe_scout({speculator, 0, 0}, request_count, elapsed_us);
    }

    bool has_scout_cost(SpeculatorKind speculator,
                        int request_count) const {
        return has_scout_cost({speculator, 0, 0}, request_count);
    }

    double scout_cost_us(SpeculatorKind speculator,
                         int request_count) const {
        return scout_cost_us({speculator, 0, 0}, request_count);
    }

    std::size_t scout_cost_samples(SpeculatorKind speculator,
                                   int request_count) const {
        return scout_cost_samples({speculator, 0, 0}, request_count);
    }

    void reset() {
        profiles_.clear();
        ar_cost_us_.clear();
        scout_cost_us_.clear();
    }

private:
    struct CostEstimate {
        double elapsed_us = 0.0;
        std::size_t samples = 0;
    };

    struct ScoutCostKey {
        ConfidenceScoutWorkKey work;
        int request_count = 0;

        bool operator<(const ScoutCostKey & other) const {
            if (work != other.work) return work < other.work;
            return request_count < other.request_count;
        }
    };

    static bool valid_cost(double elapsed_us) {
        return std::isfinite(elapsed_us) && elapsed_us > 0.0;
    }

    void update_cost(CostEstimate & estimate, double elapsed_us) {
        if (estimate.samples == 0) {
            estimate.elapsed_us = elapsed_us;
        } else {
            estimate.elapsed_us = cost_ewma_alpha_ * elapsed_us +
                (1.0 - cost_ewma_alpha_) * estimate.elapsed_us;
        }
        if (estimate.samples < std::numeric_limits<std::size_t>::max()) {
            ++estimate.samples;
        }
    }

    AdaptiveVerificationConfig config_;
    double cost_ewma_alpha_ = 0.35;
    std::map<VerifierWorkKey, AdaptiveVerificationRanker> profiles_;
    std::map<int, CostEstimate> ar_cost_us_;
    std::map<ScoutCostKey, CostEstimate> scout_cost_us_;
};

}  // namespace dflash::common
