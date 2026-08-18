#include "common/concurrency/adaptive_verification.h"
#include "common/concurrency/speculation_goodput.h"
#include "common/speculation_policy.h"
#include "host_check.h"

#include <cstdio>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    // The public policy vocabulary is intentionally small and stable across
    // DDTree, DSpark, and later speculators.
    {
        CHECK(parse_decode_mode("adaptive") ==
              SpeculationPolicy::Adaptive);
        CHECK(parse_decode_mode("speculation") ==
              SpeculationPolicy::Always);
        CHECK(parse_decode_mode("ar") ==
              SpeculationPolicy::Never);
        CHECK(!parse_decode_mode("always"));
        CHECK(decode_mode_name(SpeculationPolicy::Never) == "ar");
    }

    // Cold start measures one real speculative step and one neighboring AR
    // step, then keeps the route with higher useful-token goodput.
    {
        SpeculationGoodputController policy;
        CHECK(policy.wants_speculation());
        CHECK(policy.observe_speculation(/*emitted_tokens=*/4,
                                         /*elapsed_us=*/200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(!policy.wants_speculation());
        CHECK(policy.observe_autoregressive(/*elapsed_us=*/100.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.wants_speculation());
        CHECK(policy.has_speculative_goodput());
        CHECK(policy.has_ar_goodput());
        CHECK(policy.has_expected_emitted_tokens());
        CHECK(policy.expected_emitted_tokens() == 4.0);
    }

    // A low-yield probe loses to AR and is disabled per request.
    {
        SpeculationGoodputController policy;
        CHECK(policy.observe_speculation(/*emitted_tokens=*/1,
                                         /*elapsed_us=*/200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.observe_autoregressive(/*elapsed_us=*/100.0) ==
              SpeculationGoodputTransition::disabled);
        CHECK(!policy.wants_speculation());
    }

    // A request already benefiting from speculation tolerates one bad step,
    // then leaves the route after a second consecutive bad observation.
    {
        SpeculationGoodputConfig config;
        config.ewma_alpha = 1.0;
        SpeculationGoodputController policy(config);
        CHECK(policy.observe_speculation(4, 200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.observe_autoregressive(100.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.wants_speculation());
        CHECK(policy.observe_speculation(1, 200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.wants_speculation());
        CHECK(policy.observe_speculation(1, 200.0) ==
              SpeculationGoodputTransition::disabled);
        CHECK(!policy.wants_speculation());
    }

    // AR periodically schedules one speculative re-probe. A profitable probe
    // re-enables speculation; a losing probe returns directly to AR.
    {
        SpeculationGoodputConfig config;
        config.ewma_alpha = 1.0;
        config.ar_reprobe_steps = 2;
        SpeculationGoodputController policy(config);
        CHECK(policy.observe_speculation(1, 200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.observe_autoregressive(100.0) ==
              SpeculationGoodputTransition::disabled);
        CHECK(!policy.wants_speculation());
        CHECK(policy.observe_autoregressive(100.0) ==
              SpeculationGoodputTransition::none);
        CHECK(!policy.wants_speculation());
        CHECK(policy.observe_autoregressive(100.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.wants_speculation());
        CHECK(policy.observe_speculation(4, 200.0) ==
              SpeculationGoodputTransition::enabled);
        CHECK(policy.wants_speculation());
    }

    {
        SpeculationGoodputConfig config;
        config.ewma_alpha = 1.0;
        config.ar_reprobe_steps = 1;
        SpeculationGoodputController policy(config);
        CHECK(policy.observe_speculation(1, 200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.observe_autoregressive(100.0) ==
              SpeculationGoodputTransition::disabled);
        CHECK(policy.observe_autoregressive(100.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.wants_speculation());
        CHECK(policy.observe_speculation(1, 200.0) ==
              SpeculationGoodputTransition::none);
        CHECK(!policy.wants_speculation());
    }

    // Cheap confidence calibration does not masquerade as a measured target
    // verification or advance the request route state.
    {
        SpeculationGoodputController policy;
        policy.observe_expected_tokens(3.5);
        CHECK(policy.wants_speculation());
        CHECK(policy.has_expected_emitted_tokens());
        CHECK(policy.expected_emitted_tokens() == 3.5);
        CHECK(!policy.has_speculative_goodput());
    }

    // DSpark conditional confidence maps directly to the shared expected-token
    // value used for request ranking.
    {
        const float confidence[] = {0.8f, 0.5f};
        const double expected =
            expected_tokens_from_conditional_confidence(confidence, 2);
        CHECK(expected > 2.19 && expected < 2.21);
    }

    // The ranker first measures the all-AR hardware baseline at this exact
    // occupancy. It then explores only one uncalibrated request.
    {
        AdaptiveVerificationRanker ranker;
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {7, 1.0, 8.0, false},
            {3, 1.0, 8.0, false},
        };
        CHECK(ranker.select(5, candidates).requests.empty());
        ranker.observe_autoregressive(5, 100.0);
        const AdaptiveVerificationDecision probe =
            ranker.select(5, candidates);
        CHECK(!probe.exploring);
        CHECK(probe.requests.empty());
        CHECK(probe.calibration_request == 3);
    }

    // A measured request is admitted at C=5 only when its expected useful
    // tokens overcome the measured cost of that exact fused route shape.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 80.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {11, 4.0, 8.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(5, candidates);
        CHECK(!decision.exploring);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 11);
        CHECK(decision.predicted_gain > 1.05);
    }

    // User-forced speculation does not wait for an AR baseline and remains
    // selected even when its measured route is slower than AR.
    {
        AdaptiveVerificationRanker ranker;
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {41, 1.0, 8.0, false, 0.0, 0, 0, true},
        };
        AdaptiveVerificationDecision decision =
            ranker.select(2, candidates, 1);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 41);

        ranker.observe_autoregressive(2, 100.0);
        ranker.observe_route(2, 1, 200.0);
        decision = ranker.select(2, candidates, 1);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 41);
    }

    // Mandatory requests form the prefix. Adaptive peers may join only when
    // the exact combined width is profitable, and a user override can exceed
    // the normal efficient-lane limit without being silently ignored.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(2, 100.0);
        ranker.observe_route(2, 1, 200.0);
        ranker.observe_route(2, 2, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {51, 4.0, 8.0, true},
            {52, 1.0, 8.0, true, 0.0, 0, 0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(2, candidates, 2);
        CHECK(decision.requests.size() == 2);
        CHECK(decision.requests[0] == 52);
        CHECK(decision.requests[1] == 51);
    }
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(4, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {61, 1.0, 8.0, false, 0.0, 0, 0, true},
            {62, 1.0, 8.0, false, 0.0, 0, 0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(4, candidates, /*max_speculative_requests=*/1);
        CHECK(decision.requests.size() == 2);
        CHECK(!decision.exploring);
    }

    // A forced request makes (C,r), not the impossible all-AR route, the cold
    // baseline. Optional adaptive peers can therefore be profiled and joined.
    {
        AdaptiveVerificationRanker ranker;
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {71, 4.0, 8.0, true, 4.0, 0, 0, true},
            {72, 4.0, 8.0, true, 4.0},
        };
        AdaptiveVerificationDecision decision =
            ranker.select(2, candidates, 2);
        CHECK(decision.requests == std::vector<int>{71});
        CHECK(!ranker.has_exact_profile(
            2, 2, /*minimum_route_samples=*/1,
            /*baseline_speculative_requests=*/1));
        ranker.observe_route(2, 1, 100.0);
        decision = ranker.select(2, candidates, 2);
        CHECK(decision.exploring);
        CHECK(decision.requests == std::vector<int>({71, 72}));
        CHECK(!ranker.has_exact_profile(
            2, 2, /*minimum_route_samples=*/1,
            /*baseline_speculative_requests=*/1));
        ranker.observe_route(2, 2, 80.0);
        decision = ranker.select(2, candidates, 2);
        CHECK(!decision.exploring);
        CHECK(decision.requests == std::vector<int>({71, 72}));
        CHECK(ranker.has_exact_profile(
            2, 2, /*minimum_route_samples=*/1,
            /*baseline_speculative_requests=*/1));
    }

    // Candidates are ranked by expected value and every measured width is
    // compared against the whole-batch AR baseline.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 80.0);
        ranker.observe_route(5, 2, 95.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {2, 1.5, 8.0, true},
            {9, 4.0, 8.0, true},
            {4, 1.2, 8.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(5, candidates, /*max_speculative_requests=*/2);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 9);
    }

    // Uncalibrated requests are returned to the concrete confidence adapter;
    // they are never admitted to expensive target verification as a probe.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(16, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {1, 1.0, 8.0, false},
            {2, 1.0, 8.0, false},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(16, candidates);
        CHECK(decision.requests.empty());
        CHECK(decision.calibration_request == 1);
    }

    // Compact verifier calibration discovers every exact route width in order,
    // even when k=1 and k=2 lose. Unknown requests use deterministic neutral
    // ordering, and the globally best non-convex k=3 route wins.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {5, 1.0, 9.0, false},
            {2, 1.0, 9.0, false},
            {9, 1.0, 9.0, false},
        };
        AdaptiveVerificationDecision probe =
            ranker.select(8, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 1);
        CHECK(probe.requests[0] == 2);
        CHECK(probe.calibration_request == -1);
        ranker.observe_route(8, 1, 160.0);
        candidates[1] = {2, 4.0, 9.0, true};

        probe = ranker.select(8, candidates, 3,
                              /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 2);
        CHECK(probe.requests[0] == 2);
        CHECK(probe.requests[1] == 5);
        ranker.observe_route(8, 2, 170.0);
        candidates[0] = {5, 4.0, 9.0, true};

        probe = ranker.select(8, candidates, 3,
                              /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 3);
        CHECK(probe.requests[0] == 2);
        CHECK(probe.requests[1] == 5);
        CHECK(probe.requests[2] == 9);
        ranker.observe_route(8, 3, 105.0);
        candidates[2] = {9, 4.0, 9.0, true};

        const AdaptiveVerificationDecision decision =
            ranker.select(8, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/true);
        CHECK(!decision.exploring);
        CHECK(decision.requests.size() == 3);
        CHECK(ranker.has_exact_profile(8, 3));
    }

    // A profitable token-count estimate cannot strand slow AR peers. The same
    // route is allowed only after every active request has enough supporting
    // evidence; one cache hit or lucky accepted path is insufficient.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 150.0);
        ranker.observe_route(5, 2, 150.0);
        std::vector<AdaptiveVerificationCandidate> mixed = {
            {81, 8.0, 8.0, true},
            {82, 8.0, 8.0, true},
            {83, 1.0, 8.0, true},
            {84, 1.0, 8.0, true},
            {85, 1.0, 8.0, true},
        };
        CHECK(ranker.select(5, mixed, 2).requests.empty());

        for (AdaptiveVerificationCandidate & candidate : mixed) {
            candidate.expected_tokens = 8.0;
        }
        CHECK(ranker.select(5, mixed, 2).requests.empty());

        // Even individually useful requests cannot waive the peer guard when
        // their measured yields are too dissimilar; that creates a slow tail.
        for (AdaptiveVerificationCandidate & candidate : mixed) {
            candidate.expected_tokens = 3.0;
            candidate.evidence_samples = 4;
        }
        mixed[0].expected_tokens = 8.0;
        mixed[1].expected_tokens = 8.0;
        CHECK(ranker.select(5, mixed, 2).requests.empty());

        // A genuinely proven, comparable high-yield cohort may amortize the
        // slower mixed step because its remaining requests take verifier lanes
        // as the first requests retire.
        for (AdaptiveVerificationCandidate & candidate : mixed) {
            candidate.expected_tokens = 8.0;
        }
        CHECK(ranker.select(5, mixed, 2).requests.size() == 2);
    }

    // A continuous-batching scheduler may optimize total goodput when a
    // completed request immediately refills its slot. Closed cohorts keep the
    // AR-peer guard; backlog mode may select the measured goodput winner.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 2, 140.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {86, 5.0, 5.0, true},
            {87, 5.0, 5.0, true},
        };
        CHECK(ranker.select(
                  5, candidates, 2,
                  /*probe_uncalibrated_with_verifier=*/false,
                  /*probe_missing_routes=*/false)
                  .requests.empty());
        const AdaptiveVerificationDecision backlog = ranker.select(
            5, candidates, 2,
            /*probe_uncalibrated_with_verifier=*/false,
            /*probe_missing_routes=*/false,
            /*enforce_ar_peer_guard=*/false);
        CHECK(backlog.requests.size() == 2);
    }

    // Peer-guard relaxation follows verifier capacity rather than a hard-coded
    // concurrency cutoff. Two lanes cover C=5 but not C=6; a wider executor
    // raises the boundary automatically.
    {
        CHECK(adaptive_verification_can_relax_peer_guard(5, 2));
        CHECK(!adaptive_verification_can_relax_peer_guard(6, 2));
        CHECK(adaptive_verification_can_relax_peer_guard(7, 3));
        CHECK(!adaptive_verification_can_relax_peer_guard(8, 3));
        CHECK(adaptive_verification_can_relax_peer_guard(16, 8));
        CHECK(!adaptive_verification_can_relax_peer_guard(1, 0));
        CHECK(adaptive_verification_can_extend_stable_cohort(8, 3));
        CHECK(adaptive_verification_can_extend_stable_cohort(9, 3));
        CHECK(!adaptive_verification_can_extend_stable_cohort(10, 3));
        CHECK(!adaptive_verification_can_extend_stable_cohort(16, 3));
        CHECK(adaptive_verification_can_extend_stable_cohort(16, 6));
        CHECK(!adaptive_verification_can_extend_stable_cohort(1, 0));

        CHECK(!adaptive_verification_confidence_is_stale(63, 0, 64));
        CHECK(adaptive_verification_confidence_is_stale(64, 0, 64));
        CHECK(adaptive_verification_confidence_is_stale(192, 64, 128));
        CHECK(!adaptive_verification_confidence_is_stale(63, 64, 64));
        CHECK(!adaptive_verification_confidence_is_stale(128, 0, 0));
    }

    // Timings from a neighboring occupancy never stand in for the exact-C
    // baseline or route profile.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        ranker.observe_route(8, 1, 80.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {12, 4.0, 9.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(7, candidates, 3);
        CHECK(decision.requests.empty());
        CHECK(!decision.exploring);
        CHECK(!ranker.has_exact_profile(7, 1));
    }

    // The first graph-shape execution is a capture sample. A backlog profile
    // can require a second replay, which replaces that cold timing before the
    // normal EWMA begins.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        ranker.observe_route(8, 1, 250.0);
        CHECK(ranker.route_cost_samples(8, 1) == 1);
        CHECK(ranker.route_cost_us(8, 1) == 250.0);
        CHECK(ranker.has_exact_profile(8, 1));
        CHECK(!ranker.has_exact_profile(8, 2, 0));
        CHECK(!ranker.has_exact_profile(8, 1, 2));
        ranker.observe_route(
            8, 1, 100.0, /*discard_first_sample=*/true);
        CHECK(ranker.route_cost_samples(8, 1) == 2);
        CHECK(ranker.route_cost_us(8, 1) == 100.0);
        CHECK(ranker.has_exact_profile(8, 1, 2));
        ranker.observe_route(8, 1, 200.0);
        CHECK(ranker.route_cost_samples(8, 1) == 3);
        CHECK(std::abs(ranker.route_cost_us(8, 1) - 135.0) < 1e-9);
    }

    // Proven high-yield peers rotate scarce verifier lanes toward the request
    // with less generated progress, preventing a homogeneous cohort tail.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(2, 100.0);
        ranker.observe_route(2, 1, 150.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {91, 8.0, 8.0, true, 1.0, 4, 10},
            {92, 8.0, 8.0, true, 1.0, 4, 2},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(2, candidates, 1);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 92);
    }

    // A losing higher-C route never suppresses a fresh exact lower-C
    // measurement: GPU occupancy boundaries are non-convex.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        ranker.observe_route(8, 1, 200.0);
        ranker.observe_autoregressive(7, 95.0);

        const AdaptiveVerificationDecision decision = ranker.select(
            7, {{13, 1.0, 9.0, true}}, 1);
        CHECK(decision.exploring);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 13);
    }

    // Expected target yield, rather than a prompt category, determines order.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(2, 100.0);
        ranker.observe_route(2, 1, 100.0);
        ranker.observe_route(2, 2, 1000.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {1, 1.0, 8.0, true, 1.0},
            {2, 8.0, 8.0, true, -1.0},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(2, candidates, 2);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 2);
    }

    // DDTree softmax and DSpark confidence-head output use one sanitized
    // conditional-survival contract. Raw estimates are explicitly distinct
    // from target-verified calibration evidence.
    {
        const float confidence[] = {0.8f, 0.5f};
        const SpeculationConfidenceEstimate ddtree_confidence =
            make_speculation_confidence_estimate(
                SpeculatorKind::DDTree, confidence, 2,
                SpeculationConfidenceCost::ExtraDraftPass);
        const SpeculationConfidenceEstimate dspark_confidence =
            make_speculation_confidence_estimate(
                SpeculatorKind::DSpark, confidence, 2,
                SpeculationConfidenceCost::PiggybacksOnProposal);
        CHECK(ddtree_confidence.available());
        CHECK(dspark_confidence.available());
        CHECK(!ddtree_confidence.posthoc_calibrated);
        CHECK(!dspark_confidence.posthoc_calibrated);
        CHECK(std::abs(ddtree_confidence.expected_tokens() - 2.2) < 1e-6);
        CHECK(std::abs(dspark_confidence.expected_tokens() - 2.2) < 1e-6);
        CHECK(std::abs(conditional_prefix_survival(confidence, 2) - 0.4) <
              1e-6);
        const SpeculationConfidenceEstimate limited =
            ddtree_confidence.limited_to(1);
        CHECK(limited.maximum_tokens() == 2.0);
        CHECK(ddtree_confidence.maximum_tokens() == 3.0);
        CHECK(std::abs(ddtree_confidence.expected_tokens() - 2.2) < 1e-6);

        const float invalid[] = {
            2.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()};
        const auto sanitized = make_speculation_confidence_estimate(
            SpeculatorKind::DDTree, invalid, 3,
            SpeculationConfidenceCost::PiggybacksOnProposal);
        CHECK(sanitized.expected_tokens() == 2.0);
        const auto unavailable = make_speculation_confidence_estimate(
            SpeculatorKind::DSpark, nullptr, 0,
            SpeculationConfidenceCost::PiggybacksOnProposal);
        CHECK(!unavailable.available());
        CHECK(unavailable.expected_tokens() == 1.0);

        AdaptiveVerificationRanker ranker;
        ranker.observe_request_estimate(98, ddtree_confidence);
        auto estimate = ranker.estimate_request_yield(98);
        CHECK(estimate.has_value());
        CHECK(std::abs(estimate->expected_tokens - 2.2) < 1e-6);
        CHECK(estimate->evidence_samples == 0);
        CHECK(ranker.request_yield_samples(98) == 0);
        CHECK(!ranker.confidence_profile_expected_tokens(
                  SpeculatorKind::DDTree, 2.2).has_value());

        // Early target outcomes smoothly correct an optimistic raw estimate;
        // drafter confidence never counts as manufactured target evidence.
        ranker.observe_request_yield(98, 1.0);
        estimate = ranker.estimate_request_yield(98);
        CHECK(estimate.has_value());
        CHECK(std::abs(estimate->expected_tokens - 1.9) < 1e-6);
        CHECK(estimate->evidence_samples == 1);
        CHECK(ranker.confidence_profile_yield_samples(
                  SpeculatorKind::DDTree, 2.2) == 1);
        for (int sample = 1; sample < 4; ++sample) {
            ranker.observe_request_yield(98, 1.0);
        }
        estimate = ranker.estimate_request_yield(98);
        CHECK(estimate.has_value());
        CHECK(std::abs(estimate->expected_tokens - 1.0) < 1e-6);
        CHECK(estimate->evidence_samples == 4);

        // A material confidence-regime change invalidates local target history
        // rather than treating the request as a permanent semantic category.
        const float low_confidence_values[] = {0.1f, 0.1f};
        const SpeculationConfidenceEstimate low_confidence =
            make_speculation_confidence_estimate(
                SpeculatorKind::DDTree, low_confidence_values, 2,
                SpeculationConfidenceCost::PiggybacksOnProposal);
        AdaptiveVerificationRanker changing;
        changing.observe_request_estimate(197, ddtree_confidence);
        for (int sample = 0; sample < 4; ++sample) {
            changing.observe_request_yield(197, 4.0);
        }
        CHECK(changing.request_yield_samples(197) == 4);
        changing.observe_request_estimate(197, low_confidence);
        CHECK(changing.request_yield_samples(197) == 0);
        const auto changed = changing.estimate_request_yield(197);
        CHECK(changed.has_value());
        CHECK(std::abs(
            changed->expected_tokens - low_confidence.expected_tokens()) <
              1e-6);

        for (std::uint64_t request = 99; request <= 103; ++request) {
            ranker.observe_request_estimate(request, ddtree_confidence);
            ranker.observe_request_yield(request, 5.0);
        }
        CHECK(ranker.has_stable_confidence_yield(
            SpeculatorKind::DDTree, 2.2));
        CHECK(!ranker.has_stable_confidence_yield(
            SpeculatorKind::DSpark, 2.2));
        CHECK(ranker.forms_stable_confidence_cohort({
            {101, 5.0, 9.0, true, 2.2, 0, 0, false,
             SpeculatorKind::DDTree},
            {102, 5.0, 9.0, true, 2.2, 0, 0, false,
             SpeculatorKind::DDTree},
        }));
        CHECK(!ranker.forms_stable_confidence_cohort({
            {101, 5.0, 9.0, true, 2.2, 0, 0, false,
             SpeculatorKind::DDTree},
            {104, 5.0, 9.0, true, 2.2, 0, 0, false,
             SpeculatorKind::DSpark},
        }));

        // Per-request evidence is local to one proposal-shape ranker and is
        // explicitly forgotten at request retirement.
        AdaptiveVerificationRanker compact;
        ranker.observe_request_yield(199, 7.0);
        CHECK(ranker.request_expected_tokens(199).value() == 7.0);
        CHECK(ranker.request_yield_samples(199) == 1);
        CHECK(!compact.request_expected_tokens(199).has_value());
        ranker.forget_request(199);
        CHECK(!ranker.request_expected_tokens(199).has_value());
        ranker.reset();
        CHECK(!ranker.confidence_profile_expected_tokens(
                  SpeculatorKind::DDTree, 2.2).has_value());
    }

    // Equal-value peers outside a bounded verifier prefix receive service once
    // one member has proven the bucket useful. A materially lower-value peer
    // remains in its own bucket and cannot displace them merely for fairness.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        ranker.observe_route(8, 1, 80.0);
        ranker.observe_route(8, 2, 80.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {1, 4.0, 5.0, true,
             std::numeric_limits<double>::quiet_NaN(), 4, 12},
            {2, 4.0, 5.0, true,
             std::numeric_limits<double>::quiet_NaN(), 4, 12},
            {3, 4.0, 5.0, true,
             std::numeric_limits<double>::quiet_NaN(), 0, 4},
            {4, 4.0, 5.0, true,
             std::numeric_limits<double>::quiet_NaN(), 0, 4},
            {5, 2.0, 5.0, true,
             std::numeric_limits<double>::quiet_NaN(), 0, 0},
        };
        const AdaptiveVerificationDecision rotated = ranker.select(
            8, candidates, /*max_speculative_requests=*/2);
        CHECK(rotated.requests.size() == 2);
        CHECK(rotated.requests[0] == 3);
        CHECK(rotated.requests[1] == 4);
        CHECK(std::find(rotated.requests.begin(), rotated.requests.end(), 5) ==
              rotated.requests.end());
    }

    // Scouting and verifier menus carry only model-neutral request, confidence,
    // and structural work metadata. AR remains implicit, and a branching
    // verifier's row count does not pretend to be its maximum emitted path.
    {
        SpeculationRequestView request;
        request.request_id = 700;
        request.slot = 3;
        request.seed_token = 42;
        request.progress_tokens = 9;

        ConfidenceScoutRequest scout;
        scout.request = request;
        scout.work = {SpeculatorKind::DDTree, 7, 4};
        scout.max_candidate_tokens = 4;
        CHECK(scout.request.request_id == 700);
        CHECK(scout.work.valid());
        CHECK(scout.valid());
        CHECK(scout.max_candidate_tokens == 4);
        scout.max_candidate_tokens = 5;
        CHECK(!scout.valid());
        scout.max_candidate_tokens = 0;
        CHECK(!scout.valid());
        scout.max_candidate_tokens = 4;

        const float confidence_values[] = {0.8f, 0.5f};
        ConfidenceScoutResult ready;
        ready.request_id = request.request_id;
        ready.status = ConfidenceScoutStatus::Ready;
        ready.confidence = make_speculation_confidence_estimate(
            SpeculatorKind::DDTree, confidence_values, 2,
            SpeculationConfidenceCost::ExtraDraftPass);
        ready.elapsed_us = 12.0;
        CHECK(ready.ready());
        CHECK(ready.elapsed_us == 12.0);
        ready.status = ConfidenceScoutStatus::ProposalRequired;
        CHECK(!ready.ready());

        ConfidenceScoutBatchResult batch;
        batch.requests.push_back(ready);
        batch.elapsed_us = 25.0;
        CHECK(batch.requests.size() == 1);
        CHECK(batch.elapsed_us == 25.0);

        const VerifierWorkKey tree_work{
            SpeculatorKind::DDTree, 1, 32, 33};
        const VerifierWorkKey compact_work{
            SpeculatorKind::DDTree, 2, 4, 5};
        const VerifierWorkKey dspark_work{
            SpeculatorKind::DSpark, 1, 3, 4};
        CHECK(tree_work.valid());
        CHECK(tree_work != compact_work);
        CHECK(compact_work < tree_work || tree_work < compact_work);
        CHECK(tree_work < dspark_work);

        VerifierWorkPlan plan;
        plan.work = tree_work;
        plan.maximum_emitted_tokens = 16.0;
        plan.confidence_expected_tokens = 2.2;
        plan.preferred_for_forced_mode = true;
        plan.max_parallel_requests = 3;
        plan.adaptive_request_limit = 2;
        plan.exploration_priority = 2;
        CHECK(plan.valid());
        CHECK(plan.has_confidence());
        CHECK(plan.work.verifier_rows == 33);
        CHECK(plan.maximum_emitted_tokens == 16.0);
        CHECK(plan.max_parallel_requests == 3);
        CHECK(plan.adaptive_request_limit == 2);
        CHECK(plan.exploration_priority == 2);

        RequestVerifierWorkMenu menu;
        menu.request = request;
        menu.speculative.push_back(plan);
        CHECK(menu.request.slot == 3);
        CHECK(menu.speculative.size() == 1);
        CHECK(menu.speculative[0].preferred_for_forced_mode);
    }

    // Work profiles isolate speculative shapes while sharing the exact AR
    // baseline. Extra scouting cost has its own EWMA and cannot create or
    // mutate a verifier profile.
    {
        AdaptiveVerificationConfig config;
        config.cost_ewma_alpha = 0.5;
        AdaptiveVerificationProfileBank bank(config);
        const VerifierWorkKey short_tree{
            SpeculatorKind::DDTree, 1, 4, 5};
        const VerifierWorkKey wide_tree{
            SpeculatorKind::DDTree, 2, 8, 9};
        const VerifierWorkKey dspark_linear{
            SpeculatorKind::DSpark, 1, 3, 4};

        // An AR observation made before a profile exists is replayed when that
        // work key is first requested.
        bank.observe_autoregressive(5, 100.0);
        AdaptiveVerificationRanker & short_ranker = bank.profile(short_tree);
        CHECK(short_ranker.has_autoregressive_cost(5));
        CHECK(short_ranker.autoregressive_cost_us(5) == 100.0);
        bank.observe_route(short_tree, 5, 1, 80.0);
        CHECK(short_ranker.has_route_cost(5, 1));

        AdaptiveVerificationRanker & wide_ranker = bank.profile(wide_tree);
        CHECK(wide_ranker.has_autoregressive_cost(5));
        CHECK(!wide_ranker.has_route_cost(5, 1));
        CHECK(bank.profile_count() == 2);

        bank.observe_autoregressive(5, 120.0);
        CHECK(bank.autoregressive_cost_us(5) == 110.0);
        CHECK(bank.autoregressive_cost_samples(5) == 2);
        CHECK(short_ranker.autoregressive_cost_us(5) == 110.0);
        CHECK(wide_ranker.autoregressive_cost_us(5) == 110.0);

        const ConfidenceScoutWorkKey short_scout{
            SpeculatorKind::DDTree, 1, 4};
        const ConfidenceScoutWorkKey wide_scout{
            SpeculatorKind::DDTree, 2, 8};
        const ConfidenceScoutWorkKey dspark_scout{
            SpeculatorKind::DSpark, 1, 4};
        bank.observe_scout(short_scout, 2, 50.0);
        bank.observe_scout(short_scout, 2, 70.0);
        bank.observe_scout(wide_scout, 2, 90.0);
        bank.observe_scout(dspark_scout, 2, 110.0);
        bank.observe_scout(short_scout, 1, 20.0);
        CHECK(bank.profile_count() == 2);
        CHECK(bank.scout_cost_us(short_scout, 2) == 60.0);
        CHECK(bank.scout_cost_samples(short_scout, 2) == 2);
        CHECK(bank.scout_cost_us(wide_scout, 2) == 90.0);
        CHECK(bank.scout_cost_us(dspark_scout, 2) == 110.0);
        CHECK(bank.scout_cost_us(short_scout, 1) == 20.0);
        CHECK(short_ranker.autoregressive_cost_us(5) == 110.0);
        CHECK(short_ranker.route_cost_us(5, 1) == 80.0);

        bank.observe_request_yield(short_tree, 900, 4.0);
        bank.observe_request_yield(wide_tree, 900, 3.0);
        CHECK(short_ranker.request_expected_tokens(900).has_value());
        CHECK(wide_ranker.request_expected_tokens(900).has_value());
        bank.forget_request(900);
        CHECK(!short_ranker.request_expected_tokens(900).has_value());
        CHECK(!wide_ranker.request_expected_tokens(900).has_value());

        // A k=0 route is shared AR work and does not manufacture the referenced
        // speculative profile. It is inherited if that profile appears later.
        bank.observe_route(dspark_linear, 4, 0, 90.0);
        CHECK(bank.profile_count() == 2);
        CHECK(bank.profile(dspark_linear).autoregressive_cost_us(4) == 90.0);
        CHECK(bank.profile_count() == 3);

        bank.reset();
        CHECK(bank.profile_count() == 0);
        CHECK(!bank.has_autoregressive_cost(5));
        CHECK(!bank.has_scout_cost(short_scout, 2));
    }

    // Work-menu exploration is ordered by adapter priority. A measured
    // profitable short route suppresses a fresh wide probe, while a complete
    // losing short profile advances to the wider shape.
    {
        const VerifierWorkKey short_work{
            SpeculatorKind::DDTree, 101, 4, 5};
        const VerifierWorkKey wide_work{
            SpeculatorKind::DDTree, 102, 8, 9};
        auto menus = [&]() {
            std::vector<RequestVerifierWorkMenu> out;
            for (int slot = 0; slot < 2; ++slot) {
                RequestVerifierWorkMenu menu;
                menu.request.request_id =
                    static_cast<std::uint64_t>(1000 + slot);
                menu.request.slot = slot;
                menu.request.progress_tokens = slot;
                VerifierWorkPlan short_plan;
                short_plan.work = short_work;
                short_plan.maximum_emitted_tokens = 5.0;
                short_plan.confidence_expected_tokens =
                    slot == 0 ? 4.0 : 2.0;
                short_plan.max_parallel_requests = 1;
                short_plan.adaptive_request_limit = 1;
                short_plan.exploration_priority = 0;
                VerifierWorkPlan wide_plan;
                wide_plan.work = wide_work;
                wide_plan.maximum_emitted_tokens = 9.0;
                wide_plan.confidence_expected_tokens =
                    short_plan.confidence_expected_tokens;
                wide_plan.max_parallel_requests = 1;
                wide_plan.adaptive_request_limit = 1;
                wide_plan.exploration_priority = 1;
                menu.speculative = {short_plan, wide_plan};
                out.push_back(std::move(menu));
            }
            return out;
        }();

        AdaptiveVerificationProfileBank profitable;
        profitable.observe_autoregressive(2, 100.0);
        AdaptiveVerificationWorkDecision selected =
            profitable.select_work(2, menus);
        CHECK(selected.has_work());
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.work.value() == short_work);
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.decision.exploring);
        CHECK(selected.decision.requests.size() == 1);

        profitable.observe_route(short_work, 2, 1, 70.0);
        selected = profitable.select_work(2, menus);
        CHECK(selected.work.value() == short_work);
        CHECK(!selected.decision.exploring);
        CHECK(selected.decision.requests.size() == 1);
        CHECK(!profitable.profile(wide_work).has_route_cost(2, 1));

        AdaptiveVerificationProfileBank losing;
        losing.observe_autoregressive(2, 100.0);
        losing.observe_route(short_work, 2, 1, 300.0);
        selected = losing.select_work(2, menus);
        CHECK(selected.has_work());
        CHECK(selected.work.value() == wide_work);
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.decision.exploring);
        CHECK(selected.decision.requests.size() == 1);
    }

    // Refill relaxation is scoped to each exact work shape's executable
    // coverage. Two lanes may optimize aggregate C=5 goodput, while a
    // one-lane C=16 route still protects its fifteen AR peers even when its
    // aggregate token estimate looks attractive.
    {
        auto make_menus = [](
                int active, int eligible, const VerifierWorkKey & work,
                int adaptive_limit, double confidence) {
            std::vector<RequestVerifierWorkMenu> menus;
            menus.reserve((size_t)active);
            for (int slot = 0; slot < active; ++slot) {
                RequestVerifierWorkMenu menu;
                menu.request.request_id =
                    static_cast<std::uint64_t>(9000 + slot);
                menu.request.slot = slot;
                if (slot < eligible) {
                    VerifierWorkPlan plan;
                    plan.work = work;
                    plan.maximum_emitted_tokens = confidence;
                    plan.confidence_expected_tokens = confidence;
                    plan.max_parallel_requests = active;
                    plan.adaptive_request_limit = adaptive_limit;
                    menu.speculative = {plan};
                }
                menus.push_back(std::move(menu));
            }
            return menus;
        };

        const VerifierWorkKey broad_work{
            SpeculatorKind::DDTree, 901, 4, 5};
        AdaptiveVerificationProfileBank broad;
        broad.observe_autoregressive(5, 100.0);
        broad.observe_route(broad_work, 5, 1, 130.0);
        broad.observe_route(broad_work, 5, 2, 120.0);
        const auto broad_menus = make_menus(
            5, 2, broad_work, 2, 4.0);
        CHECK(broad.select_work(5, broad_menus).decision.requests.empty());
        const AdaptiveVerificationWorkDecision refill_broad =
            broad.select_work(
                5, broad_menus,
                /*probe_uncalibrated_with_verifier=*/false,
                /*enforce_ar_peer_guard=*/true,
                /*minimum_speculative_route_samples=*/1,
                /*trust_stable_confidence=*/true,
                /*allow_safe_peer_guard_relaxation=*/true);
        CHECK(refill_broad.decision.requests.size() == 2);

        const VerifierWorkKey narrow_work{
            SpeculatorKind::DDTree, 902, 4, 5};
        AdaptiveVerificationProfileBank narrow;
        narrow.observe_autoregressive(16, 100.0);
        narrow.observe_route(narrow_work, 16, 1, 120.0);
        const auto narrow_menus = make_menus(
            16, 1, narrow_work, 1, 21.0);
        const AdaptiveVerificationWorkDecision refill_narrow =
            narrow.select_work(
                16, narrow_menus,
                /*probe_uncalibrated_with_verifier=*/false,
                /*enforce_ar_peer_guard=*/true,
                /*minimum_speculative_route_samples=*/1,
                /*trust_stable_confidence=*/true,
                /*allow_safe_peer_guard_relaxation=*/true);
        CHECK(refill_narrow.status ==
              AdaptiveVerificationWorkStatus::Autoregressive);
        CHECK(refill_narrow.decision.requests.empty());
        CHECK(narrow.select_work(
                  16, narrow_menus,
                  /*probe_uncalibrated_with_verifier=*/false,
                  /*enforce_ar_peer_guard=*/false)
                  .decision.requests.size() == 1);
    }

    // Target-verified yield is local to each exact work shape and replaces raw
    // confidence once stable. The wider route wins despite its lower raw score
    // because only its target outcomes are useful.
    {
        const VerifierWorkKey short_work{
            SpeculatorKind::DDTree, 201, 4, 5};
        const VerifierWorkKey wide_work{
            SpeculatorKind::DDTree, 202, 8, 9};
        AdaptiveVerificationProfileBank bank;
        bank.observe_autoregressive(1, 100.0);
        bank.observe_route(short_work, 1, 1, 100.0);
        bank.observe_route(wide_work, 1, 1, 100.0);
        bank.observe_request_estimate(
            short_work, 2000, SpeculatorKind::DDTree, 8.0);
        bank.observe_request_estimate(
            wide_work, 2000, SpeculatorKind::DDTree, 2.0);
        for (int sample = 0; sample < 4; ++sample) {
            bank.observe_request_yield(short_work, 2000, 1.0);
            bank.observe_request_yield(wide_work, 2000, 4.0);
        }

        RequestVerifierWorkMenu menu;
        menu.request.request_id = 2000;
        menu.request.slot = 7;
        VerifierWorkPlan short_plan;
        short_plan.work = short_work;
        short_plan.maximum_emitted_tokens = 8.0;
        short_plan.confidence_expected_tokens = 8.0;
        short_plan.max_parallel_requests = 1;
        short_plan.adaptive_request_limit = 1;
        short_plan.exploration_priority = 0;
        VerifierWorkPlan wide_plan;
        wide_plan.work = wide_work;
        wide_plan.maximum_emitted_tokens = 8.0;
        wide_plan.confidence_expected_tokens = 2.0;
        wide_plan.max_parallel_requests = 1;
        wide_plan.adaptive_request_limit = 1;
        wide_plan.exploration_priority = 1;
        menu.speculative = {short_plan, wide_plan};

        const AdaptiveVerificationWorkDecision selected =
            bank.select_work(1, {menu});
        CHECK(selected.has_work());
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.work.value() == wide_work);
        CHECK(selected.decision.requests == std::vector<int>{7});
        CHECK(!selected.decision.exploring);
    }

    // Always requests cannot disappear into AR. The adapter's preferred plan
    // wins at cold start; without one, lower exploration priority is the
    // deterministic forced-mode fallback.
    {
        const VerifierWorkKey short_work{
            SpeculatorKind::DDTree, 301, 4, 5};
        const VerifierWorkKey wide_work{
            SpeculatorKind::DDTree, 302, 8, 9};
        RequestVerifierWorkMenu menu;
        menu.request.request_id = 3000;
        menu.request.slot = 5;
        menu.request.required = true;
        VerifierWorkPlan short_plan;
        short_plan.work = short_work;
        short_plan.maximum_emitted_tokens = 5.0;
        short_plan.max_parallel_requests = 1;
        short_plan.adaptive_request_limit = 1;
        short_plan.exploration_priority = 0;
        VerifierWorkPlan wide_plan;
        wide_plan.work = wide_work;
        wide_plan.maximum_emitted_tokens = 9.0;
        wide_plan.preferred_for_forced_mode = true;
        wide_plan.max_parallel_requests = 1;
        wide_plan.adaptive_request_limit = 1;
        wide_plan.exploration_priority = 4;
        menu.speculative = {short_plan, wide_plan};

        AdaptiveVerificationProfileBank preferred_bank;
        AdaptiveVerificationWorkDecision selected =
            preferred_bank.select_work(1, {menu});
        CHECK(selected.work.value() == wide_work);
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.decision.requests == std::vector<int>{5});

        menu.speculative[1].preferred_for_forced_mode = false;
        AdaptiveVerificationProfileBank priority_bank;
        selected = priority_bank.select_work(1, {menu});
        CHECK(selected.work.value() == short_work);
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.decision.requests == std::vector<int>{5});
    }

    // Work decisions distinguish normal AR, confidence calibration, malformed
    // adapter input, and a forced request that has no executable common shape.
    {
        const VerifierWorkKey work{
            SpeculatorKind::DDTree, 401, 4, 5};
        VerifierWorkPlan plan;
        plan.work = work;
        plan.maximum_emitted_tokens = 5.0;
        plan.max_parallel_requests = 2;
        plan.adaptive_request_limit = 1;

        RequestVerifierWorkMenu empty;
        empty.request.request_id = 4000;
        empty.request.slot = 0;
        AdaptiveVerificationProfileBank ar_bank;
        AdaptiveVerificationWorkDecision selected =
            ar_bank.select_work(1, {empty});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Autoregressive);
        CHECK(!selected.has_work());

        RequestVerifierWorkMenu unknown = empty;
        unknown.speculative = {plan};
        AdaptiveVerificationProfileBank calibration_bank;
        calibration_bank.observe_autoregressive(1, 100.0);
        selected = calibration_bank.select_work(1, {unknown});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Calibration);
        CHECK(selected.work.value() == work);
        CHECK(selected.decision.calibration_request == 0);

        selected = calibration_bank.select_work(2, {unknown});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::InvalidMenu);
        CHECK(!selected.has_work());

        RequestVerifierWorkMenu duplicate_slot = empty;
        duplicate_slot.request.request_id = 4001;
        selected = ar_bank.select_work(2, {empty, duplicate_slot});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::InvalidMenu);

        RequestVerifierWorkMenu duplicate_request = empty;
        duplicate_request.request.slot = 1;
        selected = ar_bank.select_work(2, {empty, duplicate_request});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::InvalidMenu);

        RequestVerifierWorkMenu inconsistent_a = unknown;
        RequestVerifierWorkMenu inconsistent_b = unknown;
        inconsistent_b.request.request_id = 4001;
        inconsistent_b.request.slot = 1;
        inconsistent_b.speculative[0].exploration_priority = 1;
        selected = ar_bank.select_work(
            2, {inconsistent_a, inconsistent_b});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::InvalidMenu);

        RequestVerifierWorkMenu duplicate_work = unknown;
        duplicate_work.speculative.push_back(plan);
        selected = ar_bank.select_work(1, {duplicate_work});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::InvalidMenu);

        RequestVerifierWorkMenu mixed_speculators = unknown;
        VerifierWorkPlan dspark_plan = plan;
        dspark_plan.work = {
            SpeculatorKind::DSpark, 403, 4, 5};
        mixed_speculators.speculative.push_back(dspark_plan);
        selected = ar_bank.select_work(1, {mixed_speculators});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::InvalidMenu);

        RequestVerifierWorkMenu required_empty = empty;
        required_empty.request.required = true;
        selected = ar_bank.select_work(1, {required_empty});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::RequiredUnavailable);

        const VerifierWorkKey other_work{
            SpeculatorKind::DDTree, 402, 8, 9};
        RequestVerifierWorkMenu required_a = unknown;
        required_a.request.required = true;
        RequestVerifierWorkMenu required_b = unknown;
        required_b.request.request_id = 4001;
        required_b.request.slot = 1;
        required_b.request.required = true;
        required_b.speculative[0].work = other_work;
        selected = ar_bank.select_work(2, {required_a, required_b});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::RequiredUnavailable);
    }

    // The adaptive limit bounds optional profiling, while required/Always
    // requests may cross it only when the hard executor capacity permits all.
    {
        const VerifierWorkKey work{
            SpeculatorKind::DDTree, 501, 4, 5};
        VerifierWorkPlan plan;
        plan.work = work;
        plan.maximum_emitted_tokens = 5.0;
        plan.preferred_for_forced_mode = true;
        plan.max_parallel_requests = 2;
        plan.adaptive_request_limit = 1;
        std::vector<RequestVerifierWorkMenu> required;
        for (int slot = 0; slot < 2; ++slot) {
            RequestVerifierWorkMenu menu;
            menu.request.request_id =
                static_cast<std::uint64_t>(5000 + slot);
            menu.request.slot = slot;
            menu.request.required = true;
            menu.speculative = {plan};
            required.push_back(std::move(menu));
        }
        AdaptiveVerificationProfileBank bank;
        AdaptiveVerificationWorkDecision selected =
            bank.select_work(2, required);
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.decision.requests.size() == 2);

        for (RequestVerifierWorkMenu & menu : required) {
            menu.speculative[0].max_parallel_requests = 1;
        }
        selected = bank.select_work(2, required);
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::RequiredUnavailable);
    }

    // Once required routes are measured, absolute goodput overrides the cold
    // forced-mode preference. Relative gain is one for both required baselines.
    {
        const VerifierWorkKey preferred_work{
            SpeculatorKind::DDTree, 601, 4, 5};
        const VerifierWorkKey faster_work{
            SpeculatorKind::DDTree, 602, 8, 9};
        RequestVerifierWorkMenu menu;
        menu.request.request_id = 6000;
        menu.request.slot = 0;
        menu.request.required = true;
        VerifierWorkPlan preferred;
        preferred.work = preferred_work;
        preferred.maximum_emitted_tokens = 4.0;
        preferred.confidence_expected_tokens = 2.0;
        preferred.preferred_for_forced_mode = true;
        preferred.max_parallel_requests = 1;
        preferred.adaptive_request_limit = 1;
        VerifierWorkPlan faster = preferred;
        faster.work = faster_work;
        faster.preferred_for_forced_mode = false;
        menu.speculative = {preferred, faster};

        AdaptiveVerificationProfileBank bank;
        bank.observe_route(preferred_work, 1, 1, 200.0);
        bank.observe_route(faster_work, 1, 1, 100.0);
        const AdaptiveVerificationWorkDecision selected =
            bank.select_work(1, {menu});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.work.value() == faster_work);
        CHECK(std::abs(
            selected.decision.predicted_goodput - 0.02) < 1e-12);
        CHECK(selected.decision.predicted_gain == 1.0);
    }

    // Adapter confidence cannot claim more useful output than the selected
    // work shape can emit; both calibration storage and goodput use the clamp.
    {
        const VerifierWorkKey work{
            SpeculatorKind::DSpark, 701, 1, 2};
        RequestVerifierWorkMenu menu;
        menu.request.request_id = 7000;
        menu.request.slot = 0;
        VerifierWorkPlan plan;
        plan.work = work;
        plan.maximum_emitted_tokens = 2.0;
        plan.confidence_expected_tokens = 100.0;
        plan.max_parallel_requests = 1;
        plan.adaptive_request_limit = 1;
        CHECK(plan.bounded_confidence_expected_tokens() == 2.0);
        menu.speculative = {plan};

        AdaptiveVerificationProfileBank bank;
        bank.observe_autoregressive(1, 100.0);
        bank.observe_route(work, 1, 1, 100.0);
        const AdaptiveVerificationWorkDecision selected =
            bank.select_work(1, {menu});
        CHECK(selected.work.value() == work);
        CHECK(std::abs(
            selected.decision.predicted_goodput - 0.02) < 1e-12);
        const auto estimate =
            bank.profile(work).estimate_request_yield(7000);
        CHECK(estimate.has_value());
        CHECK(estimate->expected_tokens == 2.0);
        CHECK(estimate->confidence_expected_tokens == 2.0);
    }

    // An exact losing short profile's fallback calibration cannot indefinitely
    // block bounded exploration of the next wider work shape.
    {
        const VerifierWorkKey short_work{
            SpeculatorKind::DDTree, 801, 4, 5};
        const VerifierWorkKey wide_work{
            SpeculatorKind::DDTree, 802, 8, 9};
        RequestVerifierWorkMenu menu;
        menu.request.request_id = 8000;
        menu.request.slot = 0;
        VerifierWorkPlan short_plan;
        short_plan.work = short_work;
        short_plan.maximum_emitted_tokens = 5.0;
        short_plan.max_parallel_requests = 1;
        short_plan.adaptive_request_limit = 1;
        short_plan.exploration_priority = 0;
        VerifierWorkPlan wide_plan;
        wide_plan.work = wide_work;
        wide_plan.maximum_emitted_tokens = 9.0;
        wide_plan.confidence_expected_tokens = 4.0;
        wide_plan.max_parallel_requests = 1;
        wide_plan.adaptive_request_limit = 1;
        wide_plan.exploration_priority = 1;
        menu.speculative = {short_plan, wide_plan};

        AdaptiveVerificationProfileBank bank;
        bank.observe_autoregressive(1, 100.0);
        bank.observe_route(short_work, 1, 1, 300.0);
        const AdaptiveVerificationWorkDecision selected =
            bank.select_work(1, {menu});
        CHECK(selected.status ==
              AdaptiveVerificationWorkStatus::Verification);
        CHECK(selected.work.value() == wide_work);
        CHECK(selected.decision.exploring);
    }

    // Shared cohort evidence can rank a new request, but it cannot unlock the
    // slow-route homogeneous exception. A bounded verifier probe may cross the
    // steady peer guard to gather the missing request-local evidence.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 150.0);
        std::vector<AdaptiveVerificationCandidate> confidence_backed;
        for (int request = 1; request <= 5; ++request) {
            confidence_backed.push_back(
                {request, 8.0, 8.0, true, 8.0, 0});
        }
        CHECK(ranker.select(5, confidence_backed, 1).requests.empty());

        std::vector<AdaptiveVerificationCandidate> probe = confidence_backed;
        for (AdaptiveVerificationCandidate & candidate : probe) {
            candidate.calibrated = false;
        }
        const AdaptiveVerificationDecision exploring = ranker.select(
            5, probe, 1,
            /*probe_uncalibrated_with_verifier=*/true);
        CHECK(exploring.exploring);
        CHECK(exploring.requests.size() == 1);

        for (AdaptiveVerificationCandidate & candidate : confidence_backed) {
            candidate.evidence_samples = 4;
        }
        CHECK(ranker.select(5, confidence_backed, 1).requests.size() == 1);
        confidence_backed.back().expected_tokens = 1.0;
        CHECK(ranker.select(5, confidence_backed, 1).requests.empty());
    }

    // With route costs already profiled, a no-confidence adapter calibrates
    // exactly one unknown request in the best optimistic measured width.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 80.0);
        ranker.observe_route(5, 2, 95.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {14, 4.0, 8.0, true, 1.0},
            {15, 1.0, 8.0, false},
            {16, 1.0, 8.0, false},
        };
        const AdaptiveVerificationDecision decision = ranker.select(
            5, candidates, 2,
            /*probe_uncalibrated_with_verifier=*/true);
        CHECK(decision.exploring);
        CHECK(decision.requests.size() == 2);
        CHECK(decision.requests[0] == 14);
        CHECK(decision.requests[1] == 15);
    }

    // A promising newcomer can replace one incumbent even when every executor
    // lane is occupied; calibration remains one bounded K-wide step.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 80.0);
        ranker.observe_route(5, 2, 90.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {17, 4.0, 8.0, true},
            {18, 4.0, 8.0, true},
            {19, 1.0, 8.0, false, 1.0},
        };
        const AdaptiveVerificationDecision decision = ranker.select(
            5, candidates, 2,
            /*probe_uncalibrated_with_verifier=*/true);
        CHECK(decision.exploring);
        CHECK(decision.requests.size() == 2);
        CHECK(decision.requests[0] == 17);
        CHECK(decision.requests[1] == 19);
    }

    // Selection is tied to request value, not to a lane count. When the
    // profitable request retires, speculation does not migrate to a low-yield
    // peer.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(2, 100.0);
        ranker.observe_route(2, 1, 100.0);
        std::vector<AdaptiveVerificationCandidate> mixed_yield = {
            {21, 8.0, 8.0, true, 1.0},
            {22, 1.0, 8.0, true, -1.0},
        };
        AdaptiveVerificationDecision decision =
            ranker.select(2, mixed_yield, 1);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 21);
        ranker.observe_autoregressive(1, 60.0);
        ranker.observe_route(1, 1, 100.0);
        decision = ranker.select(1, {mixed_yield[1]}, 1);
        CHECK(decision.requests.empty());
    }

    // There is no concurrency cutoff: when one request pays for the route at
    // C=16, that request alone remains speculative and all peers remain AR.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(16, 160.0);
        ranker.observe_route(16, 1, 150.0);
        ranker.observe_route(16, 2, 220.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {21, 8.0, 8.0, true},
            {22, 2.0, 8.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(16, candidates);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 21);
    }

    // Executor capacity is independent of C, and only eligible candidates
    // consume speculative lanes. Three non-candidates remain ordinary AR.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 160.0);
        ranker.observe_route(5, 2, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {51, 4.0, 8.0, true},
            {52, 4.0, 8.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(5, candidates, 3);
        CHECK(ranker.has_exact_profile(5, 2));
        CHECK(!ranker.has_exact_profile(5, 3));
        CHECK(decision.requests.size() == 2);
    }

    // The safety margin rejects a noisy 4% estimate and admits a 6% gain.
    {
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {61, 1.0, 1.0, true},
        };
        AdaptiveVerificationRanker below_margin;
        below_margin.observe_autoregressive(1, 100.0);
        below_margin.observe_route(1, 1, 96.0);
        CHECK(below_margin.select(1, candidates).requests.empty());

        AdaptiveVerificationRanker above_margin;
        above_margin.observe_autoregressive(1, 100.0);
        above_margin.observe_route(1, 1, 94.0);
        CHECK(above_margin.select(1, candidates).requests.size() == 1);
    }

    // DDTree accepted-path yield and DSpark conditional survival use the same
    // model-neutral expected-token contract and therefore make the same choice.
    {
        const float confidence[] = {0.8f, 0.5f};
        const double dspark_expected =
            expected_tokens_from_conditional_confidence(confidence, 2);
        AdaptiveVerificationRanker ddtree;
        AdaptiveVerificationRanker dspark;
        for (AdaptiveVerificationRanker * ranker : {&ddtree, &dspark}) {
            ranker->observe_autoregressive(4, 100.0);
            ranker->observe_route(4, 1, 80.0);
        }
        const auto ddtree_decision = ddtree.select(
            4, {{71, 2.2, 3.0, true}}, 1);
        const auto dspark_decision = dspark.select(
            4, {{71, dspark_expected, 3.0, true}}, 1);
        CHECK(ddtree_decision.requests == dspark_decision.requests);
        CHECK(ddtree_decision.requests.size() == 1);
    }

    // Executor capacity bounds the ranked prefix. This keeps every adaptive
    // route on the one-pass implementation even when more requests rank well.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(16, 160.0);
        ranker.observe_route(16, 1, 145.0);
        ranker.observe_route(16, 2, 150.0);
        ranker.observe_route(16, 3, 155.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {41, 8.0, 8.0, true},
            {42, 7.0, 8.0, true},
            {43, 6.0, 8.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(16, candidates, /*max_speculative_requests=*/2);
        CHECK(!decision.exploring);
        CHECK(decision.requests.size() == 2);
        CHECK(decision.requests[0] == 41);
        CHECK(decision.requests[1] == 42);
    }

    // Exact profiling and executor bounds are independent of C. Every fresh
    // occupancy from 1 through 16 discovers k=1..min(C, 3), including widths
    // whose smaller neighbors lose.
    {
        for (int concurrency = 1; concurrency <= 16; ++concurrency) {
            AdaptiveVerificationRanker ranker;
            ranker.observe_autoregressive(concurrency, 100.0);
            const int limit = std::min(concurrency, 3);
            std::vector<AdaptiveVerificationCandidate> candidates;
            for (int i = 0; i < limit; ++i) {
                candidates.push_back({
                    concurrency * 10 + i, 4.0, 8.0, true});
            }
            for (int width = 1; width <= limit; ++width) {
                const AdaptiveVerificationDecision probe = ranker.select(
                    concurrency, candidates, limit);
                CHECK(probe.exploring);
                CHECK(static_cast<int>(probe.requests.size()) == width);
                ranker.observe_route(
                    concurrency, width, 150.0 - 20.0 * width);
            }
            CHECK(ranker.has_exact_profile(concurrency, limit));
            CHECK(ranker.select(concurrency, candidates, 0)
                      .requests.empty());
        }
    }

    // With exact route costs known, request-granular selection has no C
    // cutoff: one or two useful requests remain speculative at every
    // occupancy through the supported C=16 while low-yield peers stay AR.
    {
        for (int concurrency = 1; concurrency <= 16; ++concurrency) {
            AdaptiveVerificationRanker ranker;
            ranker.observe_autoregressive(concurrency, 100.0);
            for (int width = 1; width <= concurrency; ++width) {
                ranker.observe_route(concurrency, width, 100.0);
            }
            std::vector<AdaptiveVerificationCandidate> one_useful;
            std::vector<AdaptiveVerificationCandidate> two_useful;
            for (int request = 0; request < concurrency; ++request) {
                one_useful.push_back({
                    request, request == 0 ? 4.0 : 1.0, 4.0, true});
                two_useful.push_back({
                    request, request < 2 ? 4.0 : 1.0, 4.0, true});
            }
            CHECK(ranker.select(concurrency, one_useful, concurrency)
                      .requests.size() == 1);
            CHECK(ranker.select(concurrency, two_useful, concurrency)
                      .requests.size() ==
                  static_cast<std::size_t>(std::min(2, concurrency)));
        }
    }

    // A promising calibrated prefix gets one bounded hardware-cost probe when
    // that subbatch shape has not been observed yet.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(3, 90.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {31, 6.0, 8.0, true},
        };
        const AdaptiveVerificationDecision probe =
            ranker.select(3, candidates);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 1);
        CHECK(probe.requests[0] == 31);
    }

    // A shrinking cohort may suppress a brand-new tail profile without
    // disabling a route that was already measured at that exact occupancy.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(3, 90.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {32, 6.0, 8.0, true},
        };
        AdaptiveVerificationDecision decision = ranker.select(
            3, candidates, 1,
            /*probe_uncalibrated_with_verifier=*/false,
            /*probe_missing_routes=*/false);
        CHECK(decision.requests.empty());
        CHECK(!decision.exploring);
        CHECK(!ranker.has_speculative_profile_sample(3, 1));

        ranker.observe_route(3, 1, 70.0);
        decision = ranker.select(
            3, candidates, 1,
            /*probe_uncalibrated_with_verifier=*/false,
            /*probe_missing_routes=*/false);
        CHECK(decision.requests.size() == 1);
        CHECK(ranker.has_speculative_profile_sample(3, 1));
    }

    // Kill-switch mode retains fixed speculation and ignores observations.
    {
        SpeculationGoodputController policy;
        policy.reset(/*adaptive=*/false);
        CHECK(policy.wants_speculation());
        CHECK(!policy.adaptive());
        CHECK(policy.observe_speculation(1, 1000.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.observe_autoregressive(1.0) ==
              SpeculationGoodputTransition::none);
        CHECK(policy.wants_speculation());
    }

    std::printf("speculation goodput policy: %d checks passed\n", g_checks);
    return 0;
}
