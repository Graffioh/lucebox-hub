#include "common/concurrency/adaptive_verification.h"
#include "common/concurrency/speculation_goodput.h"
#include "common/concurrency/speculation_prompt_prior.h"
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

    // The cold-start prior separates obvious structured/code requests from
    // conversational writing while leaving ambiguous requests neutral.
    {
        CHECK(speculation_prompt_hint(
                  "Complete this Python function:\n\ndef solve(values):") == 1);
        CHECK(speculation_prompt_hint(
                  "Chatting casually, write a story and avoid code.") == -1);
        CHECK(speculation_prompt_hint(
                  "What happened during the Apollo 11 mission?") == 0);
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

    // A chat-like low-yield probe loses to AR and is disabled per request.
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
        CHECK(decision.exploring);
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
    // even when k=1 and k=2 lose. The prompt prior orders unknown requests but
    // never filters them, and the globally best non-convex k=3 route wins.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {5, 1.0, 9.0, false, 1.0},
            {2, 1.0, 9.0, false, -1.0},
            {9, 1.0, 9.0, false, 0.0},
        };
        AdaptiveVerificationDecision probe =
            ranker.select(8, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 1);
        CHECK(probe.requests[0] == 5);
        CHECK(probe.calibration_request == -1);
        ranker.observe_route(8, 1, 160.0);
        candidates[0] = {5, 4.0, 9.0, true, 1.0};

        probe = ranker.select(8, candidates, 3,
                              /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 2);
        CHECK(probe.requests[0] == 5);
        CHECK(probe.requests[1] == 9);
        ranker.observe_route(8, 2, 170.0);
        candidates[2] = {9, 4.0, 9.0, true, 0.0};

        probe = ranker.select(8, candidates, 3,
                              /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 3);
        CHECK(probe.requests[0] == 5);
        CHECK(probe.requests[1] == 9);
        CHECK(probe.requests[2] == 2);
        ranker.observe_route(8, 3, 105.0);
        candidates[1] = {2, 4.0, 9.0, true, -1.0};

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

    // A prior affects cold-start order only. Once calibrated, the high-yield
    // conversational-prior request ranks ahead of a low-yield code prior.
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

    // Verified yield can seed later requests in the same coarse routing-prior
    // bucket. The cache belongs to this speculator/profile ranker and resets
    // with it; no model-specific table is required.
    {
        AdaptiveVerificationRanker ranker;
        CHECK(!ranker.routing_prior_expected_tokens(1.0).has_value());
        ranker.observe_routing_prior_yield(1.0, 4.0);
        ranker.observe_routing_prior_yield(1.0, 6.0);
        ranker.observe_routing_prior_yield(1.0, 4.0);
        CHECK(std::abs(
                  ranker.routing_prior_expected_tokens(1.0).value() -
                  14.0 / 3.0) < 1e-9);
        CHECK(ranker.routing_prior_yield_samples(1.0) == 3);
        ranker.observe_routing_prior_yield(1.0, 6.0);
        CHECK(ranker.routing_prior_expected_tokens(1.0).value() == 5.0);
        CHECK(ranker.routing_prior_yield_samples(1.0) == 4);
        CHECK(!ranker.has_stable_routing_prior_yield(1.0));
        CHECK(!ranker.routing_prior_expected_tokens(-1.0).has_value());
        CHECK(!ranker.has_stable_routing_prior_yield(-1.0));

        // Shared yield magnitude itself starts conservatively shrunk toward
        // AR; four local samples still make the request authoritative.
        const double expected[] = {2.5, 2.0, 1.5, 1.0};
        for (int sample = 1; sample <= 4; ++sample) {
            ranker.observe_request_yield(98, 1.0);
            const auto estimate =
                ranker.estimate_request_yield(98, 1.0);
            CHECK(estimate.has_value());
            CHECK(estimate->expected_tokens == expected[sample - 1]);
            CHECK(estimate->evidence_samples ==
                  static_cast<std::size_t>(sample));
        }
        const auto local = ranker.estimate_request_yield(98, 1.0);
        CHECK(local.has_value());
        CHECK(local->expected_tokens == 1.0);

        ranker.observe_routing_prior_yield(1.0, 4.0);
        CHECK(!ranker.has_stable_routing_prior_yield(1.0));
        ranker.observe_routing_prior_yield(1.0, 6.0);
        CHECK(ranker.has_stable_routing_prior_yield(1.0));
        const auto closed_prior =
            ranker.estimate_request_yield(100, 1.0);
        const auto backlog_prior =
            ranker.estimate_request_yield(
                100, 1.0, /*trust_stable_routing_prior=*/true);
        CHECK(closed_prior.has_value());
        CHECK(backlog_prior.has_value());
        CHECK(closed_prior->expected_tokens == 4.0);
        CHECK(backlog_prior->expected_tokens == 5.0);
        for (int i = 0; i < 6; ++i) {
            ranker.observe_routing_prior_yield(-1.0, 1.25);
        }
        CHECK(!ranker.has_stable_routing_prior_yield(-1.0));
        CHECK(ranker.forms_stable_routing_prior_cohort({
            {101, 5.0, 9.0, true, 1.0},
            {102, 5.0, 9.0, true, 1.0},
        }));
        CHECK(!ranker.forms_stable_routing_prior_cohort({
            {101, 5.0, 9.0, true, 1.0},
            {103, 1.25, 9.0, true, -1.0},
        }));

        // Per-request evidence is local to one proposal-shape ranker and is
        // explicitly forgotten at request retirement.
        AdaptiveVerificationRanker compact;
        ranker.observe_request_yield(99, 7.0);
        CHECK(ranker.request_expected_tokens(99).value() == 7.0);
        CHECK(ranker.request_yield_samples(99) == 1);
        CHECK(!compact.request_expected_tokens(99).has_value());
        ranker.forget_request(99);
        CHECK(!ranker.request_expected_tokens(99).has_value());
        ranker.reset();
        CHECK(!ranker.routing_prior_expected_tokens(1.0).has_value());
    }

    // Shared cohort evidence can rank a new request, but it cannot unlock the
    // slow-route homogeneous exception. A bounded verifier probe may cross the
    // steady peer guard to gather the missing request-local evidence.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(5, 100.0);
        ranker.observe_route(5, 1, 150.0);
        std::vector<AdaptiveVerificationCandidate> prior_backed;
        for (int request = 1; request <= 5; ++request) {
            prior_backed.push_back(
                {request, 8.0, 8.0, true, 1.0, 0});
        }
        CHECK(ranker.select(5, prior_backed, 1).requests.empty());

        std::vector<AdaptiveVerificationCandidate> probe = prior_backed;
        for (AdaptiveVerificationCandidate & candidate : probe) {
            candidate.calibrated = false;
        }
        const AdaptiveVerificationDecision exploring = ranker.select(
            5, probe, 1,
            /*probe_uncalibrated_with_verifier=*/true);
        CHECK(exploring.exploring);
        CHECK(exploring.requests.size() == 1);

        for (AdaptiveVerificationCandidate & candidate : prior_backed) {
            candidate.evidence_samples = 4;
        }
        CHECK(ranker.select(5, prior_backed, 1).requests.size() == 1);
        prior_backed.back().expected_tokens = 1.0;
        CHECK(ranker.select(5, prior_backed, 1).requests.empty());
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
            {15, 1.0, 8.0, false, -1.0},
            {16, 1.0, 8.0, false, 0.0},
        };
        const AdaptiveVerificationDecision decision = ranker.select(
            5, candidates, 2,
            /*probe_uncalibrated_with_verifier=*/true);
        CHECK(decision.exploring);
        CHECK(decision.requests.size() == 2);
        CHECK(decision.requests[0] == 14);
        CHECK(decision.requests[1] == 16);
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
    // profitable request retires, speculation does not migrate to chat.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(2, 100.0);
        ranker.observe_route(2, 1, 100.0);
        std::vector<AdaptiveVerificationCandidate> code_and_chat = {
            {21, 8.0, 8.0, true, 1.0},
            {22, 1.0, 8.0, true, -1.0},
        };
        AdaptiveVerificationDecision decision =
            ranker.select(2, code_and_chat, 1);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 21);
        ranker.observe_autoregressive(1, 60.0);
        ranker.observe_route(1, 1, 100.0);
        decision = ranker.select(1, {code_and_chat[1]}, 1);
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
