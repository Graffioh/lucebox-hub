#include "common/concurrency/adaptive_verification.h"
#include "common/concurrency/speculation_goodput.h"
#include "common/concurrency/speculation_prompt_prior.h"
#include "host_check.h"

#include <cstdio>

using namespace dflash::common;

static int g_checks = 0;

int main() {
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

    // Candidates are ranked by expected value. Greedy growth stops at the
    // first prefix that would reduce whole-batch throughput.
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
            ranker.select(5, candidates);
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

    // A concrete adapter may calibrate a bounded compact bundle inside useful
    // verification. The first unseen route probes up to executor capacity,
    // never the whole unknown cohort.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {5, 1.0, 9.0, false, 1.0},
            {2, 1.0, 9.0, false, -1.0},
        };
        const AdaptiveVerificationDecision probe =
            ranker.select(8, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/true);
        CHECK(probe.exploring);
        CHECK(probe.requests.size() == 2);
        CHECK(probe.requests[0] == 5);
        CHECK(probe.requests[1] == 2);
        CHECK(probe.calibration_request == -1);
    }

    // A losing route ratio suppresses repeated probes as occupancy falls.
    // Even perfect acceptance cannot make this projected route beat AR.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(16, 100.0);
        ranker.observe_route(16, 1, 200.0);
        ranker.observe_autoregressive(15, 95.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {7, 1.0, 9.0, false},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(15, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/true,
                          /*project_cost_from_higher_occupancy=*/true);
        CHECK(!decision.exploring);
        CHECK(decision.requests.empty());
        CHECK(decision.calibration_request == 7);
    }
    // A profitable measured route ratio can rank calibrated requests after
    // occupancy drops, without spending a fresh hardware probe.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        ranker.observe_route(8, 1, 80.0);
        ranker.observe_autoregressive(7, 90.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {12, 4.0, 9.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(7, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/false,
                          /*project_cost_from_higher_occupancy=*/true);
        CHECK(!decision.exploring);
        CHECK(decision.requests.size() == 1);
        CHECK(decision.requests[0] == 12);
    }

    // Prefix widths are compared independently because hardware occupancy can
    // make k=3 profitable even when the measured k=1 route loses.
    {
        AdaptiveVerificationRanker ranker;
        ranker.observe_autoregressive(8, 100.0);
        ranker.observe_route(8, 1, 120.0);
        ranker.observe_route(8, 3, 80.0);
        std::vector<AdaptiveVerificationCandidate> candidates = {
            {1, 4.0, 9.0, true},
            {2, 4.0, 9.0, true},
            {3, 4.0, 9.0, true},
        };
        const AdaptiveVerificationDecision decision =
            ranker.select(8, candidates, 3,
                          /*probe_uncalibrated_with_verifier=*/false,
                          /*project_cost_from_higher_occupancy=*/false,
                          /*evaluate_nonconvex_prefixes=*/true);
        CHECK(!decision.exploring);
        CHECK(decision.requests.size() == 3);
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
