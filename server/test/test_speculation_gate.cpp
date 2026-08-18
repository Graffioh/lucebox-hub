#include "common/concurrency/speculation_gate.h"
#include "host_check.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

namespace {

SpecCandidate candidate(
        std::uint64_t request_id, int slot,
        SpeculationPolicy policy = SpeculationPolicy::Adaptive,
        bool eligible = true,
        double prior = std::numeric_limits<double>::quiet_NaN(),
        int generated = 0) {
    return {slot, request_id, policy, eligible, prior, generated};
}

std::vector<std::pair<std::uint64_t, int>> accepted(
        std::initializer_list<std::pair<std::uint64_t, int>> values) {
    return values;
}

bool equal_slots(const std::vector<int> & actual,
                 std::initializer_list<int> expected) {
    return actual == std::vector<int>(expected);
}

SpecGateConfig permissive_config() {
    SpecGateConfig cfg;
    cfg.ema_alpha = 1.0;
    cfg.cost_ewma_alpha = 1.0;
    cfg.margin = 1.0;
    cfg.slack = 10.0;
    return cfg;
}

}  // namespace

int main() {
    // Cold start is one general all-AR baseline step. The following step
    // admits at most max_probers, ordered by score then stable request id.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.max_probers = 2;
        SpeculationGate gate(cfg, 8);
        std::vector<SpecCandidate> candidates = {
            candidate(30, 3), candidate(10, 1), candidate(20, 2),
        };
        CHECK(gate.plan(3, candidates, 3).empty());
        gate.observe_ar(3, 100.0);
        CHECK(equal_slots(gate.plan(3, candidates, 3), {1, 2}));
    }

    // Priors rank cold requests but probation still performs the real
    // measurement. Equal priors use request_id rather than reusable slot id.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.max_probers = 2;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(4, 100.0);
        std::vector<SpecCandidate> candidates = {
            candidate(9, 0, SpeculationPolicy::Adaptive, true, 3.0),
            candidate(7, 3, SpeculationPolicy::Adaptive, true, 6.0),
            candidate(5, 2, SpeculationPolicy::Adaptive, true, 6.0),
        };
        CHECK(equal_slots(gate.plan(4, candidates, 4), {2, 3}));
    }

    // Never and ineligible requests do not create request state. Always is
    // returned before a baseline and remains exempt from profitability.
    {
        SpeculationGate gate(permissive_config(), 8);
        std::vector<SpecCandidate> never = {
            candidate(1, 0, SpeculationPolicy::Never),
            candidate(2, 1, SpeculationPolicy::Never),
        };
        CHECK(gate.plan(2, never, 2).empty());
        gate.observe_ar(2, 100.0);
        std::vector<SpecCandidate> policies = {
            candidate(1, 0),
            candidate(3, 1, SpeculationPolicy::Always),
            candidate(4, 2, SpeculationPolicy::Always, false),
        };
        CHECK(equal_slots(gate.plan(2, policies, 2), {1, 0}));
        gate.observe_spec(2, 2, 1000.0, accepted({{3, 1}, {1, 1}}));
        std::vector<SpecCandidate> forced = {
            candidate(3, 1, SpeculationPolicy::Always),
        };
        CHECK(equal_slots(gate.plan(2, forced, 1), {1}));
    }

    // Forced requests beyond executor capacity are returned intact so the
    // engine can surface the configuration error instead of silently routing
    // a user-forced request through AR.
    {
        SpeculationGate gate(permissive_config(), 8);
        std::vector<SpecCandidate> forced = {
            candidate(1, 0, SpeculationPolicy::Always),
            candidate(2, 1, SpeculationPolicy::Always),
        };
        CHECK(equal_slots(gate.plan(2, forced, 1), {0, 1}));
    }

    // C=1 covers probation, the break-even transition, two-round bad-yield
    // hysteresis, AR token-cadence re-probing, and one-round re-admission.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 2;
        cfg.bad_rounds = 2;
        cfg.reprobe_tokens = 64;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(1, 100.0);

        std::vector<SpecCandidate> one = {candidate(11, 0)};
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 100.0, accepted({{11, 4}}));
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 100.0, accepted({{11, 4}}));

        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 400.0, accepted({{11, 1}}));
        CHECK(gate.plan(1, one, 1).empty());
        gate.observe_ar(1, 500.0);
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 1000.0, accepted({{11, 1}}));

        one[0].generated_tokens = 63;
        CHECK(gate.plan(1, one, 1).empty());
        one[0].generated_tokens = 64;
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 100.0, accepted({{11, 8}}));
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
    }

    // Synthetic marginal timings, rather than a universal acceptance
    // threshold, decide whether probation converges to speculation or AR.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(2, 100.0);
        std::vector<SpecCandidate> one = {candidate(21, 0)};
        CHECK(equal_slots(gate.plan(2, one, 1), {0}));
        gate.observe_spec(2, 1, 200.0, accepted({{21, 2}}));
        CHECK(gate.plan(2, one, 1).empty());
    }

    // Once an observed shape is hopeless even under max_accept, new cold
    // requests stay AR without running more speculative probes.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        cfg.max_probers = 1;
        cfg.margin = 1.05;
        SpeculationGate gate(cfg, 4);
        gate.observe_ar(8, 100.0);
        std::vector<SpecCandidate> first = {candidate(31, 0)};
        CHECK(equal_slots(gate.plan(8, first, 8), {0}));
        gate.observe_spec(8, 1, 1000.0, accepted({{31, 1}}));
        std::vector<SpecCandidate> next = {candidate(32, 1)};
        CHECK(gate.plan(8, next, 8).empty());
        CHECK(gate.plan(8, next, 8).empty());
    }

    // A missing (C,k) cost uses the nearest concurrency's per-lane affine
    // increment, then the first real measurement corrects that estimate.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(2, 100.0);
        std::vector<SpecCandidate> at_two = {candidate(41, 0)};
        CHECK(equal_slots(gate.plan(2, at_two, 1), {0}));
        gate.observe_spec(2, 1, 120.0, accepted({{41, 8}}));

        gate.observe_ar(3, 150.0);
        std::vector<SpecCandidate> at_three = {candidate(42, 1)};
        CHECK(equal_slots(gate.plan(3, at_three, 1), {1}));
        gate.observe_spec(3, 1, 1000.0, accepted({{42, 1}}));
        std::vector<SpecCandidate> corrected = {candidate(43, 2)};
        CHECK(gate.plan(3, corrected, 1).empty());
    }

    // Cost EWMAs accept valid samples and invalid observations never poison
    // either the hardware profile or request state.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.cost_ewma_alpha = 0.5;
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 2);
        gate.observe_ar(1, 100.0);
        gate.observe_ar(1, 200.0);  // T_ar = 150
        gate.observe_ar(1, 0.0);
        gate.observe_ar(1, std::numeric_limits<double>::quiet_NaN());
        std::vector<SpecCandidate> one = {candidate(51, 0)};
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1,
            std::numeric_limits<double>::quiet_NaN(), accepted({{51, 1}}));
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 300.0, accepted({{51, 2}}));
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_ar(1, 100.0);  // T_ar = 125
        CHECK(gate.plan(1, one, 1).empty());
    }

    // Eligibility may disappear for a step without discarding learned state.
    // forget() removes it on finish, and a new request reusing the same slot
    // receives independent cold probation.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(1, 100.0);
        std::vector<SpecCandidate> one = {candidate(61, 0)};
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 100.0, accepted({{61, 8}}));
        one[0].eligible = false;
        CHECK(gate.plan(1, one, 1).empty());
        one[0].eligible = true;
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));

        gate.forget(61);
        std::vector<SpecCandidate> reused = {candidate(62, 0)};
        CHECK(equal_slots(gate.plan(1, reused, 1), {0}));
        gate.forget(62);  // finishing during probation leaves no residue
        std::vector<SpecCandidate> reused_again = {candidate(63, 0)};
        CHECK(equal_slots(gate.plan(1, reused_again, 1), {0}));
    }

    // A profitable cold C=2 cohort converges to all-spec within the bounded
    // probation window, using the same mechanism as every other occupancy.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 2;
        SpeculationGate gate(cfg, 8);
        std::vector<SpecCandidate> cohort = {
            candidate(71, 0), candidate(72, 1),
        };
        CHECK(gate.plan(2, cohort, 2).empty());
        gate.observe_ar(2, 100.0);
        for (int round = 0; round < cfg.probe_rounds; ++round) {
            CHECK(equal_slots(gate.plan(2, cohort, 2), {0, 1}));
            gate.observe_spec(
                2, 2, 100.0, accepted({{71, 4}, {72, 4}}));
        }
        CHECK(equal_slots(gate.plan(2, cohort, 2), {0, 1}));
    }

    // At C=8 one expensive probation measurement makes the optimistic
    // hopeless check reject all later probes; steady state is pure AR.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 2;
        cfg.max_probers = 2;
        cfg.margin = 1.05;
        SpeculationGate gate(cfg, 8);
        std::vector<SpecCandidate> cohort;
        for (int i = 0; i < 8; ++i) cohort.push_back(candidate(80 + i, i));
        CHECK(gate.plan(8, cohort, 8).empty());
        gate.observe_ar(8, 100.0);
        CHECK(equal_slots(gate.plan(8, cohort, 8), {0, 1}));
        gate.observe_spec(8, 2, 1000.0, accepted({{80, 1}, {81, 1}}));
        CHECK(gate.plan(8, cohort, 8).empty());
        CHECK(gate.plan(8, cohort, 8).empty());
    }

    std::printf("speculation gate: %d checks passed\n", g_checks);
    return 0;
}
