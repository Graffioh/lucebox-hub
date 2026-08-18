#include "common/concurrency/speculation_gate.h"
#include "host_check.h"

#include <cstdio>
#include <limits>
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

std::vector<SpecObservation> observed(
        std::initializer_list<SpecObservation> values) {
    return values;
}

std::vector<SpecCandidate> batch(
        std::initializer_list<SpecCandidate> values) {
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
    // Cold start seeds one AR baseline, then admits only the best bounded
    // optimistic cohort. Equal scores are ordered by stable request id.
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

    // Adapter priors rank cold requests; measured acceptance remains the
    // production signal after probation.
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

    // Never and ineligible requests stay AR. Always is returned before a
    // baseline and remains exempt from profitability. Capacity overflow is
    // returned intact so the engine can report the configuration error.
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
        gate.observe_spec(2, 1, 1000.0, observed({{3, 1, 1}}));
        CHECK(equal_slots(gate.plan(2, batch({
            candidate(3, 1, SpeculationPolicy::Always)}), 1), {1}));

        std::vector<SpecCandidate> forced = {
            candidate(5, 0, SpeculationPolicy::Always),
            candidate(6, 1, SpeculationPolicy::Always),
        };
        CHECK(equal_slots(gate.plan(2, forced, 1), {0, 1}));
    }

    // C=1: one bad first sample cannot end probation. After two low samples
    // the measured score loses the cut; 64 AR tokens make it optimistic for
    // one re-probe. EMA smoothing alone keeps one later rejection from
    // immediately removing a productive request.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.ema_alpha = 0.4;
        cfg.probe_rounds = 2;
        cfg.reprobe_tokens = 64;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(1, 100.0);

        std::vector<SpecCandidate> one = {candidate(11, 0)};
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 200.0, observed({{11, 1, 1}}));
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 200.0, observed({{11, 1, 2}}));
        CHECK(gate.plan(1, one, 1).empty());

        one[0].generated_tokens = 65;
        CHECK(gate.plan(1, one, 1).empty());
        one[0].generated_tokens = 66;
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 120.0, observed({{11, 8, 67}}));
        one[0].generated_tokens = 67;
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 120.0, observed({{11, 1, 68}}));
        one[0].generated_tokens = 68;
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
    }

    // The marginal cut rejects a zero-surplus freeloader even though two
    // high-yield requests keep the aggregate route far above the AR baseline.
    {
        SpeculationGate gate(permissive_config(), 8);
        gate.observe_ar(6, 100.0);
        gate.observe_spec(6, 2, 106.0,
            observed({{21, 4, 1}, {22, 4, 1}}));
        gate.observe_spec(6, 2, 106.0,
            observed({{21, 4, 2}, {22, 4, 2}}));
        gate.observe_spec(6, 1, 103.0, observed({{23, 1, 1}}));
        gate.observe_spec(6, 1, 103.0, observed({{23, 1, 2}}));
        std::vector<SpecCandidate> cohort = {
            candidate(21, 0, SpeculationPolicy::Adaptive, true,
                      std::numeric_limits<double>::quiet_NaN(), 2),
            candidate(22, 1, SpeculationPolicy::Adaptive, true,
                      std::numeric_limits<double>::quiet_NaN(), 2),
            candidate(23, 2, SpeculationPolicy::Adaptive, true,
                      std::numeric_limits<double>::quiet_NaN(), 2),
        };
        CHECK(equal_slots(gate.plan(6, cohort, 6), {0, 1}));
    }

    // Marginal improvement alone is insufficient: the route must also clear
    // the global 5% margin and the AR-peer latency bound.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.margin = 1.05;
        cfg.probe_rounds = 1;
        SpeculationGate margin_gate(cfg, 8);
        margin_gate.observe_ar(1, 100.0);
        margin_gate.observe_spec(1, 1, 195.0, observed({{31, 2, 1}}));
        CHECK(margin_gate.plan(1, batch({candidate(
            31, 0, SpeculationPolicy::Adaptive, true,
            std::numeric_limits<double>::quiet_NaN(), 1)}), 1).empty());

        cfg.margin = 1.0;
        cfg.slack = 1.10;
        SpeculationGate slack_gate(cfg, 8);
        slack_gate.observe_ar(2, 100.0);
        slack_gate.observe_spec(2, 1, 111.0, observed({{32, 8, 1}}));
        CHECK(slack_gate.plan(2, batch({candidate(
            32, 0, SpeculationPolicy::Adaptive, true,
            std::numeric_limits<double>::quiet_NaN(), 1)}), 1).empty());
    }

    // Once an observed shape is hopeless even with max_accept, later cold
    // requests cost arithmetic only and run no speculative probe.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        cfg.max_probers = 1;
        cfg.margin = 1.05;
        SpeculationGate gate(cfg, 4);
        gate.observe_ar(8, 100.0);
        CHECK(equal_slots(gate.plan(8, batch({candidate(41, 0)}), 8), {0}));
        gate.observe_spec(8, 1, 1000.0, observed({{41, 1, 1}}));
        CHECK(gate.plan(8, batch({candidate(42, 1)}), 8).empty());
        CHECK(gate.plan(8, batch({candidate(42, 1)}), 8).empty());
    }

    // Missing shapes use the nearest concurrency's affine increment until a
    // real measurement corrects them.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(2, 100.0);
        gate.observe_spec(2, 1, 120.0, observed({{51, 8, 1}}));

        gate.observe_ar(3, 150.0);
        CHECK(equal_slots(gate.plan(3, batch({candidate(52, 1)}), 1), {1}));
        gate.observe_spec(3, 1, 1000.0, observed({{52, 1, 1}}));
        CHECK(gate.plan(3, batch({candidate(53, 2)}), 1).empty());
    }

    // Noisy exact width samples are made nondecreasing before the cut. A
    // lower measured T(2) therefore cannot make a zero-yield second lane look
    // profitable after T(1).
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(2, 100.0);
        gate.observe_spec(2, 1, 130.0, observed({{61, 4, 1}}));
        gate.observe_spec(2, 2, 120.0,
            observed({{61, 4, 2}, {62, 1, 1}}));
        std::vector<SpecCandidate> candidates = {
            candidate(61, 0, SpeculationPolicy::Adaptive, true,
                      std::numeric_limits<double>::quiet_NaN(), 2),
            candidate(62, 1, SpeculationPolicy::Adaptive, true,
                      std::numeric_limits<double>::quiet_NaN(), 1),
        };
        CHECK(equal_slots(gate.plan(2, candidates, 2), {0}));
    }

    // Invalid observations never poison costs or consume probation.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.cost_ewma_alpha = 0.5;
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(1, 100.0);
        gate.observe_ar(1, 200.0);  // T_ar = 150
        gate.observe_ar(1, 0.0);
        gate.observe_ar(1, std::numeric_limits<double>::quiet_NaN());
        std::vector<SpecCandidate> one = {candidate(71, 0)};
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1,
            std::numeric_limits<double>::quiet_NaN(),
            observed({{71, 1, 1}}));
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 300.0, observed({{71, 2, 1}}));
        one[0].generated_tokens = 1;
        CHECK(gate.plan(1, one, 1).empty());
    }

    // Eligibility can disappear without discarding measurements. forget()
    // removes state, so a new request reusing the slot starts cold.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 1;
        SpeculationGate gate(cfg, 8);
        gate.observe_ar(1, 100.0);
        std::vector<SpecCandidate> one = {candidate(81, 0)};
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));
        gate.observe_spec(1, 1, 100.0, observed({{81, 8, 1}}));
        one[0].generated_tokens = 1;
        one[0].eligible = false;
        CHECK(gate.plan(1, one, 1).empty());
        one[0].eligible = true;
        CHECK(equal_slots(gate.plan(1, one, 1), {0}));

        gate.forget(81);
        CHECK(equal_slots(gate.plan(1, batch({candidate(82, 0)}), 1), {0}));
        gate.forget(82);
        CHECK(equal_slots(gate.plan(1, batch({candidate(83, 0)}), 1), {0}));
    }

    // A profitable C=2 cohort converges through the same probation rule used
    // at every occupancy.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 2;
        SpeculationGate gate(cfg, 8);
        std::vector<SpecCandidate> cohort = {
            candidate(91, 0), candidate(92, 1),
        };
        CHECK(gate.plan(2, cohort, 2).empty());
        gate.observe_ar(2, 100.0);
        for (int round = 0; round < cfg.probe_rounds; ++round) {
            CHECK(equal_slots(gate.plan(2, cohort, 2), {0, 1}));
            gate.observe_spec(2, 2, 100.0,
                observed({{91, 4, round + 1}, {92, 4, round + 1}}));
            cohort[0].generated_tokens = round + 1;
            cohort[1].generated_tokens = round + 1;
        }
        CHECK(equal_slots(gate.plan(2, cohort, 2), {0, 1}));
    }

    // At C=8 one expensive probation round makes even optimistic requests
    // fail the cut, so steady state is pure AR.
    {
        SpecGateConfig cfg = permissive_config();
        cfg.probe_rounds = 2;
        cfg.max_probers = 2;
        cfg.margin = 1.05;
        SpeculationGate gate(cfg, 8);
        std::vector<SpecCandidate> cohort;
        for (int i = 0; i < 8; ++i) cohort.push_back(candidate(100 + i, i));
        CHECK(gate.plan(8, cohort, 8).empty());
        gate.observe_ar(8, 100.0);
        CHECK(equal_slots(gate.plan(8, cohort, 8), {0, 1}));
        gate.observe_spec(8, 2, 1000.0,
            observed({{100, 1, 1}, {101, 1, 1}}));
        CHECK(gate.plan(8, cohort, 8).empty());
        CHECK(gate.plan(8, cohort, 8).empty());
    }

    std::printf("speculation gate: %d checks passed\n", g_checks);
    return 0;
}
