#include "common/speculation/speculation_gate.h"
#include "common/speculation/speculator.h"
#include "common/speculation/survival_score.h"
#include "host_check.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

static SpecCostSeries series(int max_index, double cost) {
    SpecCostSeries out;
    for (int i = 1; i <= max_index; ++i) {
        out.indices.push_back(i);
        out.costs.push_back(cost);
    }
    return out;
}

static SpecCostTables constant_costs(double tree, double step, double draft) {
    return {series(128, tree), series(128, step), series(16, draft)};
}

static SpecStepGeometry geometry() {
    SpecStepGeometry out;
    out.tree_width = 4;
    out.bucket = [](int lanes) { return lanes; };
    return out;
}

static SpecCandidate candidate(
        uint64_t id, int slot, double activation_score,
        SpeculationPolicy policy = SpeculationPolicy::Adaptive,
        bool scoreable = true, bool can_speculate = true,
        std::vector<double> hazards = {},
        std::string score_kind = "test-score-v1") {
    return {id, slot, policy, scoreable, can_speculate, activation_score,
            std::move(hazards), std::move(score_kind)};
}

int main() {
    CHECK(!speculator_is_ready(nullptr));
    CHECK(std::string(speculator_fallback_reason(nullptr)) ==
          "no_speculator_adapter");
    const ConfidenceVectorScorer confidence_scorer;
    CHECK(std::abs(confidence_scorer.score({0.5f, 0.5f}, 4)
                       .expected_yield - 1.75) < 1e-9);
    CHECK(confidence_scorer.score({2.0f, -1.0f}, 4).expected_yield == 2.0);
    CHECK(confidence_scorer.score({}, 4).expected_yield == 1.0);

    SpeculationGate costly(constant_costs(100.0, 10.0, 100.0),
                           geometry(), 4);
    CHECK(costly.valid());
    SpecPlan plan = costly.plan(2, {
        candidate(1, 0, 4.0), candidate(2, 1, 4.0)}, 2);
    CHECK(plan.valid);
    CHECK(plan.admitted_count == 0);
    CHECK(std::abs(plan.goodput - plan.ar_goodput) < 1e-12);
    CHECK(plan.draft_lanes == 0);
    CHECK(plan.profiled_cost == 10.0);
    CHECK(plan.cost_scale == 1.0);
    CHECK(plan.predicted_cost == plan.profiled_cost);
    CHECK(costly.initial_score(1) == 4.0);

    plan = costly.plan(2, {
        candidate(1, 0, NAN), candidate(2, 1, NAN)}, 2);
    CHECK(plan.pending_evaluations.empty());
    CHECK(plan.ordered.size() == 2);
    CHECK(plan.admitted_count == 0);
    CHECK(std::abs(plan.goodput - plan.ar_goodput) < 1e-12);

    SpeculationGate prefix(constant_costs(1.0, 10.0, 1.0),
                           geometry(), 4);
    plan = prefix.plan(3, {
        candidate(10, 0, 4.0), candidate(11, 1, 1.0),
        candidate(12, 2, 4.0, SpeculationPolicy::Never)}, 3);
    CHECK(plan.admitted_count == 1);
    CHECK((plan.admitted_request_ids == std::vector<uint64_t>{10}));
    CHECK(plan.ordered.size() == 2);
    CHECK(plan.ordered[0].admitted);
    CHECK(plan.ordered[0].source == SpecScoreSource::Fresh);
    CHECK(std::string(spec_score_source_name(plan.ordered[0].source)) ==
          "fresh");
    plan = prefix.plan(3, {
        candidate(10, 0, 1.0), candidate(11, 1, 4.0),
        candidate(12, 2, 4.0, SpeculationPolicy::Never)}, 3);
    CHECK((plan.admitted_request_ids == std::vector<uint64_t>{10}));
    CHECK(plan.ordered.size() == 2);
    CHECK(!plan.ordered[0].forced);
    CHECK(plan.ordered[0].source == SpecScoreSource::Initial);
    CHECK(prefix.initial_score(10) == 4.0);
    CHECK(prefix.initial_score(11) == 1.0);

    SpecCostTables crossover = constant_costs(1.0, 4.0, 1.0);
    SpeculationGate bootstrap(crossover, geometry(), 4);
    plan = bootstrap.plan(2, {
        candidate(20, 0, NAN), candidate(21, 1, 4.0)}, 2);
    CHECK(plan.valid);
    CHECK(plan.admitted_count == 0);
    CHECK(plan.ordered.empty());
    CHECK(plan.unavailable_count == 1);
    CHECK(plan.pending_evaluations.size() == 1);
    CHECK(plan.pending_evaluations[0].request_id == 20);
    CHECK(plan.pending_evaluations[0].slot == 0);
    CHECK(plan.pending_evaluations[0].action ==
          SpecEvaluationAction::Score);
    CHECK(bootstrap.initial_score(21) == 4.0);

    plan = bootstrap.plan(2, {
        candidate(20, 0, 1.0), candidate(21, 1, NAN)}, 2);
    CHECK(plan.valid);
    CHECK(plan.pending_evaluations.empty());
    CHECK(plan.admitted_count == 1);
    CHECK((plan.admitted_request_ids == std::vector<uint64_t>{21}));
    CHECK(bootstrap.initial_score(20) == 1.0);
    CHECK(bootstrap.initial_score(21) == 4.0);

    plan = bootstrap.plan(2, {
        candidate(20, 0, 4.0), candidate(21, 1, 1.0)}, 2);
    CHECK(plan.pending_evaluations.empty());
    CHECK(plan.admitted_count == 1);
    CHECK((plan.admitted_request_ids == std::vector<uint64_t>{21}));
    CHECK(plan.ordered.size() == 2);
    CHECK(!plan.ordered[0].forced);
    CHECK(plan.ordered[0].source == SpecScoreSource::Initial);
    CHECK(bootstrap.initial_score(20) == 1.0);
    CHECK(bootstrap.initial_score(21) == 4.0);

    SpeculationGate support(crossover, geometry(), 4);
    plan = support.plan(1, {
        candidate(30, 0, NAN, SpeculationPolicy::Adaptive, true, false)}, 1);
    CHECK(plan.valid);
    CHECK(plan.pending_evaluations.size() == 1);
    CHECK(plan.pending_evaluations[0].request_id == 30);
    CHECK(plan.pending_evaluations[0].slot == 0);
    CHECK(plan.pending_evaluations[0].action ==
          SpecEvaluationAction::Score);
    plan = support.plan(1, {
        candidate(30, 0, 4.0, SpeculationPolicy::Adaptive, true, false)}, 1);
    CHECK(plan.valid);
    CHECK(plan.admitted_count == 0);
    CHECK(plan.ordered.size() == 1);
    CHECK(plan.ordered[0].execution_unsupported);
    CHECK(plan.ordered[0].source == SpecScoreSource::Fresh);
    CHECK(support.initial_score(30) == 4.0);
    plan = support.plan(1, {
        candidate(31, 0, NAN, SpeculationPolicy::Adaptive, false, false)}, 1);
    CHECK(plan.valid);
    CHECK(plan.pending_evaluations.size() == 1);
    CHECK(plan.pending_evaluations[0].request_id == 31);
    CHECK(plan.pending_evaluations[0].slot == 0);
    CHECK(plan.pending_evaluations[0].action ==
          SpecEvaluationAction::FallbackAR);
    CHECK(support.record_evaluation_failure(31));
    CHECK(!support.record_evaluation_failure(31));
    CHECK(support.evaluation_failed(31));
    CHECK(!support.has_score(31));
    CHECK(std::isnan(support.initial_score(31)));
    plan = support.plan(1, {candidate(31, 0, 4.0)}, 1);
    CHECK(plan.valid);
    CHECK(plan.pending_evaluations.empty());
    CHECK(plan.ordered.size() == 1);
    CHECK(plan.ordered[0].execution_unsupported);
    CHECK(plan.ordered[0].source == SpecScoreSource::Unavailable);

    SpeculationGate mixed_activation(crossover, geometry(), 4);
    plan = mixed_activation.plan(2, {
        candidate(32, 0, NAN, SpeculationPolicy::Adaptive, false, false),
        candidate(33, 1, 4.0)}, 2);
    CHECK(plan.valid);
    CHECK(plan.pending_evaluations.size() == 1);
    CHECK(plan.pending_evaluations[0].request_id == 32);
    CHECK(plan.pending_evaluations[0].action ==
          SpecEvaluationAction::FallbackAR);
    CHECK(mixed_activation.record_evaluation_failure(32));
    plan = mixed_activation.plan(2, {
        candidate(32, 0, NAN, SpeculationPolicy::Adaptive, false, false),
        candidate(33, 1, NAN)}, 2);
    CHECK(plan.valid);
    CHECK(plan.pending_evaluations.empty());
    CHECK(mixed_activation.evaluation_failed(32));
    CHECK(mixed_activation.initial_score(33) == 4.0);
    CHECK((plan.admitted_request_ids == std::vector<uint64_t>{33}));

    SpeculationGate policies(constant_costs(100.0, 10.0, 100.0),
                             geometry(), 4);
    plan = policies.plan(3, {
        candidate(40, 0, 1.0, SpeculationPolicy::Never),
        candidate(41, 1, NAN, SpeculationPolicy::Always),
        candidate(42, 2, 4.0)}, 2);
    CHECK(plan.valid);
    CHECK(plan.admitted_count >= 1);
    CHECK(plan.admitted_request_ids.front() == 41);
    CHECK(plan.ordered.front().forced);
    CHECK(plan.ordered.front().source == SpecScoreSource::Unavailable);
    plan = policies.plan(2, {
        candidate(43, 0, 4.0, SpeculationPolicy::Always),
        candidate(44, 1, 4.0, SpeculationPolicy::Always)}, 1);
    CHECK(!plan.valid);
    CHECK(!plan.error.empty());
    plan = policies.plan(1, {
        candidate(46, 0, NAN, SpeculationPolicy::Always,
                  /*scoreable=*/false, /*can_speculate=*/false)}, 1);
    CHECK(plan.valid);
    CHECK(plan.admitted_count == 1);
    CHECK(plan.ordered.front().forced);
    CHECK(plan.admitted_request_ids.front() == 46);

    SpeculationGate never(constant_costs(1.0, 2.0, 1.0), geometry(), 4);
    plan = never.plan(1, {
        candidate(45, 0, NAN, SpeculationPolicy::Never)}, 1);
    CHECK(plan.admitted_count == 0);
    CHECK(plan.ordered.empty());

    SpeculationGate capacity(constant_costs(1.0, 10.0, 1.0), geometry(), 4);
    plan = capacity.plan(1, {candidate(50, 0, 4.0)}, 0);
    CHECK(plan.admitted_count == 0);
    plan = capacity.plan(2, {candidate(51, 0, 4.0)}, 1);
    CHECK(!plan.valid);
    SpeculationGate always_draft(
        constant_costs(1.0, 10.0, 1.0), geometry(), 4);
    plan = always_draft.plan(1, {candidate(52, 0, 4.0)}, 1, 4);
    CHECK(plan.draft_lanes == 4);

    SpecCostSeries sparse{{2, 4}, {1.0, 2.0}};
    CHECK(sparse.valid());
    SpecCostLookup lookup = sparse.lookup(1);
    CHECK(lookup.clamped && lookup.profiled_index == 2);
    lookup = sparse.lookup(3);
    CHECK(!lookup.clamped && lookup.rounded_up && lookup.profiled_index == 4);
    lookup = sparse.lookup(9);
    CHECK(lookup.clamped && lookup.profiled_index == 4);
    SpecCostSeries invalid{{2, 1}, {1.0, 2.0}};
    CHECK(!invalid.valid());

    int clamp_logs = 0;
    SpecCostTables tiny{series(1, 1.0), series(1, 2.0), series(1, 1.0)};
    SpeculationGate clamped(tiny, geometry(), 4,
        [&](const char *, int, int) { ++clamp_logs; });
    plan = clamped.plan(2, {
        candidate(60, 0, 4.0), candidate(61, 1, 4.0)}, 2);
    CHECK(plan.cost_lookup_clamped);
    CHECK(clamp_logs > 0);

    SpeculationGate fitted(
        constant_costs(1.0, 10.0, 1.0), geometry(), 4);
    plan = fitted.plan(1, {
        candidate(
            700, 0, 2.0, SpeculationPolicy::Adaptive, true, true,
            {0.5, 0.25}, "adapter-score-v1")}, 1);
    CHECK(fitted.initial_score(700) == 2.0);
    CHECK(fitted.initial_score_kind(700) == "adapter-score-v1");
    CHECK((fitted.initial_hazards(700) ==
           std::vector<double>{0.5, 0.25}));
    CHECK(plan.ordered[0].expected_yield == 2.0);
    CHECK(plan.initial_predicted_tokens == 2.0);

    plan = fitted.plan(1, {candidate(700, 0, 1.0)}, 1);
    CHECK(!plan.ordered[0].forced);
    CHECK(plan.ordered[0].source == SpecScoreSource::Initial);
    CHECK(plan.ordered[0].expected_yield == 2.0);
    CHECK(fitted.initial_score(700) == 2.0);

    plan = fitted.plan(1, {candidate(701, 0, 4.0)}, 1);
    CHECK(plan.ordered[0].expected_yield == 4.0);
    CHECK(fitted.initial_score(701) == 4.0);
    CHECK(fitted.initial_score(700) == 2.0);

    fitted.forget(700);
    CHECK(!fitted.has_state(700));
    CHECK(!fitted.has_score(700));
    CHECK(std::isnan(fitted.initial_score(700)));
    plan = fitted.plan(1, {candidate(700, 0, NAN)}, 1);
    CHECK(plan.pending_evaluations.size() == 1);
    CHECK(plan.pending_evaluations[0].slot == 0);
    CHECK(plan.pending_evaluations[0].action ==
          SpecEvaluationAction::Score);

    SpecCostSeries cliff_step = series(32, 100.0);
    for (size_t i = 16; i < cliff_step.costs.size(); ++i) {
        cliff_step.costs[i] = 200.0;
    }
    const SpecCostTables cliff_costs{
        series(128, 1.0), cliff_step, series(16, 1.0)};
    SpeculationGate cliff_gate(cliff_costs, geometry(), 8);
    const SpecPlan below_cliff = cliff_gate.plan(4, {
        candidate(800, 0, 4.12475, SpeculationPolicy::Always),
        candidate(801, 1, 4.12475, SpeculationPolicy::Always),
        candidate(802, 2, 4.12475, SpeculationPolicy::Always),
        candidate(803, 3, 4.12475, SpeculationPolicy::Always)}, 4);
    const SpecPlan above_cliff = cliff_gate.plan(4, {
        candidate(804, 0, 4.12525, SpeculationPolicy::Always),
        candidate(805, 1, 4.12525, SpeculationPolicy::Always),
        candidate(806, 2, 4.12525, SpeculationPolicy::Always),
        candidate(807, 3, 4.12525, SpeculationPolicy::Always)}, 4);
    CHECK(std::abs(below_cliff.expected_step_rows - 16.499) < 1e-9);
    CHECK(std::abs(above_cliff.expected_step_rows - 16.501) < 1e-9);
    CHECK(std::abs(below_cliff.profiled_cost - 151.9) < 1e-9);
    CHECK(std::abs(above_cliff.profiled_cost - 152.1) < 1e-9);
    CHECK(above_cliff.profiled_cost > below_cliff.profiled_cost);
    CHECK(above_cliff.profiled_cost - below_cliff.profiled_cost < 1.0);

    SpecCostSeries replay_step = series(128, 200.0);
    for (size_t i = 0; i < 4; ++i) replay_step.costs[i] = 10.0;
    const SpecCostTables replay_priced{
        series(128, 20.0), replay_step, series(16, 1.0)};
    SpeculationGate legacy_replay_gate(
        replay_priced, geometry(), 4);
    SpeculationGate direct_commit_gate(
        replay_priced, geometry(), 4, {}, {}, true);
    std::vector<SpecCandidate> direct_candidates;
    for (int i = 0; i < 4; ++i)
        direct_candidates.push_back(candidate(850 + i, i, 4.0));
    const SpecPlan legacy_replay_plan = legacy_replay_gate.plan(
        4, direct_candidates, 4);
    const SpecPlan direct_commit_plan = direct_commit_gate.plan(
        4, direct_candidates, 4);
    CHECK(legacy_replay_plan.admitted_count == 0);
    CHECK(direct_commit_plan.admitted_count == 4);
    CHECK(direct_commit_plan.tree_rows == geometry().tree_rows(4));
    CHECK(direct_commit_plan.expected_step_rows == 0.0);
    CHECK(direct_commit_plan.profiled_cost == 21.0);

    SpeculationGate direct_mixed_gate(
        replay_priced, geometry(), 4, {}, {}, true);
    const SpecPlan direct_mixed_plan = direct_mixed_gate.plan(4, {
        candidate(870, 0, 4.0, SpeculationPolicy::Always),
        candidate(871, 1, 4.0, SpeculationPolicy::Always),
        candidate(872, 2, 0.0, SpeculationPolicy::Never),
        candidate(873, 3, 0.0, SpeculationPolicy::Never)}, 4);
    CHECK(direct_mixed_plan.admitted_count == 2);
    CHECK(direct_mixed_plan.tree_rows == geometry().tree_rows(2) + 2);
    CHECK(direct_mixed_plan.expected_step_rows == 0.0);
    CHECK(direct_mixed_plan.profiled_cost == 21.0);

    direct_commit_gate.observe_cost({4, 4, 16, 0, 4}, 84.0);
    std::vector<SpecCandidate> forced_direct_candidates;
    for (int i = 0; i < 4; ++i) {
        forced_direct_candidates.push_back(candidate(860 + i, i, 4.0,
            SpeculationPolicy::Always));
    }
    const SpecPlan observed_direct_plan = direct_commit_gate.plan(
        4, forced_direct_candidates, 4);
    CHECK(observed_direct_plan.admitted_count == 4);
    CHECK(observed_direct_plan.cost_scale == 4.0);
    CHECK(observed_direct_plan.predicted_cost == 84.0);

    SpeculationGate shape_feedback(
        constant_costs(1.0, 10.0, 1.0), geometry(), 4);
    SpecPlan expected_row_one = shape_feedback.plan(1, {
        candidate(900, 0, 1.0, SpeculationPolicy::Always)}, 1);
    CHECK(expected_row_one.expected_step_rows == 1.0);
    CHECK(expected_row_one.cost_scale == 1.0);
    shape_feedback.observe_cost({1, 1, 4, 4, 1}, 48.0);
    expected_row_one = shape_feedback.plan(1, {
        candidate(901, 0, 1.0, SpeculationPolicy::Always)}, 1);
    const SpecPlan observed_row_four = shape_feedback.plan(1, {
        candidate(902, 0, 4.0, SpeculationPolicy::Always)}, 1);
    CHECK(expected_row_one.cost_scale == 1.0);
    CHECK(expected_row_one.predicted_cost == 12.0);
    CHECK(observed_row_four.cost_scale == 4.0);
    CHECK(observed_row_four.predicted_cost == 48.0);

    SpeculationGate draft_shape_feedback(
        constant_costs(100.0, 10.0, 100.0), geometry(), 4);
    draft_shape_feedback.observe_cost({1, 0, 0, 1, 1}, 440.0);
    const SpecPlan pure_ar_after_bootstrap = draft_shape_feedback.plan(
        1, {candidate(903, 0, 4.0)}, 1);
    CHECK(pure_ar_after_bootstrap.admitted_count == 0);
    CHECK(pure_ar_after_bootstrap.cost_scale == 1.0);
    CHECK(pure_ar_after_bootstrap.predicted_cost == 10.0);

    SpeculationGate cost_feedback(
        constant_costs(1.0, 10.0, 1.0), geometry(), 4);
    plan = cost_feedback.plan(1, {candidate(950, 0, 4.0)}, 1);
    CHECK(plan.admitted_count == 1);
    CHECK(plan.profiled_cost == 12.0);
    CHECK(plan.cost_scale == 1.0);
    cost_feedback.observe_cost({1, 1, 4, 4, 1}, 48.0);
    plan = cost_feedback.plan(1, {candidate(950, 0, NAN)}, 1);
    CHECK(plan.admitted_count == 0);
    CHECK(std::abs(plan.goodput - plan.ar_goodput) < 1e-12);
    CHECK(plan.draft_lanes == 0);

    cost_feedback.forget(950);
    plan = cost_feedback.plan(1, {candidate(951, 0, 4.0)}, 1);
    CHECK(plan.admitted_count == 0);
    CHECK(std::abs(plan.goodput - plan.ar_goodput) < 1e-12);
    plan = cost_feedback.plan(2, {
        candidate(952, 0, 4.0), candidate(953, 1, 4.0)}, 2);
    CHECK(plan.admitted_count == 2);
    CHECK(plan.cost_scale == 1.0);

    SpeculationGate ar_feedback(
        constant_costs(100.0, 10.0, 100.0), geometry(), 4);
    SpecPlan ar_plan = ar_feedback.plan(1, {candidate(960, 0, 4.0)}, 1);
    CHECK(ar_plan.admitted_count == 0);
    ar_feedback.observe_cost({1, 0, 0, 1, 0}, 20.0);
    ar_plan = ar_feedback.plan(1, {candidate(960, 0, NAN)}, 1);
    CHECK(ar_plan.admitted_count == 0);
    CHECK(ar_plan.profiled_cost == 10.0);
    CHECK(ar_plan.cost_scale == 2.0);
    CHECK(ar_plan.predicted_cost == 20.0);
    CHECK(std::abs(ar_plan.goodput - ar_plan.ar_goodput) < 1e-12);

    const SpecCostTables near_break_even =
        constant_costs(1.0, 100.0, 1.0);
    SpeculationGate margin_gate(near_break_even, geometry(), 4);
    plan = margin_gate.plan(1, {candidate(970, 0, 1.03)}, 1);
    CHECK(plan.admitted_count == 0);
    plan = margin_gate.plan(1, {candidate(970, 0, 4.0)}, 1);
    CHECK(plan.admitted_count == 0);
    CHECK(plan.ordered.size() == 1);
    CHECK(!plan.ordered[0].admitted);
    CHECK(margin_gate.initial_score(970) == 1.03);

    SpecGateConfig zero_margin;
    zero_margin.adaptive_gain_margin = 0.0;
    SpeculationGate no_margin(
        zero_margin, near_break_even, geometry(), 4);
    plan = no_margin.plan(1, {candidate(971, 0, 1.03)}, 1);
    CHECK(plan.admitted_count == 1);
    plan = margin_gate.plan(1, {
        candidate(972, 0, 1.0, SpeculationPolicy::Always)}, 1);
    CHECK(plan.admitted_count == 1);
    CHECK(plan.ordered[0].forced);

    SpeculationGate pays(constant_costs(1.0, 10.0, 1.0), geometry(), 4);
    plan = pays.plan(2, {
        candidate(980, 0, 4.0), candidate(981, 1, 4.0)}, 2);
    CHECK(plan.admitted_count == 2);
    SpeculationGate cannot(constant_costs(100.0, 10.0, 100.0),
                           geometry(), 4);
    std::vector<SpecCandidate> eight;
    for (int i = 0; i < 8; ++i)
        eight.push_back(candidate(990 + i, i, 4.0));
    plan = cannot.plan(8, eight, 8);
    CHECK(plan.admitted_count == 0);

    const std::vector<SpecCandidate> cohort = {
        candidate(1000, 3, 4.0), candidate(1001, 7, 2.0)};
    SpecCohortEpoch epoch;
    epoch.id = 7;
    epoch.request_ids = SpecCohortEpoch::ids(cohort);
    epoch.plan = plan;
    CHECK(epoch.matches(cohort));
    CHECK(epoch.matches({
        candidate(1000, 9, NAN), candidate(1001, 2, NAN)}));
    CHECK(!epoch.matches({candidate(1000, 3, NAN)}));
    CHECK(epoch.matches({
        candidate(1001, 7, NAN), candidate(1000, 3, NAN)}));
    CHECK(!epoch.matches({
        candidate(1000, 3, NAN), candidate(1002, 7, NAN)}));
    CHECK((epoch.request_ids == std::vector<uint64_t>{1000, 1001}));

    std::printf("speculation gate tests passed: %d checks\n", g_checks);
    return 0;
}
