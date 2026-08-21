#include "common/dflash2_benefit.h"
#include "common/speculation/speculation_gate.h"
#include "host_check.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

using namespace dflash::common;

static int g_checks = 0;

static DFlash2BenefitModelSignature seeded_signature() {
    DFlash2BenefitModelSignature value;
    value.target_layers = 64;
    value.target_hidden = 5120;
    value.target_vocab = 248320;
    value.draft_layers = 5;
    value.draft_hidden = 5120;
    value.draft_block_size = 8;
    value.selector_rank = 256;
    value.selector_top_k = 16;
    value.selector_vocab = 248320;
    value.conv_kernel_size = 2;
    value.conv_group_size = 16;
    value.target_file_size = 15195272800ULL;
    value.draft_file_size = 2045471776ULL;
    return value;
}

static DFlash2SelectorTrace trace_from(
        std::initializer_list<std::pair<float, float>> values) {
    DFlash2SelectorTrace trace;
    for (const auto & value : values) {
        DFlash2DepthSignal signal;
        signal.selected_log_prob = value.first;
        signal.selector_winner_mass = value.second;
        trace.depths.push_back(signal);
    }
    return trace;
}

int main() {
    DFlash2BenefitProvider provider(seeded_signature());
    CHECK(provider.ready());
    CHECK(std::string(provider.score_kind()) ==
          "qwen38-dflash2-selector-benefit-v1");
    CHECK(provider.config().lm_log_weight == 0.10);
    CHECK(provider.config().hazard_scale == 1.0);
    CHECK(provider.config().yield_scale == 1.0);

    const DFlash2SelectorTrace code_like = trace_from({
        {0.0f, 0.99999988f}, {-7.6293945e-06f, 1.0f},
        {-0.44184685f, 0.64686394f}, {-2.0253334f, 1.0f},
        {-1.2170925f, 1.0f}, {-0.84755516f, 0.99942774f},
        {-3.7030106f, 0.86279351f},
    });
    const DFlash2SelectorTrace prose_like = trace_from({
        {-0.046934128f, 0.95176214f}, {-1.527895f, 0.37046611f},
        {-3.398098f, 0.98939508f}, {-3.0824165f, 0.41014573f},
        {-0.30518532f, 0.47964928f}, {-4.0057392f, 0.99999905f},
        {-1.676815f, 0.89187294f},
    });

    DFlash2BenefitEstimate code;
    DFlash2BenefitEstimate prose;
    std::string error;
    CHECK(provider.estimate(code_like, 8, code, &error));
    CHECK(error.empty());
    CHECK(provider.estimate(prose_like, 8, prose, &error));
    CHECK(code.conditional_hazards.size() == 7);
    CHECK(prose.conditional_hazards.size() == 7);
    CHECK(std::abs(code.expected_yield - 5.330600824) < 1e-6);
    CHECK(std::abs(prose.expected_yield - 2.684506084) < 1e-6);
    CHECK(code.expected_yield > prose.expected_yield + 2.5);

    const SpecCostTables observed_costs{
        {{8, 16}, {42912.6, 47572.8}},
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {29322.0, 34968.2, 38639.9, 38639.9,
          40077.6, 40077.6, 40077.6, 40077.6,
          42360.3, 42360.3, 42360.3, 42360.3,
          42360.3, 42360.3, 42360.3, 44412.2}},
        {{1, 2}, {7934.4, 13967.3}},
    };
    SpecStepGeometry observed_geometry;
    observed_geometry.tree_width = 8;
    observed_geometry.bucket = [](int lanes) { return lanes; };
    auto benefit_candidate = [](uint64_t id, int slot, double score,
                                SpeculationPolicy policy =
                                    SpeculationPolicy::Adaptive) {
        return SpecCandidate{
            id, slot, policy, true, true, score, {},
            kDFlash2BenefitAdapterVersion};
    };

    SpeculationGate code_c1(observed_costs, observed_geometry, 8);
    SpecPlan gate_plan = code_c1.plan(
        1, {benefit_candidate(100, 5, code.expected_yield)}, 1);
    CHECK(gate_plan.admitted_count == 1);
    CHECK(code_c1.initial_score_kind(100) ==
          kDFlash2BenefitAdapterVersion);

    SpeculationGate prose_c1(observed_costs, observed_geometry, 8);
    gate_plan = prose_c1.plan(
        1, {benefit_candidate(200, 3, prose.expected_yield)}, 1);
    CHECK(gate_plan.admitted_count == 0);

    SpeculationGate prose_c2(observed_costs, observed_geometry, 8);
    gate_plan = prose_c2.plan(2, {
        benefit_candidate(300, 4, prose.expected_yield),
        benefit_candidate(301, 1, prose.expected_yield)}, 2);
    CHECK(gate_plan.admitted_count == 0);

    SpeculationGate mixed_c2(observed_costs, observed_geometry, 8);
    gate_plan = mixed_c2.plan(2, {
        benefit_candidate(401, 3, prose.expected_yield),
        benefit_candidate(400, 5, code.expected_yield)}, 2);
    CHECK(gate_plan.admitted_count >= 1);
    CHECK(!gate_plan.admitted_request_ids.empty());
    CHECK(gate_plan.admitted_request_ids[0] == 400);
    CHECK(gate_plan.ordered.size() == 2);
    CHECK(gate_plan.ordered[0].request_id == 400);
    CHECK(gate_plan.ordered[0].slot == 5);
    CHECK(gate_plan.ordered[1].request_id == 401);
    CHECK(gate_plan.ordered[1].slot == 3);
    CHECK(gate_plan.ordered[0].admitted);
    CHECK(gate_plan.ordered[1].admitted ==
          (gate_plan.admitted_count == 2));

    SpeculationGate code_with_ar_peer(
        observed_costs, observed_geometry, 8);
    gate_plan = code_with_ar_peer.plan(2, {
        benefit_candidate(500, 2, code.expected_yield),
        benefit_candidate(501, 7, prose.expected_yield,
                          SpeculationPolicy::Never)}, 2);
    CHECK(gate_plan.admitted_count == 1);
    CHECK(gate_plan.admitted_request_ids.size() == 1);
    CHECK(gate_plan.admitted_request_ids[0] == 500);

    DFlash2SelectorTrace with_tail = code_like;
    with_tail.depths.push_back(trace_from({{-100.0f, 0.001f}}).depths[0]);
    DFlash2BenefitEstimate tail;
    CHECK(provider.estimate(with_tail, 8, tail, &error));
    CHECK(tail.conditional_hazards.size() == 7);
    CHECK(std::abs(tail.expected_yield - code.expected_yield) < 1e-12);
    CHECK(!provider.estimate(code_like, 9, tail, &error));
    CHECK(error.find("outside") != std::string::npos);

    DFlash2SelectorTrace missing = code_like;
    missing.depths.pop_back();
    CHECK(!provider.estimate(missing, 8, tail, &error));
    CHECK(error.find("missing") != std::string::npos);
    DFlash2SelectorTrace malformed = code_like;
    malformed.depths[2].selector_winner_mass = 0.0f;
    CHECK(!provider.estimate(malformed, 8, tail, &error));
    malformed = code_like;
    malformed.depths[2].selected_log_prob = 0.1f;
    CHECK(!provider.estimate(malformed, 8, tail, &error));
    malformed = code_like;
    malformed.depths[2].selected_log_prob =
        std::numeric_limits<float>::quiet_NaN();
    CHECK(!provider.estimate(malformed, 8, tail, &error));
    double unpublished = std::numeric_limits<double>::quiet_NaN();
    CHECK(!provider.publish_once(malformed, 8, unpublished, &error));
    CHECK(std::isnan(unpublished));

    double first_score = std::numeric_limits<double>::quiet_NaN();
    CHECK(provider.publish_once(code_like, 8, first_score, &error));
    CHECK(std::abs(first_score - code.expected_yield) < 1e-12);
    CHECK(provider.publish_once(prose_like, 8, first_score, &error));
    CHECK(std::abs(first_score - code.expected_yield) < 1e-12);
    CHECK(provider.publish_once(malformed, 8, first_score, &error));
    CHECK(std::abs(first_score - code.expected_yield) < 1e-12);

    DFlash2BenefitModelSignature unknown = seeded_signature();
    unknown.selector_rank = 128;
    DFlash2BenefitProvider unknown_provider(unknown);
    CHECK(!unknown_provider.ready());
    CHECK(unknown_provider.error().find("unsupported") != std::string::npos);
    CHECK(!unknown_provider.estimate(code_like, 8, tail, &error));

    DFlash2BenefitConfig bad_version;
    bad_version.adapter_version = "future-unfitted-adapter";
    DFlash2BenefitProvider version_provider(
        seeded_signature(), bad_version);
    CHECK(!version_provider.ready());

    DFlash2BenefitConfig conservative;
    conservative.hazard_scale = 0.9;
    DFlash2BenefitProvider conservative_provider(
        seeded_signature(), conservative);
    CHECK(conservative_provider.ready());
    CHECK(conservative_provider.estimate(code_like, 8, tail, &error));
    CHECK(tail.expected_yield < code.expected_yield);

    DFlash2BenefitConfig invalid_coefficients;
    invalid_coefficients.lm_log_weight =
        std::numeric_limits<double>::infinity();
    DFlash2BenefitProvider invalid_provider(
        seeded_signature(), invalid_coefficients);
    CHECK(!invalid_provider.ready());

    std::printf("DFlash2 benefit adapter: %d checks passed\n", g_checks);
    return 0;
}
