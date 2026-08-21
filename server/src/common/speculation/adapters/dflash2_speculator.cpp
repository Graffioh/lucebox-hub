#include "common/speculation/adapters/dflash2_speculator.h"

#include "common/dflash2_head.h"

#include <iomanip>
#include <sstream>
#include <utility>

namespace dflash::common {
namespace {

std::string depth_debug_fields(const DFlash2DepthSignal & signal) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "\"selected_logp\":" << signal.selected_log_prob
        << ",\"lm_margin\":" << signal.lm_top2_margin
        << ",\"topk_mass\":" << signal.top_k_mass
        << ",\"rank\":" << signal.selected_rank
        << ",\"lm_top1\":"
        << (signal.agrees_with_lm_top1 ? "true" : "false")
        << ",\"selector_margin\":" << signal.selector_margin
        << ",\"selector_mass\":" << signal.selector_winner_mass
        << ",\"selector_entropy\":" << signal.selector_entropy;
    return out.str();
}

}  // namespace

DFlash2Speculator::DFlash2Speculator(
        const DraftWeights & weights,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        DFlash2BenefitModelSignature signature,
        DFlash2BenefitConfig config)
    : weights_(weights), backend_(backend), lm_head_(lm_head),
      benefit_(std::move(signature), std::move(config)),
      score_kind_(benefit_.score_kind()) {
    const DraftSelectorWeights & selector = weights_.selector;
    if (!backend_ || !lm_head_ || !selector.enabled || !selector.hproj ||
        !selector.pred_cb || !selector.succ_cb || selector.rank <= 0 ||
        selector.top_k <= 0 || weights_.block_size <= 1) {
        error_ = "DFlash2 selector inputs are unavailable";
    } else if (!benefit_.ready()) {
        error_ = benefit_.error();
    }
}

int DFlash2Speculator::max_block_size() const {
    return weights_.block_size;
}

bool DFlash2Speculator::propose(
        const SpeculatorBatchInput & input,
        std::vector<SpecProposal> & output) {
    output.clear();
    if (!ready() ||
        !speculator_input_satisfies(input, input_requirements()) ||
        input.requested_depth > max_block_size()) {
        return false;
    }

    std::vector<std::vector<int32_t>> draft_tokens;
    std::vector<DFlash2SelectorTrace> traces;
    if (!dflash2_select_chains_batched(
            weights_, backend_, lm_head_, input.hidden_by_lane,
            input.requested_depth, input.seed_tokens,
            draft_tokens, &traces) ||
        static_cast<int>(draft_tokens.size()) != input.lane_count ||
        static_cast<int>(traces.size()) != input.lane_count) {
        return false;
    }

    output.resize(static_cast<size_t>(input.lane_count));
    for (int lane = 0; lane < input.lane_count; ++lane) {
        SpecProposal & proposal = output[static_cast<size_t>(lane)];
        proposal.tokens = std::move(draft_tokens[static_cast<size_t>(lane)]);

        DFlash2BenefitEstimate estimate;
        std::string estimate_error;
        if (!benefit_.estimate(
                traces[static_cast<size_t>(lane)],
                input.requested_depth, estimate, &estimate_error)) {
            proposal.error = estimate_error.empty()
                ? "DFlash2 activation estimate failed"
                : std::move(estimate_error);
            continue;
        }
        proposal.estimate.expected_yield = estimate.expected_yield;
        proposal.estimate.conditional_hazards =
            std::move(estimate.conditional_hazards);
        proposal.debug_depth_fields.reserve(
            traces[static_cast<size_t>(lane)].depths.size());
        for (const DFlash2DepthSignal & signal :
             traces[static_cast<size_t>(lane)].depths) {
            proposal.debug_depth_fields.push_back(
                depth_debug_fields(signal));
        }
    }
    return true;
}

}  // namespace dflash::common
