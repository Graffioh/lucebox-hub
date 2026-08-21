#pragma once

#include "dflash2_head.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace dflash::common {

// Structural compatibility signature for the Qwen3.8/DFlash2 benefit model.
// Exact artifact identity belongs to the benchmark metadata boundary.
struct DFlash2BenefitModelSignature {
    int target_layers = 0;
    int target_hidden = 0;
    int target_vocab = 0;
    int draft_layers = 0;
    int draft_hidden = 0;
    int draft_block_size = 0;
    int selector_rank = 0;
    int selector_top_k = 0;
    int selector_vocab = 0;
    int conv_kernel_size = 0;
    int conv_group_size = 0;
    uint64_t target_file_size = 0;
    uint64_t draft_file_size = 0;

    std::string str() const;
};

inline constexpr const char * kDFlash2BenefitAdapterVersion =
    "qwen38-dflash2-selector-benefit-v1";

struct DFlash2BenefitConfig {
    std::string adapter_version = kDFlash2BenefitAdapterVersion;

    // Conditional acceptance hazard at each depth:
    //   exp(lm_log_weight * selected_log_prob
    //       + (1-lm_log_weight) * log(selector_winner_mass))
    // The selector is the better signal on the seed traces, while the
    // LM term conservatively lowers a selector winner that has weak model
    // probability. hazard_scale may only lower the estimate.
    double lm_log_weight = 0.10;
    double hazard_scale = 1.0;
    // Offline per-adapter calibration; the generic gate never rescales yield.
    double yield_scale = 1.0;
};

struct DFlash2BenefitEstimate {
    double expected_yield = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> conditional_hazards;
};

// Stateless request-local adapter. It does not learn from prior requests.
// Request lifetime is owned by the caller via publish_once(destination): the
// first valid trace fills an empty destination, and subsequent traces cannot
// overwrite that request's activation score.
class DFlash2BenefitProvider {
public:
    DFlash2BenefitProvider(
        DFlash2BenefitModelSignature model_signature,
        DFlash2BenefitConfig config = {});

    static DFlash2BenefitConfig config_from_environment(
        std::string & error);

    bool ready() const { return error_.empty(); }
    const std::string & error() const { return error_; }
    const DFlash2BenefitModelSignature & model_signature() const {
        return model_signature_;
    }
    const DFlash2BenefitConfig & config() const { return config_; }
    const char * score_kind() const { return kDFlash2BenefitAdapterVersion; }

    bool estimate(const DFlash2SelectorTrace & trace, int max_accept,
                  DFlash2BenefitEstimate & out,
                  std::string * error = nullptr) const;

    bool publish_once(const DFlash2SelectorTrace & trace, int max_accept,
                      double & destination,
                      std::string * error = nullptr) const;

private:
    DFlash2BenefitModelSignature model_signature_;
    DFlash2BenefitConfig config_;
    std::string error_;
};

}  // namespace dflash::common
