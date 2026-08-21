#include "dflash2_benefit.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace dflash::common {
namespace {

constexpr DFlash2BenefitModelSignature kSeededQwen38DFlash2 = {
    /*target_layers=*/64,
    /*target_hidden=*/5120,
    /*target_vocab=*/248320,
    /*draft_layers=*/5,
    /*draft_hidden=*/5120,
    /*draft_block_size=*/8,
    /*selector_rank=*/256,
    /*selector_top_k=*/16,
    /*selector_vocab=*/248320,
    /*conv_kernel_size=*/2,
    /*conv_group_size=*/16,
    /*target_file_size=*/15195272800ULL,
    /*draft_file_size=*/2045471776ULL,
};

bool same_signature(const DFlash2BenefitModelSignature & a,
                      const DFlash2BenefitModelSignature & b) {
    return a.target_layers == b.target_layers &&
           a.target_hidden == b.target_hidden &&
           a.target_vocab == b.target_vocab &&
           a.draft_layers == b.draft_layers &&
           a.draft_hidden == b.draft_hidden &&
           a.draft_block_size == b.draft_block_size &&
           a.selector_rank == b.selector_rank &&
           a.selector_top_k == b.selector_top_k &&
           a.selector_vocab == b.selector_vocab &&
           a.conv_kernel_size == b.conv_kernel_size &&
           a.conv_group_size == b.conv_group_size &&
           a.target_file_size == b.target_file_size &&
           a.draft_file_size == b.draft_file_size;
}

bool parse_finite_env(const char * name, double minimum, double maximum,
                      double & value, std::string & error) {
    const char * text = std::getenv(name);
    if (!text || !*text) return true;
    errno = 0;
    char * end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || !end || *end != '\0' ||
        !std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
        error = std::string(name) + " must be finite in [" +
            std::to_string(minimum) + "," + std::to_string(maximum) + "]";
        return false;
    }
    value = parsed;
    return true;
}

void set_error(std::string * output, const std::string & value) {
    if (output) *output = value;
}

}  // namespace

std::string DFlash2BenefitModelSignature::str() const {
    std::ostringstream out;
    out << "target:l" << target_layers << ":h" << target_hidden
        << ":v" << target_vocab
        << "/draft:l" << draft_layers << ":h" << draft_hidden
        << ":b" << draft_block_size
        << "/selector:r" << selector_rank << ":k" << selector_top_k
        << ":v" << selector_vocab
        << "/conv:k" << conv_kernel_size << ":g" << conv_group_size
        << "/files:t" << target_file_size << ":d" << draft_file_size;
    return out.str();
}

DFlash2BenefitConfig DFlash2BenefitProvider::config_from_environment(
        std::string & error) {
    error.clear();
    DFlash2BenefitConfig config;
    if (const char * version =
            std::getenv("DFLASH_DFLASH2_BENEFIT_ADAPTER")) {
        config.adapter_version = version;
    }
    if (!parse_finite_env(
            "DFLASH_DFLASH2_BENEFIT_LM_WEIGHT", 0.0, 1.0,
            config.lm_log_weight, error)) {
        return config;
    }
    if (!parse_finite_env(
            "DFLASH_DFLASH2_BENEFIT_HAZARD_SCALE", 0.0, 1.0,
            config.hazard_scale, error) || config.hazard_scale <= 0.0) {
        if (error.empty()) {
            error = "DFLASH_DFLASH2_BENEFIT_HAZARD_SCALE must be in (0,1]";
        }
        return config;
    }
    if (!parse_finite_env(
            "DFLASH_DFLASH2_BENEFIT_YIELD_SCALE", 0.25, 4.0,
            config.yield_scale, error)) {
        return config;
    }
    return config;
}

DFlash2BenefitProvider::DFlash2BenefitProvider(
        DFlash2BenefitModelSignature model_signature, DFlash2BenefitConfig config)
    : model_signature_(std::move(model_signature)), config_(std::move(config)) {
    if (config_.adapter_version != kDFlash2BenefitAdapterVersion) {
        error_ = "unsupported DFlash2 benefit adapter version '" +
            config_.adapter_version + "'";
        return;
    }
    if (!std::isfinite(config_.lm_log_weight) ||
        config_.lm_log_weight < 0.0 || config_.lm_log_weight > 1.0 ||
        !std::isfinite(config_.hazard_scale) ||
        config_.hazard_scale <= 0.0 || config_.hazard_scale > 1.0 ||
        !std::isfinite(config_.yield_scale) ||
        config_.yield_scale < 0.25 || config_.yield_scale > 4.0) {
        error_ = "invalid DFlash2 benefit coefficients";
        return;
    }
    if (!same_signature(model_signature_, kSeededQwen38DFlash2)) {
        error_ = "unsupported DFlash2 model signature " +
            model_signature_.str();
    }
}

bool DFlash2BenefitProvider::estimate(
        const DFlash2SelectorTrace & trace, int max_accept,
        DFlash2BenefitEstimate & out, std::string * error) const {
    out = {};
    if (!ready()) {
        set_error(error, error_);
        return false;
    }
    if (max_accept < 2 || max_accept > model_signature_.draft_block_size) {
        set_error(error, "DFlash2 benefit depth is outside the seeded block");
        return false;
    }
    const size_t required = static_cast<size_t>(max_accept - 1);
    if (trace.depths.size() < required) {
        set_error(error, "DFlash2 selector trace is missing required depths");
        return false;
    }

    out.conditional_hazards.reserve(required);
    double survival = 1.0;
    double expected = 1.0;
    const double selector_weight = 1.0 - config_.lm_log_weight;
    for (size_t depth = 0; depth < required; ++depth) {
        const DFlash2DepthSignal & signal = trace.depths[depth];
        if (!std::isfinite(signal.selected_log_prob) ||
            signal.selected_log_prob > 1e-6f ||
            !std::isfinite(signal.selector_winner_mass) ||
            signal.selector_winner_mass <= 0.0f ||
            signal.selector_winner_mass > 1.0f + 1e-6f) {
            set_error(error, "DFlash2 selector trace contains invalid evidence");
            out = {};
            return false;
        }
        const double log_hazard =
            config_.lm_log_weight * signal.selected_log_prob +
            selector_weight *
                std::log(std::min<double>(1.0, signal.selector_winner_mass));
        const double hazard = std::clamp(
            config_.hazard_scale * std::exp(log_hazard), 0.0, 1.0);
        if (!std::isfinite(hazard)) {
            set_error(error, "DFlash2 selector trace produced a nonfinite hazard");
            out = {};
            return false;
        }
        out.conditional_hazards.push_back(hazard);
        survival *= hazard;
        expected += survival;
    }
    out.expected_yield = std::clamp(
        config_.yield_scale * expected, 1.0,
        static_cast<double>(max_accept));
    if (error) error->clear();
    return true;
}

bool DFlash2BenefitProvider::publish_once(
        const DFlash2SelectorTrace & trace, int max_accept,
        double & destination, std::string * error) const {
    if (std::isfinite(destination)) {
        if (error) error->clear();
        return true;
    }
    DFlash2BenefitEstimate estimate_result;
    if (!estimate(trace, max_accept, estimate_result, error)) return false;
    destination = estimate_result.expected_yield;
    return true;
}

}  // namespace dflash::common
