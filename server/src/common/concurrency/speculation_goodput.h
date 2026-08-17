#pragma once

// Model-neutral online controller for deciding whether a request should use
// speculative decoding.  The controller deliberately consumes only outcomes:
// useful tokens emitted and wall time for an AR or speculative step.  A
// concrete speculator (DDTree today, DSpark later) remains responsible for
// producing and verifying candidates.

#include <algorithm>
#include <cmath>

namespace dflash::common {

enum class SpeculationGoodputTransition {
    none,
    enabled,
    disabled,
};

struct SpeculationGoodputConfig {
    // Require a measured advantage large enough to survive timing noise.
    double minimum_gain = 1.05;
    // Half-life is intentionally short: generation can move between
    // predictable code/structure and high-entropy prose within one request.
    double ewma_alpha = 0.5;
    // Do not abandon a profitable route after one unlucky verification.
    int bad_speculation_steps = 2;
    // Optional re-probing after the initial route decision. This is disabled
    // by default because one speculative tree probe can be much more expensive
    // than an AR step at high occupancy. Callers may opt in when the speculator
    // has a sufficiently cheap proposal path.
    int ar_reprobe_steps = 0;
};

class SpeculationGoodputController {
public:
    SpeculationGoodputController() = default;
    explicit SpeculationGoodputController(
            SpeculationGoodputConfig config)
        : config_(sanitize(config)) {}

    void reset(bool adaptive = true) {
        adaptive_ = adaptive;
        phase_ = adaptive ? Phase::initial_spec_probe : Phase::speculate;
        speculative_goodput_ = 0.0;
        ar_goodput_ = 0.0;
        expected_emitted_tokens_ = 1.0;
        has_speculative_goodput_ = false;
        has_ar_goodput_ = false;
        has_expected_emitted_tokens_ = false;
        bad_speculation_steps_ = 0;
        ar_steps_since_probe_ = 0;
    }

    bool wants_speculation() const {
        return !adaptive_ || phase_ == Phase::initial_spec_probe ||
               phase_ == Phase::speculate ||
               phase_ == Phase::reprobe_speculation;
    }

    // Seed request value from a cheap drafter-side confidence estimate without
    // changing the measured AR/speculation route state.
    void observe_expected_tokens(double expected_tokens) {
        if (!adaptive_ || !std::isfinite(expected_tokens) ||
            expected_tokens <= 0.0) {
            return;
        }
        update_ewma(expected_emitted_tokens_,
                    has_expected_emitted_tokens_, expected_tokens);
    }

    SpeculationGoodputTransition observe_speculation(
            double emitted_tokens, double elapsed_us) {
        if (!adaptive_ || !valid_observation(emitted_tokens, elapsed_us)) {
            return SpeculationGoodputTransition::none;
        }
        update_ewma(speculative_goodput_, has_speculative_goodput_,
                    emitted_tokens / elapsed_us);
        update_ewma(expected_emitted_tokens_,
                    has_expected_emitted_tokens_, emitted_tokens);

        if (phase_ == Phase::initial_spec_probe) {
            phase_ = Phase::initial_ar_probe;
            return SpeculationGoodputTransition::none;
        }
        if (phase_ == Phase::reprobe_speculation) {
            if (speculation_profitable()) {
                phase_ = Phase::speculate;
                bad_speculation_steps_ = 0;
                return SpeculationGoodputTransition::enabled;
            }
            phase_ = Phase::autoregressive;
            ar_steps_since_probe_ = 0;
            return SpeculationGoodputTransition::none;
        }
        if (phase_ != Phase::speculate || !has_ar_goodput_) {
            return SpeculationGoodputTransition::none;
        }

        if (speculation_profitable()) {
            bad_speculation_steps_ = 0;
            return SpeculationGoodputTransition::none;
        }
        if (++bad_speculation_steps_ < config_.bad_speculation_steps) {
            return SpeculationGoodputTransition::none;
        }
        phase_ = Phase::autoregressive;
        bad_speculation_steps_ = 0;
        ar_steps_since_probe_ = 0;
        return SpeculationGoodputTransition::disabled;
    }

    SpeculationGoodputTransition observe_autoregressive(double elapsed_us) {
        if (!adaptive_ || !valid_observation(1.0, elapsed_us)) {
            return SpeculationGoodputTransition::none;
        }
        update_ewma(ar_goodput_, has_ar_goodput_, 1.0 / elapsed_us);

        if (phase_ == Phase::initial_ar_probe) {
            if (speculation_profitable()) {
                phase_ = Phase::speculate;
                bad_speculation_steps_ = 0;
                return SpeculationGoodputTransition::none;
            }
            phase_ = Phase::autoregressive;
            ar_steps_since_probe_ = 0;
            return SpeculationGoodputTransition::disabled;
        }
        if (phase_ != Phase::autoregressive ||
            config_.ar_reprobe_steps <= 0) {
            return SpeculationGoodputTransition::none;
        }
        if (++ar_steps_since_probe_ >= config_.ar_reprobe_steps) {
            phase_ = Phase::reprobe_speculation;
            ar_steps_since_probe_ = 0;
        }
        return SpeculationGoodputTransition::none;
    }

    bool adaptive() const { return adaptive_; }
    bool has_speculative_goodput() const {
        return has_speculative_goodput_;
    }
    bool has_ar_goodput() const { return has_ar_goodput_; }
    double speculative_goodput() const { return speculative_goodput_; }
    double ar_goodput() const { return ar_goodput_; }
    double expected_emitted_tokens() const {
        return expected_emitted_tokens_;
    }
    bool has_expected_emitted_tokens() const {
        return has_expected_emitted_tokens_;
    }

private:
    enum class Phase {
        initial_spec_probe,
        initial_ar_probe,
        speculate,
        autoregressive,
        reprobe_speculation,
    };

    static SpeculationGoodputConfig sanitize(
            SpeculationGoodputConfig config) {
        config.minimum_gain = std::max(1.0, config.minimum_gain);
        config.ewma_alpha = std::clamp(config.ewma_alpha, 0.0, 1.0);
        config.bad_speculation_steps =
            std::max(1, config.bad_speculation_steps);
        config.ar_reprobe_steps = std::max(0, config.ar_reprobe_steps);
        return config;
    }

    static bool valid_observation(double emitted_tokens, double elapsed_us) {
        return std::isfinite(emitted_tokens) && emitted_tokens > 0.0 &&
               std::isfinite(elapsed_us) && elapsed_us > 0.0;
    }

    void update_ewma(double & value, bool & initialized, double sample) {
        if (!initialized) {
            value = sample;
            initialized = true;
            return;
        }
        value = config_.ewma_alpha * sample +
                (1.0 - config_.ewma_alpha) * value;
    }

    bool speculation_profitable() const {
        return has_speculative_goodput_ && has_ar_goodput_ &&
               speculative_goodput_ >=
                   config_.minimum_gain * ar_goodput_;
    }

    SpeculationGoodputConfig config_;
    bool adaptive_ = true;
    Phase phase_ = Phase::initial_spec_probe;
    double speculative_goodput_ = 0.0;
    double ar_goodput_ = 0.0;
    double expected_emitted_tokens_ = 1.0;
    bool has_speculative_goodput_ = false;
    bool has_ar_goodput_ = false;
    bool has_expected_emitted_tokens_ = false;
    int bad_speculation_steps_ = 0;
    int ar_steps_since_probe_ = 0;
};

}  // namespace dflash::common
