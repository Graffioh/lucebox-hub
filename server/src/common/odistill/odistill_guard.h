// odistill_guard.h — acceptance guard state machine (ODISTILL.md §7).
//
// Pure bookkeeping, no engine dependencies: the runtime feeds it one
// accepted-length sample per verify step and tells it when an adapter
// swap happens; the guard answers with promote / rollback / disable
// decisions. Kept header-only and deterministic so the unit test can
// drive every transition.
//
//   serving(gen N) ── swap(gen N+1) ──▶ probation: cfg.probation_steps
//     AL(probation) >= AL(baseline) - epsilon  → Promote (baseline ← probation)
//     else                                     → Rollback (re-arm baseline),
//                                                next swap backs off 2×
//   >= cfg.max_consecutive_rollbacks in a row  → Disable for the session
//
// The baseline is a rolling window of accepted lengths measured while
// serving. Early in a session there may be no baseline yet; a probation
// with fewer than cfg.min_baseline_steps of baseline evidence promotes by
// default — the guard only vetoes regressions it can actually measure.

#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

namespace dflash::common::odistill {

struct ODistillGuardConfig {
    int   probation_steps          = 32;    // verify steps per probation
    float epsilon                  = 0.15f; // AL tokens of allowed noise
    int   baseline_window          = 128;   // serving AL window (steps)
    int   min_baseline_steps       = 8;     // evidence needed to veto
    int   min_steps_between_swaps  = 32;    // base swap backoff (steps)
    int   max_consecutive_rollbacks = 3;    // then disable for the session
};

enum class ODistillGuardAction {
    None,
    Promote,   // probation passed: keep the new generation
    Rollback,  // probation failed: revert to the previous generation
    Disable,   // rollback limit hit: stop training for the session
};

class ODistillGuard {
public:
    enum class State { Serving, Probation, Disabled };

    explicit ODistillGuard(const ODistillGuardConfig & cfg = {})
        : cfg_(cfg), swap_backoff_(cfg.min_steps_between_swaps) {}

    // One sample per verify step: how many drafted tokens were accepted.
    // Returns the action the runtime must carry out (slot flip on
    // Rollback is the runtime's job; the guard only decides).
    ODistillGuardAction record_step(float accept_len) {
        steps_total_++;
        if (state_ == State::Disabled) return ODistillGuardAction::None;

        if (state_ == State::Serving) {
            push_window(baseline_, accept_len, cfg_.baseline_window);
            return ODistillGuardAction::None;
        }

        // Probation.
        probation_.push_back(accept_len);
        if ((int)probation_.size() < cfg_.probation_steps) {
            return ODistillGuardAction::None;
        }
        const bool measurable =
            (int)baseline_snapshot_count_ >= cfg_.min_baseline_steps;
        const float probation_al = mean(probation_);
        // Retain the exact operands used for this decision. The state
        // transition below either replaces baseline_ or clears probation_, so
        // reading the live windows afterwards produces misleading logs.
        decision_baseline_al_ = baseline_snapshot_al_;
        decision_probation_al_ = probation_al;
        const bool pass = !measurable ||
            probation_al >= baseline_snapshot_al_ - cfg_.epsilon;

        state_ = State::Serving;
        if (pass) {
            // Promote: the probation window becomes the new baseline.
            baseline_ = probation_;
            probation_.clear();
            consecutive_rollbacks_ = 0;
            swap_backoff_ = cfg_.min_steps_between_swaps;
            promotes_++;
            return ODistillGuardAction::Promote;
        }
        probation_.clear();
        consecutive_rollbacks_++;
        rollbacks_++;
        swap_backoff_ *= 2;
        if (consecutive_rollbacks_ >= cfg_.max_consecutive_rollbacks) {
            state_ = State::Disabled;
            return ODistillGuardAction::Disable;
        }
        return ODistillGuardAction::Rollback;
    }

    // May the runtime apply a pending adapter at the next block boundary?
    bool can_swap() const {
        return state_ == State::Serving &&
               steps_total_ - last_swap_step_ >= (uint64_t)swap_backoff_;
    }

    // The runtime flipped to `generation`; probation starts now.
    void on_swap(uint64_t generation) {
        last_swap_step_ = steps_total_;
        current_generation_ = generation;
        baseline_snapshot_al_    = baseline_.empty() ? 0.0f : mean(baseline_);
        baseline_snapshot_count_ = baseline_.size();
        probation_.clear();
        state_ = State::Probation;
        swaps_++;
    }

    State    state() const              { return state_; }
    uint64_t current_generation() const { return current_generation_; }
    float    baseline_al() const { return baseline_.empty() ? 0.0f : mean(baseline_); }
    float    probation_al() const { return probation_.empty() ? 0.0f : mean(probation_); }
    float    decision_baseline_al() const { return decision_baseline_al_; }
    float    decision_probation_al() const { return decision_probation_al_; }
    uint64_t swaps() const     { return swaps_; }
    uint64_t promotes() const  { return promotes_; }
    uint64_t rollbacks() const { return rollbacks_; }
    int      swap_backoff() const { return swap_backoff_; }

private:
    static float mean(const std::vector<float> & v) {
        return v.empty() ? 0.0f
             : std::accumulate(v.begin(), v.end(), 0.0f) / (float)v.size();
    }
    static void push_window(std::vector<float> & v, float x, int cap) {
        v.push_back(x);
        if ((int)v.size() > cap) v.erase(v.begin());
    }

    ODistillGuardConfig cfg_;
    State state_ = State::Serving;
    std::vector<float> baseline_;
    std::vector<float> probation_;
    float    baseline_snapshot_al_ = 0.0f;
    size_t   baseline_snapshot_count_ = 0;
    float    decision_baseline_al_ = 0.0f;
    float    decision_probation_al_ = 0.0f;
    int      swap_backoff_ = 0;
    uint64_t steps_total_ = 0;
    uint64_t last_swap_step_ = 0;
    uint64_t current_generation_ = 0;
    int      consecutive_rollbacks_ = 0;
    uint64_t swaps_ = 0, promotes_ = 0, rollbacks_ = 0;
};

}  // namespace dflash::common::odistill
