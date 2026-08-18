#pragma once

// Model- and executor-neutral admission policy for mixed speculative/AR
// decode. The gate consumes only measured wall times, accepted-token yield,
// executor capacity, and an optional speculator-provided request prior.

#include "common/speculation_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace dflash::common {

struct SpecGateConfig {
    double ema_alpha       = 0.4;
    double cost_ewma_alpha = 0.35;
    int probe_rounds        = 2;
    int max_probers         = 2;
    int bad_rounds          = 2;
    int reprobe_tokens      = 64;
    double margin           = 1.05;
    double slack            = 1.10;
};

struct SpecCandidate {
    int slot = -1;
    std::uint64_t request_id = 0;
    SpeculationPolicy policy = SpeculationPolicy::Adaptive;
    bool eligible = false;
    double prior_accept = std::numeric_limits<double>::quiet_NaN();
    int generated_tokens = 0;
};

class SpeculationGate {
public:
    explicit SpeculationGate(SpecGateConfig config, double max_accept)
        : cfg_(sanitize(config)),
          max_accept_(valid_positive(max_accept)
                          ? std::max(1.0, max_accept) : 1.0) {}

    // The project is C++17, so these public range-taking methods provide the
    // same contiguous/range call shape as the spec's std::span API without
    // raising the language level of every CUDA/HIP translation unit.
    template <typename CandidateRange>
    std::vector<int> plan(int C, const CandidateRange & candidates,
                          int k_cap) {
        if (C <= 0) return {};

        struct Ranked {
            const SpecCandidate * candidate = nullptr;
            double score = 1.0;
            bool probing = false;
            bool reprobe = false;
        };
        std::vector<Ranked> forced;
        std::vector<Ranked> steady;
        std::vector<Ranked> probers;

        for (const SpecCandidate & candidate : candidates) {
            if (!candidate.eligible ||
                candidate.policy == SpeculationPolicy::Never) {
                continue;
            }
            RequestState & state = requests_[candidate.request_id];
            state.request_id = candidate.request_id;
            if (candidate.policy == SpeculationPolicy::Always) {
                forced.push_back({&candidate,
                                  score_for(state, candidate), false, false});
                continue;
            }
            if (state.mode == Mode::spec) {
                steady.push_back(
                    {&candidate, state.ema_yield, false, false});
                continue;
            }
            if (state.mode == Mode::probing) {
                probers.push_back(
                    {&candidate, probe_score(candidate), true, false});
                continue;
            }
            const long long generated = candidate.generated_tokens;
            const long long last = state.tokens_at_last_spec;
            if (generated - last >= cfg_.reprobe_tokens) {
                probers.push_back(
                    {&candidate, probe_score(candidate), true, true});
            }
        }

        // In particular, an all-Never/ineligible cohort must not allocate
        // either request or per-concurrency state.
        if (forced.empty() && steady.empty() && probers.empty()) return {};
        ensure_cost(C);

        const auto rank = [](const Ranked & a, const Ranked & b) {
            if (a.score != b.score) return a.score > b.score;
            return a.candidate->request_id < b.candidate->request_id;
        };
        std::sort(forced.begin(), forced.end(), rank);
        std::sort(probers.begin(), probers.end(), rank);
        if ((int)probers.size() > cfg_.max_probers) {
            probers.resize((size_t)cfg_.max_probers);
        }
        steady.insert(steady.end(), probers.begin(), probers.end());
        std::sort(steady.begin(), steady.end(), rank);

        std::vector<Ranked> selected = forced;
        const int capacity = std::max(0, std::min(C, k_cap));
        if (costs_[(size_t)C].ar.samples > 0 &&
            (int)forced.size() <= capacity) {
            double surplus = 0.0;
            for (const Ranked & item : forced) surplus += item.score - 1.0;
            const double ar_us = costs_[(size_t)C].ar.value;
            for (const Ranked & item : steady) {
                const int next_k = (int)selected.size() + 1;
                if (next_k > capacity) break;
                const double next_surplus = surplus + item.score - 1.0;
                const double spec_us = estimated_spec_us(C, next_k);
                const bool goodput = valid_positive(spec_us) &&
                    (C + next_surplus) / spec_us >=
                        cfg_.margin * C / ar_us;
                const bool peers_ok = next_k == C ||
                    spec_us <= cfg_.slack * ar_us;
                if (!goodput || !peers_ok) break;
                selected.push_back(item);
                surplus = next_surplus;
            }
        }

        std::vector<int> slots;
        slots.reserve(selected.size());
        std::vector<RequestState *> reprobed;
        for (const Ranked & item : selected) {
            const SpecCandidate & candidate = *item.candidate;
            RequestState & state = requests_[candidate.request_id];
            state.pending_generated_tokens = candidate.generated_tokens;
            state.pending_forced =
                candidate.policy == SpeculationPolicy::Always;
            if (item.reprobe && state.mode == Mode::ar) {
                state.mode = Mode::probing;
                state.bad_rounds = 0;
                reprobed.push_back(&state);
            }
            slots.push_back(candidate.slot);
        }
        for (RequestState * state : reprobed) {
            log_transition(state->request_id, Mode::ar, Mode::probing,
                           state->ema_yield, C, (int)selected.size());
        }
        return slots;
    }

    void observe_ar(int C, double step_us) {
        if (C <= 0 || !valid_positive(step_us)) return;
        ensure_cost(C);
        costs_[(size_t)C].ar.observe(step_us, cfg_.cost_ewma_alpha);
        trace(C, 0);
    }

    template <typename AcceptedRange>
    void observe_spec(int C, int k, double step_us,
                      const AcceptedRange & accepted) {
        if (C <= 0 || k < 1 || k > C || !valid_positive(step_us)) return;
        ensure_cost(C);
        CostState & cost = costs_[(size_t)C];
        cost.ensure_k(k);
        cost.spec[(size_t)k].observe(step_us, cfg_.cost_ewma_alpha);
        if (cost.ar.samples > 0) {
            cost.delta.observe(
                (step_us - cost.ar.value) / k, cfg_.cost_ewma_alpha);
        }

        for (const std::pair<std::uint64_t, int> & sample : accepted) {
            if (sample.second < 1) continue;
            RequestState & state = requests_[sample.first];
            state.request_id = sample.first;
            const double emitted = std::min(
                max_accept_, static_cast<double>(sample.second));
            if (state.rounds == 0) state.ema_yield = emitted;
            else state.ema_yield = cfg_.ema_alpha * emitted +
                (1.0 - cfg_.ema_alpha) * state.ema_yield;
            ++state.rounds;
            if (state.pending_generated_tokens >= 0) {
                state.tokens_at_last_spec =
                    state.pending_generated_tokens;
            }
            state.pending_generated_tokens = -1;
            const bool forced = state.pending_forced;
            state.pending_forced = false;
            if (forced || cost.ar.samples == 0) continue;

            const double marginal_us =
                estimated_spec_us(C, k) - estimated_spec_us(C, k - 1);
            const bool pays = state.ema_yield - 1.0 >=
                C * marginal_us / cost.ar.value;
            const Mode before = state.mode;
            if (state.mode == Mode::probing &&
                state.rounds >= cfg_.probe_rounds) {
                state.mode = pays ? Mode::spec : Mode::ar;
                state.bad_rounds = 0;
            } else if (state.mode == Mode::spec) {
                state.bad_rounds = pays ? 0 : state.bad_rounds + 1;
                if (state.bad_rounds >= cfg_.bad_rounds) {
                    state.mode = Mode::ar;
                    state.bad_rounds = 0;
                }
            }
            if (before != state.mode) {
                log_transition(sample.first, before, state.mode,
                               state.ema_yield, C, k);
            }
        }
        trace(C, k);
    }

    void forget(std::uint64_t request_id) { requests_.erase(request_id); }

private:
    enum class Mode { probing, spec, ar };

    struct Ewma {
        double value = 0.0;
        int samples = 0;

        void observe(double sample, double alpha) {
            value = samples == 0 ? sample
                                 : alpha * sample + (1.0 - alpha) * value;
            ++samples;
        }
    };

    struct RequestState {
        std::uint64_t request_id = 0;
        Mode mode = Mode::probing;
        double ema_yield = 1.0;
        int rounds = 0;
        int bad_rounds = 0;
        int tokens_at_last_spec = 0;
        int pending_generated_tokens = -1;
        bool pending_forced = false;
    };

    struct CostState {
        Ewma ar;
        std::vector<Ewma> spec{1};
        Ewma delta;

        void ensure_k(int k) {
            if ((int)spec.size() <= k) spec.resize((size_t)k + 1);
        }
    };

    static bool valid_positive(double value) {
        return std::isfinite(value) && value > 0.0;
    }

    static SpecGateConfig sanitize(SpecGateConfig cfg) {
        cfg.ema_alpha = std::isfinite(cfg.ema_alpha)
            ? std::clamp(cfg.ema_alpha, 0.0, 1.0) : 0.4;
        cfg.cost_ewma_alpha = std::isfinite(cfg.cost_ewma_alpha)
            ? std::clamp(cfg.cost_ewma_alpha, 0.0, 1.0) : 0.35;
        cfg.probe_rounds = std::max(1, cfg.probe_rounds);
        cfg.max_probers = std::max(0, cfg.max_probers);
        cfg.bad_rounds = std::max(1, cfg.bad_rounds);
        cfg.reprobe_tokens = std::max(0, cfg.reprobe_tokens);
        if (!valid_positive(cfg.margin)) cfg.margin = 1.05;
        if (!valid_positive(cfg.slack)) cfg.slack = 1.10;
        return cfg;
    }

    void ensure_cost(int C) {
        if ((int)costs_.size() <= C) costs_.resize((size_t)C + 1);
    }

    double probe_score(const SpecCandidate & candidate) const {
        return std::isfinite(candidate.prior_accept)
            ? std::clamp(candidate.prior_accept, 1.0, max_accept_)
            : max_accept_;
    }

    double score_for(const RequestState & state,
                     const SpecCandidate & candidate) const {
        return state.rounds > 0
            ? std::clamp(state.ema_yield, 1.0, max_accept_)
            : probe_score(candidate);
    }

    double nearest_delta(int C) const {
        int best_distance = std::numeric_limits<int>::max();
        double best = 0.0;
        for (int other = 1; other < (int)costs_.size(); ++other) {
            if (costs_[(size_t)other].delta.samples == 0) continue;
            const int distance = std::abs(other - C);
            if (distance < best_distance) {
                best_distance = distance;
                best = costs_[(size_t)other].delta.value;
            }
        }
        return best;
    }

    double estimated_spec_us(int C, int k) const {
        const CostState & cost = costs_[(size_t)C];
        if (k == 0) return cost.ar.value;
        if (k < (int)cost.spec.size() &&
            cost.spec[(size_t)k].samples > 0) {
            return cost.spec[(size_t)k].value;
        }
        const double delta = cost.delta.samples > 0
            ? cost.delta.value : nearest_delta(C);
        return cost.ar.value + k * delta;
    }

    static const char * mode_name(Mode mode) {
        switch (mode) {
        case Mode::probing: return "probing";
        case Mode::spec:    return "spec";
        case Mode::ar:      return "ar";
        }
        return "unknown";
    }

    static void log_transition(std::uint64_t request_id, Mode before,
                               Mode after, double ema_yield, int C, int k) {
        std::fprintf(stderr,
            "[speculation-gate] request=%llu mode=%s->%s "
            "ema_yield=%.3f C=%d k=%d\n",
            (unsigned long long)request_id, mode_name(before),
            mode_name(after), ema_yield, C, k);
    }

    void trace(int C, int k) const {
        if (std::getenv("DFLASH_SPECULATION_GATE_TRACE") == nullptr) return;
        const CostState & cost = costs_[(size_t)C];
        const double spec = k > 0 && k < (int)cost.spec.size() &&
                cost.spec[(size_t)k].samples > 0
            ? cost.spec[(size_t)k].value
            : std::numeric_limits<double>::quiet_NaN();
        const double delta = cost.delta.samples > 0
            ? cost.delta.value
            : std::numeric_limits<double>::quiet_NaN();
        std::fprintf(stderr,
            "[speculation-gate] C=%d k=%d T_ar=%.3f T_spec=%.3f "
            "delta=%.3f requests=%zu\n",
            C, k, cost.ar.samples > 0 ? cost.ar.value
                                      : std::numeric_limits<double>::quiet_NaN(),
            spec, delta, requests_.size());
        for (const auto & entry : requests_) {
            const RequestState & state = entry.second;
            std::fprintf(stderr,
                "[speculation-gate] request=%llu mode=%s ema_yield=%.3f "
                "rounds=%d bad_rounds=%d\n",
                (unsigned long long)entry.first, mode_name(state.mode),
                state.ema_yield, state.rounds, state.bad_rounds);
        }
    }

    SpecGateConfig cfg_;
    double max_accept_ = 1.0;
    std::map<std::uint64_t, RequestState> requests_;
    std::vector<CostState> costs_{1};
};

}  // namespace dflash::common
