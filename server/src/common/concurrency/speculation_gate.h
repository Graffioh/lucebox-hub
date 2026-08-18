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
#include <vector>

namespace dflash::common {

struct SpecGateConfig {
    double ema_alpha       = 0.4;
    double cost_ewma_alpha = 0.35;
    int probe_rounds        = 2;
    int max_probers         = 2;
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

struct SpecObservation {
    std::uint64_t request_id = 0;
    int emitted_tokens = 0;
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
            if (candidate.policy == SpeculationPolicy::Always) {
                forced.push_back({&candidate, state.rounds > 0
                    ? measured_score(state) : optimistic_score(candidate)});
                continue;
            }
            const long long generated = candidate.generated_tokens;
            const long long last = state.tokens_at_last_spec;
            const bool stale =
                generated - last >= cfg_.reprobe_tokens;
            const bool optimistic =
                state.rounds < cfg_.probe_rounds || stale;
            (optimistic ? probers : steady).push_back({
                &candidate,
                optimistic ? optimistic_score(candidate)
                           : measured_score(state),
            });
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
            const double ar_us = costs_[(size_t)C].ar.value;
            std::vector<double> estimated_us((size_t)capacity + 1, ar_us);
            for (int k = 1; k <= capacity; ++k) {
                const double raw = estimated_spec_us(C, k);
                estimated_us[(size_t)k] = valid_positive(raw)
                    ? std::max(estimated_us[(size_t)k - 1], raw)
                    : estimated_us[(size_t)k - 1];
            }

            double emitted = C;
            for (const Ranked & item : forced) emitted += item.score - 1.0;
            int current_k = (int)forced.size();
            double current_goodput =
                emitted / estimated_us[(size_t)current_k];
            const double ar_goodput = C / ar_us;
            for (const Ranked & item : steady) {
                const int next_k = current_k + 1;
                if (next_k > capacity) break;
                const double next_emitted = emitted + item.score - 1.0;
                const double next_us = estimated_us[(size_t)next_k];
                const double next_goodput = next_emitted / next_us;
                const bool marginal = next_goodput > current_goodput;
                const bool overall =
                    next_goodput >= cfg_.margin * ar_goodput;
                const bool peers_ok = next_k == C ||
                    next_us <= cfg_.slack * ar_us;
                if (!marginal || !overall || !peers_ok) break;
                selected.push_back(item);
                current_k = next_k;
                emitted = next_emitted;
                current_goodput = next_goodput;
            }
        }

        std::vector<int> slots;
        slots.reserve(selected.size());
        for (const Ranked & item : selected) {
            slots.push_back(item.candidate->slot);
        }
        trace_plan(C, capacity, slots);
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

        for (const SpecObservation & sample : accepted) {
            if (sample.emitted_tokens < 1) continue;
            RequestState & state = requests_[sample.request_id];
            const double emitted = std::min(
                max_accept_, static_cast<double>(sample.emitted_tokens));
            if (state.rounds == 0) state.ema_yield = emitted;
            else state.ema_yield = cfg_.ema_alpha * emitted +
                (1.0 - cfg_.ema_alpha) * state.ema_yield;
            ++state.rounds;
            state.tokens_at_last_spec =
                std::max(0, sample.generated_tokens);
        }
        trace(C, k);
    }

    void forget(std::uint64_t request_id) { requests_.erase(request_id); }

private:
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
        double ema_yield = 1.0;
        int rounds = 0;
        int tokens_at_last_spec = 0;
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
        cfg.reprobe_tokens = std::max(0, cfg.reprobe_tokens);
        if (!valid_positive(cfg.margin)) cfg.margin = 1.05;
        if (!valid_positive(cfg.slack)) cfg.slack = 1.10;
        return cfg;
    }

    void ensure_cost(int C) {
        if ((int)costs_.size() <= C) costs_.resize((size_t)C + 1);
    }

    double optimistic_score(const SpecCandidate & candidate) const {
        return std::isfinite(candidate.prior_accept)
            ? std::clamp(candidate.prior_accept, 1.0, max_accept_)
            : max_accept_;
    }

    double measured_score(const RequestState & state) const {
        return std::clamp(state.ema_yield, 1.0, max_accept_);
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

    static void trace_plan(int C, int capacity,
                           const std::vector<int> & slots) {
        if (std::getenv("DFLASH_SPECULATION_GATE_TRACE") == nullptr) return;
        std::fprintf(stderr,
            "[speculation-gate] plan C=%d capacity=%d selected=%zu slots=",
            C, capacity, slots.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", slots[i]);
        }
        std::fputc('\n', stderr);
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
                "[speculation-gate] request=%llu ema_yield=%.3f "
                "rounds=%d tokens_at_last_spec=%d\n",
                (unsigned long long)entry.first, state.ema_yield,
                state.rounds, state.tokens_at_last_spec);
        }
    }

    SpecGateConfig cfg_;
    double max_accept_ = 1.0;
    std::map<std::uint64_t, RequestState> requests_;
    std::vector<CostState> costs_{1};
};

}  // namespace dflash::common
