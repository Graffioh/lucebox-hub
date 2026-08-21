// Cohort-planned adaptive speculation policy over startup-profiled costs.
// Request-local state contains only immutable activation knowledge. The engine
// owns the current cohort epoch and decides when to run plan() again.
// Pure host code: no graph, backend, or scheduler types belong here.

#pragma once

#include "common/speculation_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dflash::common {

struct SpecGateConfig {
    double cost_ema_alpha = 0.20;
    double adaptive_gain_margin = 0.01;
};

struct SpecCostLookup {
    double cost = std::numeric_limits<double>::infinity();
    int requested_index = 0;
    int profiled_index = 0;
    bool clamped = false;
    bool rounded_up = false;
};

struct SpecCostSeries {
    std::vector<int> indices;
    std::vector<double> costs;

    bool valid() const {
        if (indices.empty() || indices.size() != costs.size()) return false;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || !std::isfinite(costs[i]) || costs[i] <= 0.0)
                return false;
            if (i > 0 && (indices[i] <= indices[i - 1] ||
                          costs[i] < costs[i - 1]))
                return false;
        }
        return true;
    }

    SpecCostLookup lookup(int index) const {
        SpecCostLookup result;
        result.requested_index = index;
        if (!valid()) return result;
        auto it = std::lower_bound(indices.begin(), indices.end(), index);
        if (it == indices.end()) {
            result.profiled_index = indices.back();
            result.cost = costs.back();
            result.clamped = true;
            return result;
        }
        const size_t pos = static_cast<size_t>(it - indices.begin());
        result.profiled_index = *it;
        result.cost = costs[pos];
        result.clamped = index < indices.front() || index > indices.back();
        result.rounded_up = *it != index && !result.clamped;
        return result;
    }
};

struct SpecCostTables {
    SpecCostSeries tree_cost;
    SpecCostSeries step_cost;
    SpecCostSeries draft_cost;
    std::string speculator_id;

    bool valid() const {
        return tree_cost.valid() && step_cost.valid() && draft_cost.valid();
    }
};

inline constexpr const char * kUnspecifiedScoreKind = "unspecified";

struct SpecCandidate {
    uint64_t request_id = 0;
    int slot = -1;
    SpeculationPolicy policy = SpeculationPolicy::Adaptive;
    // scoreable is a request-lifetime capability: the engine can produce the
    // one-time activation score from the committed feature mirror.
    bool scoreable = false;
    // can_speculate is also request-lifetime (for example, false for an
    // unsupported sampler or thinking hook). The min-token EOS policy is
    // enforced inside the speculative path. An incompatible prefill graph may
    // suspend speculative execution for one explicitly telemetered AR service
    // round without changing this capability or the chosen request mode.
    bool can_speculate = false;
    // NaN requests a one-time bootstrap when no cached score exists. A finite
    // value is the preferred activation measurement. Evaluation failure keeps
    // the request AR without inventing a score. Adapter estimates are already
    // calibrated; the gate only clamps executor bounds.
    double activation_yield = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> conditional_hazards;
    std::string score_kind = kUnspecifiedScoreKind;
};

struct SpecStepGeometry {
    int tree_width = 1;
    std::function<int(int)> bucket = [](int lanes) {
        return std::max(0, lanes);
    };

    int bucketed_lanes(int lanes) const {
        return lanes <= 0 ? 0 : std::max(lanes, bucket ? bucket(lanes) : lanes);
    }
    int tree_rows(int spec_lanes) const {
        return bucketed_lanes(spec_lanes) * std::max(1, tree_width);
    }
    double expected_step_rows(int concurrency, int spec_lanes,
                              double expected_spec_tokens) const {
        const double accepted_rows = std::max(
            static_cast<double>(spec_lanes), expected_spec_tokens);
        return accepted_rows + bucketed_lanes(concurrency - spec_lanes);
    }
};

enum class SpecScoreSource : uint8_t {
    Fresh,
    Initial,
    Unavailable,
};

inline const char * spec_score_source_name(SpecScoreSource source) {
    switch (source) {
        case SpecScoreSource::Fresh: return "fresh";
        case SpecScoreSource::Initial: return "initial";
        case SpecScoreSource::Unavailable: return "unavailable";
    }
    return "unknown";
}

enum class SpecEvaluationAction : uint8_t {
    Score,
    FallbackAR,
};

struct SpecPendingEvaluation {
    uint64_t request_id = 0;
    int slot = -1;
    SpecEvaluationAction action = SpecEvaluationAction::Score;
};

struct SpecPlanScore {
    uint64_t request_id = 0;
    int slot = -1;
    double expected_yield = 1.0;
    SpecScoreSource source = SpecScoreSource::Unavailable;
    bool forced = false;
    bool admitted = false;
    std::string score_kind = kUnspecifiedScoreKind;
    bool execution_unsupported = false;
};

// Discrete graph shape that actually ran. Unlike a SpecPlan's fractional
// expected replay rows, every field here comes from executor telemetry and is
// safe to use as an online timing key.
struct SpecExecutionShape {
    int concurrency = 0;
    int admitted_count = 0;
    int tree_rows = 0;
    int step_rows = 0;
    int draft_lanes = 0;

    bool operator==(const SpecExecutionShape & other) const {
        return concurrency == other.concurrency &&
               admitted_count == other.admitted_count &&
               tree_rows == other.tree_rows &&
               step_rows == other.step_rows &&
               draft_lanes == other.draft_lanes;
    }
};

struct SpecPlan {
    bool valid = true;
    std::string error;
    int concurrency = 0;
    int admitted_count = 0;
    int tree_rows = 0;
    double expected_step_rows = 0.0;
    int draft_lanes = 0;
    double expected_tokens = 0.0;
    // Startup-profiled cost before online correction, and the shape-local
    // correction applied to it. predicted_cost is their product.
    double profiled_cost = 0.0;
    double cost_scale = 1.0;
    double predicted_cost = 0.0;
    // Fixed-scale expected yield for admitted activation-scored lanes. This is
    // directly comparable with realized emitted tokens in telemetry.
    double initial_predicted_tokens = 0.0;
    double goodput = 0.0;
    double ar_goodput = 0.0;
    int unavailable_count = 0;
    // The engine resolves pending actions, immediately replans, and caches
    // only the completed result as the current cohort epoch.
    bool cost_lookup_clamped = false;
    std::vector<SpecPlanScore> ordered;
    std::vector<uint64_t> admitted_request_ids;
    std::vector<int> admitted_slots;
    // Score actions are batched for one-time activation-score initialization.
    // FallbackAR actions cannot attempt scoring and instead record an
    // explicit failed evaluation. One tagged record keeps
    // request identity and slot inseparable on all failure paths.
    std::vector<SpecPendingEvaluation> pending_evaluations;
};

struct SpecCohortEpoch {
    uint64_t id = 0;
    std::vector<uint64_t> request_ids;
    SpecPlan plan;

    bool matches(const std::vector<SpecCandidate> & candidates) const {
        return request_ids == ids(candidates);
    }

    static std::vector<uint64_t> ids(
            const std::vector<SpecCandidate> & candidates) {
        std::vector<uint64_t> out;
        out.reserve(candidates.size());
        for (const SpecCandidate & candidate : candidates)
            out.push_back(candidate.request_id);
        std::sort(out.begin(), out.end());
        return out;
    }
};

class SpeculationGate {
private:
    struct RequestState {
        double initial_score =
            std::numeric_limits<double>::quiet_NaN();
        bool evaluation_failed = false;
        std::string score_kind = kUnspecifiedScoreKind;
        std::vector<double> conditional_hazards;
    };

    struct ExecutionShapeHash {
        size_t operator()(const SpecExecutionShape & shape) const {
            size_t seed = 0;
            auto mix = [&](int value) {
                seed ^= std::hash<int>{}(value) +
                    static_cast<size_t>(0x9e3779b9U) +
                    (seed << 6) + (seed >> 2);
            };
            mix(shape.concurrency);
            mix(shape.admitted_count);
            mix(shape.tree_rows);
            mix(shape.step_rows);
            mix(shape.draft_lanes);
            return seed;
        }
    };

    struct CostState {
        double scale = 1.0;
        uint64_t observations = 0;
    };

    struct CostPrice {
        double profiled = std::numeric_limits<double>::infinity();
        double predicted = std::numeric_limits<double>::infinity();
    };

    struct CandidateScore {
        double expected_yield = 1.0;
        SpecScoreSource source = SpecScoreSource::Unavailable;
        std::string score_kind = kUnspecifiedScoreKind;
    };

public:
    using ClampLogger = std::function<void(
        const char * table, int requested, int profiled)>;

    SpeculationGate(SpecCostTables costs, SpecStepGeometry geometry,
                    int max_accept, ClampLogger clamp_logger = {},
                    SpecGateConfig config = {},
                    bool direct_commit = false)
        : config_(config), costs_(std::move(costs)),
          geometry_(std::move(geometry)),
          max_accept_(std::max(1, max_accept)),
          direct_commit_(direct_commit),
          clamp_logger_(std::move(clamp_logger)) {}

    SpeculationGate(SpecGateConfig config, SpecCostTables costs,
                    SpecStepGeometry geometry, int max_accept,
                    ClampLogger clamp_logger = {},
                    bool direct_commit = false)
        : SpeculationGate(std::move(costs), std::move(geometry), max_accept,
                          std::move(clamp_logger), config, direct_commit) {}

    bool valid() const {
        auto valid_alpha = [](double value) {
            return std::isfinite(value) && value > 0.0 && value <= 1.0;
        };
        return valid_alpha(config_.cost_ema_alpha) &&
               std::isfinite(config_.adaptive_gain_margin) &&
               config_.adaptive_gain_margin >= 0.0 &&
               costs_.valid() && geometry_.tree_width >= 1 &&
               max_accept_ >= 1;
    }

    // draft_lanes_override prices always-drafting. -1 means admitted-only.
    // The engine calls this only for a new cohort epoch and once more after
    // resolving any cold-score actions.
    SpecPlan plan(int concurrency,
                  const std::vector<SpecCandidate> & candidates,
                  int k_cap, int draft_lanes_override = -1) {
        SpecPlan out;
        out.concurrency = concurrency;
        if (!valid() || concurrency < 0 ||
            static_cast<size_t>(concurrency) != candidates.size() ||
            k_cap < 0) {
            out.valid = false;
            out.error = "invalid speculation gate inputs";
            return out;
        }

        struct Ranked {
            const SpecCandidate * candidate = nullptr;
            double score = 1.0;
            SpecScoreSource source = SpecScoreSource::Unavailable;
            std::string score_kind = kUnspecifiedScoreKind;
            bool forced = false;
        };
        std::vector<Ranked> forced;
        std::vector<Ranked> adaptive;
        std::vector<Ranked> forced_ar;
        forced.reserve(candidates.size());
        adaptive.reserve(candidates.size());
        forced_ar.reserve(candidates.size());

        for (const SpecCandidate & candidate : candidates) {
            if (candidate.policy == SpeculationPolicy::Never) continue;
            if (candidate.policy == SpeculationPolicy::Adaptive &&
                evaluation_failed(candidate.request_id)) {
                forced_ar.push_back({
                    &candidate, 1.0, SpecScoreSource::Unavailable,
                    initial_score_kind(candidate.request_id), false});
                continue;
            }

            const CandidateScore score = score_candidate(candidate);
            if (candidate.policy == SpeculationPolicy::Adaptive &&
                score.source == SpecScoreSource::Unavailable) {
                ++out.unavailable_count;
                out.pending_evaluations.push_back({
                    candidate.request_id, candidate.slot,
                    candidate.scoreable
                        ? SpecEvaluationAction::Score
                        : SpecEvaluationAction::FallbackAR,
                });
                continue;
            }
            if (candidate.policy == SpeculationPolicy::Adaptive &&
                !candidate.can_speculate) {
                forced_ar.push_back({
                    &candidate, score.expected_yield, score.source,
                    score.score_kind, false});
                continue;
            }
            Ranked ranked{
                &candidate, score.expected_yield, score.source,
                score.score_kind,
                candidate.policy == SpeculationPolicy::Always};
            (ranked.forced ? forced : adaptive).push_back(ranked);
        }

        // A cohort plan is publishable only after every cold adaptive request
        // has resolved. Explicit Always lanes may still run in the bootstrap
        // service plan, but the engine never caches that partial result.
        if (!out.pending_evaluations.empty()) adaptive.clear();

        auto request_order = [](const Ranked & a, const Ranked & b) {
            return a.candidate->request_id < b.candidate->request_id;
        };
        std::sort(forced.begin(), forced.end(), request_order);
        std::sort(forced_ar.begin(), forced_ar.end(), request_order);
        std::sort(adaptive.begin(), adaptive.end(),
            [](const Ranked & a, const Ranked & b) {
                if (a.score != b.score) return a.score > b.score;
                return a.candidate->request_id < b.candidate->request_id;
            });

        if (static_cast<int>(forced.size()) > k_cap) {
            out.valid = false;
            out.error = "forced speculation exceeds executor capacity";
            return out;
        }

        std::vector<Ranked> ranked;
        ranked.reserve(forced.size() + adaptive.size());
        ranked.insert(ranked.end(), forced.begin(), forced.end());
        ranked.insert(ranked.end(), adaptive.begin(), adaptive.end());
        for (const Ranked & item : ranked) {
            out.ordered.push_back({
                item.candidate->request_id,
                item.candidate->slot,
                item.score,
                item.source,
                item.forced,
                false,
                item.score_kind,
                false,
            });
        }

        const int forced_count = static_cast<int>(forced.size());
        const int max_k = std::min<int>(k_cap, ranked.size());
        double expected_sum = 0.0;
        for (int i = 0; i < forced_count; ++i) expected_sum += ranked[i].score;

        const SpecExecutionShape ar_shape{
            concurrency, 0, 0, geometry_.bucketed_lanes(concurrency), 0};
        const CostPrice ar_price = price_execution_shape(ar_shape, &out);
        out.ar_goodput = concurrency == 0 ? 0.0
            : static_cast<double>(concurrency) / ar_price.predicted;

        struct PlanPoint {
            int k = 0;
            double goodput = -1.0;
            double profiled_cost = 0.0;
            double predicted_cost = 0.0;
            double cost_scale = 1.0;
            double expected_tokens = 0.0;
            int tree_rows = 0;
            double expected_step_rows = 0.0;
            int draft_lanes = 0;
        };
        PlanPoint baseline;
        PlanPoint best;

        for (int k = forced_count; k <= max_k; ++k) {
            if (k > forced_count) expected_sum += ranked[k - 1].score;
            const double expected_tokens =
                static_cast<double>(concurrency - k) + expected_sum;
            int tree_rows = 0;
            double expected_step_rows = geometry_.bucketed_lanes(concurrency);
            const int draft_lanes = draft_lanes_override >= 0
                ? draft_lanes_override : k;

            if (k > 0) {
                if (direct_commit_) {
                    tree_rows = geometry_.tree_rows(k) + concurrency - k;
                    expected_step_rows = 0.0;
                } else {
                    tree_rows = geometry_.tree_rows(k);
                    expected_step_rows = geometry_.expected_step_rows(
                        concurrency, k, expected_sum);
                }
            }
            const CostPrice price = price_expected_shape(
                {concurrency, k, tree_rows, 0, draft_lanes},
                expected_step_rows, &out);
            const double scale = price.predicted / price.profiled;
            const double goodput = expected_tokens / price.predicted;
            const PlanPoint point{
                k, goodput, price.profiled, price.predicted, scale,
                expected_tokens, tree_rows, expected_step_rows, draft_lanes};
            if (k == forced_count) baseline = point;
            if (goodput > best.goodput) best = point;
        }

        // Explicit Always lanes establish the non-negotiable baseline. The
        // safety margin applies to this epoch's adaptive subset selection.
        if (best.k > forced_count &&
            best.goodput < baseline.goodput *
                (1.0 + config_.adaptive_gain_margin)) {
            best = baseline;
        }

        out.admitted_count = best.k;
        out.goodput = std::max(0.0, best.goodput);
        out.profiled_cost = best.profiled_cost;
        out.cost_scale = best.cost_scale;
        out.predicted_cost = best.predicted_cost;
        out.expected_tokens = best.expected_tokens;
        out.tree_rows = best.tree_rows;
        out.expected_step_rows = best.expected_step_rows;
        out.draft_lanes = best.draft_lanes;
        for (int i = 0; i < best.k; ++i) {
            out.ordered[static_cast<size_t>(i)].admitted = true;
            out.admitted_request_ids.push_back(
                ranked[static_cast<size_t>(i)].candidate->request_id);
            out.admitted_slots.push_back(
                ranked[static_cast<size_t>(i)].candidate->slot);
            if (ranked[static_cast<size_t>(i)].source !=
                SpecScoreSource::Unavailable) {
                out.initial_predicted_tokens +=
                    ranked[static_cast<size_t>(i)].score;
            }
        }
        for (const Ranked & item : forced_ar) {
            out.ordered.push_back({
                item.candidate->request_id,
                item.candidate->slot,
                item.score,
                item.source,
                false,
                false,
                item.score_kind,
                true,
            });
        }
        return out;
    }

    void observe_cost(const SpecExecutionShape & executed,
                      double measured_us) {
        if (!std::isfinite(measured_us) || measured_us <= 0.0 ||
            executed.concurrency < 0 || executed.admitted_count < 0 ||
            executed.admitted_count > executed.concurrency ||
            executed.tree_rows < 0 || executed.step_rows < 0 ||
            (executed.step_rows == 0 &&
             (!direct_commit_ || executed.admitted_count == 0)) ||
            executed.draft_lanes < 0) {
            return;
        }
        const double profiled_cost =
            profile_execution_shape(executed, nullptr);
        if (!std::isfinite(profiled_cost) || profiled_cost <= 0.0) return;
        const double ratio = std::clamp(
            measured_us / profiled_cost,
            kCostScaleMin, kCostScaleMax);
        CostState & state = cost_states_[executed];
        update_ema(state.scale, state.observations, ratio,
                   config_.cost_ema_alpha);
    }

    // Record a cold evaluation failure without inventing a score. The
    // request remains AR in every later cohort because it cannot be ranked.
    bool record_evaluation_failure(uint64_t request_id) {
        RequestState & state = request_states_[request_id];
        if (state.evaluation_failed ||
            std::isfinite(state.initial_score)) return false;
        state.evaluation_failed = true;
        return true;
    }

    void forget(uint64_t request_id) {
        request_states_.erase(request_id);
    }

    bool has_state(uint64_t request_id) const {
        return request_states_.find(request_id) != request_states_.end();
    }
    bool has_score(uint64_t request_id) const {
        auto state = request_states_.find(request_id);
        return state != request_states_.end() &&
               std::isfinite(state->second.initial_score);
    }
    bool evaluation_failed(uint64_t request_id) const {
        auto state = request_states_.find(request_id);
        return state != request_states_.end() &&
               state->second.evaluation_failed;
    }
    double initial_score(uint64_t request_id) const {
        auto state = request_states_.find(request_id);
        return state == request_states_.end()
            ? std::numeric_limits<double>::quiet_NaN()
            : state->second.initial_score;
    }
    std::string initial_score_kind(uint64_t request_id) const {
        auto state = request_states_.find(request_id);
        return state == request_states_.end()
            ? kUnspecifiedScoreKind : state->second.score_kind;
    }
    const std::vector<double> & initial_hazards(uint64_t request_id) const {
        static const std::vector<double> empty;
        auto state = request_states_.find(request_id);
        return state == request_states_.end()
            ? empty : state->second.conditional_hazards;
    }
    const SpecCostTables & costs() const { return costs_; }

private:
    CandidateScore score_candidate(const SpecCandidate & candidate) {
        bool accepted_initial_score = false;
        if (std::isfinite(candidate.activation_yield)) {
            const double raw = std::clamp(
                candidate.activation_yield, 1.0,
                static_cast<double>(max_accept_));
            RequestState & state = request_states_[candidate.request_id];
            if (!std::isfinite(state.initial_score)) {
                state.initial_score = raw;
                state.score_kind = candidate.score_kind;
                state.conditional_hazards = candidate.conditional_hazards;
                accepted_initial_score = true;
            }
            return {
                state.initial_score,
                accepted_initial_score ? SpecScoreSource::Fresh
                                       : SpecScoreSource::Initial,
                state.score_kind,
            };
        }
        auto state = request_states_.find(candidate.request_id);
        if (state != request_states_.end() &&
            std::isfinite(state->second.initial_score)) {
            return {
                state->second.initial_score,
                SpecScoreSource::Initial,
                state->second.score_kind,
            };
        }
        return {1.0, SpecScoreSource::Unavailable,
                kUnspecifiedScoreKind};
    }

    static void update_ema(double & value, uint64_t & observations,
                           double sample, double alpha) {
        value = observations == 0
            ? sample : (1.0 - alpha) * value + alpha * sample;
        ++observations;
    }

    double cost_scale(const SpecExecutionShape & shape) const {
        auto state = cost_states_.find(shape);
        return state == cost_states_.end() || state->second.observations == 0
            ? 1.0 : state->second.scale;
    }

    double profile_execution_shape(const SpecExecutionShape & shape,
                                   SpecPlan * plan) const {
        double cost = 0.0;
        auto add = [&](const char * name, const SpecCostLookup & lookup) {
            if (plan) report_clamp(name, lookup, *plan);
            cost += lookup.cost;
        };

        if (shape.admitted_count > 0) {
            add("tree", costs_.tree_cost.lookup(shape.tree_rows));
            if (!direct_commit_) {
                add("step", costs_.step_cost.lookup(shape.step_rows));
            }
        } else {
            add("step", costs_.step_cost.lookup(shape.step_rows));
        }
        if (shape.draft_lanes > 0) {
            add("draft", costs_.draft_cost.lookup(shape.draft_lanes));
        }
        return cost;
    }

    CostPrice price_execution_shape(const SpecExecutionShape & shape,
                                    SpecPlan * plan) const {
        const double profiled = profile_execution_shape(shape, plan);
        return {profiled, profiled * cost_scale(shape)};
    }

    CostPrice price_expected_shape(SpecExecutionShape shape,
                                   double expected_step_rows,
                                   SpecPlan * plan) const {
        if (!std::isfinite(expected_step_rows) || expected_step_rows < 0.0)
            return {};

        // The executor can only launch an integer row count, while the gate
        // owns an expected count. Price that expectation continuously across
        // the two neighboring executable shapes so nonlinear profile cliffs
        // retain their cost without an lround() decision discontinuity. Apply
        // each neighbor's own online correction before interpolating it.
        const int lower_rows =
            static_cast<int>(std::floor(expected_step_rows));
        const int upper_rows =
            static_cast<int>(std::ceil(expected_step_rows));
        shape.step_rows = lower_rows;
        const CostPrice lower = price_execution_shape(shape, plan);
        if (lower_rows == upper_rows) return lower;

        shape.step_rows = upper_rows;
        const CostPrice upper = price_execution_shape(shape, plan);
        const double upper_weight = expected_step_rows - lower_rows;
        return {
            lower.profiled + upper_weight * (upper.profiled - lower.profiled),
            lower.predicted +
                upper_weight * (upper.predicted - lower.predicted),
        };
    }

    void report_clamp(const char * name, const SpecCostLookup & lookup,
                      SpecPlan & plan) const {
        if (!lookup.clamped) return;
        plan.cost_lookup_clamped = true;
        if (clamp_logger_)
            clamp_logger_(name, lookup.requested_index, lookup.profiled_index);
    }

    SpecGateConfig config_;
    SpecCostTables costs_;
    SpecStepGeometry geometry_;
    int max_accept_ = 1;
    bool direct_commit_ = false;
    static constexpr double kCostScaleMin = 0.25;
    static constexpr double kCostScaleMax = 4.0;
    std::unordered_map<uint64_t, RequestState> request_states_;
    std::unordered_map<SpecExecutionShape, CostState, ExecutionShapeHash>
        cost_states_;
    ClampLogger clamp_logger_;
};

}  // namespace dflash::common
