#pragma once

// Model-neutral value types for adaptive speculative planning.
//
// A concrete speculator owns proposal generation and durable state. These
// types only describe request identity, optional confidence scouting, and the
// verifier work shapes that the concrete executor can offer for each request.
// Ordinary autoregressive decoding is deliberately implicit in every menu.

#include "speculation_confidence.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace dflash::common {

struct SpeculationRequestView {
    // Stable across slot reuse and therefore suitable for confidence caches.
    std::uint64_t request_id = 0;
    // Engine-local execution handle. It must not be used as a cache key.
    int slot = -1;
    std::int32_t seed_token = -1;
    int progress_tokens = 0;
    // Always/forced speculation is represented per request. A work selector
    // must not choose a shape that omits any required request.
    bool required = false;
};

// Adapter-stable identity for one confidence-scout execution shape. The
// adapter owns shape_id (for example DDTree top-k projection or a DSpark
// confidence-head executor); candidate_tokens keeps different decode depths
// from contaminating the same timing profile.
struct ConfidenceScoutWorkKey {
    SpeculatorKind speculator = SpeculatorKind::DDTree;
    std::uint32_t shape_id = 0;
    int candidate_tokens = 0;

    bool valid() const { return candidate_tokens > 0; }

    bool operator==(const ConfidenceScoutWorkKey & other) const {
        return speculator == other.speculator &&
            shape_id == other.shape_id &&
            candidate_tokens == other.candidate_tokens;
    }

    bool operator!=(const ConfidenceScoutWorkKey & other) const {
        return !(*this == other);
    }

    bool operator<(const ConfidenceScoutWorkKey & other) const {
        if (speculator != other.speculator) {
            return static_cast<std::uint8_t>(speculator) <
                static_cast<std::uint8_t>(other.speculator);
        }
        if (shape_id != other.shape_id) return shape_id < other.shape_id;
        return candidate_tokens < other.candidate_tokens;
    }
};

enum class ConfidenceScoutStatus : std::uint8_t {
    Ready,
    // The speculator can expose confidence only after generating a proposal.
    ProposalRequired,
    // A transient setup/compute failure. The caller may preserve service with
    // AR and retry according to its normal calibration policy.
    RetryableFailure,
    Ineligible,
};

struct ConfidenceScoutRequest {
    SpeculationRequestView request;
    ConfidenceScoutWorkKey work;
    // Conditional candidate positions requested, excluding the mandatory
    // root/AR token. A concrete adapter may return a shorter estimate.
    int max_candidate_tokens = 0;

    bool valid() const {
        return work.valid() && max_candidate_tokens > 0 &&
            max_candidate_tokens <= work.candidate_tokens;
    }
};

struct ConfidenceScoutResult {
    std::uint64_t request_id = 0;
    ConfidenceScoutStatus status = ConfidenceScoutStatus::RetryableFailure;
    SpeculationConfidenceEstimate confidence;
    // Exact request-local time when the adapter executes scouts separately.
    // A fused adapter may leave this zero and report only batch elapsed_us.
    double elapsed_us = 0.0;

    bool ready() const {
        return status == ConfidenceScoutStatus::Ready &&
            confidence.available();
    }
};

struct ConfidenceScoutBatchResult {
    // One entry per request, in input order. request_id makes association
    // explicit even when a concrete implementation partially fails.
    std::vector<ConfidenceScoutResult> requests;
    // Extra scouting time is intentionally not part of a verifier route cost.
    // It may instead inform calibration cadence and cold-start accounting.
    double elapsed_us = 0.0;
};

// Stable structural identity for one executor/profile shape. shape_id is
// scoped by speculator; proposal_nodes and verifier_rows stay separate because
// a branching tree can verify many rows while exposing a much shorter path.
struct VerifierWorkKey {
    SpeculatorKind speculator = SpeculatorKind::DDTree;
    std::uint32_t shape_id = 0;
    int proposal_nodes = 0;
    int verifier_rows = 0;

    bool valid() const {
        return proposal_nodes > 0 && verifier_rows > 0;
    }

    bool operator==(const VerifierWorkKey & other) const {
        return speculator == other.speculator &&
            shape_id == other.shape_id &&
            proposal_nodes == other.proposal_nodes &&
            verifier_rows == other.verifier_rows;
    }

    bool operator!=(const VerifierWorkKey & other) const {
        return !(*this == other);
    }

    bool operator<(const VerifierWorkKey & other) const {
        if (speculator != other.speculator) {
            return static_cast<std::uint8_t>(speculator) <
                static_cast<std::uint8_t>(other.speculator);
        }
        if (shape_id != other.shape_id) return shape_id < other.shape_id;
        if (proposal_nodes != other.proposal_nodes) {
            return proposal_nodes < other.proposal_nodes;
        }
        return verifier_rows < other.verifier_rows;
    }
};

struct VerifierWorkPlan {
    VerifierWorkKey work;
    // Maximum useful output including the ordinary root token. This cannot be
    // inferred from verifier_rows for branching proposals.
    double maximum_emitted_tokens = 1.0;
    // Raw request-local confidence for this exact work budget. NaN means the
    // plan is executable but currently uncalibrated.
    double confidence_expected_tokens =
        std::numeric_limits<double>::quiet_NaN();
    // Forced speculation uses the adapter's preferred executable plan when no
    // measured adaptive decision is available.
    bool preferred_for_forced_mode = false;
    // Hard executor capacity. No route, including Always/required requests,
    // may exceed this bound.
    int max_parallel_requests = std::numeric_limits<int>::max();
    // Efficiency/policy bound for adaptive peers. Required requests may exceed
    // this value when the hard executor capacity still permits them.
    int adaptive_request_limit = std::numeric_limits<int>::max();
    // Lower values are explored first. Adapters should place cheap/short work
    // before wider work so wider shapes are reached only after cheaper shapes
    // have a complete losing profile.
    std::uint32_t exploration_priority = 0;

    bool has_confidence() const {
        return std::isfinite(confidence_expected_tokens) &&
            confidence_expected_tokens >= 1.0;
    }

    double bounded_confidence_expected_tokens() const {
        return has_confidence()
            ? std::clamp(
                  confidence_expected_tokens, 1.0,
                  maximum_emitted_tokens)
            : std::numeric_limits<double>::quiet_NaN();
    }

    bool valid() const {
        return work.valid() && std::isfinite(maximum_emitted_tokens) &&
            maximum_emitted_tokens >= 1.0 && max_parallel_requests > 0 &&
            adaptive_request_limit > 0 &&
            adaptive_request_limit <= max_parallel_requests;
    }
};

struct RequestVerifierWorkMenu {
    SpeculationRequestView request;
    // AR is implicit. Empty means this request has no executable speculative
    // plan at the current position/capacity.
    std::vector<VerifierWorkPlan> speculative;
};

}  // namespace dflash::common
