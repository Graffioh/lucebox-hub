// Model-agnostic speculation adapter contract.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace dflash::common {

enum SpeculatorInputRequirement : uint32_t {
    SpeculatorInputNone = 0,
    SpeculatorInputHidden = 1u << 0,
    SpeculatorInputPrenorm = 1u << 1,
};

struct ActivationEstimate {
    double expected_yield = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> conditional_hazards;
};

struct SpeculatorBatchInput {
    int lane_count = 0;
    int requested_depth = 0;
    std::vector<const float *> hidden_by_lane;
    std::vector<const float *> prenorm_by_lane;
    std::vector<int32_t> seed_tokens;
};

struct SpecProposal {
    std::vector<int32_t> tokens;
    ActivationEstimate estimate;
    std::string error;
    // Optional per-depth JSON field fragments used only by debug telemetry.
    std::vector<std::string> debug_depth_fields;
};

inline bool speculator_input_satisfies(
        const SpeculatorBatchInput & input, uint32_t requirements) {
    if (input.lane_count <= 0 || input.requested_depth < 2 ||
        static_cast<int>(input.seed_tokens.size()) != input.lane_count) {
        return false;
    }
    auto has_lanes = [&](const std::vector<const float *> & lanes) {
        if (static_cast<int>(lanes.size()) != input.lane_count) return false;
        for (const float * lane : lanes) {
            if (!lane) return false;
        }
        return true;
    };
    if ((requirements & SpeculatorInputHidden) != 0 &&
        !has_lanes(input.hidden_by_lane)) {
        return false;
    }
    if ((requirements & SpeculatorInputPrenorm) != 0 &&
        !has_lanes(input.prenorm_by_lane)) {
        return false;
    }
    return true;
}

class Speculator {
public:
    virtual ~Speculator() = default;

    // Opaque, versioned identity. The activation engine never enumerates it.
    virtual const std::string & score_kind() const = 0;
    virtual int max_block_size() const = 0;
    virtual uint32_t input_requirements() const = 0;
    virtual bool ready() const = 0;
    virtual const std::string & error() const = 0;

    // Draft and score each lane in one adapter call. A false return means the
    // whole batch failed; lane-local failures use SpecProposal::error.
    virtual bool propose(const SpeculatorBatchInput & input,
                         std::vector<SpecProposal> & output) = 0;
};

inline constexpr const char * kNoSpeculatorAdapterReason =
    "no_speculator_adapter";

inline bool speculator_is_ready(const Speculator * speculator) {
    return speculator != nullptr && speculator->ready();
}

inline const char * speculator_fallback_reason(const Speculator * speculator) {
    return speculator_is_ready(speculator)
        ? nullptr : kNoSpeculatorAdapterReason;
}

}  // namespace dflash::common
