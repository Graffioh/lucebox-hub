#pragma once

// User-selectable, speculator-neutral decode policy. Concrete backends map
// this to DDTree, DSpark, or any later speculative implementation.

#include <optional>
#include <string_view>

namespace dflash::common {

enum class SpeculationPolicy {
    Adaptive,
    Always,
    Never,
};

constexpr std::string_view decode_mode_name(
        SpeculationPolicy policy) {
    switch (policy) {
    case SpeculationPolicy::Adaptive: return "adaptive";
    case SpeculationPolicy::Always:   return "speculation";
    case SpeculationPolicy::Never:    return "ar";
    }
    return "adaptive";
}

inline std::optional<SpeculationPolicy> parse_decode_mode(
        std::string_view value) {
    if (value == "adaptive") return SpeculationPolicy::Adaptive;
    if (value == "speculation") return SpeculationPolicy::Always;
    if (value == "ar") return SpeculationPolicy::Never;
    return std::nullopt;
}

}  // namespace dflash::common
