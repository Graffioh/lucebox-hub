// Decode policy shared by the HTTP layer, scheduler, and sequence engines.

#pragma once

#include <optional>
#include <string_view>

namespace dflash::common {

enum class SpeculationPolicy {
    Adaptive,
    Always,
    Never,
};

// Runtime capabilities for the concurrent decode path. Forced speculation
// needs an executable draft/verify chain. Adaptive means the server can honor
// the one-shot activation contract; a configured chain may satisfy that by
// recording a request-local sticky-AR fallback when scoring or profiling is
// unavailable. AR is always supported.
struct ConcurrentDecodeCapabilities {
    bool forced_speculation = false;
    bool adaptive = false;

    constexpr bool supports(SpeculationPolicy policy) const {
        switch (policy) {
            case SpeculationPolicy::Always:   return forced_speculation;
            case SpeculationPolicy::Adaptive: return adaptive;
            case SpeculationPolicy::Never:    return true;
        }
        return false;
    }
};

inline const char * speculation_policy_name(SpeculationPolicy policy) {
    switch (policy) {
        case SpeculationPolicy::Adaptive: return "adaptive";
        case SpeculationPolicy::Always:   return "speculation";
        case SpeculationPolicy::Never:    return "ar";
    }
    return "adaptive";
}

inline bool parse_speculation_policy(
        std::string_view value, SpeculationPolicy & policy) {
    if (value == "adaptive") {
        policy = SpeculationPolicy::Adaptive;
        return true;
    }
    if (value == "speculation") {
        policy = SpeculationPolicy::Always;
        return true;
    }
    if (value == "ar") {
        policy = SpeculationPolicy::Never;
        return true;
    }
    return false;
}

inline SpeculationPolicy resolve_speculation_policy(
        SpeculationPolicy server_default,
        const std::optional<SpeculationPolicy> & request_override) {
    return request_override.value_or(server_default);
}

}  // namespace dflash::common
