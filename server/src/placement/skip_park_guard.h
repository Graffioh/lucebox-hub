// Skip-park guard + startup policy resolution for PFlash compression.
//
// During PFlash compression the server normally parks the resident target (and
// decode draft), loads the scoring drafter, compresses, frees the drafter and
// reloads everything. When VRAM is ample the park/unpark round-trip is pure
// overhead: the drafter can coexist with the resident models for the whole
// compression window ("skip park").
//
// This header owns the policy, not the mechanics:
//   - SkipParkMode: the requested policy (--prefill-skip-park auto|on|off)
//   - skip_park_allowed: hardware crash guard (<32GiB with ctx>64K)
//   - resolve_skip_park: the single startup decision (auto estimate vs. free
//     VRAM with margin, with explicit on/off precedence)
//
// The per-drafter footprint inputs come from inspect_drafter_footprint()
// (common/gguf_inspect.h), which reads dims from the drafter GGUF header and
// mirrors the buffers the scorers actually allocate (qwen35_drafter.cpp,
// qwen3_graph.cpp).

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace luce::common {

// Requested policy for --prefill-skip-park. Bare `--prefill-skip-park` (no
// value) keeps the historical boolean meaning = On. Auto is the default:
// the startup estimate decides; On/Off are explicit operator overrides that
// bypass the estimate (the hardware guard still applies to On).
enum class SkipParkMode { Auto, On, Off };

inline const char * skip_park_mode_name(SkipParkMode mode) {
    switch (mode) {
    case SkipParkMode::Auto: return "auto";
    case SkipParkMode::On:   return "on";
    case SkipParkMode::Off:  return "off";
    }
    return "auto";
}

inline bool parse_skip_park_mode(const std::string & value, SkipParkMode & out) {
    if (value == "auto") { out = SkipParkMode::Auto; return true; }
    if (value == "on")   { out = SkipParkMode::On;   return true; }
    if (value == "off")  { out = SkipParkMode::Off;  return true; }
    return false;
}

// Hardware crash guard: skipping park on small cards with large contexts has
// produced cuMemSetAccess-style failures. Applies to Auto and On alike — a
// forced-on that reliably crashes is worse than ignoring the request.
inline bool skip_park_allowed(bool requested, size_t total_vram_bytes, int max_ctx) {
    return requested &&
           (total_vram_bytes >= 32ull*1024*1024*1024 || max_ctx <= 65536);
}

// Worst-case incremental VRAM while the PFlash drafter scores a window, on top
// of the resident target + decode draft it skips parking for. Populated by
// inspect_drafter_footprint() from the drafter GGUF header.
struct SkipParkDrafterInfo {
    bool    recognized = false;           // false → auto resolves off
    int64_t weights_bytes = 0;            // GGUF file size (upper bound)
    int64_t runtime_bytes_per_token = 0;  // KV + activations + score buffers
    int64_t fixed_bytes = 0;              // SSM/conv state, gallocr, chunk transients
    int     context_length = 0;           // drafter native ctx — caps the window
};

// Headroom multiplier applied to the whole footprint. Covers fragmentation,
// gallocr slack and per-arch details the estimator intentionally ignores.
constexpr int64_t kSkipParkSafetyMarginPercent = 25;

inline int64_t skip_park_required_bytes(const SkipParkDrafterInfo & info,
                                        int64_t window_tokens) {
    const int64_t raw = info.weights_bytes + info.fixed_bytes +
                        info.runtime_bytes_per_token * window_tokens;
    return raw + raw * kSkipParkSafetyMarginPercent / 100;
}

struct SkipParkDecision {
    bool        enabled = false;
    std::string reason;                    // short human-readable log token
    int64_t     required_bytes = 0;        // footprint incl. margin (auto only)
    int64_t     window_tokens = 0;         // min(max_ctx, drafter ctx)
};

// The one authoritative decision, resolved once at startup:
//   off  → never skip park
//   on   → skip park unless the hardware guard blocks it
//   auto → skip park iff the guard passes AND the estimated footprint fits
//          measured free VRAM with margin
inline SkipParkDecision resolve_skip_park(
        SkipParkMode mode, bool drafter_configured,
        const SkipParkDrafterInfo & info,
        int64_t free_vram_bytes, int64_t total_vram_bytes, int64_t max_ctx) {
    SkipParkDecision d;
    if (mode == SkipParkMode::Off) {
        d.reason = "off (explicit)";
        return d;
    }
    if (!drafter_configured) {
        d.reason = "off (no local pflash drafter)";
        return d;
    }
    if (!skip_park_allowed(true, size_t(std::max<int64_t>(total_vram_bytes, 0)),
                           int(std::min<int64_t>(max_ctx, INT32_MAX)))) {
        d.reason = "off (guard: <32GiB VRAM with ctx>64K)";
        return d;
    }
    if (mode == SkipParkMode::On) {
        d.enabled = true;
        d.reason = "on (explicit)";
        return d;
    }
    if (!info.recognized || info.context_length <= 0 ||
        info.runtime_bytes_per_token <= 0 || free_vram_bytes < 0) {
        d.reason = "off (auto: drafter footprint unknown)";
        return d;
    }
    d.window_tokens = std::min<int64_t>(max_ctx, info.context_length);
    d.required_bytes = skip_park_required_bytes(info, d.window_tokens);
    d.enabled = d.required_bytes <= free_vram_bytes;
    d.reason = d.enabled
        ? "on (auto: estimate fits free VRAM)"
        : "off (auto: estimate exceeds free VRAM)";
    return d;
}

}  // namespace luce::common
