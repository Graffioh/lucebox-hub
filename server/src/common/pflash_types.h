#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace luce::common {

inline constexpr size_t kPFlashMaxInstructionSpans = 64;

// Half-open token range in the drafter-tokenized prompt.
struct PFlashTokenSpan {
    int begin = 0;
    int end = 0;
};

inline bool operator==(
        const PFlashTokenSpan & left,
        const PFlashTokenSpan & right) noexcept {
    return left.begin == right.begin && left.end == right.end;
}

inline bool operator!=(
        const PFlashTokenSpan & left,
        const PFlashTokenSpan & right) noexcept {
    return !(left == right);
}

// PFLASH_SELECT_QUERY_PROBE=1: the content spans of the human user
// messages (role "user" after normalisation; tool results are role "tool"),
// latest first, at most two; ``error`` says why the server could not map
// them (the request then falls back to today's windows).
// PFLASH_SELECT_QUERY_IDENT=heur travels with them: the
// identifier tokens of the last turns (ascending positions), the turns read,
// and why the server found none it could map (``ident_error``).
struct PFlashRequestMessages {
    std::vector<PFlashTokenSpan> human;
    std::string error;
    std::vector<int> ident_tokens;
    int ident_turns = 0;
    std::string ident_error;
};

} // namespace luce::common
