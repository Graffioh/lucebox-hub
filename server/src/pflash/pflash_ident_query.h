// Identifier query (PFLASH_SELECT_QUERY_IDENT=heur, strict selection with the
// head scorer): the scorer query is the identifiers written in the
// conversation's last turns, alone, instead of today's query windows.
// It is the runtime form of arm (viii) of the round-10 agent-cue diagnostic
// (luce_box 99c91fc1, scripts/agent_decay/cues_rows.py, "heur"): every
// identifier in the LAST 3 non-system turns, pooled as ONE query, scored
// with today's key mask (identifier tokens are not removed from the keys)
// and NOT pinned; today's pins are unchanged.
//
// Identifiers (the diagnostic's regular expressions, matched left to right
// with Python ``re`` semantics; ASCII word characters, bytes >= 0x80 count as
// non-word):
//   paths / file names      [\w./-]*[A-Za-z_]\w*\.[A-Za-z]{1,6}\b
//   dotted symbols          \b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b
//   snake_case              \b[a-z]\w*_\w+\b
//   camelCase / CapWords    \b[a-z]+[A-Z]\w*\b | \b[A-Z][a-z]+[A-Z]\w*\b
//   call names              \b[A-Za-z_]\w{3,}(?=\()
// kept when >= 4 characters, plus every whole line containing Error,
// Exception or Traceback (>= 8 characters once stripped). A turn's text is
// its rendered content with the vocabulary's added (control / user-defined)
// tokens blanked, so template markup such as <tool_call> never matches.

#pragma once

#include "common/pflash_types.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace luce::pflash {

using luce::common::PFlashTokenSpan;

inline constexpr int kIdentQueryTurns = 3;      // last turns read
inline constexpr int kIdentMinChars = 4;        // identifier matches
inline constexpr int kIdentErrorLineMinChars = 8;

enum class PFlashIdentQuery {
    Off,        // today's windows (default)
    Heur,       // the identifier tokens alone (arm viii)
};

// "off" | "heur".
const char * pflash_ident_query_name(PFlashIdentQuery mode) noexcept;

// Character spans [begin, end) of the identifiers and error lines of
// ``text``, in match order (identifiers first, then error lines).
std::vector<std::pair<size_t, size_t>> pflash_identifier_char_spans(const std::string & text);

// The identifier tokens of ``turns`` (content spans in token positions of a
// prompt whose per-token texts are ``token_texts``; ``markup[t]`` blanks
// token t): every token overlapping an identifier, ascending, deduplicated.
std::vector<int> pflash_ident_query_tokens(
    const std::vector<std::string> & token_texts,
    const std::vector<unsigned char> & markup,
    const std::vector<PFlashTokenSpan> & turns);

// What the identifier query did on one scoring call (trace "ident_query").
struct PFlashIdentQueryTrace {
    PFlashIdentQuery mode = PFlashIdentQuery::Off;
    int turns = 0;                  // turns read
    int ident_tokens = 0;           // identifier tokens found
    int query_tokens = 0;           // scored query tokens (0 on fallback)
    int query_tiles = 0;
    // "" when scored; else why today's windows were scored instead:
    // the server's mapping error, "no_identifiers", "rows_missing".
    std::string fallback;
    std::vector<int> tokens;        // the scored query tokens
};

} // namespace luce::pflash
