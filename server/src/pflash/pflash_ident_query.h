// EVALUATION REFERENCE ARM ONLY -- not a product path (user rule: no regex or
// fixed behaviour in the product). PFLASH_EVAL_REF_IDENT_QUERY=heur (strict
// selection with the head scorer) replaces today's query windows with the
// identifiers written in the conversation's last 3 non-system turns, pooled
// as ONE query, scored with today's key mask (identifier tokens are not
// removed from the keys) and NOT pinned; today's pins are unchanged. It is the
// runtime form of the agent-cue diagnostic's identifier-heuristic arm (viii),
// the reference a learned request probe must match or beat on agent turns.
//
// Identifiers = ident_v2 (luce_box lucebox_training/pflash/src/pflash_scorer/
// request_probe/identifiers.py, IDENTIFIER_VERSION "ident_v2", 06fd8041):
// the SET of spans (>= 4 characters) of, left to right with Python ``re``
// semantics (ASCII word characters; bytes >= 0x80 are non-word -- parity is
// defined on ASCII text):
//   ident_v1   [\w./-]*[A-Za-z_]\w*\.[A-Za-z]{1,6}\b
//              | \b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b
//              | \b[a-z]\w*_\w+\b | \b[a-z]+[A-Z]\w*\b | \b[A-Z][a-z]+[A-Z]\w*\b
//              | \b[A-Za-z_]\w{3,}(?=\()
//   PATH_V2    (?:\.{1,2}/|/)?[\w.-]+(?:/[\w.-]+)+/? | /[\w.-]+/?
//   UNDERSCORE (?<![\w.])_{1,2}[A-Za-z0-9]\w*
//   DOTTED_V2  \.?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\.?  plus every [A-Za-z_]\w*
//              segment of each match
// plus every whole line containing Error, Exception or Traceback (>= 8
// characters once stripped). A turn's text is its rendered content with the
// vocabulary's added (control / user-defined) tokens blanked, so template
// markup such as <tool_call> never matches.

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
    Heur,       // evaluation reference: the identifier tokens alone (arm viii)
};

enum class PFlashIdentVersion { V1, V2 };      // ident_v1 (pre-registered label), ident_v2 (current)

// "off" | "heur".
const char * pflash_ident_query_name(PFlashIdentQuery mode) noexcept;

// Character spans [begin, end) of the identifiers and error lines of
// ``text``: the sorted, deduplicated set over every rule of ``version``.
std::vector<std::pair<size_t, size_t>> pflash_identifier_char_spans(
    const std::string & text, PFlashIdentVersion version = PFlashIdentVersion::V2);

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
