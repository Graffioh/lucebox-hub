// PFlash fold selection (PFLASH_SELECT_STRATEGY=fold): anchor + structural
// completion over the probe segments.
//
//   1. Structure. The drafter prompt is decoded and cut into message bodies
//      (ChatML markers); inside each body a light line/brace scanner finds
//      code blocks (Python-like def/class by indentation, C-like
//      function/type bodies by brace matching) and text blocks (numbered
//      records such as "Document N:", else markdown sections, else
//      blank-line paragraphs).
//   2. Route. A prompt is code-like when most of its non-blank lines sit
//      inside code blocks; JSON, markdown and prose have none.
//   3. Select. Anchors are the optional segments ranked by mass (density x
//      length) on code-like prompts, by density otherwise. Each anchor is
//      completed to its smallest enclosing block; a block over ``cap``
//      tokens keeps its first ``head`` tokens plus the anchor. The top ``k``
//      distinct completions are kept under the token budget, mandatory
//      segments always, in document order.
//
// Limits of the structure scanner, by design (no parser, no grammar):
//   - Python blocks end at the first non-comment line at or left of the
//     header's indentation; a bracket left open past a blank line followed
//     by a column-0 line is treated as closed.
//   - C-like blocks need a balanced brace; a signature is recognised by its
//     shape (a type keyword, a function keyword, or ``name(...)`` at bracket
//     depth 0 with no assignment), so macros that expand to braces, K&R
//     declarations and brace-less bodies are missed; lambdas and callbacks
//     inside a call's parentheses are not blocks.
//   - Content parts of one message are joined with no separator by the
//     chat template and cannot be told apart in the text.

#pragma once

#include "pflash_selection.h"
#include "common/pflash_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::common { class Tokenizer; }

namespace luce::pflash {

// Byte range [begin, end) of a text.
struct PFlashTextSpan {
    size_t begin = 0;
    size_t end = 0;
};

inline bool operator==(const PFlashTextSpan & a, const PFlashTextSpan & b) noexcept {
    return a.begin == b.begin && a.end == b.end;
}

// Code-like: at least this share of the non-blank lines inside code blocks.
inline constexpr double kPFlashFoldCodeLineShare = 0.5;
// Paragraph fallback: a blank-line block longer than this (bytes, ~1500
// tokens) is split into its lines.
inline constexpr size_t kPFlashFoldParagraphBytes = 6000;

// Message bodies: the text between ChatML markers (<|im_start|>role\n ...
// <|im_end|>), or the whole text when it has none.
std::vector<PFlashTextSpan> pflash_message_bodies(const std::string & text);

// Function/method/class bodies, each from its header line (decorators and
// annotations included) through its last line; nested blocks are reported
// too. Python-like indentation blocks and C-like brace blocks.
std::vector<PFlashTextSpan> pflash_python_blocks(
    const std::string & text, PFlashTextSpan body);
std::vector<PFlashTextSpan> pflash_brace_blocks(
    const std::string & text, PFlashTextSpan body);

// Text blocks of one body outside ``code``: the widest numbered header
// family ("Document 3:", "Passage 12.", "[4]": a capitalised word or two, a
// space, a number, then ':', '.', ')', ']', '-' or the line end; at least
// three distinct numbers), each record from its header to the next; else,
// in a body whose code (if any) sits in markdown fences, markdown sections,
// else blank-line paragraphs with each fenced example one block
// (``paragraphs``; PFLASH_SELECT_FOLD_PARAGRAPHS=0 leaves such a body with
// no blocks, as the offline reference had).
// A body that is mostly code gets none. ``source`` names the kind found.
std::vector<PFlashTextSpan> pflash_text_blocks(
    const std::string & text, PFlashTextSpan body,
    const std::vector<PFlashTextSpan> & code, const char ** source,
    bool paragraphs = true);

struct PFlashFoldStructure {
    std::vector<PFlashTextSpan> blocks;   // code and text blocks, may nest
    size_t code_blocks = 0;
    int nonblank_lines = 0;
    int code_lines = 0;                   // non-blank lines inside a code block
    bool code_like = false;
    const char * text_source = "none";    // records, headings, paragraphs, none
};

PFlashFoldStructure pflash_fold_structure(const std::string & text, bool paragraphs = true);

// Query blocks of a user message (PFLASH_SELECT_QUERY_HEAD=1): where the
// user's own prose sits around pasted material.
//   head  the leading prose block, from the message start to the first
//         paste-like line (a code fence, a markdown heading, a separator
//         rule, a record/file header, a tag, JSON/table data, an indented or
//         code-shaped line), trimmed; empty when the message opens with
//         pasted material. Without any paste-like line, a short first
//         paragraph ending in '?' before a text at least eight times longer
//         is the head too (a question before a plain prose paste).
//   tail  the last blank-line paragraph when it follows the last paste-like
//         line and has none itself; empty otherwise.
// ``structured`` says a paste-like line was found; without one only the
// '?' head rule applies and the tail stays empty. A block holding more than
// half the text is not a question block and is left empty.
struct PFlashQueryBlocks {
    bool structured = false;
    PFlashTextSpan head;
    PFlashTextSpan tail;
    const char * head_rule = "none";   // paste, question, none
};

PFlashQueryBlocks pflash_query_blocks(const std::string & text, PFlashTextSpan message);

// Token span covering byte span [begin, end): from the token holding
// ``begin`` through the token holding ``end - 1``. ``token_begin`` holds
// each token's first byte, ascending.
luce::common::PFlashTokenSpan pflash_text_to_token_span(
    const std::vector<size_t> & token_begin, PFlashTextSpan span);

enum class PFlashFoldRank { Density, Mass };

struct PFlashFoldPolicy {
    int token_budget = 0;
    int k = 20;
    int cap = 1500;
    int head = 150;
    PFlashFoldRank rank = PFlashFoldRank::Density;
};

struct PFlashFoldResult {
    bool ok = false;
    std::vector<luce::common::PFlashTokenSpan> kept;   // merged, ascending
    std::vector<size_t> anchors;          // ordinals of the anchors taken
    int retained_tokens = 0;
    int skipped = 0;                      // completions over the budget
    int capped = 0;                       // completions of a block over cap
    int contained = 0;                    // anchors inside some block
    PFlashSelectionStop stop = PFlashSelectionStop::InvalidInput;
    std::string error;
};

// Blocks in token coordinates for the selector: blocks never cross a
// mandatory candidate (a block ends where one begins and starts after one
// it begins in); empty spans are dropped.
std::vector<luce::common::PFlashTokenSpan> pflash_fold_token_blocks(
    const std::vector<PFlashTextSpan> & blocks,
    const std::vector<size_t> & token_begin,
    const std::vector<PFlashSelectionCandidate> & candidates);

// The fold rule over candidates (contiguous probe segments with density
// scores) and token blocks.
PFlashFoldResult select_pflash_fold(
    const std::vector<PFlashSelectionCandidate> & candidates,
    const std::vector<luce::common::PFlashTokenSpan> & blocks,
    const PFlashFoldPolicy & policy);

// What the drafter runs: decode ``ids`` with ``vocab``, find the structure,
// route, and select. ``info`` (optional) receives the structure.
PFlashFoldResult select_pflash_fold_for_ids(
    const luce::common::Tokenizer & vocab,
    const std::vector<int32_t> & ids,
    const std::vector<PFlashSelectionCandidate> & candidates,
    const PFlashSelectionConfig & config,
    int token_budget,
    PFlashFoldStructure * info = nullptr,
    std::vector<luce::common::PFlashTokenSpan> * token_blocks = nullptr);

// Adaptive selection (PFLASH_SELECT_STRATEGY=adaptive): every structural
// level of the prompt, nested, per ChatML message body:
//   records     from a record header to the next one: the widest numbered
//               family ("Document 3:", at least three numbers in sequence
//               from 0-2, outside code and headings; not caption or
//               reference families such as "Figure 2:", "Table 3.",
//               "[12] Author"; the family's label without a number, as in
//               "Document: <title>", starts a record too), "File: <path>"
//               lines, and whole-line "[label]" lines; the text before the
//               first header is not a record
//   (levels never cross a ChatML message boundary; markers quoted inside a
//   message do not count as boundaries)
//   code        Python-like def/class and C-like function/type blocks
//               (inside a "File:" record, nothing else is looked for)
//   fences      markdown ```/~~~ fenced blocks
//   sections    markdown headings outside fences and code, nested (a heading
//               runs to the next heading of its level or above)
//   turns       "User:" / "Assistant:" / "Human:" / "AI:" line starts, two or
//               more in a record or a leaf section, each to the next
//   paragraphs  blank-line blocks up to 6000 bytes (a longer block gives its
//               lines up to 6000 bytes)
// Spans are [line start, block end) in bytes; duplicates removed.
std::vector<PFlashTextSpan> pflash_adaptive_levels(const std::string & text);

// Levels in token coordinates: [first token starting at or after begin,
// first token starting at or after end); empty spans dropped, duplicates
// removed.
std::vector<luce::common::PFlashTokenSpan> pflash_adaptive_token_levels(
    const std::vector<PFlashTextSpan> & levels, const std::vector<size_t> & token_begin);

struct PFlashAdaptivePolicy {
    int token_budget = 0;
    int k = 20;
    double tau = 0.95;
    int cap = 1500;
    int head = 150;
};

// Adaptive granularity by score concentration. Anchors are the optional
// candidates by density (ordinal tie-break). An anchor already fully kept is
// skipped. From the anchor's own segment the region climbs its chain of
// enclosing levels (each strictly longer than the one below): with
// mass(level) = sum of density x length over the optional candidates whose
// midpoint lies in it, the climb stops when mass(current) / mass(next) >= tau
// (or the next level has no mass); a next level over ``cap`` tokens is
// never taken whole -- the region becomes the current one plus that level's
// first ``head`` tokens, and the climb stops. A region whose new tokens do
// not fit the budget is skipped; ``k`` regions are kept, mandatory
// candidates always. ``anchors`` lists the anchors taken, ``contained`` how
// many were promoted above their segment, ``capped`` how many hit the cap.
PFlashFoldResult select_pflash_adaptive(
    const std::vector<PFlashSelectionCandidate> & candidates,
    const std::vector<luce::common::PFlashTokenSpan> & levels,
    const PFlashAdaptivePolicy & policy);

PFlashFoldResult select_pflash_adaptive_for_ids(
    const luce::common::Tokenizer & vocab,
    const std::vector<int32_t> & ids,
    const std::vector<PFlashSelectionCandidate> & candidates,
    const PFlashSelectionConfig & config,
    int token_budget,
    std::vector<luce::common::PFlashTokenSpan> * token_levels = nullptr);

} // namespace luce::pflash
