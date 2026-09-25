#pragma once

#include "common/pflash_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::pflash {

enum class PFlashSelectionMode {
    Legacy,
    BudgetOnly,
    CumulativeTopP,
    // Rank rule: keep the K highest-scoring optional candidates, the token
    // budget still a hard ceiling -- min(K segments, the budget).
    TopK,
};

enum class PFlashQueryParser {
    SemanticUser,
    ArbitraryTail,
};

enum class PFlashSelectionStop {
    TopPReached,
    TopKReached,
    BudgetReached,
    CandidatesExhausted,
    InvalidInput,
    MandatoryQueryExceedsBudget,
};

struct PFlashSelectionCandidate {
    size_t ordinal = 0;
    int begin = 0;
    int end = 0;
    double score = 0.0;
    bool mandatory = false;
};

struct PFlashSelectionPolicy {
    int token_budget = 0;
    double top_p = 0.95;
    // Variable-length segments: a candidate that does not fit the remaining
    // budget is skipped instead of ending the fill, so smaller segments
    // ranked below it can still be kept.
    bool skip_oversized = false;
    // TopK mode only: how many optional candidates to keep. Must be positive
    // in that mode and is ignored in the others.
    int top_k = 0;
};

struct PFlashSelectionResult {
    bool ok = false;
    std::vector<size_t> ordinals;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    PFlashSelectionStop stop = PFlashSelectionStop::InvalidInput;
    std::string error;
};

// A chunk is kept whatever its score when it overlaps the query window or a
// required instruction span, or -- with ``query_suffix_structural`` -- any
// token after the query window.
bool pflash_chunk_is_structurally_required(
    int begin,
    int end,
    int query_begin,
    int query_end,
    int input_tokens,
    const std::vector<luce::common::PFlashTokenSpan> &
        required_instruction_spans = {},
    bool query_suffix_structural = true) noexcept;

bool validate_pflash_instruction_spans(
    const std::vector<luce::common::PFlashTokenSpan> & spans,
    int input_tokens,
    std::string & error) noexcept;

PFlashSelectionResult select_pflash_candidates(
    const std::vector<PFlashSelectionCandidate> & candidates,
    const PFlashSelectionPolicy & policy,
    PFlashSelectionMode mode);

const char * pflash_selection_mode_name(PFlashSelectionMode mode) noexcept;
const char * pflash_selection_stop_name(PFlashSelectionStop stop) noexcept;
const char * pflash_query_parser_name(PFlashQueryParser parser) noexcept;

// How the context is cut into candidates and how a candidate is scored.
// ``Auto`` resolves at scoring time: probe segments when a segment probe is
// loaded, fixed chunks otherwise; density with probe segments, sum otherwise.
enum class PFlashSegmentation { Auto, Fixed, Probe };
enum class PFlashCandidateScore { Auto, Sum, Density };
// Which scorer ranks the candidates: the block-15 attention-mass head, the
// original all-layer running-max scorer, or both with a split budget (the
// head fills ``split_fraction`` of the budget first, the other scorer the rest).
enum class PFlashScorer { Head, Legacy, Split };

struct PFlashSelectionConfig {
    PFlashSelectionMode mode = PFlashSelectionMode::Legacy;
    // Chat-first default: the scorer query is the tail of the latest user
    // turn. latest_user stays selectable for benchmark experiments.
    PFlashQueryParser query_parser = PFlashQueryParser::ArbitraryTail;
    int chunk_size = 0;
    int query_tokens = 8;
    double top_p = 0.95;
    int top_k = 0;
    PFlashSegmentation segmentation = PFlashSegmentation::Auto;
    PFlashCandidateScore candidate_score = PFlashCandidateScore::Auto;
    PFlashScorer scorer = PFlashScorer::Head;
    double split_fraction = 0.5;
    bool configured = false;
    bool selection_active = false;
    // Per request, never from the environment: the tokens after the query
    // window are candidates scored against it instead of a kept suffix (a
    // chat whose latest user turn is followed by assistant and tool turns).
    // The caller pins whatever of that suffix must stay.
    bool query_suffix_candidates = false;
    // Per request: earlier user questions' scorer windows, most recent
    // first. The head scores the context against each and mixes the masses
    // with the query's at weights 1/2, 1/4, ... (multi-turn chats).
    std::vector<luce::common::PFlashTokenSpan> history_queries;
    // Per request: the tail of the latest user turn, scored as a second
    // query window at full weight next to the prompt-end query. The last
    // token reads the whole request; the user's own tokens match literal
    // strings (an identifier, a function description) the last token does
    // not carry.
    luce::common::PFlashTokenSpan turn_query{-1, -1};
    // Assembly experiments, both off by default. Container headers
    // (PFLASH_SELECT_CONTAINER_HEADERS=1) keep the header line of every
    // container (document, record, file) the selection keeps content from;
    // cut markers (PFLASH_ASSEMBLY_CUT_MARKER=1) put kPFlashCutMarkerText
    // between kept ranges that were not adjacent in the prompt. Both are
    // paid for inside the token budget.
    bool container_headers = false;
    int container_header_max = 48;
    bool cut_markers = false;
    // Request-structure headers (PFLASH_SELECT_STRUCT_HEADERS=1): the same
    // header rule and budget refit, with record starts taken from the
    // request itself -- where each message, text content part and tool
    // result begins -- instead of (or, with container headers on too,
    // besides) the probe's record-start head. Off by default.
    bool struct_headers = false;
    // Per request, from the server: those starts, in the drafter's token
    // positions, ascending, none inside a kept span. Empty unless the switch
    // is on and the request has at least two units in the compressed region.
    std::vector<int> struct_starts;
    // Per process, filled by the drafter: the cut marker's token ids.
    std::vector<int32_t> cut_marker_ids;
};

// Text of the cut marker; the drafter tokenizes it once, the server's
// paragraph join inserts it verbatim.
inline constexpr const char * kPFlashCutMarkerText = "\n[...]\n";

// The assembly switches straight from the environment ("1" on; unset or "0"
// off; anything else is off here and a resolution error in
// resolve_pflash_selection).
bool pflash_container_headers_requested() noexcept;
bool pflash_cut_markers_requested() noexcept;
bool pflash_struct_headers_requested() noexcept;

// Container starts from per-token start probabilities (the probe's
// record-start head: document, file, email, session), decoded like unit
// boundaries: a token whose probability is above ``threshold`` (strict, the
// offline threshold-selection rule) starts a container unless it lies within
// ``min_spacing`` tokens of the previous accepted start. Tokens inside any
// ``excluded`` span (the query window, instruction spans, the kept suffix)
// never start one. Ascending.
std::vector<int> pflash_container_starts(
    const std::vector<float> & container_probs,
    int input_tokens,
    float threshold,
    int min_spacing,
    const std::vector<luce::common::PFlashTokenSpan> & excluded);

// Container extents and header spans. Container i runs from its start to the
// next start, the next excluded span or the input end, whichever comes
// first. Its header runs from the start through the first newline-bearing
// token after the start (``newline_vocab[id] != 0``), capped at
// ``header_max`` tokens and at the container end.
struct PFlashContainers {
    std::vector<int> starts;
    std::vector<luce::common::PFlashTokenSpan> extents;
    std::vector<luce::common::PFlashTokenSpan> headers;
    // Where the starts came from, for the trace (set by the caller when
    // request-structure starts are merged in; with none, every start is a
    // probe record start): the probe's record starts and the
    // request-structure starts.
    std::vector<int> record_starts;
    std::vector<int> struct_starts;
};

// Record starts from two sources (the probe's record head, the request's
// structure), merged: ascending, duplicates once.
std::vector<int> pflash_union_starts(const std::vector<int> & a,
                                     const std::vector<int> & b);
PFlashContainers pflash_container_headers(
    const std::vector<int32_t> & ids,
    const std::vector<int> & starts,
    const std::vector<luce::common::PFlashTokenSpan> & excluded,
    const std::vector<uint8_t> & newline_vocab,
    int header_max);

struct PFlashAssemblyPolicy {
    int token_budget = 0;
    // Containers (extents and header spans, parallel, ascending, disjoint);
    // empty when container headers are off.
    const PFlashContainers * containers = nullptr;
    // Cut marker length in tokens; 0 when cut markers are off.
    int marker_tokens = 0;
};

struct PFlashAssemblyResult {
    bool ok = false;
    std::vector<size_t> ordinals;          // final selection, source order
    std::vector<size_t> dropped;           // ordinals dropped to pay, in drop order
    std::vector<luce::common::PFlashTokenSpan> kept;   // merged kept ranges
    std::vector<luce::common::PFlashTokenSpan> headers_added;
    int header_tokens = 0;                 // header tokens not already kept
    int cut_markers = 0;
    int retained_tokens = 0;               // kept tokens plus marker tokens
    int passes = 0;
    std::string error;
};

// After a normal selection: add the header span of every container that
// holds a selected non-mandatory candidate, count a cut marker between every
// two kept ranges that are not adjacent, and pay for both inside the budget
// by dropping the lowest-ranked selected non-mandatory candidates (the
// reverse of the selection order), one per pass. A dropped candidate can
// leave a container without content, whose header then goes too, or remove
// a marker: every pass recomputes both, until the result fits.
// Mandatory candidates are never dropped; when only they remain the result
// may exceed the budget by the markers between them.
PFlashAssemblyResult pflash_assemble_selection(
    const std::vector<PFlashSelectionCandidate> & candidates,
    const std::vector<size_t> & selected_ordinals,
    const PFlashAssemblyPolicy & policy);

// The kept ranges' ids in order, ``marker_ids`` between two ranges that are
// not adjacent (never before the first or after the last).
std::vector<int32_t> pflash_assemble_ids(
    const std::vector<int32_t> & ids,
    const std::vector<luce::common::PFlashTokenSpan> & kept,
    const std::vector<int32_t> & marker_ids);

// Segment probe: cut the context before every token whose boundary score is
// above ``threshold``; ``forced_cuts`` (query start, instruction span edges)
// are always cut; a cut closer than ``min_segment`` tokens to the previous
// accepted cut is dropped unless forced; a span longer than ``max_segment``
// is split at its best-scoring interior token, or evenly when no interior
// token scores above zero. ``split_scores`` (the sub-unit logit when the
// probe artifact carries one) feeds only the oversize interior argmax; when
// empty the unit boundary scores are used. Returns contiguous spans covering
// [0, input_tokens), or an empty vector on invalid input.
std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
    const std::vector<float> & boundary_scores,
    int input_tokens,
    float threshold,
    int min_segment,
    int max_segment,
    const std::vector<int> & forced_cuts,
    const std::vector<float> & split_scores = {});

// Two-scorer selection: ``head`` candidates fill ``head_fraction`` of the
// budget (mandatory candidates first, charged once), then ``other``
// candidates (same spans and ordinals, scored by the other scorer) fill what
// remains, skipping ordinals already selected. Both lists must describe the
// same spans in the same order.
PFlashSelectionResult select_pflash_split(
    const std::vector<PFlashSelectionCandidate> & head,
    const std::vector<PFlashSelectionCandidate> & other,
    const PFlashSelectionPolicy & policy,
    double head_fraction,
    PFlashSelectionMode mode);

const char * pflash_scorer_name(PFlashScorer scorer) noexcept;
const char * pflash_segmentation_name(PFlashSegmentation segmentation) noexcept;
const char * pflash_candidate_score_name(PFlashCandidateScore score) noexcept;

// Presence, rather than validity, gates cache and continuation policy so an
// empty or invalid experiment variable cannot silently fall back to legacy.
bool has_pflash_selection_environment() noexcept;

bool resolve_pflash_selection(
    int input_tokens,
    int legacy_chunk_size,
    PFlashSelectionConfig & out,
    std::string & error);

} // namespace luce::pflash
