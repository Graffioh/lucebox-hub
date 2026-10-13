// Relative query candidates and the verbatim request pin (PFlash query redesign, step 1; runtime only, no
// training). Both need the request probe (PFLASH_SELECT_QUERY_PROBE=1, a cand_v5 stage-1 GGUF: the server maps
// every message) and the segment probe; both are off unless their flag is set, and then nothing changes.
//
//   PFLASH_QUERY_CANDIDATES=K (1..8)   the scorer query becomes up to K candidate query sets (one per message),
//                                      ranked RELATIVELY from the stage-1 logits; the head scores the context
//                                      against each candidate and the per-candidate token masses are aggregated by
//                                      the candidates' confidence. No absolute threshold, never empty, no fallback
//                                      to fixed windows.
//   PFLASH_QUERY_PIN=N (0..4096)       a separate high-recall verbatim pin: the latest user message's top-ranked
//                                      WHOLE units, up to N tokens, charged to the budget.
//   PFLASH_QUERY_UNIT_SCORE=max|mean   unit score = max (default) or mean of the stage-1 logit over the unit.
//   PFLASH_QUERY_EXTENT=d|gap          a candidate's extent: its message's units scoring within d (>= 0) logits of
//                                      the message's top unit (0 = the top unit alone), or above the message's
//                                      largest score gap (gap).
//   PFLASH_QUERY_PIN_MARGIN=d|inf|gap  the pin's extent: the latest message's units within d logits of its top unit,
//                                      every unit in rank order until the cap (inf), or above the largest gap.
//   PFLASH_QUERY_AGG=softmax|wmax      aggregation: softmax-weighted mean of the masses (default) or weighted max.
//   PFLASH_QUERY_TEMP=T (> 0)          softmax temperature over the candidates' scores.
//   PFLASH_QUERY_NULL_MARGIN=z (>= 0)  NULL option of the latest user message (below); 0 = never NULL.
//   PFLASH_QUERY_TW=s|score            today's windows (TW) as ONE more candidate of the blend (USER DECISION 9 Oct:
//                                      allowed as a candidate, never as an override): a fixed share s in (0, 1) of
//                                      the blend, or "score": TW joins the softmax with its own score (the unit
//                                      score of s1 over the TW window tokens inside message content).
//   PFLASH_QUERY_END=s                 the prompt-end token (the generation prompt's last token, a 1-token query)
//                                      as one more candidate, fixed share s in (0, 1). TW + END shares < 1.
//
// Definitions (the model-free part lives here so unit tests and offline replays run the runtime's own code):
//  - UNIT: one segment of the S3 segment probe (the segmentation the selector uses for the request, cut at the
//    caller's pinned spans) intersected with one message's content span. A segment crossing a message boundary
//    gives one unit per message. Units come from every non-system message (system messages only when no other
//    message has a unit), so a short latest message is one or two whole units.
//  - UNIT SCORE: max (or mean) of the finite stage-1 raw logits s1 over the unit's tokens. It only RANKS units
//    against each other and against their message's top unit; it is never compared with tau or any absolute
//    threshold.
//  - EXTENT of a message: its units in rank order (score descending, ties to the later position) that the rule
//    keeps -- margin d: score >= top - d (top = its best unit's score); gap: the units ranked above the largest
//    drop between consecutive scores, or every unit when the message has fewer than kQueryNullMinUnits -- each
//    taken WHOLE while it fits the token cap (a unit that does not fit is skipped, never cut; when even the top
//    unit exceeds the cap, its cap best tokens by s1). A short message whose units score alike, or with fewer
//    than kQueryNullMinUnits units under gap, is its own extent.
//  - NULL (latest user message only): with u >= kQueryNullMinUnits units, median m and robust spread
//    s = max(1.4826 * MAD, kQuerySpreadFloor) of its unit scores, z = (top - m) / s. The message is NULL when
//    z < margin: even its best unit does not stand out from the message's own distribution (a paste without a
//    request). Messages with fewer units are never NULL (their units ARE the distribution: a short follow-up
//    always competes). margin 0 disables NULL.
//  - CANDIDATES: one per message. The latest user message's extent first (unless NULL), then the other messages'
//    extents by their top unit's score, descending (ties to the later message); K at most. Query tokens per
//    candidate: kQueryExtentQueryCap. Weights w = softmax(top score / T). A NULL latest message steers only when
//    no other message has a unit (never empty). Without any unit at all the set is empty and the caller fails the
//    request (fail closed; never fixed windows).
//  - MASSES: candidate k is scored like the request set (pflash_request_probe.h): softmax over today's key range
//    minus candidate k's own tokens, mean over heads and the candidate's tokens. Tiles never mix candidates, so the
//    cost is one tile stream over sum_k ceil(|q_k| / 32) tiles (no repeated forward, no repeated keys).
//  - BLEND WEIGHTS: probe candidates w = softmax(score / T); with a TW / END share s_tw / s_end those are scaled by
//    1 - s_tw - s_end and TW / END take their shares; with TW "score" the softmax runs over the probe scores plus
//    TW's score (then scaled by 1 - s_end). TW's mass is today's windows scored and mixed exactly as today; END's is
//    the request-set scoring of the single last query-window token.
//  - AGGREGATE: softmax: m(t) = sum_k w_k m_k(t). wmax: m(t) = max_k w_k m_k(t), renormalised to sum 1.
//  - PIN: the latest user message's extent at the pin margin, capped at N tokens (whole units), merged ascending.
//    Charged. When the pins exceed the selector budget the cap halves until they fit (0 at worst). The steering
//    candidates are not pinned by themselves.

#pragma once

#include "common/pflash_types.h"
#include "pflash/pflash_request_probe.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace luce::pflash {

using luce::common::PFlashTokenSpan;

enum class PFlashUnitScore { Max, Mean };
enum class PFlashQueryAggregate { Softmax, WeightedMax };

inline constexpr int kQueryCandidatesMax = 8;
inline constexpr int kQueryPinMax = 4096;
inline constexpr int kQueryExtentQueryCap = 512;   // query tokens per candidate
inline constexpr int kQueryNullMinUnits = 3;       // NULL is tested on messages with at least this many units
inline constexpr float kQuerySpreadFloor = 0.25f;  // robust spread floor (logit units) of the NULL z

// The step-1 query knobs (resolve_pflash_selection reads them from the environment). Defaults = the development
// operating point of the candidate replay (k1-gap-tw50-end25-soft-pgap: extent by the largest gap, TW share .5, END share .25, softmax, pin by the gap rule). That config did NOT pass the frozen
// no-regression rule (it loses one FU-layout item vs today's windows); no config did. Both switches stay off by default.
struct PFlashQueryConfig {
    int candidates = 0;                                   // K; 0 = off
    PFlashUnitScore unit_score = PFlashUnitScore::Max;
    PFlashQueryAggregate aggregate = PFlashQueryAggregate::Softmax;
    float temperature = 1.0f;
    float null_margin = 0.0f;
    float extent_margin = 0.0f;
    bool extent_gap = true;                               // extent rule = largest gap (extent_margin unused)
    int pin_tokens = 0;                                   // 0 = off
    float pin_margin = std::numeric_limits<float>::infinity();
    bool pin_gap = true;                                  // pin rule = largest gap (pin_margin unused)
    float tw_share = 0.5f;                                // TW candidate share (0 = no TW candidate unless tw_score)
    bool tw_score = false;                                // TW joins the softmax with its own score
    float end_share = 0.25f;                              // prompt-end candidate share (0 = none)
    bool tw() const { return tw_score || tw_share > 0.0f; }
    bool any() const { return candidates > 0 || pin_tokens > 0; }
};

// Reads PFLASH_QUERY_* (fail closed: false + error on an invalid value, or on a knob set without
// PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN).
bool pflash_query_config_from_env(PFlashQueryConfig & out, std::string & error);
const char * pflash_unit_score_name(PFlashUnitScore mode) noexcept;
const char * pflash_query_aggregate_name(PFlashQueryAggregate mode) noexcept;
bool pflash_unit_score_from_name(const std::string & name, PFlashUnitScore & out);
bool pflash_query_aggregate_from_name(const std::string & name, PFlashQueryAggregate & out);

struct PFlashQueryUnit {
    PFlashTokenSpan span{0, 0};
    int message = -1;          // index in the normalised message list
    bool latest = false;       // inside the latest user message
    float score_max = 0.0f;    // max / mean of the finite s1 over the unit
    float score_mean = 0.0f;
    float score(PFlashUnitScore mode) const { return mode == PFlashUnitScore::Max ? score_max : score_mean; }
};

struct PFlashQueryUnits {
    std::vector<PFlashQueryUnit> units;   // ascending positions
    int latest = -1;                      // latest user message index, -1 = none
    bool system_used = false;             // no non-system unit: the system messages' units
    std::string error;                    // "" = ok
};

// Units of ``segments`` (a contiguous cover of [0, n)) inside each message's content span (see the header
// comment). Units without a finite s1 are dropped. error: "segments_invalid", "logits_missing" (a message
// span outside s1).
PFlashQueryUnits pflash_query_units(const std::vector<PFlashTokenSpan> & segments,
                                    const std::vector<PFlashRequestMessage> & messages,
                                    const std::vector<float> & s1);

struct PFlashQueryCandidate {
    int message = -1;              // index in the normalised message list
    std::vector<int> units;        // indices into PFlashQueryUnits::units, rank order
    std::vector<int> tokens;       // ascending query tokens
    float score = 0.0f;            // the message's top unit score
    double weight = 0.0;
};

struct PFlashQueryCandidateSet {
    std::vector<PFlashQueryCandidate> candidates;   // rank order
    int latest = -1;
    int latest_units = 0;
    float latest_top = 0.0f;
    float latest_median = 0.0f;
    float latest_spread = 0.0f;
    float latest_z = 0.0f;         // NaN when the message has fewer than kQueryNullMinUnits units
    bool latest_null = false;
};

// The relative ranking (header comment). config.candidates >= 1; temperature > 0.
PFlashQueryCandidateSet pflash_query_candidate_set(const PFlashQueryUnits & units,
                                                   const std::vector<float> & s1,
                                                   const PFlashQueryConfig & config);

// The extent of message ``message`` at ``margin`` under ``cap`` tokens (header comment): unit indices in rank
// order, and the tokens (ascending). Empty when the message has no unit.
struct PFlashQueryExtent {
    std::vector<int> units;
    std::vector<int> tokens;
    float top = 0.0f;
    int skipped = 0;               // units within the margin that did not fit whole
    bool cut = false;              // the top unit alone exceeded the cap: its cap best tokens
};
PFlashQueryExtent pflash_query_extent(const PFlashQueryUnits & units, int message, const std::vector<float> & s1,
                                      PFlashUnitScore mode, float margin, int cap, bool gap = false);
// The units of ``message`` the extent rule keeps, in rank order (before any cap).
std::vector<int> pflash_query_extent_units(const PFlashQueryUnits & units, int message, PFlashUnitScore mode,
                                           float margin, bool gap);

// Unit indices ordered by (score descending, position descending).
std::vector<int> pflash_query_rank(const PFlashQueryUnits & units, const std::vector<int> & which,
                                   PFlashUnitScore mode);

// The tokens of ``span``: all of them, or the ``cap`` best by s1 (ties later first), ascending.
std::vector<int> pflash_query_unit_tokens(PFlashTokenSpan span, const std::vector<float> & s1, int cap);

// softmax(score / T), max-subtracted; NaN-free for finite scores.
std::vector<double> pflash_query_softmax(const std::vector<float> & scores, float temperature);

// The blend weights (header comment): ``probe`` per probe candidate (rank order), then TW and END.
struct PFlashQueryBlend {
    std::vector<double> probe;
    double tw = 0.0;
    double end = 0.0;
};
PFlashQueryBlend pflash_query_blend(const PFlashQueryCandidateSet & set, const PFlashQueryConfig & config,
                                    float tw_score_value);

// Aggregates per-candidate masses (each of size n) with ``weights``. false + error on a size mismatch or no
// candidate.
bool pflash_query_aggregate(const std::vector<std::vector<float>> & masses,
                            const std::vector<double> & weights,
                            PFlashQueryAggregate mode,
                            std::vector<float> & out,
                            std::string & error);

struct PFlashQueryPin {
    std::vector<PFlashTokenSpan> spans;   // ascending, merged
    int tokens = 0;
    int units = 0;                        // units pinned
    int skipped = 0;                      // units that did not fit whole
};

// The verbatim pin: the latest user message's extent at ``margin`` under ``cap`` (whole units only; a top unit
// over the cap is skipped, not cut). Empty when cap <= 0 or there is no latest user message.
PFlashQueryPin pflash_query_pin(const PFlashQueryUnits & units, int cap, PFlashUnitScore mode, float margin,
                                bool gap = false);

// FNV-1a 64 over the ids as uint32 little-endian bytes, 16 lowercase hex digits (the server's
// pflash_token_fingerprint, repeated here for the drafter's dev-only replay grid).
std::string pflash_ids_fingerprint(const std::vector<int32_t> & ids);

// Dev-only replay grid (PFLASH_QUERY_GRID=<json>, PFLASH_QUERY_GRID_OUT=<jsonl>, PFLASH_QUERY_ORACLE_DIR=<dir>):
// one scoring call evaluates the selection of every grid config on the same forward (S1 = today's windows and
// pins, P1 = the served request-probe set, CAND = the query step above, ORACLE = a given query) and appends one
// JSON record of their kept spans; the request itself is served by the process's own flags. Never set in
// production; the grid does not change the served selection.
//   {"configs": [{"name": "S1", "mode": "s1"}, {"name": "P1", "mode": "p1"},
//                {"name": "c3", "mode": "cand", "k": 3, "unit_score": "max", "agg": "softmax", "temp": 1,
//                 "null_margin": 0, "extent": 2, "pin": 512, "pin_margin": "inf"},
//                {"name": "ORACLE", "mode": "oracle", "oracle_pin": true}]}
// "pin" / "pin_margin" / "unit_score" apply to every mode (the verbatim pin on top of that mode's own pins).
enum class PFlashGridMode { S1, P1, Candidates, Oracle };
struct PFlashGridConfig {
    std::string name;
    PFlashGridMode mode = PFlashGridMode::S1;
    PFlashQueryConfig query;
    bool oracle_pin = true;
};
bool pflash_query_grid_parse(const std::string & text, std::vector<PFlashGridConfig> & out, std::string & error);
const char * pflash_grid_mode_name(PFlashGridMode mode) noexcept;

// An oracle file ({"query": [[b,e]...], "pin": [[b,e]...]}, spans inside [0, n_tokens)): query tokens ascending,
// pin spans merged.
struct PFlashQueryOracle {
    std::vector<int> query;
    std::vector<PFlashTokenSpan> pin;
};
bool pflash_query_oracle_parse(const std::string & text, int n_tokens, PFlashQueryOracle & out, std::string & error);

// What the query step did on one scoring call (trace object request_probe.query; flag on only).
struct PFlashQueryTrace {
    bool evaluated = false;
    PFlashQueryConfig config;
    int units = 0;
    bool system_used = false;
    PFlashQueryCandidateSet set;
    PFlashQueryPin pin;
    int pin_cap_used = 0;          // the cap after any shrink to fit the budget
    int query_tokens = 0;          // sum over candidates
    int tiles = 0;
    int64_t rank_us = 0;
    int64_t score_us = 0;
    float tw_score = 0.0f;         // TW's unit score (NaN when TW is not a candidate or has no content token)
    PFlashQueryBlend blend;
    std::string error;
    // Drafter sessions on (PFLASH_DRAFTER_SESSIONS > 0) only, so a sessions-off record stays as before: the token
    // the scoring resumed from (pflash_scoring_resume.h) and whether this request was redone from token 0 because a
    // query row it needed was not exactly available (redo_reason names the missing input).
    bool sessions = false;
    int session_resume = 0;
    bool redo = false;
    std::string redo_reason;
};

} // namespace luce::pflash
