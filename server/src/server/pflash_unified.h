#pragma once

// PFLASH_UNIFIED=1: one view/recall mechanism for chat and agent sessions
// (.benchmark-work/round13-data/agent-pager/STAGE1-DESIGN.md, amendments A1-A4, A6).
// The pure parts: the tool list, line-structure detection, region layout,
// marker and label text, greedy admission of units under a cap, re-read
// matching and the view store. Nothing here knows a harness's output format,
// and nothing here tokenises: callers pass byte offsets of the rendered prompt
// text and token costs.

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace pflash_unified {

// ── Tool list (PFLASH_REGION_TOOLS) ───────────────────────────────────

// Which tool outputs may become regions: tools the agent can call again before the next user question (recall
// only runs on a user question). Names only; ``all`` for "*".
struct ToolList {
    bool all = false;
    std::vector<std::string> names;
    std::string identity;      // canonical text (part of the config identity)
    bool listed(const std::string & name) const;
};

// ``spec``: comma-separated tool names, or "*". Empty = none. False with ``error`` on an empty or malformed entry
// (the server refuses to start).
bool parse_tool_list(const std::string & spec, ToolList & out, std::string & error);

// "NAME ARGS" for a recall label: the arguments as compact ASCII JSON (``arguments`` may be an object or a
// JSON-encoded string), cut at ``max_args`` bytes with "…".
std::string call_brief(const std::string & name, const nlohmann::json & arguments, size_t max_args = 120);

// ── Line structure ─────────────────────────────────────────────────────

// The lines of text[begin, end): plain ('\n' separates; a CR stays in its line) or escaped (inside a JSON string:
// the two bytes "\n" separate, other escapes are skipped whole). ``starts[i]`` is where line i begins, ``ends[i]``
// where its separator begins (``end`` for the last line). A separator at ``end`` opens no empty last line.
struct Lines {
    bool escaped = false;
    size_t begin = 0;
    size_t end = 0;
    std::vector<size_t> starts;
    std::vector<size_t> ends;
    size_t count() const { return starts.size(); }
    size_t sep_size() const { return escaped ? 2 : 1; }
    const char * sep() const { return escaped ? "\\n" : "\n"; }
};
Lines split_lines(const std::string & text, size_t begin, size_t end, bool escaped);

// The raw byte ranges (between the quotes) of the string values of text[begin, end) when it is exactly one JSON
// value, surrounding whitespace allowed; keys are not values. Empty when it is not JSON.
std::vector<std::pair<size_t, size_t>> json_string_values(const std::string & text, size_t begin, size_t end);

// The structure to page (A6): the plain lines of the content, or the escaped lines of one of its JSON string
// values, whichever has the most lines; ties go to plain.
Lines detect_lines(const std::string & text, size_t begin, size_t end);

// ── Region layout ───────────────────────────────────────────────────────

// A region's lines split into a pinned head (at least its first two lines), a pinned tail (at least its last two
// lines) and units between cuts. Every cut is a line_cut(): a position that starts a token in both tokenisations
// (``common``), at the start of the line it opens or of the separator before it.
struct Layout {
    size_t head_end = 0;                 // pinned head [content begin, head_end)
    size_t tail_begin = 0;               // pinned tail [tail_begin, content end)
    std::vector<size_t> cuts;            // cuts.front() == head_end, cuts.back() == tail_begin
    std::vector<int> cut_line;           // the line each cut opens (cut_line.back(): the tail's first line)
    int units() const { return cuts.empty() ? 0 : (int) cuts.size() - 1; }
    int first_line(int unit) const { return cut_line[(size_t) unit]; }
    int line_count(int first_unit, int last_unit) const {
        return cut_line[(size_t) last_unit + 1] - cut_line[(size_t) first_unit];
    }
};

// The cut opening ``line`` (> 0), so that kept lines stay whole: the line's start when that is a common boundary,
// else the start of the separator before it when that is one and lies after the previous line's start; npos
// otherwise (the line opens no unit).
size_t line_cut(const Lines & lines, const std::vector<size_t> & common, int line);

// ``common``: ascending byte offsets that start a token in both tokenisations. ``proposed``: candidate cut offsets
// (the drafter's candidate starts: probe segments or fixed chunks); each opens the line it falls in and
// is kept when that line has a cut; a boundary at a separator opens the next line. The head ends at the first line
// >= 2 with a cut, the tail starts at the last line <= count - 2 with one. False with ``why`` when there are fewer
// than five lines, no head/tail cut, or no unit between them.
bool layout_region(const Lines & lines, const std::vector<size_t> & common,
                   const std::vector<size_t> & proposed, Layout & out, std::string & why);

// ── Text ────────────────────────────────────────────────────────────────

enum class RegionKind { UserContent, ToolOutput };

// ``text`` escaped for the inside of a JSON string.
std::string json_escape(const std::string & text);
// The decoded bytes of text[begin, end) read as the inside of a JSON string; malformed escapes stay as written.
std::string json_unescape(const std::string & text, size_t begin, size_t end);

std::string marker_text(int lines_not_shown);
std::string preamble_text(RegionKind kind, int shown_lines, int total_lines);
std::string recall_label_text(RegionKind kind, const std::string & brief);
inline const char * recall_open() { return "[Earlier in this conversation]\n"; }
inline const char * recall_close() { return "\n[End of earlier excerpts]\n\n"; }

// ``body`` as a line of its own at a cut in ``lines``' structure: a separator before it unless the text before
// ``at`` ends with one, a separator after it unless the text from ``next`` starts with one; JSON-escaped in
// escaped mode.
std::string inserted_line(const std::string & text, const Lines & lines, size_t at, size_t next,
                          const std::string & body);

// The served text of a paged region's content text[content_begin, content_end): the pinned head, the preamble,
// each unit raw when ``admitted`` and one marker per run of dropped units (counting their lines), the pinned tail.
// The server serves the same pieces as token slices; this is their text, for parity checks and tests.
std::string render_region_text(const std::string & text, size_t content_begin, size_t content_end,
                               const Lines & lines, const Layout & layout, const std::vector<char> & admitted,
                               RegionKind kind);

// ── Admission ──────────────────────────────────────────────────────────

struct Admission {
    bool ok = false;
    std::vector<char> admitted;          // per unit (locked units included)
    int cost = 0;                        // fixed + optional units + markers (or labels)
    int taken = 0;                       // optional units admitted
};

// Paging: ``locked`` units are kept whatever the cap (mandatory, served on top). The rest start dropped and
// ``fixed`` + Σ marker_cost(first, last) over the gaps is the starting cost. Units are taken in ``order`` (best
// first) while the cost after taking one -- its tokens plus the change in marker cost of the gap it splits -- stays
// within ``cap``. ``ok`` false when even the starting cost exceeds the cap.
Admission admit_paging(const std::vector<int> & unit_cost, const std::vector<char> & locked,
                       const std::vector<int> & order, int fixed, int cap,
                       const std::function<int(int, int)> & marker_cost);

// Recall: ``held`` units are never taken; the others in ``order`` while Σ unit tokens + Σ label_cost(first, last)
// over the runs of consecutive taken units stays within ``cap``.
Admission admit_recall(const std::vector<int> & unit_cost, const std::vector<char> & held,
                       const std::vector<int> & order, int cap,
                       const std::function<int(int, int)> & label_cost);

// Units best first: higher score first, document order on ties; ``skip`` units are left out.
std::vector<int> rank_units(const std::vector<double> & score, const std::vector<char> & skip);

// ── Re-reads (A6) ──────────────────────────────────────────────────────

// Line fingerprints of a structure (CR stripped; blank lines 0).
std::vector<uint64_t> line_hashes(const std::string & text, const Lines & lines);

// Lines of ``region`` matched by ``candidate``. Informative lines: non-blank and occurring at most ``max_repeat``
// times in the region (blank and over-frequent lines such as "}" are skipped on both sides, never breaking a run).
// A match is a run of >= ``min_run`` consecutive informative candidate lines equal to consecutive informative
// region lines. A skipped region line counts as covered when the informative lines on both sides of it are.
struct RereadMatch {
    std::vector<char> region_covered;    // per region line
    int matched_candidate = 0;           // informative candidate lines inside a match
    int candidate_informative = 0;
    bool is_reread() const { return candidate_informative > 0 && 2 * matched_candidate >= candidate_informative; }
};
RereadMatch match_reread(const std::vector<uint64_t> & candidate, const std::vector<uint64_t> & region,
                         int min_run = 3, int max_repeat = 32);

// ── View ───────────────────────────────────────────────────────────────

struct Region {
    RegionKind kind = RegionKind::UserContent;
    std::string brief;                   // call_brief() of a tool output ("" for user content)
    std::string tool_call_id;
    std::string content_sha;             // hex digest of the raw content bytes
    size_t cb = 0, ce = 0;               // content bytes in the rendered prompt text
    int tb = 0, te = 0;                  // target tokens of the content
    int db = 0, de = 0;                  // drafter tokens of the content
    Lines lines;
    Layout layout;
    std::vector<uint64_t> hashes;        // line_hashes(lines)
    std::vector<int> unit_tb, unit_te;   // per unit, target tokens
    std::vector<int> unit_db, unit_de;   // per unit, drafter tokens
    std::vector<char> mandatory;         // per unit
    std::vector<char> kept;              // served when paged
    std::vector<char> held;              // kept, recalled or re-read
    std::vector<char> reread;            // per line, covered by a re-read
    int cap = 0;                         // optional served tokens at paging (target)
    int droppable = 0;                   // target tokens of the optional units
    int recalled_tokens = 0;             // cumulative appended by recall (target)
};

struct Piece {
    int begin = 0;                       // served target tokens [begin, end)
    int end = 0;
    std::string kind;                    // raw, paged_text, preamble, marker, recall_wrapper, recall_label, recall_text, generation
    int region = -1;
    int turn = -1;
};

struct View {
    uint64_t lineage = 0;
    int version = 0;
    std::string config;
    std::vector<int32_t> raw_tokens;     // target, raw prompt
    int raw_gen_begin = -1;
    std::vector<int32_t> drafter_ids;
    int drafter_gen_begin = -1;
    std::vector<int32_t> view_tokens;    // target, served
    int view_gen_begin = -1;
    std::vector<Region> regions;
    std::vector<Piece> pieces;
    int turns = 0;
    size_t bytes() const;
};

// Views by raw prefix. ``find``: the longest stored view whose raw tokens up to its generation prompt prefix this
// request's (target and drafter alike) and whose config identity matches; newest on ties; a copy. ``remember``
// adds a view and evicts the least recently used views over ``capacity`` or ``byte_cap``, never the newest one;
// the view it continued stays, so a fork or an edited request continues it (A4).
class ViewStore {
public:
    ViewStore(size_t capacity = 16, size_t byte_cap = (size_t) 256 << 20)
        : capacity_(capacity), byte_cap_(byte_cap) {}
    bool find(const std::vector<int32_t> & raw_tokens, const std::vector<int32_t> & drafter_ids,
              const std::string & config, View & out);
    void remember(View view);
    size_t size() const;
    size_t bytes() const;

private:
    struct Entry {
        View view;
        uint64_t used = 0;
    };
    mutable std::mutex mutex_;
    size_t capacity_;
    size_t byte_cap_;
    uint64_t clock_ = 0;
    std::vector<Entry> entries_;
};

// Hex SHA-1 of ``bytes`` (region identity).
std::string sha1_hex(const std::string & bytes);

}  // namespace pflash_unified
