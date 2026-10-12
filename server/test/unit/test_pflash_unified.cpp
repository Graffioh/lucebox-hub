#include "CppUnitTestFramework.hpp"

#include "server/pflash_unified.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace pflash_unified;

namespace {

struct PFlashUnifiedFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// "line 1" .. "line n", '\n'-joined, with a trailing newline.
std::string numbered_text(int n, const char * prefix = "line ") {
    std::string out;
    for (int i = 1; i <= n; ++i) out += prefix + std::to_string(i) + "\n";
    return out;
}

// Every line start and separator start: both tokenisations agree everywhere.
std::vector<size_t> all_boundaries(const Lines & lines) {
    std::vector<size_t> out;
    for (size_t i = 0; i < lines.count(); ++i) {
        out.push_back(lines.starts[i]);
        out.push_back(lines.ends[i]);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string line_at(const std::string & text, const Lines & lines, size_t i) {
    return text.substr(lines.starts[i], lines.ends[i] - lines.starts[i]);
}

}  // namespace

TEST_CASE(PFlashUnifiedFixture, test_unified_region_tools_parse) {
    ToolList t;
    std::string err;
    REQUIRE(parse_tool_list("", t, err));
    REQUIRE(!t.all);
    REQUIRE(t.names.empty());
    REQUIRE(!t.listed("read_file"));

    REQUIRE(parse_tool_list(" search_files , read_file,read_file ", t, err));
    REQUIRE(t.names == std::vector<std::string>({"read_file", "search_files"}));
    REQUIRE(t.identity == "read_file,search_files");
    REQUIRE(t.listed("read_file"));
    REQUIRE(!t.listed("terminal"));

    REQUIRE(parse_tool_list("*", t, err));
    REQUIRE(t.all);
    REQUIRE(t.identity == "*");
    REQUIRE(t.listed("anything"));

    REQUIRE(!parse_tool_list("*,read_file", t, err));
    REQUIRE(!parse_tool_list("a,,b", t, err));
    REQUIRE(!parse_tool_list("read file", t, err));
    REQUIRE(!parse_tool_list("a[0]", t, err));
}

TEST_CASE(PFlashUnifiedFixture, test_unified_labels_name_and_arguments) {
    const nlohmann::json obj = {{"path", "docs/a.md"}, {"offset", 10}};
    REQUIRE(call_brief("read_file", obj) == "read_file {\"offset\":10,\"path\":\"docs/a.md\"}");
    // JSON-encoded arguments (the OpenAI wire form) read the same.
    REQUIRE(call_brief("read_file", nlohmann::json(obj.dump())) == call_brief("read_file", obj));
    // Newlines and non-ASCII never reach the label raw; long arguments are cut.
    const std::string brief = call_brief("run\ncmd", {{"cmd", "echo a\nb \xC3\xA9"}});
    REQUIRE(brief.find('\n') == std::string::npos);
    REQUIRE(brief.find("\\n") != std::string::npos);
    REQUIRE(brief.find("\\u00e9") != std::string::npos);
    const std::string long_brief = call_brief("t", {{"q", std::string(500, 'x')}}, 40);
    REQUIRE(long_brief.size() == 2 + 40 + 3);
    REQUIRE(long_brief.substr(long_brief.size() - 3) == "\xE2\x80\xA6");
    REQUIRE(call_brief("list", nlohmann::json::object()) == "list");
    REQUIRE(call_brief("list", nullptr) == "list");

    REQUIRE(recall_label_text(RegionKind::ToolOutput, "read_file {\"path\":\"a\"}") ==
            "[From the earlier result of read_file {\"path\":\"a\"}]");
    REQUIRE(recall_label_text(RegionKind::UserContent, "") == "[From an earlier user message]");
}

TEST_CASE(PFlashUnifiedFixture, unified_split_lines_plain_and_escaped) {
    const std::string plain = "a\r\nbb\n\nc\n";
    const Lines p = split_lines(plain, 0, plain.size(), false);
    REQUIRE(p.count() == 4);
    REQUIRE(line_at(plain, p, 0) == "a\r");
    REQUIRE(line_at(plain, p, 1) == "bb");
    REQUIRE(line_at(plain, p, 2) == "");
    REQUIRE(line_at(plain, p, 3) == "c");
    REQUIRE(split_lines(plain, 0, plain.size() - 1, false).count() == 4);
    REQUIRE(split_lines(plain, 0, 0, false).count() == 0);

    // Inside a JSON string: "\n" separates, an escaped backslash followed by n does not, \u escapes are skipped.
    const std::string esc = "1|a\\n2|b\\\\n still 2\\n3|\\u005cn";
    const Lines e = split_lines(esc, 0, esc.size(), true);
    REQUIRE(e.count() == 3);
    REQUIRE(line_at(esc, e, 1) == "2|b\\\\n still 2");
    REQUIRE(line_at(esc, e, 2) == "3|\\u005cn");
    REQUIRE(std::string(e.sep()) == "\\n");
}

TEST_CASE(PFlashUnifiedFixture, test_unified_line_model_json_string) {
    // A JSON-wrapped file read (the shape Hermes and many MCP tools return): the lines live in a string value.
    std::string body;
    for (int i = 1; i <= 40; ++i) body += std::to_string(i) + "|text " + std::to_string(i) + "\n";
    const nlohmann::json result = {{"content", body}, {"total_lines", 40}, {"_hint", "a\nb"}};
    const std::string text = result.dump();
    const Lines lines = detect_lines(text, 0, text.size());
    REQUIRE(lines.escaped);
    REQUIRE(lines.count() == 40);
    REQUIRE(line_at(text, lines, 0) == "1|text 1");
    REQUIRE(line_at(text, lines, 39) == "40|text 40");

    // The same lines as plain text are found as plain lines; a pretty-printed JSON whose plain lines outnumber
    // its string's lines stays plain.
    const Lines plain = detect_lines(body, 0, body.size());
    REQUIRE(!plain.escaped);
    REQUIRE(plain.count() == 40);
    nlohmann::json many = nlohmann::json::array();
    for (int i = 0; i < 30; ++i) many.push_back({{"k", i}, {"v", "x\ny"}});
    const std::string pretty = many.dump(2);
    REQUIRE(!detect_lines(pretty, 0, pretty.size()).escaped);
    REQUIRE(json_string_values("not json {", 0, 10).empty());
    const std::string keys = "{\"a\\nb\\nc\\nd\\ne\\nf\": 1}";
    REQUIRE(json_string_values(keys, 0, keys.size()).empty());

    // Paging inside the string keeps the result valid JSON with every other field unchanged.
    std::vector<size_t> common = all_boundaries(lines);
    std::vector<size_t> proposed;
    for (int l = 5; l < 40; l += 5) proposed.push_back(lines.starts[(size_t) l]);
    Layout layout;
    std::string why;
    REQUIRE(layout_region(lines, common, proposed, layout, why));
    std::vector<char> admitted((size_t) layout.units(), 0);
    admitted[1] = 1;
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::ToolOutput);
    const nlohmann::json parsed = nlohmann::json::parse(served, nullptr, false);
    REQUIRE(!parsed.is_discarded());
    REQUIRE(parsed["total_lines"] == 40);
    REQUIRE(parsed["_hint"] == "a\nb");
    const std::string content = parsed["content"].get<std::string>();
    REQUIRE(content.find("1|text 1\n2|text 2\n[Server note: this tool result was shortened:") == 0);
    REQUIRE(content.find("[\xE2\x80\xA6 3 lines not shown \xE2\x80\xA6]\n6|text 6\n") != std::string::npos);
    REQUIRE(content.find("\n39|text 39\n40|text 40\n") != std::string::npos);
}

TEST_CASE(PFlashUnifiedFixture, test_unified_markers_count_lines) {
    // CRLF content without a trailing newline: served bytes keep CRLF; markers count the dropped lines.
    std::string text;
    for (int i = 1; i <= 12; ++i) text += "row " + std::to_string(i) + (i < 12 ? "\r\n" : "");
    const Lines lines = detect_lines(text, 0, text.size());
    REQUIRE(lines.count() == 12);
    std::vector<size_t> proposed;
    for (int l = 3; l < 12; l += 2) proposed.push_back(lines.starts[(size_t) l] + 1);  // mid-line: opens that line
    Layout layout;
    std::string why;
    REQUIRE(layout_region(lines, all_boundaries(lines), proposed, layout, why));
    REQUIRE(layout.cut_line == std::vector<int>({2, 3, 5, 7, 9, 10}));
    std::vector<char> admitted = {1, 0, 0, 1, 0};
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::UserContent);
    const std::string expect =
        "row 1\r\nrow 2\r\n"
        "[Server note: this message was shortened: 7 of 12 lines are shown; each omitted range is marked inline.]\n"
        "row 3\r\n"
        "[\xE2\x80\xA6 4 lines not shown \xE2\x80\xA6]\n"
        "row 8\r\nrow 9\r\n"
        "[\xE2\x80\xA6 1 line not shown \xE2\x80\xA6]\n"
        "row 11\r\nrow 12";
    REQUIRE(served == expect);
    REQUIRE(marker_text(2) == "[\xE2\x80\xA6 2 lines not shown \xE2\x80\xA6]");
}

TEST_CASE(PFlashUnifiedFixture, unified_layout_cuts_only_at_clean_line_boundaries) {
    const std::string text = numbered_text(20);
    const Lines lines = detect_lines(text, 0, text.size());
    // Separator starts only (a tokenizer that glues the newline to the next word): cuts sit before the separator.
    std::vector<size_t> seps;
    for (size_t i = 0; i < lines.count(); ++i) seps.push_back(lines.ends[i]);
    Layout layout;
    std::string why;
    REQUIRE(layout_region(lines, seps, {lines.starts[10]}, layout, why));
    REQUIRE(layout.cuts.front() == lines.ends[1]);
    REQUIRE(layout.cuts[1] == lines.ends[9]);
    REQUIRE(layout.cuts.back() == lines.ends[17]);
    std::vector<char> admitted = {0, 1};
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::UserContent);
    REQUIRE(served.find("line 2\n[Server note:") != std::string::npos);
    REQUIRE(served.find("inline.]\n[\xE2\x80\xA6 8 lines not shown \xE2\x80\xA6]\nline 11\n") != std::string::npos);

    // A line with no clean boundary opens no unit; the edges move inward to the first clean line.
    std::vector<size_t> sparse;
    for (int l : {3, 8, 16}) sparse.push_back(lines.starts[(size_t) l]);
    REQUIRE(layout_region(lines, sparse, {lines.starts[5], lines.starts[8]}, layout, why));
    REQUIRE(layout.cut_line == std::vector<int>({3, 8, 16}));

    // Too few lines, or no clean cut at all: never a region.
    const std::string four = numbered_text(4);
    const Lines four_lines = detect_lines(four, 0, four.size());
    REQUIRE(!layout_region(four_lines, all_boundaries(four_lines), {}, layout, why));
    REQUIRE(!layout_region(lines, {}, {}, layout, why));
    const std::string one_line(5000, 'x');
    REQUIRE(detect_lines(one_line, 0, one_line.size()).count() == 1);
}

TEST_CASE(PFlashUnifiedFixture, unified_admit_paging_counts_markers_and_respects_cap) {
    // Five units of 10 tokens, marker cost 3 per gap, preamble 4.
    const std::vector<int> cost = {10, 10, 10, 10, 10};
    auto marker = [](int, int) { return 3; };
    const std::vector<int> order = {2, 0, 4, 1, 3};
    // Start: one gap (3) + preamble (4) = 7. Unit 2 splits the gap: +10 +3 = 20. Unit 0: +10 +0 = 30. Unit 4:
    // +10 + 0 = 40 > 35, skipped; unit 1 joins [0..2]: +10 -3 = 37 > 35, skipped; unit 3: +10 -3 +3 = 40, skipped.
    Admission a = admit_paging(cost, {}, order, 4, 35, marker);
    REQUIRE(a.ok);
    REQUIRE(a.admitted == std::vector<char>({1, 0, 1, 0, 0}));
    REQUIRE(a.cost == 30);
    REQUIRE(a.taken == 2);
    // Locked units are kept and cost nothing against the cap.
    a = admit_paging(cost, {0, 0, 0, 0, 1}, order, 4, 7, marker);
    REQUIRE(a.ok);
    REQUIRE(a.admitted == std::vector<char>({0, 0, 0, 0, 1}));
    REQUIRE(a.cost == 7);
    // A cap below the preamble plus one marker cannot page.
    REQUIRE(!admit_paging(cost, {}, order, 4, 6, marker).ok);
}

TEST_CASE(PFlashUnifiedFixture, unified_admit_recall_skips_held_and_charges_labels_per_run) {
    const std::vector<int> cost = {10, 10, 10, 10, 10};
    auto label = [](int, int) { return 5; };
    // Held 1; order 2, 3, 0, 4: unit 2 = 15; unit 3 joins the run: +10 = 25; unit 0 new run: +15 = 40 > 30.
    Admission a = admit_recall(cost, {0, 1, 0, 0, 0}, {2, 3, 0, 4}, 30, label);
    REQUIRE(a.admitted == std::vector<char>({0, 0, 1, 1, 0}));
    REQUIRE(a.cost == 25);
    // Unit 4 fits after unit 0 is skipped.
    a = admit_recall(cost, {0, 1, 0, 0, 0}, {2, 3, 0, 4}, 35, label);
    REQUIRE(a.admitted == std::vector<char>({0, 0, 1, 1, 1}));
    REQUIRE(a.cost == 35);
    // Bridging two runs merges their labels.
    a = admit_recall(cost, {}, {0, 2, 1}, 1000, label);
    REQUIRE(a.cost == 35);
    // Worst case: every unheld unit comes back once, nothing more.
    a = admit_recall(cost, {1, 0, 1, 0, 0}, {0, 1, 2, 3, 4}, 1000, label);
    REQUIRE(a.admitted == std::vector<char>({0, 1, 0, 1, 1}));
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_appends_nothing_when_the_present_context_suffices) {
    RecallRegion r;
    r.unit_cost = {10, 10, 10, 10};
    r.held = {1, 0, 0, 1};
    r.score = {9.0, 2.0, 1.0, 9.0};
    r.mass = {0.40, 0.03, 0.01, 0.30};   // held units carry most of the mass; D = 0.04
    r.cap = 1000;
    r.label_cost = 5;
    GatedRecall g = admit_recall_gated({r}, 1.0, 0.05);
    REQUIRE(g.gated);
    REQUIRE(g.regions[0].taken == 0);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 0, 0, 0}));
    REQUIRE(std::fabs(g.deficit_before - 0.04) < 1e-12);
    REQUIRE(std::fabs(g.deficit_after - 0.04) < 1e-12);
    // Held units never count towards D, whatever their mass.
    r.mass = {5.0, 0.03, 0.01, 5.0};
    REQUIRE(std::fabs(admit_recall_gated({r}, 1.0, 0.05).deficit_before - 0.04) < 1e-12);
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_stops_once_the_deficit_is_covered) {
    RecallRegion r;
    r.unit_cost = {10, 10, 10, 10, 10};
    r.held = {0, 0, 0, 0, 0};
    r.score = {1.0, 8.0, 2.0, 0.5, 0.4};
    r.mass = {0.02, 0.30, 0.05, 0.01, 0.01};   // D = 0.39 of S = 1
    r.cap = 1000;
    r.label_cost = 5;
    // theta .10: unit 1 leaves .09 < .10: one unit, nothing more.
    GatedRecall g = admit_recall_gated({r}, 1.0, 0.10);
    REQUIRE(!g.gated);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1, 0, 0, 0}));
    REQUIRE(g.regions[0].cost == 15);
    REQUIRE(std::fabs(g.deficit_after - 0.09) < 1e-12);
    // theta .05: unit 1 (.09 left), unit 2 (.04 left): stop.
    g = admit_recall_gated({r}, 1.0, 0.05);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1, 1, 0, 0}));
    REQUIRE(g.regions[0].cost == 25);
    // S scales the share: the same units against S = 4 are under .10 at the start.
    REQUIRE(admit_recall_gated({r}, 4.0, 0.10).gated);
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_at_zero_fills_like_admit_recall_and_never_past_the_cap) {
    RecallRegion r;
    r.unit_cost = {10, 10, 10, 10, 10};
    r.held = {0, 1, 0, 0, 0};
    r.score = {3.0, 9.0, 5.0, 4.0, 1.0};
    r.mass = {0.1, 0.5, 0.1, 0.1, 0.1};
    r.cap = 30;
    r.label_cost = 5;
    const auto label = [](int, int) { return 5; };
    const Admission fill = admit_recall(r.unit_cost, r.held, rank_units(r.score, r.held), r.cap, label);
    const GatedRecall g = admit_recall_gated({r}, 1.0, 0.0);
    REQUIRE(g.regions[0].admitted == fill.admitted);
    REQUIRE(g.regions[0].cost == fill.cost);
    REQUIRE(g.regions[0].cost <= r.cap);
    // Diffuse mass under a strict theta: still bounded by the cap.
    REQUIRE(admit_recall_gated({r}, 1.0, 0.01).regions[0].cost <= r.cap);
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_orders_units_across_regions_under_their_own_caps) {
    RecallRegion a;
    a.unit_cost = {10, 10};
    a.held = {0, 0};
    a.score = {1.0, 6.0};
    a.mass = {0.05, 0.10};
    a.cap = 15;
    a.label_cost = 5;
    RecallRegion b;
    b.unit_cost = {10, 10};
    b.held = {0, 0};
    b.score = {7.0, 2.0};
    b.mass = {0.20, 0.05};
    b.cap = 1000;
    b.label_cost = 5;
    // D = .40. Order: b0 (7), a1 (6), b1 (2), a0 (1). theta .12: b0 -> .20, a1 -> .10 < .12: stop.
    GatedRecall g = admit_recall_gated({a, b}, 1.0, 0.12);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1}));
    REQUIRE(g.regions[1].admitted == std::vector<char>({1, 0}));
    // theta .01: a0 does not fit a's cap (15 + 10 + label change) and is skipped; b1 is taken.
    g = admit_recall_gated({a, b}, 1.0, 0.01);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1}));
    REQUIRE(g.regions[0].cost <= a.cap);
    REQUIRE(g.regions[1].admitted == std::vector<char>({1, 1}));
}

TEST_CASE(PFlashUnifiedFixture, unified_rank_units_orders_by_score_then_document) {
    REQUIRE(rank_units({0.5, 2.0, 0.5, NAN, 3.0}, {}) == std::vector<int>({4, 1, 0, 2, 3}));
    REQUIRE(rank_units({0.5, 2.0, 0.5}, {0, 1, 0}) == std::vector<int>({0, 2}));
}

TEST_CASE(PFlashUnifiedFixture, test_unified_reread_by_content) {
    std::string file;
    for (int i = 1; i <= 100; ++i) file += "def f" + std::to_string(i) + "():\n    return " + std::to_string(i) + "\n}\n\n";
    const Lines region_lines = detect_lines(file, 0, file.size());
    const auto region = line_hashes(file, region_lines);

    // A partial re-read of the same file, delivered JSON-wrapped: decoded lines compare equal.
    std::string part;
    for (int i = 40; i <= 60; ++i) part += "def f" + std::to_string(i) + "():\n    return " + std::to_string(i) + "\n}\n\n";
    const std::string wrapped = nlohmann::json({{"content", part}}).dump();
    const Lines part_lines = detect_lines(wrapped, 0, wrapped.size());
    REQUIRE(part_lines.escaped);
    RereadMatch m = match_reread(line_hashes(wrapped, part_lines), region);
    REQUIRE(m.is_reread());
    REQUIRE(m.matched_candidate == m.candidate_informative);
    REQUIRE(m.candidate_informative == 21 * 2);             // "}" (100 copies) and blank lines are skipped
    int covered = 0;
    for (char c : m.region_covered) covered += c;
    REQUIRE(covered == 21 * 2 + 20 * 2);                    // + the "}" and blank lines between matched blocks
    REQUIRE(!m.region_covered[(size_t) (59 * 4 + 2)]);      // the "}" after the last block: next line unmatched
    REQUIRE(m.region_covered[(size_t) (39 * 4)]);           // "def f40():"
    REQUIRE(!m.region_covered[(size_t) (38 * 4)]);          // "def f39():"
    REQUIRE(m.region_informative[(size_t) (39 * 4)]);
    REQUIRE(!m.region_informative[(size_t) (39 * 4 + 2)]);  // "}"
    REQUIRE(!m.region_informative[(size_t) (39 * 4 + 3)]);  // blank
    // A unit ending in the last block's "}" and blank line is covered: its informative lines are; one reaching
    // into f61 is not.
    REQUIRE(lines_covered(m.region_covered, m.region_informative, 59 * 4, 60 * 4));
    REQUIRE(!lines_covered(m.region_covered, m.region_informative, 59 * 4, 60 * 4 + 1));
    // Lines with nothing informative count only when covered themselves (between two covered blocks).
    REQUIRE(lines_covered(m.region_covered, m.region_informative, 39 * 4 + 2, 40 * 4));
    REQUIRE(!lines_covered(m.region_covered, m.region_informative, 59 * 4 + 2, 60 * 4));

    // A changed file matches only where it is unchanged; "}" lines (more than 32 copies) never match alone.
    std::string changed = part;
    for (char & c : changed) if (c == 'r') c = 'R';
    m = match_reread(line_hashes(changed, detect_lines(changed, 0, changed.size())), region);
    REQUIRE(m.matched_candidate == 0);
    REQUIRE(!m.is_reread());
    const std::string unrelated = numbered_text(30, "other ");
    m = match_reread(line_hashes(unrelated, detect_lines(unrelated, 0, unrelated.size())), region);
    REQUIRE(!m.is_reread());
}

TEST_CASE(PFlashUnifiedFixture, test_unified_reread_multiple_regions) {
    const std::string v1 = numbered_text(50, "alpha ");
    std::string v2 = v1;
    v2.replace(v2.find("alpha 20\n"), 9, "ALPHA 20\n");
    const auto h1 = line_hashes(v1, detect_lines(v1, 0, v1.size()));
    const auto h2 = line_hashes(v2, detect_lines(v2, 0, v2.size()));
    const std::string reread = numbered_text(25, "alpha ");
    const auto hr = line_hashes(reread, detect_lines(reread, 0, reread.size()));
    const RereadMatch m1 = match_reread(hr, h1);
    const RereadMatch m2 = match_reread(hr, h2);
    REQUIRE(m1.region_covered[19]);
    REQUIRE(!m2.region_covered[19]);
    REQUIRE(m2.region_covered[18]);
    REQUIRE(m2.region_covered[20]);
}

TEST_CASE(PFlashUnifiedFixture, unified_view_store_finds_longest_prefix_and_keeps_ancestors) {
    ViewStore store(3, (size_t) 1 << 30);
    auto make = [](std::vector<int32_t> raw, int gen, const std::string & cfg) {
        View v;
        v.raw_tokens = raw;
        v.raw_gen_begin = gen;
        v.drafter_ids = raw;
        v.drafter_gen_begin = gen;
        v.config = cfg;
        return v;
    };
    store.remember(make({1, 2, 3, 9}, 3, "c"));
    store.remember(make({1, 2, 3, 4, 5, 9}, 5, "c"));
    View out;
    REQUIRE(store.find({1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7}, "c", out));
    REQUIRE(out.raw_gen_begin == 5);
    REQUIRE(store.find({1, 2, 3, 7}, {1, 2, 3, 7}, "c", out));   // a fork continues the ancestor
    REQUIRE(out.raw_gen_begin == 3);
    REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, "other", out));
    REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 9, 4, 5, 6}, "c", out));   // drafter ids must match too
    // A retry replaces its own entry; capacity evicts the least recently used, never the newest.
    store.remember(make({1, 2, 3, 4, 5, 9}, 5, "c"));
    REQUIRE(store.size() == 2);
    store.remember(make({7, 7}, 1, "c"));
    store.remember(make({8, 8}, 1, "c"));
    REQUIRE(store.size() == 3);
    REQUIRE(!store.find({1, 2, 3, 0}, {1, 2, 3, 0}, "c", out));
    REQUIRE(store.find({8, 8, 1}, {8, 8, 1}, "c", out));
}

TEST_CASE(PFlashUnifiedFixture, unified_text_helpers) {
    REQUIRE(json_escape("a\"b\\c\nd\te\x01 \xC3\xA9") == "a\\\"b\\\\c\\nd\\te\\u0001 \xC3\xA9");
    REQUIRE(sha1_hex("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d");
    REQUIRE(preamble_text(RegionKind::ToolOutput, 3, 10) ==
            "[Server note: this tool result was shortened: 3 of 10 lines are shown; each omitted range is marked "
            "inline. Call the tool again if you need an omitted part.]");
}
