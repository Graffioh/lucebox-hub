#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_ident_query.h"
#include "pflash/pflash_selection.h"
#include "scoped_env.h"

#include <string>
#include <utility>
#include <vector>

using namespace luce::pflash;

namespace {

struct PFlashIdentQueryFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

using Spans = std::vector<std::pair<size_t, size_t>>;

} // namespace

// Expected spans from Python ``re`` with the diagnostic's patterns (cues_rows.py identifiers()).
TEST_CASE(PFlashIdentQueryFixture, identifiers_match_the_diagnostic_patterns) {
    const std::string text =
        "open src/pkg/io_utils.py at line 10\nfoo.bar.baz = getValue(x) and HttpClient\n"
        "run_tests() abc(1)\n  ValueError: bad thing\nok";
    const Spans expected = {{5, 24}, {36, 47}, {50, 58}, {66, 76}, {77, 86}, {98, 108}, {96, 119}};
    REQUIRE(pflash_identifier_char_spans(text) == expected);
}

TEST_CASE(PFlashIdentQueryFixture, short_names_and_plain_words_are_not_identifiers) {
    REQUIRE(pflash_identifier_char_spans("the cat sat on a mat. a.b x_y f(x) abc(1)").empty());
    // A file name needs a 1-6 letter extension ending the word.
    REQUIRE(pflash_identifier_char_spans("archive.tarball2 data.verylongext").size() == 2);  // dotted symbols
    REQUIRE(pflash_identifier_char_spans("Traceback").size() == 1);
    REQUIRE(pflash_identifier_char_spans("Error").empty());   // a 5-character error line is too short
}

TEST_CASE(PFlashIdentQueryFixture, tokens_overlap_identifiers_and_markup_is_blanked) {
    // Tokens: "<tool_call>" (markup), "call", " my", "_func", "(", ")", "</tool_call>" (markup).
    const std::vector<std::string> texts = {"<tool_call>", "call", " my", "_func", "(", ")", "</tool_call>"};
    std::vector<unsigned char> markup = {1, 0, 0, 0, 0, 0, 1};
    const std::vector<PFlashTokenSpan> turn = {{0, 7}};
    REQUIRE(pflash_ident_query_tokens(texts, markup, turn) == std::vector<int>({2, 3}));
    // Without the blanking the markup itself reads as snake_case.
    markup.assign(7, 0);
    const auto unmasked = pflash_ident_query_tokens(texts, markup, turn);
    REQUIRE(unmasked.size() > 2);
    REQUIRE(unmasked.front() == 0);
    // Only the given turns are read, in token positions of the prompt.
    REQUIRE(pflash_ident_query_tokens(texts, std::vector<unsigned char>(7, 1), turn).empty());
    REQUIRE(pflash_ident_query_tokens(texts, {}, {{2, 6}}) == std::vector<int>({2, 3}));
}

TEST_CASE(PFlashIdentQueryFixture, flag_parses_and_needs_strict_head_selection_without_the_probe) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar scorer{"PFLASH_SELECT_SCORER", nullptr};
    luce_test::ScopedEnvVar probe{"PFLASH_SELECT_QUERY_PROBE", nullptr};
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_IDENT", nullptr};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(config.query_ident == PFlashIdentQuery::Off);
    }
    for (const char * value : {"off", "heur"}) {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_IDENT", value};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(std::string(pflash_ident_query_name(config.query_ident)) == value);
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_IDENT", "today_heur"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(1000, 32, config, error));
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_IDENT", "heur"};
        luce_test::ScopedEnvVar probe_on{"PFLASH_SELECT_QUERY_PROBE", "1"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(1000, 32, config, error));
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_IDENT", "heur"};
        luce_test::ScopedEnvVar split{"PFLASH_SELECT_SCORER", "split"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(1000, 32, config, error));
    }
}
