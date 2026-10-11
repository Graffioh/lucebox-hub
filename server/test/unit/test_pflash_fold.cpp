#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_compress.h"
#include "pflash/pflash_fold.h"
#include "pflash/pflash_selection.h"
#include "scoped_env.h"
#include "server/tokenizer.h"

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace luce::pflash;
using luce::common::PFlashTokenSpan;

namespace {

struct PFlashFoldFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

PFlashTextSpan whole(const std::string & text) { return {0, text.size()}; }

// (first line, last line) of every span.
std::vector<std::pair<std::string, std::string>> ends_of(
        const std::string & text, const std::vector<PFlashTextSpan> & spans) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto & s : spans) {
        std::string body = text.substr(s.begin, s.end - s.begin);
        while (!body.empty() && (body.back() == '\n' || body.back() == ' ')) body.pop_back();
        const size_t first_nl = body.find('\n');
        const size_t last_nl = body.rfind('\n');
        out.push_back({body.substr(0, first_nl),
                       last_nl == std::string::npos ? body : body.substr(last_nl + 1)});
    }
    return out;
}

using Ends = std::vector<std::pair<std::string, std::string>>;

PFlashSelectionCandidate seg(size_t ordinal, int begin, int end, double density,
                             bool mandatory = false) {
    return {ordinal, begin, end, density, mandatory};
}

// Ten 10-token segments over [0, 100); the last one mandatory.
std::vector<PFlashSelectionCandidate> ten(const std::vector<double> & density) {
    std::vector<PFlashSelectionCandidate> out;
    for (size_t i = 0; i < 10; ++i) {
        out.push_back(seg(i, (int) i * 10, (int) i * 10 + 10, density[i], i == 9));
    }
    return out;
}

bool same(const std::vector<PFlashTokenSpan> & a, const std::vector<PFlashTokenSpan> & b) {
    return a == b;
}

} // namespace

TEST_CASE(PFlashFoldFixture, python_blocks_follow_indentation_brackets_and_strings) {
    const std::string text =
        "import os\n"
        "\n"
        "@decorator\n"
        "def top(a,\n"
        "        b):\n"
        "    \"\"\"Doc.\n"
        "Column-0 docstring line.\n"
        "    \"\"\"\n"
        "    x = {\n"
        "'k': 1}\n"
        "    return x\n"
        "    # trailing comment\n"
        "\n"
        "class A(Base):\n"
        "    y = 1\n"
        "\n"
        "    async def m(self):\n"
        "        pass\n"
        "\n"
        "def tail(): return 1\n";
    const auto blocks = pflash_python_blocks(text, whole(text));
    REQUIRE(ends_of(text, blocks) == Ends({
        {"@decorator", "    return x"},
        {"class A(Base):", "        pass"},
        {"    async def m(self):", "        pass"},
        {"def tail(): return 1", "def tail(): return 1"},
    }));
    // Python has no brace bodies.
    REQUIRE(pflash_brace_blocks(text, whole(text)).empty());
}

TEST_CASE(PFlashFoldFixture, python_blocks_survive_a_broken_signature) {
    // An unclosed bracket (the deliberate bug of a CodeDebug file) must not
    // swallow the next def.
    const std::string text =
        "def broken(a, b:\n"
        "    return a\n"
        "\n"
        "    def after(self):\n"
        "        return 2\n";
    const auto blocks = pflash_python_blocks(text, whole(text));
    REQUIRE(ends_of(text, blocks) == Ends({
        {"def broken(a, b:", "        return 2"},
        {"    def after(self):", "        return 2"},
    }));
}

TEST_CASE(PFlashFoldFixture, brace_blocks_find_c_and_cpp_functions_and_types) {
    const std::string text =
        "#include <vector>\n"
        "namespace ns {\n"
        "\n"
        "// Adds.\n"
        "static int add(int a, int b) {\n"
        "    if (a > b) {\n"
        "        return a;\n"
        "    }\n"
        "    return a + b;\n"
        "}\n"
        "\n"
        "class Widget : public Base {\n"
        "public:\n"
        "    void draw() const override {\n"
        "        for (int i = 0; i < 3; ++i) { paint(i); }\n"
        "    }\n"
        "};\n"
        "\n"
        "struct Point { int x; int y; };\n"
        "int table[] = {1, 2, 3};\n"
        "\n"
        "void\n"
        "allman(int x)\n"
        "{\n"
        "    char c = '{';\n"
        "    const char * s = \"}\";\n"
        "}\n"
        "\n"
        "}  // namespace ns\n";
    const auto blocks = pflash_brace_blocks(text, whole(text));
    REQUIRE(ends_of(text, blocks) == Ends({
        {"static int add(int a, int b) {", "}"},
        {"class Widget : public Base {", "};"},
        {"    void draw() const override {", "    }"},
        {"struct Point { int x; int y; };", "struct Point { int x; int y; };"},
        {"allman(int x)", "}"},
    }));
}

TEST_CASE(PFlashFoldFixture, brace_blocks_find_js_and_ts_declarations_not_callbacks) {
    const std::string text =
        "import x from 'y';\n"
        "export function load(path: string): Promise<Data> {\n"
        "  return fetch(path).then((r) => {\n"
        "    return r.json();\n"
        "  });\n"
        "}\n"
        "export const handler = async (event) => {\n"
        "  const obj = { a: 1, b: { c: 2 } };\n"
        "  return obj;\n"
        "};\n"
        "class Store extends Base {\n"
        "  get size() {\n"
        "    return `${this.items.length}}`;\n"
        "  }\n"
        "}\n"
        "interface Props {\n"
        "  name: string;\n"
        "}\n"
        "describe('store', () => {\n"
        "  it('works', () => {});\n"
        "});\n";
    const auto blocks = pflash_brace_blocks(text, whole(text));
    REQUIRE(ends_of(text, blocks) == Ends({
        {"export function load(path: string): Promise<Data> {", "}"},
        {"export const handler = async (event) => {", "};"},
        {"class Store extends Base {", "}"},
        {"  get size() {", "  }"},
        {"interface Props {", "}"},
    }));
}

TEST_CASE(PFlashFoldFixture, brace_blocks_find_go_functions_methods_and_types) {
    const std::string text =
        "package main\n"
        "\n"
        "import \"fmt\"\n"
        "\n"
        "type Server struct {\n"
        "\taddr string\n"
        "}\n"
        "\n"
        "func (s *Server) Start(port int) error {\n"
        "\tif port == 0 {\n"
        "\t\treturn fmt.Errorf(\"bad\")\n"
        "\t}\n"
        "\tfor i := range s.addr {\n"
        "\t\t_ = i\n"
        "\t}\n"
        "\treturn nil\n"
        "}\n"
        "\n"
        "func main() {\n"
        "\tgo func() {\n"
        "\t\tfmt.Println(\"x\")\n"
        "\t}()\n"
        "}\n";
    const auto blocks = pflash_brace_blocks(text, whole(text));
    REQUIRE(ends_of(text, blocks) == Ends({
        {"type Server struct {", "}"},
        {"func (s *Server) Start(port int) error {", "}"},
        {"func main() {", "}"},
    }));
}

TEST_CASE(PFlashFoldFixture, brace_blocks_find_rust_items_through_lifetimes_and_match_arms) {
    const std::string text =
        "use std::fmt;\n"
        "\n"
        "pub struct Wrapper<'a> {\n"
        "    inner: &'a str,\n"
        "}\n"
        "\n"
        "impl<'a> fmt::Display for Wrapper<'a> {\n"
        "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
        "        match self.inner {\n"
        "            \"\" => write!(f, \"empty\"),\n"
        "            s => {\n"
        "                write!(f, \"{}\", s)\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
        "\n"
        "fn main() {\n"
        "    let c = '}';\n"
        "    println!(\"{}\", c);\n"
        "}\n";
    const auto blocks = pflash_brace_blocks(text, whole(text));
    REQUIRE(ends_of(text, blocks) == Ends({
        {"pub struct Wrapper<'a> {", "}"},
        {"impl<'a> fmt::Display for Wrapper<'a> {", "}"},
        {"    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {", "    }"},
        {"fn main() {", "}"},
    }));
}

TEST_CASE(PFlashFoldFixture, code_gate_takes_code_and_rejects_json_markdown_and_prose) {
    std::string python;
    for (int i = 0; i < 6; ++i) {
        python += "def f" + std::to_string(i) + "(x):\n"
                  "    y = x + 1\n"
                  "    return {'v': y}\n\n";
    }
    const auto code = pflash_fold_structure("import os\n\n" + python);
    REQUIRE(code.code_like);
    REQUIRE(code.code_blocks == 6);

    // JSON dialogue and few-shot data: every line looks like code to a
    // punctuation count (the old gate routed such prompts to the code
    // branch), none is inside a function.
    std::string json = "[\n";
    for (int i = 0; i < 20; ++i) {
        json += "  {\"role\": \"user\", \"content\": \"def f(x): return {x}\"},\n"
                "  {\"role\": \"assistant\", \"content\": \"f(1) { ok }\"},\n";
    }
    json += "  {\"examples\": {\"input\": [1, 2, {\"k\": \"v\"}], \"output\": null}}\n]\n";
    const auto dialog = pflash_fold_structure(json);
    REQUIRE(!dialog.code_like);
    REQUIRE(dialog.code_blocks == 0);

    const std::string markdown =
        "# Setup\n\nInstall the package, then run it.\n\n"
        "```python\ndef main():\n    print('hi')\n```\n\n"
        "## Usage\n\nCall `main()` from your script. It prints a greeting.\n"
        "The function takes no arguments and returns nothing useful.\n";
    const auto md = pflash_fold_structure(markdown);
    REQUIRE(!md.code_like);
    REQUIRE(std::string(md.text_source) == "headings");

    const std::string prose =
        "The committee met on Tuesday (after a long delay) to review the plan.\n"
        "Its members argued about the budget {which was late} and the schedule.\n\n"
        "In the end they agreed to meet again; the chair closed the session.\n";
    const auto p = pflash_fold_structure(prose);
    REQUIRE(!p.code_like);
    REQUIRE(p.code_blocks == 0);
}

TEST_CASE(PFlashFoldFixture, text_blocks_prefer_the_widest_numbered_family) {
    const std::string text =
        "Answer from the documents.\n\n"
        "Document 1: Alpha\nLine 1: a\nLine 2: b\nLine 3: c\n\n"
        "Document 2: Beta\nbeta text\n\n"
        "Document 3: Gamma\ngamma text\n";
    const char * source = nullptr;
    const auto blocks = pflash_text_blocks(text, whole(text), {}, &source);
    REQUIRE(std::string(source) == "records");
    REQUIRE(ends_of(text, blocks) == Ends({
        {"Document 1: Alpha", "Line 3: c"},
        {"Document 2: Beta", "beta text"},
        {"Document 3: Gamma", "gamma text"},
    }));
    // Two numbers are not a family; citations wrapped to a line start and
    // numbers glued to a word are not headers.
    const std::string weak =
        "Chapter 1: One\ntext\nChapter 4: Four\ntext\n[36], as shown\nks2.run()\n";
    const auto none = pflash_text_blocks(weak, whole(weak), {}, &source, /*paragraphs=*/false);
    REQUIRE(none.empty());
}

TEST_CASE(PFlashFoldFixture, text_blocks_fall_back_to_sections_then_paragraphs) {
    const std::string md = "# A\na\n## B\nb\n# C\nc\n";
    const char * source = nullptr;
    const auto sections = pflash_text_blocks(md, whole(md), {}, &source);
    REQUIRE(std::string(source) == "headings");
    REQUIRE(ends_of(md, sections) == Ends({{"# A", "b"}, {"## B", "b"}, {"# C", "c"}}));

    const std::string para = "one\none more\n\n\ntwo\n";
    const auto paragraphs = pflash_text_blocks(para, whole(para), {}, &source);
    REQUIRE(std::string(source) == "paragraphs");
    REQUIRE(ends_of(para, paragraphs) == Ends({{"one", "one more"}, {"two", "two"}}));
    REQUIRE(pflash_text_blocks(para, whole(para), {}, &source, false).empty());

    // A body with code gets no sections or paragraphs (Python comments are
    // not headings).
    const std::string code = "# comment\ndef f():\n    return 1\n\n# other\nx = 2\n";
    const auto py = pflash_python_blocks(code, whole(code));
    REQUIRE(pflash_text_blocks(code, whole(code), py, &source).empty());
}

TEST_CASE(PFlashFoldFixture, message_bodies_exclude_chat_markers) {
    const std::string text =
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU1\nU2<|im_end|>\n"
        "<|im_start|>assistant\n";
    const auto bodies = pflash_message_bodies(text);
    std::vector<std::string> got;
    for (const auto & b : bodies) got.push_back(text.substr(b.begin, b.end - b.begin));
    REQUIRE(got == std::vector<std::string>({"S", "\n", "U1\nU2", "\n"}));
    REQUIRE(pflash_message_bodies("plain").size() == 1);
}

TEST_CASE(PFlashFoldFixture, token_blocks_map_bytes_and_stop_at_kept_spans) {
    // Tokens start at bytes 0, 4, 8, ... 36.
    std::vector<size_t> token_begin;
    for (size_t i = 0; i < 10; ++i) token_begin.push_back(i * 4);
    REQUIRE(pflash_text_to_token_span(token_begin, {5, 13}) == PFlashTokenSpan({1, 4}));
    REQUIRE(pflash_text_to_token_span(token_begin, {8, 12}) == PFlashTokenSpan({2, 3}));
    const std::vector<PFlashSelectionCandidate> candidates{
        seg(0, 0, 3, 1.0, true), seg(1, 3, 6, 1.0), seg(2, 6, 8, 1.0, true), seg(3, 8, 10, 1.0)};
    const auto blocks = pflash_fold_token_blocks({{0, 20}, {16, 40}, {26, 30}}, token_begin, candidates);
    // [0,5) starts in a kept span -> [3,5); [4,10) runs into one -> [4,6);
    // [6,8) lies inside one -> dropped.
    REQUIRE(same(blocks, {{3, 5}, {4, 6}}));
}

TEST_CASE(PFlashFoldFixture, fold_completes_anchors_to_blocks_with_cap_head_and_budget) {
    // Segment 1 is the densest; its block [0, 30) fits under the cap.
    auto candidates = ten({1, 5, 1, 1, 1, 4, 1, 1, 1, 0});
    PFlashFoldPolicy policy;
    policy.token_budget = 100;
    policy.k = 1;
    policy.cap = 50;
    policy.head = 5;
    const std::vector<PFlashTokenSpan> blocks{{0, 30}, {30, 90}};
    auto r = select_pflash_fold(candidates, blocks, policy);
    REQUIRE(r.ok);
    REQUIRE(same(r.kept, {{0, 30}, {90, 100}}));
    REQUIRE(r.retained_tokens == 40);
    REQUIRE(r.anchors == std::vector<size_t>({1}));
    REQUIRE(r.stop == PFlashSelectionStop::TopKReached);

    // Over the cap: the block's first five tokens plus the anchor.
    candidates = ten({1, 1, 1, 1, 1, 5, 1, 1, 1, 0});
    r = select_pflash_fold(candidates, blocks, policy);
    REQUIRE(same(r.kept, {{30, 35}, {50, 60}, {90, 100}}));
    REQUIRE(r.capped == 1);

    // Two anchors in one block count once; the third anchor goes on.
    candidates = ten({5, 4, 1, 1, 1, 1, 3, 1, 1, 0});
    policy.k = 2;
    r = select_pflash_fold(candidates, blocks, policy);
    REQUIRE(same(r.kept, {{0, 35}, {60, 70}, {90, 100}}));
    REQUIRE(r.anchors == std::vector<size_t>({0, 6}));

    // A completion over the budget is skipped, a smaller one later fits.
    policy.k = 1;
    policy.token_budget = 25;
    candidates = ten({5, 1, 1, 1, 1, 4, 1, 1, 1, 0});
    r = select_pflash_fold(candidates, blocks, policy);
    REQUIRE(same(r.kept, {{30, 35}, {50, 60}, {90, 100}}));
    REQUIRE(r.skipped == 1);

    // Mandatory spans over the budget fail closed.
    policy.token_budget = 5;
    r = select_pflash_fold(candidates, blocks, policy);
    REQUIRE(!r.ok);
    REQUIRE(r.stop == PFlashSelectionStop::MandatoryQueryExceedsBudget);
}

TEST_CASE(PFlashFoldFixture, fold_ranks_by_mass_on_code_and_completes_straddled_blocks) {
    // A short dense segment and a long less dense one: density takes the
    // short, mass (density x length) the long.
    const std::vector<PFlashSelectionCandidate> candidates{
        seg(0, 0, 5, 4.0), seg(1, 5, 45, 1.0), seg(2, 45, 50, 0.0, true)};
    PFlashFoldPolicy policy;
    policy.token_budget = 50;
    policy.k = 1;
    auto r = select_pflash_fold(candidates, {}, policy);
    REQUIRE(same(r.kept, {{0, 5}, {45, 50}}));
    policy.rank = PFlashFoldRank::Mass;
    r = select_pflash_fold(candidates, {}, policy);
    REQUIRE(same(r.kept, {{5, 50}}));
    REQUIRE(r.contained == 0);

    // An anchor across two blocks completes both.
    policy.rank = PFlashFoldRank::Density;
    const std::vector<PFlashSelectionCandidate> across{
        seg(0, 0, 10, 0.0), seg(1, 10, 20, 2.0), seg(2, 20, 30, 0.0), seg(3, 30, 40, 0.0, true)};
    r = select_pflash_fold(across, {{0, 15}, {15, 25}}, policy);
    REQUIRE(same(r.kept, {{0, 25}, {30, 40}}));
    REQUIRE(r.contained == 1);
}

TEST_CASE(PFlashFoldFixture, fold_strategy_config_is_explicit_and_fails_closed) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", nullptr};
    luce_test::ScopedEnvVar strategy{"PFLASH_SELECT_STRATEGY", "fold"};
    luce_test::ScopedEnvVar scorer{"PFLASH_SELECT_SCORER", nullptr};
    luce_test::ScopedEnvVar k{"PFLASH_SELECT_FOLD_K", nullptr};
    luce_test::ScopedEnvVar cap{"PFLASH_SELECT_FOLD_CAP", nullptr};
    luce_test::ScopedEnvVar head{"PFLASH_SELECT_FOLD_HEAD", nullptr};
    luce_test::ScopedEnvVar paragraphs{"PFLASH_SELECT_FOLD_PARAGRAPHS", nullptr};
    PFlashSelectionConfig config;
    std::string error;
    REQUIRE(has_pflash_selection_environment());
    REQUIRE(pflash_fold_requested());
    // Fold needs the budget rule.
    REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
    REQUIRE(error.find("budget_only") != std::string::npos);
    {
        luce_test::ScopedEnvVar budget{"PFLASH_SELECT_MODE", "budget_only"};
        REQUIRE(resolve_pflash_selection(32768, 32, config, error));
        REQUIRE(config.strategy == PFlashSelectStrategy::Fold);
        REQUIRE(config.fold_k == 20);
        REQUIRE(config.fold_cap == 1500);
        REQUIRE(config.fold_head == 150);
        REQUIRE(config.fold_paragraphs);
        {
            luce_test::ScopedEnvVar split{"PFLASH_SELECT_SCORER", "split"};
            REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
        }
        {
            luce_test::ScopedEnvVar big_head{"PFLASH_SELECT_FOLD_HEAD", "2000"};
            REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
        }
        {
            luce_test::ScopedEnvVar bad_k{"PFLASH_SELECT_FOLD_K", "0"};
            REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
        }
        {
            luce_test::ScopedEnvVar off{"PFLASH_SELECT_FOLD_PARAGRAPHS", "0"};
            REQUIRE(resolve_pflash_selection(32768, 32, config, error));
            REQUIRE(!config.fold_paragraphs);
        }
        {
            luce_test::ScopedEnvVar unknown{"PFLASH_SELECT_STRATEGY", "folds"};
            REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
            REQUIRE(error.find("segments or fold") != std::string::npos);
        }
    }
    luce_test::ScopedEnvVar segments{"PFLASH_SELECT_STRATEGY", "segments"};
    luce_test::ScopedEnvVar budget{"PFLASH_SELECT_MODE", "budget_only"};
    REQUIRE(resolve_pflash_selection(32768, 32, config, error));
    REQUIRE(config.strategy == PFlashSelectStrategy::Segments);
    REQUIRE(std::string(pflash_select_strategy_name(PFlashSelectStrategy::Fold)) == "fold");
}

TEST_CASE(PFlashFoldFixture, select_chunks_folds_through_the_drafter_vocabulary) {
    // A byte-level vocabulary: "Document 1: a\nx\n\nDocument 2: b\ny\n\n
    // Document 3: c\nz\n\nQ", one piece per token.
    const char * tokens[] = {"Document", "\xc4\xa0" "1", ":", "\xc4\xa0" "a", "\xc4\x8a", "x",
                             "\xc4\x8a\xc4\x8a", "\xc4\xa0" "2", "\xc4\xa0" "b", "y",
                             "\xc4\xa0" "3", "\xc4\xa0" "c", "z", "Q"};
    gguf_context * g = gguf_init_empty();
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tokens, sizeof(tokens) / sizeof(tokens[0]));
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen35");
    const std::string path = (std::filesystem::temp_directory_path() /
        ("luce_test_pflash_fold_" +
         std::to_string(reinterpret_cast<uintptr_t>(&g)) + ".gguf")).string();
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    luce::common::Tokenizer vocab;
    REQUIRE(vocab.load_from_gguf(path.c_str()));
    std::remove(path.c_str());

    const std::vector<int32_t> ids{0, 1, 2, 3, 4, 5, 6, 0, 7, 2, 8, 4, 9, 6,
                                   0, 10, 2, 11, 4, 12, 6, 13};
    REQUIRE(vocab.decode(ids) == "Document 1: a\nx\n\nDocument 2: b\ny\n\nDocument 3: c\nz\n\nQ");
    const std::vector<PFlashTokenSpan> segments{
        {0, 4}, {4, 7}, {7, 10}, {10, 14}, {14, 18}, {18, 21}, {21, 22}};
    std::vector<float> mass(ids.size(), 0.01f);
    mass[12] = 0.5f;   // "y", inside Document 2
    PFlashSelectionConfig config;
    config.mode = PFlashSelectionMode::BudgetOnly;
    config.selection_active = true;
    config.strategy = PFlashSelectStrategy::Fold;
    config.fold_k = 1;
    const auto run = [&](const luce::common::Tokenizer * v) {
        return luce::common::select_pflash_chunks(
            ids, mass, /*keep_ratio=*/0.5f, /*n_lookahead=*/1, /*score_query_end=*/22,
            /*pool_kernel=*/1, config, {}, /*direct_mass=*/true, /*write_trace=*/false,
            &segments, /*density=*/true, nullptr, 0.0, v);
    };
    const auto out = run(&vocab);
    // The anchor [10, 14) completes to its record, Document 2 = [7, 14); the
    // query token stays.
    REQUIRE(out == std::vector<int32_t>({0, 7, 2, 8, 4, 9, 6, 13}));
    const auto & kept = luce::common::pflash_last_kept_spans();
    REQUIRE(same(kept, {{7, 14}, {21, 22}}));
    // Without the vocabulary fold fails closed.
    REQUIRE(run(nullptr).empty());
}

namespace {

std::string block_text(const std::string & text, PFlashTextSpan s) {
    return text.substr(s.begin, s.end - s.begin);
}

std::string pasted_paper() {
    std::string out = "## Paper: A Study\n\n";
    for (int i = 0; i < 40; ++i) {
        out += "Paragraph " + std::to_string(i) + " of the paper describes the method in detail.\n\n";
    }
    return out;
}

} // namespace

TEST_CASE(PFlashFoldFixture, query_blocks_find_the_question_before_the_paste) {
    const std::string text = "Which KV budget does the paper use on LiveCodeBench?\n\n"
                             "Here is the material I'm working with.\n\n" + pasted_paper();
    const auto blocks = pflash_query_blocks(text, whole(text));
    REQUIRE(blocks.structured);
    REQUIRE(std::string(blocks.head_rule) == "paste");
    REQUIRE(block_text(text, blocks.head) ==
            "Which KV budget does the paper use on LiveCodeBench?\n\n"
            "Here is the material I'm working with.");
    // The paste ends in its own prose: that paragraph is the tail block.
    REQUIRE(block_text(text, blocks.tail) ==
            "Paragraph 39 of the paper describes the method in detail.");
}

TEST_CASE(PFlashFoldFixture, query_blocks_find_the_question_after_and_on_both_sides) {
    const std::string after = pasted_paper() + "What does paragraph 3 describe?\n";
    auto blocks = pflash_query_blocks(after, whole(after));
    REQUIRE(blocks.structured);
    REQUIRE(blocks.head.end == blocks.head.begin);          // opens with the paste
    REQUIRE(block_text(after, blocks.tail) == "What does paragraph 3 describe?");

    const std::string both = "I am comparing two methods.\n\n```python\ndef f():\n    return 1\n```\n\n"
        "Which one is faster, and why?\n";
    const std::string padded = both + std::string();   // short but structured
    blocks = pflash_query_blocks(padded, whole(padded));
    REQUIRE(block_text(padded, blocks.head) == "I am comparing two methods.");
    REQUIRE(block_text(padded, blocks.tail) == "Which one is faster, and why?");
}

TEST_CASE(PFlashFoldFixture, query_blocks_keep_a_long_question_and_leave_plain_text_alone) {
    // A long question (a few hundred words) is still one head block; the
    // server caps it at its first and last cap/2 tokens.
    std::string question;
    for (int i = 0; i < 60; ++i) question += "Consider constraint number " + std::to_string(i) + " carefully. ";
    question += "Which setting satisfies all of them?";
    std::string paste;
    for (int i = 0; i < 6; ++i) paste += pasted_paper();
    const std::string text = question + "\n\n" + paste;
    auto blocks = pflash_query_blocks(text, whole(text));
    REQUIRE(block_text(text, blocks.head) == question);

    // No paste structure and no opening question: nothing changes.
    std::string plain;
    for (int i = 0; i < 50; ++i) plain += "A plain sentence of a long letter, line " + std::to_string(i) + ".\n";
    blocks = pflash_query_blocks(plain, whole(plain));
    REQUIRE(!blocks.structured);
    REQUIRE(blocks.head.end == blocks.head.begin);
    REQUIRE(blocks.tail.end == blocks.tail.begin);

    // A plain prose paste after an opening question: the question is the head.
    const std::string asked = "Who wrote the letter?\n\n" + plain;
    blocks = pflash_query_blocks(asked, whole(asked));
    REQUIRE(!blocks.structured);
    REQUIRE(std::string(blocks.head_rule) == "question");
    REQUIRE(block_text(asked, blocks.head) == "Who wrote the letter?");
    REQUIRE(blocks.tail.end == blocks.tail.begin);
}

TEST_CASE(PFlashFoldFixture, query_blocks_give_code_only_messages_no_blocks) {
    std::string code;
    for (int i = 0; i < 20; ++i) {
        code += "def f" + std::to_string(i) + "(x):\n    return x + " + std::to_string(i) + "\n\n";
    }
    const auto blocks = pflash_query_blocks(code, whole(code));
    REQUIRE(blocks.structured);
    REQUIRE(blocks.head.end == blocks.head.begin);
    REQUIRE(blocks.tail.end == blocks.tail.begin);
    // JSON data is pasted material too.
    std::string data = "Summarise these records for me please.\n\n[\n";
    for (int i = 0; i < 10; ++i) data += "  {\"id\": " + std::to_string(i) + ", \"ok\": true},\n";
    data += "]\n";
    const auto json_blocks = pflash_query_blocks(data, whole(data));
    REQUIRE(block_text(data, json_blocks.head) == "Summarise these records for me please.");
    REQUIRE(json_blocks.tail.end == json_blocks.tail.begin);
}

TEST_CASE(PFlashFoldFixture, query_head_config_needs_latest_user_and_bounded_caps) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar parser{"PFLASH_SELECT_QUERY_PARSER", nullptr};
    luce_test::ScopedEnvVar head{"PFLASH_SELECT_QUERY_HEAD", "1"};
    luce_test::ScopedEnvVar head_tokens{"PFLASH_SELECT_QUERY_HEAD_TOKENS", nullptr};
    luce_test::ScopedEnvVar tail_tokens{"PFLASH_SELECT_QUERY_TAIL_TOKENS", nullptr};
    PFlashSelectionConfig config;
    std::string error;
    // The chat tail parser has no user-message boundary to read.
    REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
    REQUIRE(error.find("latest_user") != std::string::npos);
    luce_test::ScopedEnvVar latest{"PFLASH_SELECT_QUERY_PARSER", "latest_user"};
    REQUIRE(resolve_pflash_selection(32768, 32, config, error));
    REQUIRE(config.query_head);
    REQUIRE(config.query_head_tokens == 256);
    REQUIRE(config.query_tail_tokens == 256);
    {
        luce_test::ScopedEnvVar big{"PFLASH_SELECT_QUERY_TAIL_TOKENS", "1024"};
        REQUIRE(!resolve_pflash_selection(32768, 32, config, error));
    }
    {
        luce_test::ScopedEnvVar off{"PFLASH_SELECT_QUERY_HEAD", "0"};
        luce_test::ScopedEnvVar caps{"PFLASH_SELECT_QUERY_HEAD_TOKENS", "128"};
        REQUIRE(resolve_pflash_selection(32768, 32, config, error));
        REQUIRE(!config.query_head);
        REQUIRE(config.query_head_tokens == 128);
    }
}
