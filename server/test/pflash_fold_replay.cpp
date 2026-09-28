// PFlash fold replay: runs the runtime's fold selection on recorded
// compression traces (CPU only, no model weights; the drafter GGUF supplies
// the vocabulary).
//
//   pflash_fold_replay <drafter.gguf> traces <trace.jsonl>...
//       One JSON line per trace line: the fold selection the runtime makes
//       from the trace's input_ids, segments, chunk_scores (density),
//       forced_chunks and token_budget, with PFLASH_SELECT_FOLD_* knobs.
//   pflash_fold_replay <drafter.gguf> text <rows.jsonl>
//       One JSON line per {"id", "prompt"} row: the structure and route of
//       the raw prompt text (no template, no selection).

#include "pflash/pflash_fold.h"
#include "pflash/pflash_selection.h"
#include "server/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace luce::pflash;
using luce::common::PFlashTokenSpan;

namespace {

json spans_json(const std::vector<PFlashTokenSpan> & spans) {
    json out = json::array();
    for (const auto & s : spans) out.push_back({s.begin, s.end});
    return out;
}

json structure_json(const PFlashFoldStructure & s) {
    return {{"code", s.code_like}, {"code_lines", s.code_lines},
            {"nonblank_lines", s.nonblank_lines}, {"code_blocks", s.code_blocks},
            {"text_source", s.text_source}};
}

int replay_traces(const luce::common::Tokenizer & vocab, int argc, char ** argv,
                  const PFlashSelectionConfig & config) {
    for (int a = 3; a < argc; ++a) {
        std::ifstream in(argv[a]);
        if (!in) {
            std::fprintf(stderr, "cannot read %s\n", argv[a]);
            return 1;
        }
        std::string line;
        int index = 0;
        while (std::getline(in, line)) {
            const json t = json::parse(line);
            const auto ids = t.at("input_ids").get<std::vector<int32_t>>();
            const auto & segments = t.at("segments");
            const auto & scores = t.at("chunk_scores");
            std::vector<uint8_t> forced(segments.size(), 0);
            for (const auto & f : t.at("forced_chunks")) forced[f.get<size_t>()] = 1;
            std::vector<PFlashSelectionCandidate> candidates;
            for (size_t i = 0; i < segments.size(); ++i) {
                const double score = scores[i].is_null() ? 0.0 : scores[i].get<double>();
                candidates.push_back({i, segments[i][0].get<int>(), segments[i][1].get<int>(),
                                      score, forced[i] != 0});
            }
            PFlashFoldStructure structure;
            std::vector<PFlashTokenSpan> blocks;
            const auto result = select_pflash_fold_for_ids(
                vocab, ids, candidates, config, t.at("token_budget").get<int>(),
                &structure, &blocks);
            json out = {{"file", argv[a]}, {"line", index++}, {"ok", result.ok},
                        {"error", result.error}, {"kept", spans_json(result.kept)},
                        {"anchors", result.anchors}, {"retained", result.retained_tokens},
                        {"skipped", result.skipped}, {"capped", result.capped},
                        {"contained", result.contained},
                        {"stop", pflash_selection_stop_name(result.stop)},
                        {"blocks", spans_json(blocks)},
                        {"structure", structure_json(structure)}};
            std::cout << out.dump() << "\n";
        }
    }
    return 0;
}

int replay_text(int argc, char ** argv) {
    for (int a = 3; a < argc; ++a) {
        std::ifstream in(argv[a]);
        std::string line;
        while (std::getline(in, line)) {
            const json row = json::parse(line);
            const std::string text = row.at("prompt").get<std::string>();
            const auto s = pflash_fold_structure(text);
            json blocks = json::array();
            for (const auto & b : s.blocks) blocks.push_back({b.begin, b.end});
            json out = structure_json(s);
            out["id"] = row.value("id", std::string());
            out["blocks"] = blocks;
            std::cout << out.dump() << "\n";
        }
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: pflash_fold_replay <drafter.gguf> traces|text <file.jsonl>...\n");
        return 2;
    }
    const std::string mode = argv[2];
    if (mode == "text") return replay_text(argc, argv);
    luce::common::Tokenizer vocab;
    if (!vocab.load_from_gguf(argv[1])) return 1;
    PFlashSelectionConfig config;
    std::string error;
    if (!resolve_pflash_selection(1, 32, config, error)) {
        std::fprintf(stderr, "config: %s\n", error.c_str());
        return 1;
    }
    return replay_traces(vocab, argc, argv, config);
}
