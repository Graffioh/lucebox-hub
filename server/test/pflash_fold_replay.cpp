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
//   pflash_fold_replay <drafter.gguf> query <rows.jsonl>
//       One JSON line per {"id", "text"} user message: its query blocks
//       (PFLASH_SELECT_QUERY_HEAD) with token counts.
//   pflash_fold_replay - request <rows.jsonl>...
//       The request probe's query set (PFLASH_SELECT_QUERY_PROBE=1) from
//       per-token request logits, with the runtime's own code. One row:
//         {"id", "tau": <f32 logit threshold>, "logits": [per token, null =
//          NaN], "messages": [{"role", "span": [begin, end)}, ...]}
//       messages in conversation order, spans in the logits' token
//       coordinates (the content of each message). Human = role "user"
//       exactly (tool results must carry role "tool"). Prints per row the
//       human messages read (latest first) with their runs, the query
//       spans and tokens, the fallback reason and the counts. No vocabulary
//       is loaded (the first argument is ignored). v2: an optional row
//       "policy" object (the PFLASH_REQUEST_PROBE_POLICY keys) and
//       "prompt_tokens" (default: the logits length) apply the runtime's
//       abstention (fallback "abstain:<criterion>", "abstain" stats) and
//       recency pin ("recency_pin").

#include "pflash/pflash_fold.h"
#include "pflash/pflash_request_probe.h"
#include "pflash/pflash_selection.h"
#include "server/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cmath>
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

// {"id", "text"} rows: the query blocks of a user message, with token counts.
int replay_query(const luce::common::Tokenizer & vocab, int argc, char ** argv) {
    for (int a = 3; a < argc; ++a) {
        std::ifstream in(argv[a]);
        std::string line;
        while (std::getline(in, line)) {
            const json row = json::parse(line);
            const std::string text = row.at("text").get<std::string>();
            const auto blocks = pflash_query_blocks(text, {0, text.size()});
            const auto block = [&](PFlashTextSpan s) {
                const std::string piece = text.substr(s.begin, s.end - s.begin);
                return json{{"bytes", {s.begin, s.end}},
                            {"tokens", s.end > s.begin ? vocab.encode(piece).size() : 0},
                            {"text", piece}};
            };
            json out = {{"id", row.value("id", std::string())}, {"structured", blocks.structured},
                        {"head_rule", blocks.head_rule}, {"head", block(blocks.head)},
                        {"tail", block(blocks.tail)}, {"message_tokens", vocab.encode(text).size()}};
            std::cout << out.dump() << "\n";
        }
    }
    return 0;
}

json request_spans_json(const std::vector<PFlashTokenSpan> & spans) {
    json out = json::array();
    for (const auto & s : spans) out.push_back({s.begin, s.end});
    return out;
}

int replay_request(int argc, char ** argv) {
    for (int a = 3; a < argc; ++a) {
        std::ifstream in(argv[a]);
        if (!in) {
            std::fprintf(stderr, "cannot read %s\n", argv[a]);
            return 1;
        }
        std::string line;
        int index = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const json row = json::parse(line);
            json out = {{"file", argv[a]}, {"line", index++}, {"id", row.value("id", json(nullptr))}};
            if (!row.contains("tau") || !row["tau"].is_number()) {
                out["error"] = "row needs a numeric tau";
                std::cout << out.dump() << "\n";
                continue;
            }
            const float tau = row["tau"].get<float>();
            std::vector<float> logits;
            for (const auto & v : row.at("logits")) {
                logits.push_back(v.is_null() ? std::nanf("") : v.get<float>());
            }
            // Human messages, latest first.
            std::vector<PFlashTokenSpan> human;
            const auto & messages = row.at("messages");
            for (size_t m = messages.size(); m-- > 0;) {
                if (messages[m].value("role", std::string()) != "user") continue;
                const auto & span = messages[m].at("span");
                human.push_back({span[0].get<int>(), span[1].get<int>()});
            }
            const auto set = pflash_request_query_set(logits, human, tau);
            PFlashRequestPolicy policy;
            if (row.contains("policy")) {
                std::string perr;
                if (!pflash_request_policy_apply_json(row["policy"].dump(), policy, perr)) {
                    out["error"] = perr;
                    std::cout << out.dump() << "\n";
                    continue;
                }
            }
            std::string fallback = set.fallback;
            json abstain = nullptr, pin = nullptr;
            if (fallback.empty()) {
                const int prompt_tokens = row.value("prompt_tokens", (int) logits.size());
                const auto a = pflash_request_abstain(set, logits, tau, prompt_tokens, policy);
                abstain = {{"reason", a.reason.empty() ? json(nullptr) : json(a.reason)}, {"mean_margin", a.mean_margin},
                           {"max_margin", a.max_margin}, {"latest_share", a.latest_share}, {"runs", a.runs},
                           {"prompt_tokens", a.prompt_tokens}};
                if (!a.reason.empty()) {
                    fallback = "abstain:" + a.reason;
                } else if (!set.messages.empty()) {
                    const auto p = pflash_request_recency_pin(set.messages[0].message, policy.recency_pin_tokens);
                    if (p.begin >= 0) pin = {p.begin, p.end};
                }
            }
            json read = json::array();
            for (const auto & m : set.messages) {
                read.push_back({{"span", {m.message.begin, m.message.end}},
                                {"runs", request_spans_json(m.runs)},
                                {"request_tokens", m.request_tokens},
                                {"at_or_above_tau", m.logits_at_or_above_tau},
                                {"turn_tokens", pflash_request_turn_tokens(m).size()}});
            }
            out["human_messages"] = human.size();
            out["messages"] = read;
            out["fallback"] = fallback.empty() ? json(nullptr) : json(fallback);
            out["abstain"] = abstain;
            out["recency_pin"] = pin;
            out["query_spans"] = request_spans_json(pflash_request_token_spans(set.tokens));
            out["query_tokens"] = set.tokens.size();
            out["tokens"] = set.tokens;
            out["previous_used"] = set.previous_used;
            out["latest_tokens"] = set.latest_tokens;
            out["previous_tokens"] = set.previous_tokens;
            out["uncapped_tokens"] = set.uncapped_tokens;
            out["capped"] = set.capped;
            out["extraction_version"] = kRequestExtractionVersion;
            out["min_run"] = kRequestMinRun;
            std::cout << out.dump() << "\n";
        }
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: pflash_fold_replay <drafter.gguf> traces|text|query <file.jsonl>...\n"
            "       pflash_fold_replay - request <rows.jsonl>...\n");
        return 2;
    }
    const std::string mode = argv[2];
    if (mode == "text") return replay_text(argc, argv);
    if (mode == "request") return replay_request(argc, argv);
    luce::common::Tokenizer vocab;
    if (!vocab.load_from_gguf(argv[1])) return 1;
    if (mode == "query") return replay_query(vocab, argc, argv);
    PFlashSelectionConfig config;
    std::string error;
    if (!resolve_pflash_selection(1, 32, config, error)) {
        std::fprintf(stderr, "config: %s\n", error.c_str());
        return 1;
    }
    return replay_traces(vocab, argc, argv, config);
}
