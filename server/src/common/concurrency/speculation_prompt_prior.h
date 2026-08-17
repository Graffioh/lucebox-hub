#pragma once

// Training-free request prior for cold-start speculative routing.
//
// This is deliberately a weak prior, not the final decision: obvious
// structured/code prompts are ranked ahead of neutral prompts, obvious
// conversational/creative prompts stay on AR, and measured verifier goodput
// remains authoritative after admission. The policy is model-neutral and can
// be replaced by a learned prompt ranker without changing the verifier.

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace dflash::common {

inline int speculation_prompt_hint(std::string_view prompt) {
    constexpr size_t kMaxPromptChars = 16 * 1024;
    std::string text(prompt.substr(0, kMaxPromptChars));
    std::transform(text.begin(), text.end(), text.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    auto has = [&](std::string_view cue) {
        return text.find(cue) != std::string::npos;
    };

    // Explicit exclusions override incidental structured words such as
    // "avoid code" in a creative-writing request.
    if (has("avoid code") || has("chatting casually") ||
        has("casual conversation") || has("small talk") ||
        has("keep it conversational") || has("write a story") ||
        has("invent a story") || has("write a poem") ||
        has("roleplay")) {
        return -1;
    }

    int score = 0;
    score += has("```") ? 4 : 0;
    score += has("\ndef ") || has("\nclass ") || has("#include") ? 4 : 0;
    score += has("public static") || has("fn ") || has("function ") ? 3 : 0;
    score += has("implement") || has("debug") || has("unit test") ? 3 : 0;
    score += has("algorithm") || has("sql query") ||
             has("regular expression") || has("json schema") ? 2 : 0;
    score += has("python") || has("javascript") || has("typescript") ||
             has("rust") || has("c++") ? 2 : 0;
    score += has("code") ? 1 : 0;

    if (score >= 3) return 1;
    if (has("story") || has("conversational") || has("brainstorm") ||
        has("opinion") || has("friendly chat")) {
        return -1;
    }
    return 0;
}

}  // namespace dflash::common
