#include "pin_friendly_prompt.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

int PinFriendlyPrompt::longest_common_prefix_len(
        const std::vector<int32_t> & a,
        const std::vector<int32_t> & b) {
    const int n = (int)std::min(a.size(), b.size());
    int i = 0;
    while (i < n && a[(size_t)i] == b[(size_t)i]) ++i;
    return i;
}

int PinFriendlyPrompt::longest_common_suffix_len(
        const std::vector<int32_t> & a,
        const std::vector<int32_t> & b,
        int prefix_len) {
    int i = (int)a.size() - 1;
    int j = (int)b.size() - 1;
    int s = 0;
    while (i >= prefix_len && j >= prefix_len &&
           a[(size_t)i] == b[(size_t)j]) {
        --i;
        --j;
        ++s;
    }
    return s;
}

TokenDiffSplit PinFriendlyPrompt::diff_split(
        const std::vector<int32_t> & reference,
        const std::vector<int32_t> & current) {
    TokenDiffSplit out;
    out.prefix_len = longest_common_prefix_len(reference, current);
    out.suffix_len = longest_common_suffix_len(
        reference, current, out.prefix_len);
    out.middle_begin = out.prefix_len;
    out.middle_end = (int)current.size() - out.suffix_len;
    if (out.middle_end < out.middle_begin) {
        out.middle_end = out.middle_begin;
        out.suffix_len = (int)current.size() - out.prefix_len;
    }
    return out;
}

int PinFriendlyPrompt::safe_boundary_cut(
        int n, const std::vector<int> & boundaries) {
    if (n <= 0) return 0;
    int best = 0;
    for (int b : boundaries) {
        if (b > 0 && b <= n) best = std::max(best, b);
    }
    return best;
}

int PinFriendlyPrompt::choose_pin_end(
        int lcp,
        const std::vector<int> & boundaries,
        int min_pin_tokens) {
    if (lcp < min_pin_tokens) return 0;
    const int boundary = safe_boundary_cut(lcp, boundaries);
    if (boundary >= min_pin_tokens) return boundary;
    return lcp;
}

int PinFriendlyPrompt::annotate_pin_end(
        const std::vector<int32_t> & tokens,
        const std::vector<int> & boundaries,
        const std::vector<std::vector<int32_t>> & recent_tool_prefixes,
        int window,
        int min_pin_tokens) {
    if (tokens.empty() || recent_tool_prefixes.empty() || window <= 0) {
        return 0;
    }
    const int n = std::min(window, (int)recent_tool_prefixes.size());
    int common = 0;
    for (int i = 0; i < n; ++i) {
        const int idx = (int)recent_tool_prefixes.size() - n + i;
        common = std::max(
            common,
            longest_common_prefix_len(tokens, recent_tool_prefixes[(size_t)idx]));
    }
    return choose_pin_end(common, boundaries, min_pin_tokens);
}

int PinFriendlyPrompt::trailing_end_marker_len(
        const std::vector<int32_t> & ids,
        const ChatMarkers & markers) {
    if (ids.empty()) return 0;
    int best = 0;
    for (const auto & seq : markers.end_msg_seqs) {
        if (seq.empty() || (int)seq.size() > (int)ids.size()) continue;
        const int start = (int)ids.size() - (int)seq.size();
        bool match = true;
        for (int i = 0; i < (int)seq.size(); ++i) {
            if (ids[(size_t)(start + i)] != seq[(size_t)i]) {
                match = false;
                break;
            }
        }
        if (match) best = std::max(best, (int)seq.size());
    }
    // Qwen boundaries often include a trailing '\n' token after im_end.
    // Peek one extra equal-length attempt is unnecessary; keep marker only.
    return best;
}

int PinFriendlyPrompt::tools_system_head_end(
        const std::vector<int32_t> & tokens,
        const ChatMarkers & markers) {
    if (tokens.empty() || markers.end_msg_seqs.empty()) return 0;
    int best = -1;
    int best_len = 0;
    for (const auto & seq : markers.end_msg_seqs) {
        if (seq.empty() || (int)seq.size() > (int)tokens.size()) continue;
        for (int i = 0; i + (int)seq.size() <= (int)tokens.size(); ++i) {
            bool match = true;
            for (int k = 0; k < (int)seq.size(); ++k) {
                if (tokens[(size_t)(i + k)] != seq[(size_t)k]) {
                    match = false;
                    break;
                }
            }
            if (!match) continue;
            if (best < 0 || i < best) {
                best = i;
                best_len = (int)seq.size();
            }
            break;  // first occurrence of this seq; keep scanning other seqs
        }
    }
    if (best < 0) return 0;
    return best + best_len;
}

PinFriendlyRewrite PinFriendlyPrompt::diff_make_pin_friendly(
        const std::vector<int32_t> & tokens,
        const std::vector<int> & boundaries,
        const std::vector<std::vector<int32_t>> & recent_tool_prefixes,
        const ChatMarkers & markers,
        int window,
        int min_pin_tokens,
        int max_ephemeral_tokens) {
    PinFriendlyRewrite out;
    out.tokens = tokens;
    if (tokens.empty() || recent_tool_prefixes.empty() || window <= 0) {
        return out;
    }
    // Custom / unstructured templates: do not rewrite the whole prompt.
    if (boundaries.empty()) return out;

    // Only rewrite through the first end-of-message marker. find_all_boundaries
    // returns cuts *after* the next role-start; using those would let the
    // volatile middle float past user/assistant markers.
    const int head_end = tools_system_head_end(tokens, markers);
    if (head_end <= 0 || head_end < min_pin_tokens) return out;

    std::vector<int32_t> head(tokens.begin(), tokens.begin() + head_end);
    const std::vector<int32_t> rest(tokens.begin() + head_end, tokens.end());

    const int trailer_len = trailing_end_marker_len(head, markers);
    std::vector<int32_t> trailer;
    if (trailer_len > 0) {
        trailer.assign(head.end() - trailer_len, head.end());
        head.resize((size_t)((int)head.size() - trailer_len));
    }
    if ((int)head.size() < min_pin_tokens) return out;

    // Pick the reference that yields the largest stable span
    // (prefix + suffix) with a small ephemeral middle.
    TokenDiffSplit best{};
    int best_stable = -1;
    bool found = false;
    const int n = std::min(window, (int)recent_tool_prefixes.size());
    for (int i = 0; i < n; ++i) {
        const int idx = (int)recent_tool_prefixes.size() - n + i;
        const auto & ref_full = recent_tool_prefixes[(size_t)idx];
        // Align reference to body length (strip its own trailer if present).
        std::vector<int32_t> ref = ref_full;
        if ((int)ref.size() > head_end) {
            ref.resize((size_t)head_end);
        }
        const int ref_trailer = trailing_end_marker_len(ref, markers);
        if (ref_trailer > 0 && ref_trailer < (int)ref.size()) {
            ref.resize((size_t)((int)ref.size() - ref_trailer));
        }
        if (ref.empty()) continue;

        const TokenDiffSplit split = diff_split(ref, head);
        const int middle = split.middle_end - split.middle_begin;
        const int stable = split.prefix_len + split.suffix_len;
        if (middle < 0 || middle > max_ephemeral_tokens) continue;
        if (stable < min_pin_tokens) continue;
        // Require a real middle relocation opportunity (suffix beyond prefix).
        if (middle == 0) continue;
        if (stable > best_stable) {
            best_stable = stable;
            best = split;
            found = true;
        }
    }

    if (!found) {
        // Fall back: LCP pin without rewrite.
        out.pin_end = annotate_pin_end(
            tokens, boundaries, recent_tool_prefixes, window, min_pin_tokens);
        return out;
    }

    std::vector<int32_t> new_head;
    new_head.reserve(head.size() + trailer.size());
    new_head.insert(new_head.end(),
                    head.begin(), head.begin() + best.prefix_len);
    if (best.suffix_len > 0) {
        new_head.insert(new_head.end(),
                        head.end() - best.suffix_len, head.end());
    }
    const int pin_body = (int)new_head.size();
    new_head.insert(new_head.end(),
                    head.begin() + best.middle_begin,
                    head.begin() + best.middle_end);
    new_head.insert(new_head.end(), trailer.begin(), trailer.end());

    out.tokens.clear();
    out.tokens.insert(out.tokens.end(), new_head.begin(), new_head.end());
    out.tokens.insert(out.tokens.end(), rest.begin(), rest.end());
    out.pin_end = pin_body;  // contiguous stable blob; volatile + trailer after
    // Prefer including trailer in the pin when it yields a chat boundary cut.
    if (!trailer.empty()) {
        const int with_trailer = pin_body + (int)trailer.size();
        // Only if volatile sits after trailer we would have moved wrong;
        // here volatile is before trailer, so pin stays at pin_body.
        (void)with_trailer;
    }
    out.rewritten = (out.tokens != tokens);
    out.prefix_len = best.prefix_len;
    out.suffix_len = best.suffix_len;
    out.middle_len = best.middle_end - best.middle_begin;

    if (out.pin_end < min_pin_tokens) {
        out.tokens = tokens;
        out.rewritten = false;
        out.pin_end = annotate_pin_end(
            tokens, boundaries, recent_tool_prefixes, window, min_pin_tokens);
    }
    return out;
}

static std::string trim_leading_newlines(std::string s) {
    while (!s.empty() && (s[0] == '\n' || s[0] == '\r')) {
        s.erase(s.begin());
    }
    return s;
}

std::pair<std::string, std::string>
PinFriendlyPrompt::split_ephemeral_system_tail(const std::string & system) {
    static const char * kMarkers[] = {
        "\nConversation started:",
        "\n\nConversation started:",
        "\nSession started:",
        "\n\nSession started:",
    };
    size_t cut = std::string::npos;
    for (const char * marker : kMarkers) {
        const size_t p = system.rfind(marker);
        if (p == std::string::npos) continue;
        if (cut == std::string::npos || p < cut) cut = p;
    }
    if (cut == std::string::npos || cut == 0) {
        return {system, {}};
    }
    std::string stable = system.substr(0, cut);
    while (!stable.empty() &&
           (stable.back() == ' ' || stable.back() == '\t' ||
            stable.back() == '\n' || stable.back() == '\r')) {
        stable.pop_back();
    }
    std::string ephemeral = trim_leading_newlines(system.substr(cut));
    if (ephemeral.empty()) return {system, {}};
    return {std::move(stable), std::move(ephemeral)};
}

PinFriendlyLayout PinFriendlyPrompt::rearrange(
        const std::vector<ChatMessage> & messages, bool enable) {
    PinFriendlyLayout layout;
    layout.messages = messages;
    if (!enable || messages.empty() || messages[0].role != "system") {
        return layout;
    }
    auto [stable, ephemeral] = split_ephemeral_system_tail(messages[0].content);
    if (ephemeral.empty() || stable.empty()) {
        return layout;
    }
    layout.messages[0].content = std::move(stable);
    ChatMessage meta;
    meta.role = "system";
    meta.content = std::move(ephemeral);
    layout.messages.insert(layout.messages.begin() + 1, std::move(meta));
    layout.rearranged = true;
    return layout;
}

void PinFriendlyPrompt::remember_tool_prefix(
        std::vector<std::vector<int32_t>> & ring,
        const std::vector<int32_t> & prompt_ids,
        int max_prefix_tokens,
        int window) {
    if (prompt_ids.empty() || window <= 0) return;
    const int n = std::min((int)prompt_ids.size(),
                           max_prefix_tokens > 0 ? max_prefix_tokens
                                                 : (int)prompt_ids.size());
    std::vector<int32_t> prefix(prompt_ids.begin(), prompt_ids.begin() + n);
    ring.push_back(std::move(prefix));
    while ((int)ring.size() > window) {
        ring.erase(ring.begin());
    }
}

}  // namespace dflash::common
