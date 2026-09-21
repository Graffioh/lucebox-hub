#include "qwen35_image_prompt.h"

#include <algorithm>
#include <cstring>

namespace dflash::common {

bool qwen35_expand_image_tokens(std::vector<int32_t> & tokens, int32_t image_pad,
                                std::vector<Qwen35ImageSlot> & slots, uint64_t max_tokens,
                                std::string & error) {
    const size_t pads = (size_t) std::count(tokens.begin(), tokens.end(), image_pad);
    if (pads != slots.size()) {
        error = "prompt has " + std::to_string(pads) + " image markers for " +
                std::to_string(slots.size()) + " images";
        return false;
    }
    uint64_t total = tokens.size() - pads;
    for (const Qwen35ImageSlot & slot : slots) {
        if (slot.tokens() <= 0) { error = "image has no tokens"; return false; }
        total += (uint64_t) slot.tokens();
    }
    if (total > max_tokens) {
        error = "prompt with images needs " + std::to_string(total) + " tokens, the limit is " +
                std::to_string(max_tokens);
        return false;
    }
    std::vector<int32_t> expanded;
    expanded.reserve((size_t) total);
    size_t next_slot = 0;
    for (int32_t token : tokens) {
        if (token != image_pad) { expanded.push_back(token); continue; }
        Qwen35ImageSlot & slot = slots[next_slot++];
        slot.begin = (int) expanded.size();
        expanded.insert(expanded.end(), (size_t) slot.tokens(), image_pad);
    }
    tokens.swap(expanded);
    return true;
}

Qwen35RopePositions qwen35_image_rope_positions(int prompt_tokens, const std::vector<Qwen35ImageSlot> & slots) {
    Qwen35RopePositions out;
    out.temporal.resize((size_t) prompt_tokens);
    out.height.resize((size_t) prompt_tokens);
    out.width.resize((size_t) prompt_tokens);
    int position = 0, token = 0;
    const auto text_until = [&](int end) {
        for (; token < end; ++token, ++position) {
            out.temporal[token] = out.height[token] = out.width[token] = position;
        }
    };
    for (const Qwen35ImageSlot & slot : slots) {
        text_until(slot.begin);
        for (int row = 0; row < slot.rows; ++row) {
            for (int column = 0; column < slot.columns; ++column, ++token) {
                out.temporal[token] = position;
                out.height[token] = position + row;
                out.width[token] = position + column;
            }
        }
        position += std::max(slot.rows, slot.columns);
    }
    text_until(prompt_tokens);
    out.next = position;
    return out;
}

void qwen35_overwrite_image_rows(const Qwen35ImageSlot & slot, const float * rows, int hidden,
                                 float * embeddings, int first, int count) {
    const int begin = std::max(first, slot.begin);
    const int end = std::min(first + count, slot.begin + slot.tokens());
    if (begin >= end) return;
    std::memcpy(embeddings + (size_t) (begin - first) * hidden,
                rows + (size_t) (begin - slot.begin) * hidden,
                sizeof(float) * (size_t) (end - begin) * hidden);
}

void Qwen35RopePositions::fill(int32_t * out, int first, int count) const {
    std::copy_n(temporal.begin() + first, count, out);
    std::copy_n(height.begin() + first, count, out + count);
    std::copy_n(width.begin() + first, count, out + 2 * count);
    std::fill_n(out + 3 * count, count, 0);
}

}  // namespace dflash::common
