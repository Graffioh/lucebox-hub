// What an image request carries through the Qwen3.5 backend: the prompt
// binding built on the request thread, and the encoded rows built on the
// worker thread just before prefill.
#pragma once

#include "common/image_prompt.h"
#include "qwen35_image_prompt.h"
#include "qwen35_vision.h"

#include <cstdint>
#include <vector>

namespace luce::common {

class Qwen35ImagePrompt final : public ImagePromptPayload {
public:
    bool matches(const std::vector<int32_t> & tokens) const override { return tokens == expanded_tokens; }

    const void * owner = nullptr;              // the backend that prepared it
    std::vector<int32_t> expanded_tokens;      // prompt with one pad per image token
    std::vector<Qwen35ImageSlot> slots;        // one per image, in prompt order
    std::vector<vision::Qwen35Pixels> pixels;  // same order as slots
    Qwen35RopePositions positions;
};

struct Qwen35ImageRows {
    const Qwen35ImagePrompt * prompt = nullptr;
    std::vector<std::vector<float>> rows;  // per image: tokens() x hidden floats

    // Writes the image rows that fall inside prompt tokens [first, first +
    // count) over `embeddings`, which holds `count` rows of `hidden` floats.
    void overwrite(float * embeddings, int first, int count, int hidden) const {
        for (size_t i = 0; i < rows.size(); ++i) {
            qwen35_overwrite_image_rows(prompt->slots[i], rows[i].data(), hidden, embeddings, first, count);
        }
    }
};

}  // namespace luce::common
