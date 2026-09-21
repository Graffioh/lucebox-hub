// Prompt-side half of Qwen3.5 / Qwen3.8 image input: where each image sits in
// the token stream and which rotary positions its tokens take. Pure functions
// of the token ids and the image grids; no GPU, no projector.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

// What the chat template carries for one image. The middle token is repeated
// once per image token by qwen35_expand_image_tokens().
inline constexpr const char * QWEN35_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>";

struct Qwen35ImageSlot {
    int begin = 0;              // first image token in the expanded prompt
    int columns = 0, rows = 0;  // image tokens across and down

    int tokens() const { return columns * rows; }
};

// `tokens` holds one `image_pad` per image, in order. Each becomes
// slot.tokens() pads and slot.begin is filled in. Fails when the number of
// pads differs from the number of slots or the result exceeds `max_tokens`.
bool qwen35_expand_image_tokens(std::vector<int32_t> & tokens, int32_t image_pad,
                                std::vector<Qwen35ImageSlot> & slots, uint64_t max_tokens,
                                std::string & error);

// Rotary positions of every prompt token. Text advances all three axes
// together. An image starting at position p holds the temporal axis at p and
// lays its rows and columns out from p on the other two; the text after it
// resumes at p + max(rows, columns). So positions run behind token indices
// after an image, by `next - prompt tokens`.
struct Qwen35RopePositions {
    std::vector<int32_t> temporal, height, width;
    int next = 0;  // position of the first generated token

    // [4 * count] in the axis-major layout the target graph reads.
    void fill(int32_t * out, int first, int count) const;
};

Qwen35RopePositions qwen35_image_rope_positions(int prompt_tokens, const std::vector<Qwen35ImageSlot> & slots);

// Copies the part of one image's `rows` that falls inside prompt tokens
// [first, first + count) over `embeddings`, which holds those `count` rows.
// Prefill runs in chunks, so an image can straddle several calls.
void qwen35_overwrite_image_rows(const Qwen35ImageSlot & slot, const float * rows, int hidden,
                                 float * embeddings, int first, int count);

}  // namespace luce::common
