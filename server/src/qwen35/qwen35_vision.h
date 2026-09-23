// Vision tower for Qwen3.5 / Qwen3.8 image input: the "qwen3vl_merger"
// projector published next to the model as an mmproj GGUF. Turns one image
// into rows of language-model embeddings, one per 2x2 block of patches.
//
// Split in two because the halves run on different threads: preprocessing is
// a CPU-only function of the config, safe while the GPU is busy or the tower
// is unloaded; encoding runs the tower graph and must be serialised with the
// rest of the backend's GPU work.
#pragma once

#include "../common/vision/image_decode.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::vision {

struct Qwen35VisionConfig {
    int layers = 0, dimension = 0, heads = 0, intermediate = 0;
    int patch_size = 0, merge = 0;
    int position_side = 0;       // the learned position table is side x side
    int language_dimension = 0;  // width of one output row
    float epsilon = 1e-6f;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float deviation[3] = {0.5f, 0.5f, 0.5f};
    // Image tokens per image. The lower bound is the model's own; the upper
    // bound keeps the tower's attention scratch near 1 GiB.
    int min_image_tokens = 64;
    int max_image_tokens = 1024;

    // Everything preprocessing and prompt expansion depend on.
    bool same_geometry(const Qwen35VisionConfig & o) const {
        return patch_size == o.patch_size && merge == o.merge &&
               language_dimension == o.language_dimension &&
               min_image_tokens == o.min_image_tokens && max_image_tokens == o.max_image_tokens &&
               mean[0] == o.mean[0] && mean[1] == o.mean[1] && mean[2] == o.mean[2] &&
               deviation[0] == o.deviation[0] && deviation[1] == o.deviation[1] &&
               deviation[2] == o.deviation[2];
    }
};

// A resized, normalised image. `planar` is channel major ([3, height, width]).
struct Qwen35Pixels {
    int width = 0, height = 0;
    int grid_columns = 0, grid_rows = 0;  // image tokens across and down
    std::vector<float> planar;

    int tokens() const { return grid_columns * grid_rows; }
};

// Size the model resizes a width x height image to: both sides a multiple of
// patch_size * merge, area within the configured token bounds.
bool qwen35_vision_target_size(const Qwen35VisionConfig & config, int width, int height,
                               int & out_width, int & out_height, std::string & error);

// Resizes the way the model was trained (bicubic, straight to the target
// size) and normalises. CPU only.
bool qwen35_vision_preprocess(const Qwen35VisionConfig & config, const DecodedRgb & image,
                              Qwen35Pixels & out, std::string & error);

// Learned position rows for a columns x rows patch grid, bilinearly resampled
// from the side x side table with aligned corners and written in the tower's
// merged patch order. `table` and `out` hold `dimension` floats per position.
void qwen35_vision_position_rows(const std::vector<float> & table, int side, int dimension,
                                 int columns, int rows, int merge, std::vector<float> & out);

// Rotary positions of the patches, in merged order: [4 * patches] as the four
// consecutive axes (y, x, y, x) that ggml's vision rope expects.
void qwen35_vision_rope_positions(int columns, int rows, int merge, std::vector<int32_t> & out);

class Qwen35VisionTower {
public:
    Qwen35VisionTower();
    ~Qwen35VisionTower();
    Qwen35VisionTower(const Qwen35VisionTower &) = delete;
    Qwen35VisionTower & operator=(const Qwen35VisionTower &) = delete;

    // `backend` must outlive the tower. `language_dimension` is the text
    // model's hidden size; a projector built for another model is refused.
    bool load(const std::string & path, ggml_backend_t backend, int language_dimension,
              std::string & error);

    const Qwen35VisionConfig & config() const;
    size_t weight_bytes() const;

    // Runs the tower. `rows` receives tokens() * language_dimension floats in
    // reading order. One caller at a time.
    bool encode(const Qwen35Pixels & pixels, std::vector<float> & rows, std::string & error);

    // Frees the graph scratch between requests; the weights stay loaded.
    void release_scratch();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace luce::vision
