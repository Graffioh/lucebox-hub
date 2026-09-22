#include "qwen35_vision.h"

#include "../common/vision/image_resize.h"
#include "../common/vision/mmproj_file.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace luce::vision {

namespace {

constexpr const char * PROJECTOR_TYPE = "qwen3vl_merger";
// The model's own preprocessing refuses anything more elongated.
constexpr int MAX_ASPECT_RATIO = 200;
constexpr size_t GRAPH_NODES = 2048;

struct Block {
    ggml_tensor * ln1_w, * ln1_b, * qkv_w, * qkv_b, * out_w, * out_b;
    ggml_tensor * ln2_w, * ln2_b, * up_w, * up_b, * down_w, * down_b;
};

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(ctx, ggml_mul_mat(ctx, w, x), b);
}

struct ContextDeleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

}  // namespace

bool qwen35_vision_target_size(const Qwen35VisionConfig & config, int width, int height,
                               int & out_width, int & out_height, std::string & error) {
    const int factor = config.patch_size * config.merge;
    if (factor <= 0 || width <= 0 || height <= 0) {
        error = "image has no pixels";
        return false;
    }
    if (std::max(width, height) > (int64_t) MAX_ASPECT_RATIO * std::min(width, height)) {
        error = "image aspect ratio exceeds 200:1";
        return false;
    }
    const double area = double(width) * height;
    const double min_pixels = double(config.min_image_tokens) * factor * factor;
    const double max_pixels = double(config.max_image_tokens) * factor * factor;
    // nearbyint rounds halves to even, like the reference implementation.
    const auto nearest = [factor](double v) { return std::max(factor, int(std::nearbyint(v / factor)) * factor); };
    int w = nearest(width), h = nearest(height);
    if (double(w) * h > max_pixels) {
        const double shrink = std::sqrt(area / max_pixels);
        w = std::max(factor, int(std::floor(width / shrink / factor)) * factor);
        h = std::max(factor, int(std::floor(height / shrink / factor)) * factor);
    } else if (double(w) * h < min_pixels) {
        const double grow = std::sqrt(min_pixels / area);
        w = int(std::ceil(width * grow / factor)) * factor;
        h = int(std::ceil(height * grow / factor)) * factor;
    }
    out_width = w;
    out_height = h;
    return true;
}

void qwen35_vision_position_rows(const std::vector<float> & table, int side, int dimension,
                                 int columns, int rows, int merge, std::vector<float> & out) {
    out.resize(size_t(columns) * rows * dimension);
    // Aligned corners: point i of n lands on i * (side - 1) / (n - 1).
    const auto source = [side](int i, int n) { return n > 1 ? double(i) * (side - 1) / (n - 1) : 0.0; };
    const auto row_of = [&](int y, int x) { return table.data() + (size_t(y) * side + x) * dimension; };
    float * dst = out.data();
    for (int by = 0; by < rows; by += merge) {
        for (int bx = 0; bx < columns; bx += merge) {
            for (int dy = 0; dy < merge; ++dy) {
                for (int dx = 0; dx < merge; ++dx, dst += dimension) {
                    const double fy = source(by + dy, rows), fx = source(bx + dx, columns);
                    const int y0 = int(fy), x0 = int(fx);
                    const int y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
                    const float wy = float(fy - y0), wx = float(fx - x0);
                    const float * a = row_of(y0, x0), * b = row_of(y0, x1);
                    const float * c = row_of(y1, x0), * d = row_of(y1, x1);
                    for (int i = 0; i < dimension; ++i) {
                        dst[i] = (1.0f - wy) * ((1.0f - wx) * a[i] + wx * b[i]) +
                                 wy * ((1.0f - wx) * c[i] + wx * d[i]);
                    }
                }
            }
        }
    }
}

void qwen35_vision_rope_positions(int columns, int rows, int merge, std::vector<int32_t> & out) {
    const size_t patches = size_t(columns) * rows;
    out.resize(4 * patches);
    size_t at = 0;
    for (int by = 0; by < rows; by += merge) {
        for (int bx = 0; bx < columns; bx += merge) {
            for (int dy = 0; dy < merge; ++dy) {
                for (int dx = 0; dx < merge; ++dx, ++at) {
                    out[at] = out[2 * patches + at] = by + dy;
                    out[patches + at] = out[3 * patches + at] = bx + dx;
                }
            }
        }
    }
}

struct Qwen35VisionTower::Impl {
    Qwen35VisionConfig config;
    MmprojFile file;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::vector<float> position_table;
    std::vector<Block> blocks;
    ggml_tensor * patch_first = nullptr, * patch_second = nullptr, * patch_bias = nullptr;
    ggml_tensor * post_w = nullptr, * post_b = nullptr;
    ggml_tensor * merge_in_w = nullptr, * merge_in_b = nullptr;
    ggml_tensor * merge_out_w = nullptr, * merge_out_b = nullptr;

    ~Impl() {
        if (allocator) ggml_gallocr_free(allocator);
    }

    // Looks a tensor up and checks its shape; records the first failure.
    ggml_tensor * need(const std::string & name, std::initializer_list<int64_t> shape, std::string & error) {
        ggml_tensor * t = file.tensor(name);
        bool ok = t != nullptr && shape.size() <= GGML_MAX_DIMS;
        int axis = 0;
        for (int64_t extent : shape) ok = ok && t->ne[axis++] == extent;
        for (; ok && axis < GGML_MAX_DIMS; ++axis) ok = t->ne[axis] == 1;
        if (!ok && error.empty()) error = "projector tensor is missing or has the wrong shape: " + name;
        return ok ? t : nullptr;
    }

    bool read_config(int language_dimension, std::string & error) {
        uint32_t layers = 0, dimension = 0, heads = 0, intermediate = 0, patch = 0, merge = 0, projection = 0;
        std::vector<float> mean, deviation;
        if (!file.u32("clip.vision.block_count", layers) ||
            !file.u32("clip.vision.embedding_length", dimension) ||
            !file.u32("clip.vision.attention.head_count", heads) ||
            !file.u32("clip.vision.feed_forward_length", intermediate) ||
            !file.u32("clip.vision.patch_size", patch) ||
            !file.u32("clip.vision.spatial_merge_size", merge) ||
            !file.u32("clip.vision.projection_dim", projection) ||
            !file.f32("clip.vision.attention.layer_norm_epsilon", config.epsilon) ||
            !file.f32_array("clip.vision.image_mean", mean) || mean.size() != 3 ||
            !file.f32_array("clip.vision.image_std", deviation) || deviation.size() != 3) {
            error = "projector metadata is incomplete";
            return false;
        }
        // The patch reordering in encode() is written for 2x2 merging, and
        // the vision rope splits each head into four equal sections.
        if (layers == 0 || layers > 256 || heads == 0 || dimension == 0 || dimension > 16384 ||
            dimension % heads != 0 || (dimension / heads) % 4 != 0 || intermediate == 0 ||
            patch == 0 || patch > 64 || merge != 2) {
            error = "projector geometry is not supported";
            return false;
        }
        if ((int) projection != language_dimension) {
            error = "projector was built for a model of width " + std::to_string(projection) +
                    ", this model is " + std::to_string(language_dimension);
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            if (!(deviation[i] > 0.0f)) { error = "projector image_std must be positive"; return false; }
            config.mean[i] = mean[i];
            config.deviation[i] = deviation[i];
        }
        config.layers = (int) layers;
        config.dimension = (int) dimension;
        config.heads = (int) heads;
        config.intermediate = (int) intermediate;
        config.patch_size = (int) patch;
        config.merge = (int) merge;
        config.language_dimension = language_dimension;
        return true;
    }

    bool bind_tensors(std::string & error) {
        const int64_t d = config.dimension, ff = config.intermediate, p = config.patch_size;
        const int64_t merged = d * config.merge * config.merge;
        patch_first = need("v.patch_embd.weight", {p, p, 3, d}, error);
        patch_second = need("v.patch_embd.weight.1", {p, p, 3, d}, error);
        patch_bias = need("v.patch_embd.bias", {d}, error);
        post_w = need("v.post_ln.weight", {d}, error);
        post_b = need("v.post_ln.bias", {d}, error);
        merge_in_w = need("mm.0.weight", {merged, merged}, error);
        merge_in_b = need("mm.0.bias", {merged}, error);
        merge_out_w = need("mm.2.weight", {merged, config.language_dimension}, error);
        merge_out_b = need("mm.2.bias", {config.language_dimension}, error);
        blocks.resize(config.layers);
        for (int i = 0; i < config.layers; ++i) {
            const std::string prefix = "v.blk." + std::to_string(i) + ".";
            Block & b = blocks[i];
            b.ln1_w = need(prefix + "ln1.weight", {d}, error);
            b.ln1_b = need(prefix + "ln1.bias", {d}, error);
            b.qkv_w = need(prefix + "attn_qkv.weight", {d, 3 * d}, error);
            b.qkv_b = need(prefix + "attn_qkv.bias", {3 * d}, error);
            b.out_w = need(prefix + "attn_out.weight", {d, d}, error);
            b.out_b = need(prefix + "attn_out.bias", {d}, error);
            b.ln2_w = need(prefix + "ln2.weight", {d}, error);
            b.ln2_b = need(prefix + "ln2.bias", {d}, error);
            b.up_w = need(prefix + "ffn_up.weight", {d, ff}, error);
            b.up_b = need(prefix + "ffn_up.bias", {ff}, error);
            b.down_w = need(prefix + "ffn_down.weight", {ff, d}, error);
            b.down_b = need(prefix + "ffn_down.bias", {d}, error);
        }
        if (!error.empty()) return false;

        ggml_tensor * table = file.tensor("v.position_embd.weight");
        const int side = table ? (int) std::lround(std::sqrt((double) table->ne[1])) : 0;
        if (!table || table->type != GGML_TYPE_F32 || table->ne[0] != d || side < 2 ||
            (int64_t) side * side != table->ne[1]) {
            error = "projector position table is missing or not a square F32 grid";
            return false;
        }
        config.position_side = side;
        position_table.resize(size_t(side) * side * d);
        ggml_backend_tensor_get(table, position_table.data(), 0, ggml_nbytes(table));
        return true;
    }
};

Qwen35VisionTower::Qwen35VisionTower() : impl_(std::make_unique<Impl>()) {}
Qwen35VisionTower::~Qwen35VisionTower() = default;

const Qwen35VisionConfig & Qwen35VisionTower::config() const { return impl_->config; }
size_t Qwen35VisionTower::weight_bytes() const { return impl_->file.weight_bytes(); }

bool Qwen35VisionTower::load(const std::string & path, ggml_backend_t backend, int language_dimension,
                             std::string & error) {
    Impl & m = *impl_;
    if (!m.file.load(path, backend, error)) return false;
    if (m.file.projector_type() != PROJECTOR_TYPE) {
        error = "projector type '" + m.file.projector_type() + "' is not " + PROJECTOR_TYPE;
        return false;
    }
    if (m.file.has_tensor_prefix("v.deepstack.")) {
        error = "projectors with deepstack branches are not supported";
        return false;
    }
    m.backend = backend;
    return m.read_config(language_dimension, error) && m.bind_tensors(error);
}

bool qwen35_vision_preprocess(const Qwen35VisionConfig & c, const DecodedRgb & image,
                              Qwen35Pixels & out, std::string & error) {
    int width = 0, height = 0;
    if (!qwen35_vision_target_size(c, (int) image.width, (int) image.height, width, height, error)) return false;
    std::vector<uint8_t> resized;
    if (!resize_rgb_bicubic(image.pixels, (int) image.width, (int) image.height, width, height, resized, error)) {
        return false;
    }
    out.width = width;
    out.height = height;
    out.grid_columns = width / (c.patch_size * c.merge);
    out.grid_rows = height / (c.patch_size * c.merge);
    const size_t plane = size_t(width) * height;
    out.planar.resize(3 * plane);
    for (int channel = 0; channel < 3; ++channel) {
        float * dst = out.planar.data() + channel * plane;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = (resized[3 * i + channel] / 255.0f - c.mean[channel]) / c.deviation[channel];
        }
    }
    return true;
}

bool Qwen35VisionTower::encode(const Qwen35Pixels & pixels, std::vector<float> & rows, std::string & error) {
    Impl & m = *impl_;
    const Qwen35VisionConfig & c = m.config;
    const int columns = pixels.width / c.patch_size, lines = pixels.height / c.patch_size;
    const int64_t patches = int64_t(columns) * lines;
    const int64_t tokens = pixels.tokens();
    if (!m.backend || tokens <= 0 || tokens > c.max_image_tokens ||
        columns != pixels.grid_columns * c.merge || lines != pixels.grid_rows * c.merge ||
        pixels.planar.size() != size_t(3) * pixels.width * pixels.height) {
        error = "image was not prepared for this projector";
        return false;
    }

    const int64_t d = c.dimension, head = d / c.heads;
    ggml_init_params params{ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false),
                            nullptr, /*no_alloc=*/true};
    std::unique_ptr<ggml_context, ContextDeleter> owner(ggml_init(params));
    ggml_context * ctx = owner.get();
    if (!ctx) { error = "vision graph context allocation failed"; return false; }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GRAPH_NODES, false);

    ggml_tensor * image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, pixels.width, pixels.height, 3, 1);
    ggml_tensor * position_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, patches);
    ggml_tensor * rope_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * patches);
    ggml_set_input(image);
    ggml_set_input(position_rows);
    ggml_set_input(rope_positions);

    // Patch embedding. The model was trained on two-frame clips; a still
    // image is the same frame twice, so both temporal kernels see it.
    const int p = c.patch_size;
    ggml_tensor * x = ggml_add(ctx, ggml_conv_2d(ctx, m.patch_first, image, p, p, 0, 0, 1, 1),
                                    ggml_conv_2d(ctx, m.patch_second, image, p, p, 0, 0, 1, 1));
    // [columns, lines, d] in raster order -> [d, patches] with the four
    // patches of each 2x2 block adjacent, which is what the merger expects.
    x = ggml_permute(ctx, x, 1, 2, 0, 3);
    x = ggml_cont_4d(ctx, x, d * 2, columns / 2, lines, 1);
    x = ggml_reshape_4d(ctx, x, d * 2, columns / 2, 2, lines / 2);
    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    x = ggml_cont_2d(ctx, x, d, patches);
    x = ggml_add(ctx, x, m.patch_bias);
    x = ggml_add(ctx, x, position_rows);

    int sections[4] = {int(head / 4), int(head / 4), int(head / 4), int(head / 4)};
    const float attention_scale = 1.0f / std::sqrt(float(head));
    for (const Block & b : m.blocks) {
        ggml_tensor * qkv = linear(ctx, b.qkv_w, b.qkv_b, layer_norm(ctx, x, b.ln1_w, b.ln1_b, c.epsilon));
        const auto part = [&](int index) {
            return ggml_view_3d(ctx, qkv, head, c.heads, patches, ggml_row_size(qkv->type, head), qkv->nb[1],
                                ggml_row_size(qkv->type, d) * index);
        };
        const auto rotate = [&](ggml_tensor * t) {
            return ggml_rope_multi(ctx, t, rope_positions, nullptr, int(head / 2), sections,
                                   GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        };
        // Fused attention with half-precision keys and values, as the
        // reference implementation runs it: the score matrix of a large
        // image (3,900 patches squared per head) never touches memory.
        ggml_tensor * q = ggml_permute(ctx, rotate(part(0)), 0, 2, 1, 3);
        ggml_tensor * k = ggml_cast(ctx, ggml_permute(ctx, rotate(part(1)), 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * v = ggml_cast(ctx, ggml_permute(ctx, part(2), 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor * mixed = ggml_flash_attn_ext(ctx, q, k, v, nullptr, attention_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(mixed, GGML_PREC_F32);
        x = ggml_add(ctx, x, linear(ctx, b.out_w, b.out_b, ggml_reshape_2d(ctx, mixed, d, patches)));

        ggml_tensor * hidden = linear(ctx, b.up_w, b.up_b, layer_norm(ctx, x, b.ln2_w, b.ln2_b, c.epsilon));
        x = ggml_add(ctx, x, linear(ctx, b.down_w, b.down_b, ggml_gelu(ctx, hidden)));
    }

    // Merger: each block of four patches becomes one language-model row.
    x = layer_norm(ctx, x, m.post_w, m.post_b, c.epsilon);
    x = ggml_reshape_2d(ctx, x, d * c.merge * c.merge, tokens);
    x = linear(ctx, m.merge_out_w, m.merge_out_b, ggml_gelu(ctx, linear(ctx, m.merge_in_w, m.merge_in_b, x)));
    ggml_set_output(x);
    ggml_build_forward_expand(graph, x);

    if (!m.allocator) m.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!m.allocator || !ggml_gallocr_alloc_graph(m.allocator, graph)) {
        error = "not enough device memory for the vision tower scratch";
        return false;
    }

    std::vector<float> positions;
    std::vector<int32_t> rope;
    qwen35_vision_position_rows(m.position_table, c.position_side, c.dimension, columns, lines, c.merge, positions);
    qwen35_vision_rope_positions(columns, lines, c.merge, rope);
    ggml_backend_tensor_set(image, pixels.planar.data(), 0, ggml_nbytes(image));
    ggml_backend_tensor_set(position_rows, positions.data(), 0, ggml_nbytes(position_rows));
    ggml_backend_tensor_set(rope_positions, rope.data(), 0, ggml_nbytes(rope_positions));

    if (ggml_backend_graph_compute(m.backend, graph) != GGML_STATUS_SUCCESS) {
        error = "vision tower compute failed";
        return false;
    }
    rows.resize(size_t(tokens) * c.language_dimension);
    ggml_backend_tensor_get(x, rows.data(), 0, ggml_nbytes(x));
    return true;
}

void Qwen35VisionTower::release_scratch() {
    if (impl_->allocator) {
        ggml_gallocr_free(impl_->allocator);
        impl_->allocator = nullptr;
    }
}

}  // namespace luce::vision
