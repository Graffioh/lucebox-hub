#include "qwen35_btflash.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace dflash::common {

Qwen35BTFlashStateBank::~Qwen35BTFlashStateBank() {
    clear();
}

void Qwen35BTFlashStateBank::clear() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    width = 0;
    ssm_state.clear();
    conv_state.clear();
    ssm_slots.clear();
    conv_slots.clear();
}

bool Qwen35BTFlashStateBank::allocate(const TargetWeights & weights,
                                      ggml_backend_t backend,
                                      int branch_width) {
    clear();
    if (!backend || branch_width <= 1) return false;

    const int n_full_attn = weights.n_layer / weights.full_attention_interval;
    const int n_delta = weights.n_layer - n_full_attn;
    const int head_v_dim = weights.ssm_d_inner / weights.ssm_dt_rank;
    const int conv_ch = weights.ssm_d_inner +
        2 * weights.ssm_n_group * weights.ssm_d_state;

    ggml_init_params params{};
    params.mem_size = (size_t)(2 * n_delta * (branch_width + 1) + 32) *
        ggml_tensor_overhead();
    params.no_alloc = true;
    ctx = ggml_init(params);
    if (!ctx) return false;

    width = branch_width;
    ssm_state.reserve((size_t)n_delta);
    conv_state.reserve((size_t)n_delta);
    for (int layer = 0; layer < n_delta; ++layer) {
        ggml_tensor * ssm = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, head_v_dim, head_v_dim,
            weights.ssm_dt_rank, width);
        ggml_tensor * conv = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, weights.ssm_d_conv - 1, conv_ch, width);
        char name[64];
        std::snprintf(name, sizeof(name), "btflash_ssm_%d", layer);
        ggml_set_name(ssm, name);
        std::snprintf(name, sizeof(name), "btflash_conv_%d", layer);
        ggml_set_name(conv, name);
        ssm_state.push_back(ssm);
        conv_state.push_back(conv);
    }

    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        clear();
        return false;
    }

    ssm_slots.reserve((size_t)n_delta * (size_t)width);
    conv_slots.reserve((size_t)n_delta * (size_t)width);
    for (int layer = 0; layer < n_delta; ++layer) {
        ggml_tensor * ssm = ssm_state[(size_t)layer];
        ggml_tensor * conv = conv_state[(size_t)layer];
        for (int branch = 0; branch < width; ++branch) {
            ggml_tensor * ssm_slot = ggml_view_3d(
                ctx, ssm, ssm->ne[0], ssm->ne[1], ssm->ne[2],
                ssm->nb[1], ssm->nb[2], (size_t)branch * ssm->nb[3]);
            ggml_tensor * conv_slot = ggml_view_2d(
                ctx, conv, conv->ne[0], conv->ne[1], conv->nb[1],
                (size_t)branch * conv->nb[2]);
            // These views are created after ggml_backend_alloc_ctx_tensors(),
            // so the allocator cannot initialize their buffer metadata for us.
            if (ggml_backend_view_init(ssm_slot) != GGML_STATUS_SUCCESS ||
                ggml_backend_view_init(conv_slot) != GGML_STATUS_SUCCESS) {
                clear();
                return false;
            }
            ssm_slots.push_back(ssm_slot);
            conv_slots.push_back(conv_slot);
        }
    }
    ggml_backend_buffer_clear(buffer, 0);
    return true;
}

bool Qwen35BTFlashStateBank::capture(const TargetCache & cache) {
    if (!buffer || width <= 1 || cache.ssm_state.size() != ssm_state.size() ||
        cache.conv_state.size() != conv_state.size()) {
        return false;
    }
    for (size_t layer = 0; layer < ssm_state.size(); ++layer) {
        if (!cache.ssm_state[layer] || !cache.conv_state[layer]) return false;
        for (int branch = 0; branch < width; ++branch) {
            const size_t slot = layer * (size_t)width + (size_t)branch;
            ggml_backend_tensor_copy(cache.ssm_state[layer], ssm_slots[slot]);
            ggml_backend_tensor_copy(cache.conv_state[layer], conv_slots[slot]);
        }
    }
    return true;
}

bool Qwen35BTFlashStateBank::restore(TargetCache & cache, int winner) const {
    if (!buffer || winner < 0 || winner >= width ||
        cache.ssm_state.size() != ssm_state.size() ||
        cache.conv_state.size() != conv_state.size()) {
        return false;
    }
    for (size_t layer = 0; layer < ssm_state.size(); ++layer) {
        const size_t slot = layer * (size_t)width + (size_t)winner;
        if (!cache.ssm_state[layer] || !cache.conv_state[layer]) return false;
        ggml_backend_tensor_copy(ssm_slots[slot], cache.ssm_state[layer]);
        ggml_backend_tensor_copy(conv_slots[slot], cache.conv_state[layer]);
    }
    return true;
}

namespace {

bool copy_tensor_range(ggml_tensor * tensor, size_t dst_offset,
                       size_t src_offset, size_t bytes,
                       std::vector<uint8_t> & scratch) {
    if (!tensor || bytes == 0 || dst_offset == src_offset) return tensor != nullptr;
    if (src_offset + bytes > ggml_nbytes(tensor) ||
        dst_offset + bytes > ggml_nbytes(tensor)) {
        return false;
    }
    scratch.resize(bytes);
    ggml_backend_tensor_get(tensor, scratch.data(), src_offset, bytes);
    ggml_backend_tensor_set(tensor, scratch.data(), dst_offset, bytes);
    return true;
}

}  // namespace

bool compact_qwen35_btflash_winner(TargetCache & cache,
                                   int branch_start,
                                   int width,
                                   int fed_steps,
                                   int winner) {
    if (branch_start < 0 || width <= 1 || fed_steps < 0 ||
        winner < 0 || winner >= width) {
        return false;
    }
    std::vector<uint8_t> scratch;

    for (int step = 0; step < fed_steps; ++step) {
        const int src_pos = branch_start + step * width + winner;
        const int dst_pos = branch_start + step;
        for (size_t layer = 0; layer < cache.attn_k.size(); ++layer) {
            ggml_tensor * key = cache.attn_k[layer];
            ggml_tensor * value = cache.attn_v[layer];
            if (!key || !value) return false;
            for (int head = 0; head < key->ne[2]; ++head) {
                const size_t src_key = (size_t)src_pos * key->nb[1] +
                    (size_t)head * key->nb[2];
                const size_t dst_key = (size_t)dst_pos * key->nb[1] +
                    (size_t)head * key->nb[2];
                const size_t src_value = (size_t)src_pos * value->nb[1] +
                    (size_t)head * value->nb[2];
                const size_t dst_value = (size_t)dst_pos * value->nb[1] +
                    (size_t)head * value->nb[2];
                if (!copy_tensor_range(key, dst_key, src_key, key->nb[1], scratch) ||
                    !copy_tensor_range(value, dst_value, src_value,
                                       value->nb[1], scratch)) {
                    return false;
                }
            }
        }

        if (cache.target_feat) {
            const int cap = cache.target_feat_cap;
            const size_t src = (size_t)(src_pos % cap) * cache.target_feat->nb[1];
            const size_t dst = (size_t)(dst_pos % cap) * cache.target_feat->nb[1];
            if (!copy_tensor_range(cache.target_feat, dst, src,
                                   cache.target_feat->nb[1], scratch)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace dflash::common
