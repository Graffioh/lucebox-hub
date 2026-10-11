// Qwen3.5-0.8B drafter scoring for pflash speculative prefill.
//
// Two scorers share these weights:
//   - qwen35_score_and_compress        : the original all-layer running-max
//                                        scorer, on the Qwen3.5 architecture
//   - qwen35_strict_score_and_compress : blocks 0..14 plus the block-15 NoPE
//                                        Q/K scoring head, under strict
//                                        budget selection
//
// Loading lives in qwen35_loader.cpp; pflash_drafter.cpp dispatches into
// qwen35_drafter_score_and_compress.

#include "qwen35_drafter.h"

#include "pflash_drafter.h"
#include "pflash_compress.h"
#include "pflash_selection.h"
#include "common/gguf_inspect.h"
#include "anchor_params.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace luce::common {

namespace {

static int align_up_i(int x, int a) { return ((x + a - 1) / a) * a; }

// Bytes of build_qwen35_causal_mask's mask, the part of a layer graph that
// grows with the context.
static size_t causal_mask_bytes(int kv_start, int n_tokens) {
    return (size_t)align_up_i(kv_start + n_tokens, 256) * align_up_i(n_tokens, 32) * sizeof(uint16_t);
}

// create_target_cache honours LUCE_KV_TQ3; the drafter cache never wants
// the TurboQuant rotation, so force it off while the cache is created.
struct ScopedKvTq3Off {
    ScopedKvTq3Off() {
#if defined(_WIN32)
        char * raw = nullptr;
        size_t len = 0;
        _dupenv_s(&raw, &len, "LUCE_KV_TQ3");
        had_ = raw != nullptr;
        old_ = had_ ? raw : "";
        free(raw);
        _putenv_s("LUCE_KV_TQ3", "0");
#else
        const char * raw = std::getenv("LUCE_KV_TQ3");
        had_ = raw != nullptr;
        old_ = had_ ? raw : "";
        setenv("LUCE_KV_TQ3", "0", 1);
#endif
    }
    ~ScopedKvTq3Off() {
#if defined(_WIN32)
        // _putenv_s with empty value removes the variable on MSVCRT.
        _putenv_s("LUCE_KV_TQ3", had_ ? old_.c_str() : "");
#else
        if (had_) setenv("LUCE_KV_TQ3", old_.c_str(), 1);
        else unsetenv("LUCE_KV_TQ3");
#endif
    }
    bool had_ = false;
    std::string old_;
};

} // namespace

ggml_tensor * build_qwen35_causal_mask(ggml_context * ctx, ggml_cgraph * gf,
                                       int kv_start, int n_tokens) {
    // Keys padded to 256: the full-attention block reads a K view padded to
    // the 256-token flash-attention stride for masked prefill spans
    // (build_full_attn_block, padded_prefill), and every key column it reads
    // must lie inside a mask row.
    const int kv_pad = align_up_i(kv_start + n_tokens, 256);
    const int q_pad = align_up_i(n_tokens, 32);
    // Keys [kv_start, kv_pad): row q is zero through key kv_start + q, then
    // -inf, the kv padding included; n_tokens + 255 keys at most.
    const int tail = kv_pad - kv_start;
    // ggml_fill and ggml_tri take F32 only; the big tensor is written in F16.
    ggml_tensor * zero = ggml_cast(ctx,
        ggml_fill_inplace(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 0.0f), GGML_TYPE_F16);
    ggml_tensor * mask = ggml_repeat_4d(ctx, zero, kv_pad, q_pad, 1, 1);
    ggml_tensor * diag = ggml_tri(ctx,
        ggml_fill_inplace(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tail, tail), -INFINITY),
        GGML_TRI_TYPE_UPPER);
    ggml_build_forward_expand(gf, ggml_cpy(ctx,
        ggml_view_2d(ctx, diag, tail, n_tokens, diag->nb[1], 0),
        ggml_view_2d(ctx, mask, tail, n_tokens, mask->nb[1], (size_t)kv_start * mask->nb[0])));
    if (q_pad > n_tokens) {
        ggml_tensor * pad_rows = ggml_fill_inplace(ctx,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_pad, q_pad - n_tokens), -INFINITY);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, pad_rows,
            ggml_view_2d(ctx, mask, kv_pad, q_pad - n_tokens, mask->nb[1],
                         (size_t)n_tokens * mask->nb[1])));
    }
    // The attention reads the mask after these copies: they come first in gf.
    return mask;
}

std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_scores_out) {

    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    if (S < n_lookahead + 1) return ids;
    const int query_end = score_query_end < 0 ? S : score_query_end;
    if (n_lookahead < 1 || query_end < n_lookahead || query_end > S) {
        set_last_error("qwen35 scorer query window out of range");
        return {};
    }
    const int query_start = query_end - n_lookahead;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max((size_t)n_lookahead * S, -INFINITY);

    TargetCache cache;
    {
        ScopedKvTq3Off tq3_off;
        if (!create_target_cache(w, S, 0, w.backend, cache, true)) {
            return {};
        }
    }

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < S; i += batch) {
            const int n = std::min(batch, S - i);
            if (!w.embedder.embed(ids.data() + i, n, emb.data())) {
                ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter embedding failed");
                return {};
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1], (size_t)hidden * n * sizeof(float));
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    // Graph of layer il over tokens [start, start + n), writing act_out.
    // nullptr (error set) on failure.
    const auto layer_graph = [&](int il, int start, int n, ggml_cgraph *& gf,
                                 ggml_tensor *& pos) -> ggml_context * {
        ggml_init_params ip{};
        ip.mem_size = 512 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            set_last_error("qwen35 drafter layer graph ctx init failed");
            return nullptr;
        }
        gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1], (size_t)start * act_in->nb[1]);
        pos = nullptr;
        ggml_tensor * mask = nullptr;
        if (((il + 1) % w.full_attention_interval) == 0) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
            ggml_set_input(pos);
            mask = build_qwen35_causal_mask(ctx, gf, start, n);
        }
        ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask, start, n, false, 0);
        ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1], (size_t)start * act_out->nb[1]);
        if (ggml_nelements(out) != ggml_nelements(dst)) {
            std::fprintf(stderr,
                "[qwen35-drafter] layer output shape mismatch il=%d start=%d out=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                il, start,
                (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3]);
            ggml_free(ctx);
            set_last_error("qwen35 layer output shape mismatch");
            return nullptr;
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
        return ctx;
    };
    // Size the compute buffer once, for a full-attention layer over the
    // ubatch with the largest mask: allocating per ubatch would regrow it
    // with every ubatch of the first attention layer.
    if (w.full_attention_interval <= w.n_layer) {
        int worst = 0;
        for (int start = 0; start < S; start += ubatch) {
            if (causal_mask_bytes(start, std::min(ubatch, S - start)) >=
                causal_mask_bytes(worst, std::min(ubatch, S - worst))) worst = start;
        }
        ggml_cgraph * gf = nullptr;
        ggml_tensor * pos = nullptr;
        ggml_context * ctx = layer_graph(w.full_attention_interval - 1, worst,
                                         std::min(ubatch, S - worst), gf, pos);
        if (!ctx) {
            ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
            return {};
        }
        const bool reserved = ggml_gallocr_reserve(alloc, gf);
        ggml_free(ctx);
        if (!reserved) {
            ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
            set_last_error("qwen35 drafter graph allocation failed");
            return {};
        }
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        int fa_idx = 0;
        if (is_attn) {
            for (int k = 0; k < il; ++k) if (((k + 1) % w.full_attention_interval) == 0) ++fa_idx;
        }
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            ggml_cgraph * gf = nullptr;
            ggml_tensor * pos = nullptr;
            ggml_context * ctx = layer_graph(il, start, n, gf, pos);
            if (!ctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                return {};
            }
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph allocation failed");
                return {};
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
            }
            auto st = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph compute failed");
                return {};
            }
        }

        if (is_attn) {
            ggml_init_params sip{};
            sip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(1024, false) + 64 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph ctx allocation failed");
                return {};
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 1024, false);
            const int K_len = (int) cache.attn_k[(size_t)fa_idx]->ne[1];
            ggml_tensor * mask_tail = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, K_len, n_lookahead);
            ggml_tensor * K_f32 = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, w.n_head_kv);
            ggml_tensor * K_cast = ggml_cpy(sctx, cache.attn_k[(size_t)fa_idx], K_f32);
            ggml_tensor * K_score = nullptr;
            if (w.n_head != w.n_head_kv) {
                const int gqa = w.n_head / w.n_head_kv;
                ggml_tensor * K_4d = ggml_reshape_4d(sctx, K_cast, w.n_embd_head_k, K_len, 1, w.n_head_kv);
                ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, gqa, w.n_head_kv);
                ggml_tensor * K_rep = ggml_repeat(sctx, K_4d, K_tpl);
                K_score = ggml_reshape_3d(sctx, K_rep, w.n_embd_head_k, K_len, w.n_head);
            } else {
                K_score = K_cast;
            }
            const TargetLayer & L = w.layers[il];
            ggml_tensor * inp_tail = ggml_view_2d(sctx, act_in, hidden, n_lookahead,
                act_in->nb[1], (size_t)query_start * act_in->nb[1]);
            ggml_tensor * q_cur = ggml_rms_norm(sctx, inp_tail, w.rms_eps);
            q_cur = ggml_mul(sctx, q_cur, L.attn_norm);
            ggml_tensor * QG = ggml_mul_mat(sctx, L.wq, q_cur);
            QG = ggml_reshape_3d(sctx, QG, w.n_embd_head_k * 2, w.n_head, n_lookahead);
            ggml_tensor * Q = ggml_view_3d(sctx, QG,
                w.n_embd_head_k, w.n_head, n_lookahead,
                ggml_element_size(QG) * w.n_embd_head_k * 2,
                ggml_element_size(QG) * w.n_embd_head_k * 2 * w.n_head,
                0);
            Q = ggml_rms_norm(sctx, Q, w.rms_eps);
            Q = ggml_mul(sctx, Q, L.q_norm);
            ggml_tensor * pos_tail = ggml_new_tensor_1d(sctx, GGML_TYPE_I32, 4 * n_lookahead);
            int sections[4];
            for (int k = 0; k < 4; ++k) sections[k] = w.rope_sections[k];
            Q = ggml_rope_multi(sctx, Q, pos_tail, nullptr,
                                w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE,
                                0, w.rope_theta, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * Q_tail_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));
            ggml_tensor * attn_score = ggml_mul_mat(sctx, K_score, Q_tail_perm);
            ggml_tensor * probs = ggml_soft_max_ext(sctx, attn_score, mask_tail, 1.0f / std::sqrt((float)w.n_embd_head_k), 0.0f);
            ggml_set_output(probs);
            ggml_build_forward_expand(sgf, probs);
            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            if (!ggml_gallocr_alloc_graph(salloc, sgf)) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph allocation failed");
                return {};
            }
            std::vector<int32_t> pos4((size_t)4 * n_lookahead, 0);
            for (int i = 0; i < n_lookahead; ++i) {
                const int p = query_start + i;
                pos4[(size_t)0 * n_lookahead + i] = p;
                pos4[(size_t)1 * n_lookahead + i] = p;
                pos4[(size_t)2 * n_lookahead + i] = p;
            }
            ggml_backend_tensor_set(pos_tail, pos4.data(), 0, pos4.size() * sizeof(int32_t));
            std::vector<float> mask((size_t)n_lookahead * K_len, 0.0f);
            for (int t = 0; t < n_lookahead; ++t) {
                const int visible_end = query_start + t + 1;
                for (int j = 0; j < K_len; ++j) {
                    mask[(size_t)t * K_len + j] = (j < visible_end) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask_tail, mask.data(), 0, mask.size() * sizeof(float));
            auto st = ggml_backend_graph_compute(w.backend, sgf);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph compute failed");
                return {};
            }
            std::vector<float> tmp((size_t)K_len * n_lookahead * w.n_head);
            ggml_backend_tensor_get(probs, tmp.data(), 0, tmp.size() * sizeof(float));
            const size_t nonfinite =
                count_nonfinite_scores(tmp.data(), tmp.size());
            if (nonfinite != 0) {
                const std::string message =
                    "non-finite Qwen3.5 PFlash scores at layer " +
                    std::to_string(il) + ": " + std::to_string(nonfinite) +
                    "/" + std::to_string(tmp.size());
                std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
                std::fflush(stderr);
                ggml_gallocr_free(salloc); ggml_free(sctx);
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf);
                ggml_free(act_ctx); free_target_cache(cache);
                set_last_error(message);
                return {};
            }
            for (int h = 0; h < w.n_head; ++h) {
                for (int t = 0; t < n_lookahead; ++t) {
                    for (int j = 0; j < S; ++j) {
                        const size_t src = (size_t)h * K_len * n_lookahead + (size_t)t * K_len + j;
                        const size_t dst = (size_t)t * S + j;
                        running_max[dst] = std::max(running_max[dst], tmp[src]);
                    }
                }
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_target_cache(cache);

    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) s += running_max[(size_t)t * S + j];
        score[(size_t)j] = s / (float)n_lookahead;
    }

    const int n_chunks = (S + chunk_size - 1) / chunk_size;
    const int n_keep = std::max(1, (int)((float)n_chunks * keep_ratio));
    
    std::vector<float> smooth_score = score;
    // Caller pool_kernel takes precedence; if zero/negative, fall back to env or 5.
    const int pk = (pool_kernel > 0)
        ? pool_kernel
        : std::max(3, env_int("LUCE_COMPRESS_POOL_KERNEL", 5));
    std::vector<float> smoothed((size_t)S, 0.0f);
    int half = pk / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[(size_t)k]; ++n; }
        smoothed[(size_t)j] = (n > 0) ? (s / (float)n) : 0.0f;
    }
    smooth_score.swap(smoothed);

    if (token_scores_out) {
        // Scoring only (two-scorer selection): hand the smoothed per-token
        // scores back and let the caller select.
        *token_scores_out = smooth_score;
        return ids;
    }

    if (experiment.selection_active) {
        return select_pflash_chunks(
            ids, smooth_score, keep_ratio, n_lookahead, score_query_end,
            pk, experiment, required_instruction_spans, false, true);
    }
    
    std::vector<std::pair<float, int>> chunk_means;
    for (int c = 0; c < n_chunks; ++c) {
        int lo = c * chunk_size, hi = std::min(S, lo + chunk_size);
        float s = 0.0f;
        for (int j = lo; j < hi; ++j) s += smooth_score[(size_t)j];
        chunk_means.push_back({s / std::max(1, hi - lo), c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(), [](auto a, auto b) { return a.first > b.first; });
    
    std::vector<uint8_t> selected((size_t)n_chunks, 0);
    int count = 0;
    // Scale head/tail forced chunks so they don't crowd out top-K scoring.
    {
        const int h_raw = env_int("LUCE_COMPRESS_HEAD_CHUNKS", 8);
        const int t_raw = env_int("LUCE_COMPRESS_TAIL_CHUNKS", 24);
        int h_n = h_raw, t_n = t_raw;
        if (h_n + t_n >= n_keep) {
            const int budget = std::max(1, n_keep - 1);
            h_n = std::max(0, h_raw * budget / (h_raw + t_raw));
            t_n = std::max(0, budget - h_n);
        }
        for (int c = 0; c < std::min(n_chunks, h_n); ++c) { selected[(size_t)c] = 1; ++count; }
        for (int c = std::max(0, n_chunks - t_n); c < n_chunks; ++c) if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }

    const int query_tokens = env_int("LUCE_COMPRESS_QUERY_TOKENS", 96);
    const auto ap = resolve_anchor_params(n_chunks,
        env_int("PFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("PFLASH_COMPRESS_MAX_ANCHOR_HITS", -1),
        env_int("LUCE_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("LUCE_COMPRESS_MAX_ANCHOR_HITS", -1));
    const int anchor_radius   = ap.radius;
    const int max_anchor_hits = ap.max_hits;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    const int q0 = std::max(0, S - query_tokens);
    constexpr int NGRAM = 4;
    for (int q = q0; q + NGRAM <= S; ++q) {
        int hits = 0;
        std::vector<int> hit_pos(max_anchor_hits);
        const int search_end = std::max(0, q0 - NGRAM);
        for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < NGRAM; ++k) {
                if (ids[(size_t)p + k] != ids[(size_t)q + k]) { same = false; break; }
            }
            if (same) {
                if (hits < max_anchor_hits) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= max_anchor_hits) {
            for (int i = 0; i < hits && i < max_anchor_hits; ++i) {
                force_chunk_neighborhood(forced, n_chunks, hit_pos[i] / chunk_size, anchor_radius);
            }
        }
    }

    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c] && !selected[(size_t)c]) {
            selected[(size_t)c] = 1;
            ++count;
        }
    }

    // Global aggregation tasks often depend on repeated rare tokens that do
    // not appear in the final query. Preserve high-frequency-but-not-filler
    // token chunks before filling with model-score top-K.
    const int repeat_min = env_int("LUCE_COMPRESS_REPEAT_MIN", 4);
    const int repeat_max = env_int("LUCE_COMPRESS_REPEAT_MAX", 32);
    const int repeat_limit = env_int("LUCE_COMPRESS_REPEAT_CHUNKS", n_keep);
    if (repeat_min > 1 && count < repeat_limit) {
        std::unordered_map<int32_t, int> freq;
        freq.reserve((size_t)S);
        const int repeat_scan_end = std::max(0, S - query_tokens);
        for (int j = 0; j < repeat_scan_end; ++j) {
            ++freq[ids[(size_t)j]];
        }
        std::vector<std::pair<int, int32_t>> repeated;
        repeated.reserve(freq.size());
        for (const auto & kv : freq) {
            if (kv.second >= repeat_min && kv.second <= repeat_max) {
                repeated.push_back({kv.second, kv.first});
            }
        }
        std::sort(repeated.begin(), repeated.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        for (const auto & rp : repeated) {
            if (count >= repeat_limit) break;
            const int32_t tok = rp.second;
            for (int j = 0; j < repeat_scan_end && count < repeat_limit; ++j) {
                if (ids[(size_t)j] != tok) continue;
                const int c = j / chunk_size;
                if (!selected[(size_t)c]) {
                    selected[(size_t)c] = 1;
                    ++count;
                }
            }
        }
    }
    
    for (auto [_, c] : chunk_means) {
        if (count >= n_keep) break;
        if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }
    
    std::vector<int32_t> out_ids;
    std::vector<int> selected_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (selected[(size_t)c]) selected_chunks.push_back(c);
    }
    int span_start = -1, span_end = -1;
    for (int c : selected_chunks) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[qwen35-drafter] forward+compress %.2fs S=%d kept=%zu (%d/%d chunks)\n",
                 std::chrono::duration<double>(t1 - t0).count(), S, out_ids.size(), count, n_chunks);
    std::fflush(stderr);
    return out_ids;
}

void free_qwen35_scoring_session(Qwen35ScoringSession & session) {
    free_target_cache(session.cache);
    session.cache = TargetCache{};
    if (session.key_buf) ggml_backend_buffer_free(session.key_buf);
    if (session.key_ctx) ggml_free(session.key_ctx);
    session.key_buf = nullptr;
    session.key_ctx = nullptr;
    session.keys = nullptr;
    session.capacity = 0;
    session.ids.clear();
    session.checkpoint = 0;
    session.probe_raw.clear();
    session.subunit_raw.clear();
    session.request_raw.clear();
    session.request_rows.clear();
    session.query_windows.clear();
}

namespace {

int scoring_session_limit() {
    const char * raw = std::getenv("PFLASH_DRAFTER_SESSIONS");
    if (!raw || !*raw) return 2;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) return 2;
    return (int) std::min<long>(value, 16);
}

bool allocate_scoring_session(TargetWeights & w, int capacity,
                              Qwen35ScoringSession & session) {
    {
        ScopedKvTq3Off tq3_off;
        if (!create_target_cache_partial(w, capacity, 0, w.backend, session.cache,
                                         /*prefill_only=*/true, 0, kQwen35HeadBlock,
                                         /*allocate_target_feat=*/false)) {
            return false;
        }
    }
    if (!ensure_ssm_snapshot(session.cache, w.backend)) {
        free_qwen35_scoring_session(session);
        return false;
    }
    ggml_init_params kp{};
    kp.mem_size = ggml_tensor_overhead() + 1024;
    kp.no_alloc = true;
    session.key_ctx = ggml_init(kp);
    if (session.key_ctx) {
        session.keys = ggml_new_tensor_3d(session.key_ctx, GGML_TYPE_F32,
                                          w.n_embd_head_k, w.n_head_kv, capacity);
        session.key_buf = ggml_backend_alloc_ctx_tensors(session.key_ctx, w.backend);
    }
    if (!session.key_buf) {
        free_qwen35_scoring_session(session);
        return false;
    }
    session.capacity = capacity;
    return true;
}

void zero_recurrent_state(TargetCache & cache) {
    for (size_t i = 0; i < cache.ssm_state.size(); ++i) {
        if (cache.ssm_state[i]) {
            ggml_backend_tensor_memset(cache.ssm_state[i], 0, 0,
                                       ggml_nbytes(cache.ssm_state[i]));
        }
        if (i < cache.conv_state.size() && cache.conv_state[i]) {
            ggml_backend_tensor_memset(cache.conv_state[i], 0, 0,
                                       ggml_nbytes(cache.conv_state[i]));
        }
    }
}

// The session to score ``ids`` with, and the token it resumes from: the
// longest prefix a stored session already covers -- its live end, or its
// checkpoint when the prompt diverged before the end (the previous turn's
// generation prompt) -- provided the query rows and probe logits the
// scoring needs are covered too. Otherwise the least recently used session
// (or ``scratch`` with sessions off) starts over from token 0.
Qwen35ScoringSession * acquire_scoring_session(
        Qwen35DrafterState & st,
        const std::vector<int32_t> & ids,
        int query_start,
        int query_end,
        bool need_probe,
        bool need_subunit,
        bool need_request,
        int & resume,
        int & shared_prefix,
        std::unique_ptr<Qwen35ScoringSession> & scratch) {
    TargetWeights & w = st.weights;
    const int S = (int) ids.size();
    const int limit = scoring_session_limit();
    const size_t row_floats = (size_t) w.n_embd * (size_t) (query_end - query_start);
    resume = 0;
    shared_prefix = 0;

    Qwen35ScoringSession * best = nullptr;
    bool best_restore = false;
    int best_shared = 0;
    for (auto & owned : st.sessions) {
        Qwen35ScoringSession * session = owned.get();
        if (!session || session->capacity < S || session->ids.empty() ||
            session->keys_trained != st.head_loaded) {
            continue;
        }
        const size_t n = std::min(session->ids.size(), ids.size());
        const int shared = (int) (std::mismatch(session->ids.begin(),
            session->ids.begin() + (long) n, ids.begin()).first -
            session->ids.begin());
        int r = 0;
        bool restore = false;
        if (shared == (int) session->ids.size()) {
            r = shared;
        } else if (session->checkpoint > 0 && session->checkpoint <= shared) {
            r = session->checkpoint;
            restore = true;
        }
        if (r > query_start) {
            bool rows = false;
            for (const auto & window : session->query_windows) {
                rows = rows || (window.begin == query_start &&
                    window.end == query_end && query_end <= shared &&
                    window.rows.size() == row_floats);
            }
            if (!rows) {
                r = session->checkpoint > 0 && session->checkpoint <= query_start &&
                        session->checkpoint <= shared
                    ? session->checkpoint : 0;
                restore = r > 0;
            }
        }
        if ((need_probe && (int) session->probe_raw.size() < r) ||
            (need_subunit && (int) session->subunit_raw.size() < r) ||
            (need_request && (int) session->request_raw.size() < r)) {
            r = 0;
        }
        if (r > resume) {
            resume = r;
            best = session;
            best_restore = restore;
            best_shared = shared;
        }
    }
    if (best) {
        if (best_restore && !restore_ssm_state(best->cache, w.backend)) {
            resume = 0;
        } else {
            shared_prefix = best_shared;
            return best;
        }
    }

    Qwen35ScoringSession * target = best;
    if (!target) {
        if (limit == 0) {
            scratch = std::make_unique<Qwen35ScoringSession>();
            target = scratch.get();
        } else if ((int) st.sessions.size() < limit) {
            st.sessions.push_back(std::make_unique<Qwen35ScoringSession>());
            target = st.sessions.back().get();
        } else {
            target = std::min_element(st.sessions.begin(), st.sessions.end(),
                [] (const auto & a, const auto & b) {
                    return a->last_used < b->last_used;
                })->get();
        }
    }
    if (target->capacity < S) {
        free_qwen35_scoring_session(*target);
        // Headroom so the next turns append without reallocating.
        const int capacity = limit == 0 ? S : S + S / 2 + 4096;
        if (!allocate_scoring_session(w, capacity, *target)) {
            set_last_error("qwen35 scoring session allocation failed");
            return nullptr;
        }
    }
    zero_recurrent_state(target->cache);
    target->ids.clear();
    target->checkpoint = 0;
    target->probe_raw.clear();
    target->subunit_raw.clear();
    target->request_raw.clear();
    target->request_rows.clear();
    target->query_windows.clear();
    return target;
}

} // namespace

// Scoring-head selection for the Qwen3.5-0.8B drafter: run blocks 0..14, then
// score every context token against the query window with block 15's NoPE
// Q/K (or a trained replacement) and select chunks by attention mass. This
// is the runtime counterpart of the Python retention screen (trial 0075).
std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_mass_out,
    std::vector<PFlashTokenSpan> * segments_out,
    bool * density_out) {

    TargetWeights & w = st.weights;
    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    const int H = w.n_head;
    const int Hk = w.n_head_kv;
    const int D = w.n_embd_head_k;
    std::string block_error;
    if (!qwen35_head_block_available(w, block_error)) {
        set_last_error(block_error);
        return {};
    }
    if (n_lookahead < 1 || S < n_lookahead + 1) {
        set_last_error("qwen35 scoring head input is too short");
        return {};
    }
    const int query_end = score_query_end < 0 ? S : score_query_end;
    if (query_end < n_lookahead || query_end > S) {
        set_last_error("qwen35 scoring head query window out of range");
        return {};
    }
    const int query_start = query_end - n_lookahead;
    const TargetLayer & L = w.layers[(size_t)kQwen35HeadBlock];
    const bool use_probe = st.probe_loaded &&
        experiment.segmentation != luce::pflash::PFlashSegmentation::Fixed;
    // PFLASH_SELECT_QUERY_PROBE=1: the request probe's own trunk runs on
    // the same tap whatever the segmentation mode.
    const bool request_probe = experiment.query_probe;
    if (request_probe && !st.request_loaded) {
        set_last_error("PFLASH_SELECT_QUERY_PROBE=1: the drafter has no request probe");
        return {};
    }

    auto t0 = std::chrono::steady_clock::now();
    int resume = 0;
    int shared_prefix = 0;
    std::unique_ptr<Qwen35ScoringSession> scratch;
    Qwen35ScoringSession * session = acquire_scoring_session(
        st, ids, query_start, query_end, use_probe,
        use_probe && st.probe_sub_fc2_w != nullptr, request_probe, resume,
        shared_prefix, scratch);
    if (!session) return {};
    // Retained request rows past the shared prefix belong to a prompt that
    // diverged there (an edit, a truncation, another conversation).
    const int request_rows_invalidated = request_probe
        ? session->request_rows.invalidate_from(shared_prefix) : 0;
    // A session is released (freed or kept) on every exit below.
    struct SessionExit {
        std::unique_ptr<Qwen35ScoringSession> & scratch;
        ~SessionExit() {
            if (scratch) free_qwen35_scoring_session(*scratch);
        }
    } session_exit{scratch};
    TargetCache & cache = session->cache;
    const int n_new = S - resume;
    // The next turn replaces this prompt's generation prompt; checkpoint the
    // recurrent state a little before the end so it can resume there. The
    // same prompt again (a retry) keeps the checkpoint it has.
    const int checkpoint = n_new == 0 && session->checkpoint > 0
        ? session->checkpoint : std::max(resume, S - 64);

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        session->ids.clear();
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, std::max(1, n_new));
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, std::max(1, n_new));
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        session->ids.clear();
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }
    // Any failure below leaves the session's state half-written: forget its
    // prompt so the next call starts it over.
    auto cleanup = [&]() {
        ggml_backend_buffer_free(act_buf);
        ggml_free(act_ctx);
    };
    auto fail = [&](const char * message) -> std::vector<int32_t> {
        cleanup();
        session->ids.clear();
        set_last_error(message);
        return {};
    };

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < n_new; i += batch) {
            const int n = std::min(batch, n_new - i);
            if (!w.embedder.embed(ids.data() + resume + i, n, emb.data())) {
                return fail("qwen35 drafter embedding failed");
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1],
                                    (size_t)hidden * n * sizeof(float));
        }
    }

    // Blocks 0..14 over the new tokens only, layer by layer. Each
    // DeltaNet layer's recurrent state is copied once it reaches the
    // checkpoint.
    const auto snapshot_layer = [&](int il) {
        int dn = 0;
        for (int l = 0; l < il; ++l) {
            if (((l + 1) % w.full_attention_interval) != 0) ++dn;
        }
        if (dn < (int) cache.ssm_state.size() && cache.ssm_state[(size_t) dn] &&
            cache.ssm_state_snap[(size_t) dn]) {
            ggml_backend_tensor_copy(cache.ssm_state[(size_t) dn],
                                     cache.ssm_state_snap[(size_t) dn]);
            ggml_backend_tensor_copy(cache.conv_state[(size_t) dn],
                                     cache.conv_state_snap[(size_t) dn]);
        }
    };
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    // Ubatches stop at the checkpoint, so the state there can be copied.
    const auto ubatch_at = [&](int start) {
        return std::min(ubatch, (start < checkpoint ? checkpoint : S) - start);
    };
    // Graph of layer il over tokens [start, start + n), writing act_out.
    // nullptr (error set) on failure.
    const auto layer_graph = [&](int il, int start, int n, ggml_cgraph *& gf,
                                 ggml_tensor *& pos) -> ggml_context * {
        ggml_init_params ip{};
        ip.mem_size = 512 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            set_last_error("qwen35 drafter layer graph ctx init failed");
            return nullptr;
        }
        gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1],
                                         (size_t)(start - resume) * act_in->nb[1]);
        pos = nullptr;
        ggml_tensor * mask = nullptr;
        if (((il + 1) % w.full_attention_interval) == 0) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
            ggml_set_input(pos);
            mask = build_qwen35_causal_mask(ctx, gf, start, n);
        }
        ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask,
                                               start, n, false, 0);
        ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1],
                                         (size_t)(start - resume) * act_out->nb[1]);
        if (ggml_nelements(out) != ggml_nelements(dst)) {
            ggml_free(ctx);
            set_last_error("qwen35 layer output shape mismatch");
            return nullptr;
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
        return ctx;
    };
    const auto fail_graph = [&]() -> std::vector<int32_t> {
        ggml_gallocr_free(alloc);
        cleanup();
        session->ids.clear();
        return {};
    };
    // Size the compute buffer once, for a full-attention layer over the
    // ubatch with the largest mask: allocating per ubatch would regrow it
    // with every ubatch of the first attention layer.
    if (n_new > 0 && w.full_attention_interval <= kQwen35HeadBlock) {
        int worst = resume;
        for (int start = resume; start < S; start += ubatch_at(start)) {
            if (causal_mask_bytes(start, ubatch_at(start)) >=
                causal_mask_bytes(worst, ubatch_at(worst))) worst = start;
        }
        ggml_cgraph * gf = nullptr;
        ggml_tensor * pos = nullptr;
        ggml_context * ctx = layer_graph(w.full_attention_interval - 1, worst,
                                         ubatch_at(worst), gf, pos);
        if (!ctx) return fail_graph();
        const bool reserved = ggml_gallocr_reserve(alloc, gf);
        ggml_free(ctx);
        if (!reserved) {
            ggml_gallocr_free(alloc);
            return fail("qwen35 drafter graph allocation failed");
        }
    }
    for (int il = 0; il < kQwen35HeadBlock; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        if (!is_attn && checkpoint == resume) snapshot_layer(il);
        for (int start = resume; start < S;) {
            const int n = ubatch_at(start);
            ggml_cgraph * gf = nullptr;
            ggml_tensor * pos = nullptr;
            ggml_context * ctx = layer_graph(il, start, n, gf, pos);
            if (!ctx) return fail_graph();
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc);
                return fail("qwen35 drafter graph allocation failed");
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    const int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
            }
            const auto status = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (status != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc);
                return fail("qwen35 drafter graph compute failed");
            }
            start += n;
            if (!is_attn && start == checkpoint && checkpoint > resume) {
                snapshot_layer(il);
            }
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    auto t1 = std::chrono::steady_clock::now();

    // Block-14 rows of each query window -- the query, then earlier
    // questions -- from this call when it computed them, else from the
    // session while they sit in the shared prefix. The query's are always
    // available (the session was chosen for them); a history window whose
    // rows are gone is skipped.
    struct ScoredWindow {
        int begin = 0;
        int end = 0;
        double weight = 1.0;
        std::vector<float> rows;
    };
    std::vector<ScoredWindow> windows;
    const auto rows_for = [&](int begin, int end, std::vector<float> & rows) {
        rows.assign((size_t)hidden * (size_t)(end - begin), 0.0f);
        if (begin >= resume) {
            ggml_backend_tensor_get(act_in, rows.data(),
                                    (size_t)(begin - resume) * act_in->nb[1],
                                    rows.size() * sizeof(float));
            return true;
        }
        for (const auto & stored : session->query_windows) {
            if (stored.begin == begin && stored.end == end &&
                end <= shared_prefix && stored.rows.size() == rows.size()) {
                rows = stored.rows;
                return true;
            }
        }
        return false;
    };
    {
        ScoredWindow query;
        query.begin = query_start;
        query.end = query_end;
        if (!rows_for(query_start, query_end, query.rows)) {
            return fail("qwen35 scorer query rows unavailable");
        }
        windows.push_back(std::move(query));
        const auto & turn = experiment.turn_query;
        if (turn.begin >= 0 && turn.end <= query_start && turn.end > turn.begin) {
            ScoredWindow tail;
            tail.begin = turn.begin;
            tail.end = turn.end;
            if (rows_for(turn.begin, turn.end, tail.rows)) {
                windows.push_back(std::move(tail));
            }
        }
        // The user's prose around pasted material (query head), at the
        // query's own weight.
        for (const auto & span : experiment.extra_queries) {
            if (span.end > query_start || span.end - span.begin < 1) continue;
            ScoredWindow extra;
            extra.begin = span.begin;
            extra.end = span.end;
            if (rows_for(span.begin, span.end, extra.rows)) {
                windows.push_back(std::move(extra));
            }
        }
        double weight = 1.0;
        for (const auto & span : experiment.history_queries) {
            weight *= 0.5;
            if (span.end > query_start || span.end - span.begin < 1) continue;
            ScoredWindow history;
            history.begin = span.begin;
            history.end = span.end;
            history.weight = weight;
            if (rows_for(span.begin, span.end, history.rows)) {
                windows.push_back(std::move(history));
            }
        }
    }

    ggml_tensor * wk_src = st.head_loaded ? st.head_wk : L.wk;
    session->keys_trained = st.head_loaded;
    session->probe_raw.resize(use_probe ? (size_t) resume : 0);
    session->subunit_raw.resize(
        use_probe && st.probe_sub_fc2_w ? (size_t) resume : 0);
    session->request_raw.resize(request_probe ? (size_t) resume : 0);

    // Keys (and probe logits) of the new tokens, into the session. Keys are
    // projected in chunks so no intermediate tensor puts the sequence length
    // into a HIP grid y/z dimension (65,535 limit).
    const int key_chunk = 8192;
    if (n_new > 0) {
        ggml_init_params nip{};
        nip.mem_size = (size_t)4 * ggml_tensor_overhead() + 4096;
        nip.no_alloc = true;
        ggml_context * nctx = ggml_init(nip);
        ggml_tensor * probe_new = use_probe ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_tensor * subunit_new = use_probe && st.probe_sub_fc2_w
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_tensor * request_new = request_probe
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_backend_buffer_t nbuf = use_probe || request_probe
            ? ggml_backend_alloc_ctx_tensors(nctx, w.backend) : nullptr;
        if ((use_probe || request_probe) && !nbuf) {
            ggml_free(nctx);
            return fail("qwen35 probe buffer allocation failed");
        }
        const int n_key_chunks = (n_new + key_chunk - 1) / key_chunk;
        ggml_init_params kip{};
        kip.mem_size = ggml_tensor_overhead() *
                           (size_t)(64 + (request_probe ? 40 : 24) * n_key_chunks) +
                       ggml_graph_overhead_custom(4096, false) + 64 * 1024;
        kip.no_alloc = true;
        ggml_context * kctx = ggml_init(kip);
        ggml_cgraph * kgf = ggml_new_graph_custom(kctx, 4096, false);
        for (int b = 0; b < n_new; b += key_chunk) {
            const int n = std::min(key_chunk, n_new - b);
            ggml_tensor * x_c = ggml_view_2d(kctx, act_in, hidden, n, act_in->nb[1],
                                             (size_t)b * act_in->nb[1]);
            ggml_tensor * x_norm = ggml_mul(kctx, ggml_rms_norm(kctx, x_c, w.rms_eps), L.attn_norm);
            ggml_tensor * K = ggml_reshape_3d(kctx, ggml_mul_mat(kctx, wk_src, x_norm), D, Hk, n);
            K = ggml_mul(kctx, ggml_rms_norm(kctx, K, w.rms_eps), L.k_norm);
            ggml_tensor * k_dst = ggml_view_3d(kctx, session->keys, D, Hk, n,
                                               session->keys->nb[1], session->keys->nb[2],
                                               (size_t)(resume + b) * session->keys->nb[2]);
            ggml_build_forward_expand(kgf, ggml_cpy(kctx, K, k_dst));
            if (use_probe) {
                // Segment probe on the same tap: LayerNorm -> fc1 -> GELU trunk,
                // then one fc2 row per head (unit always; sub-unit when shipped).
                ggml_tensor * p = ggml_norm(kctx, x_c, st.probe_norm_eps);
                p = ggml_add(kctx, ggml_mul(kctx, p, st.probe_norm_w), st.probe_norm_b);
                p = ggml_gelu(kctx, ggml_add(kctx, ggml_mul_mat(kctx, st.probe_fc1_w, p),
                                             st.probe_fc1_b));                      // [width, n]
                ggml_tensor * unit = ggml_add(kctx, ggml_mul_mat(kctx, st.probe_fc2_w, p),
                                              st.probe_fc2_b);                      // [1, n]
                ggml_tensor * p_dst = ggml_view_1d(kctx, probe_new, n,
                                                   (size_t)b * ggml_element_size(probe_new));
                ggml_build_forward_expand(kgf, ggml_cpy(kctx, ggml_reshape_1d(kctx, unit, n), p_dst));
                if (subunit_new) {
                    ggml_tensor * sub = ggml_add(kctx,
                        ggml_mul_mat(kctx, st.probe_sub_fc2_w, p), st.probe_sub_fc2_b);
                    ggml_tensor * s_dst = ggml_view_1d(kctx, subunit_new, n,
                        (size_t)b * ggml_element_size(subunit_new));
                    ggml_build_forward_expand(kgf,
                        ggml_cpy(kctx, ggml_reshape_1d(kctx, sub, n), s_dst));
                }
            }
            if (request_new) {
                // Request probe: its own LayerNorm -> fc1 -> GELU(erf) -> fc2
                // trunk on the same tap, one raw logit per token.
                ggml_tensor * r = ggml_norm(kctx, x_c, st.request_norm_eps);
                r = ggml_add(kctx, ggml_mul(kctx, r, st.request_norm_w), st.request_norm_b);
                r = ggml_gelu_erf(kctx, ggml_add(kctx, ggml_mul_mat(kctx, st.request_fc1_w, r),
                                                 st.request_fc1_b));                // [width, n]
                ggml_tensor * req = ggml_add(kctx, ggml_mul_mat(kctx, st.request_fc2_w, r),
                                             st.request_fc2_b);                     // [1, n]
                ggml_tensor * r_dst = ggml_view_1d(kctx, request_new, n,
                    (size_t)b * ggml_element_size(request_new));
                ggml_build_forward_expand(kgf,
                    ggml_cpy(kctx, ggml_reshape_1d(kctx, req, n), r_dst));
            }
        }
        ggml_gallocr_t kalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        const bool key_ok = ggml_gallocr_alloc_graph(kalloc, kgf) &&
            ggml_backend_graph_compute(w.backend, kgf) == GGML_STATUS_SUCCESS;
        ggml_gallocr_free(kalloc);
        ggml_free(kctx);
        if (key_ok && use_probe) {
            session->probe_raw.resize((size_t) S);
            ggml_backend_tensor_get(probe_new, session->probe_raw.data() + resume, 0,
                                    (size_t) n_new * sizeof(float));
            if (subunit_new) {
                session->subunit_raw.resize((size_t) S);
                ggml_backend_tensor_get(subunit_new, session->subunit_raw.data() + resume, 0,
                                        (size_t) n_new * sizeof(float));
            }
        }
        if (key_ok && request_new) {
            session->request_raw.resize((size_t) S);
            ggml_backend_tensor_get(request_new, session->request_raw.data() + resume, 0,
                                    (size_t) n_new * sizeof(float));
        }
        if (nbuf) ggml_backend_buffer_free(nbuf);
        ggml_free(nctx);
        if (!key_ok) return fail("qwen35 key graph compute failed");
    }

    // Request probe (PFLASH_SELECT_QUERY_PROBE=1): the query set from the
    // request logits of the human messages, its block-14 rows, and the rows
    // later turns will need -- all read here, while this call's activations
    // (act_in) still exist. Rows before ``resume`` come from the session's
    // store; a query row found nowhere falls the whole request back to
    // today's windows.
    luce::pflash::PFlashRequestProbeTrace request_trace;
    std::vector<float> request_query_rows;      // [hidden, query tokens]
    if (request_probe) {
        request_trace.tau = st.request_tau;
        request_trace.stub = st.request_stub;
        request_trace.store_invalidated_turns = request_rows_invalidated;
        const auto gather = [&](const std::vector<int> & positions,
                                std::vector<float> & rows, int & fresh,
                                int & stored) -> bool {
            rows.assign((size_t) hidden * positions.size(), 0.0f);
            fresh = 0;
            stored = 0;
            std::vector<int> old;
            std::vector<size_t> old_index;
            size_t index = 0;
            for (const auto & span : luce::pflash::pflash_request_token_spans(positions)) {
                int begin = span.begin;
                for (; begin < std::min(span.end, resume); ++begin) {
                    old.push_back(begin);
                    old_index.push_back(index++);
                }
                if (begin < span.end) {
                    const int n = span.end - begin;
                    ggml_backend_tensor_get(act_in, rows.data() + index * (size_t) hidden,
                                            (size_t) (begin - resume) * act_in->nb[1],
                                            (size_t) hidden * (size_t) n * sizeof(float));
                    fresh += n;
                    index += (size_t) n;
                }
            }
            if (!old.empty()) {
                std::vector<float> old_rows;
                if (!session->request_rows.lookup(old, hidden, old_rows)) return false;
                for (size_t k = 0; k < old.size(); ++k) {
                    std::copy_n(old_rows.data() + k * (size_t) hidden, (size_t) hidden,
                                rows.data() + old_index[k] * (size_t) hidden);
                }
                stored = (int) old.size();
            }
            return true;
        };
        if (!experiment.request_messages_error.empty()) {
            request_trace.fallback = "message_mapping: " + experiment.request_messages_error;
        } else {
            request_trace.set = luce::pflash::pflash_request_query_set(
                session->request_raw, experiment.request_messages, st.request_tau);
            request_trace.fallback = request_trace.set.fallback;
        }
        if (request_trace.fallback.empty()) {
            // v2 calibrated abstention: an unreliable extraction restores today's complete behaviour.
            request_trace.policy_evaluated = true;
            request_trace.abstain = luce::pflash::pflash_request_abstain(
                request_trace.set, session->request_raw, st.request_tau, S, st.request_policy);
            if (!request_trace.abstain.reason.empty()) {
                request_trace.fallback = "abstain:" + request_trace.abstain.reason;
            }
        }
        if (request_trace.fallback.empty() &&
            !gather(request_trace.set.tokens, request_query_rows,
                    request_trace.rows_new, request_trace.rows_stored)) {
            request_trace.fallback = "rows_missing";
            request_query_rows.clear();
            request_trace.rows_new = request_trace.rows_stored = 0;
        }
        // Retain each message's turn (previous first, so the latest is the
        // most recently used), whatever this call decided.
        for (size_t k = request_trace.set.messages.size(); k-- > 0;) {
            const auto & message = request_trace.set.messages[k];
            const std::vector<int> turn = luce::pflash::pflash_request_turn_tokens(message);
            if (turn.empty()) continue;
            std::vector<float> rows;
            int fresh = 0;
            int stored = 0;
            if (gather(turn, rows, fresh, stored)) {
                request_trace.store_evicted_turns +=
                    session->request_rows.put(message.message, turn, rows.data(), hidden);
            }
        }
        request_trace.store_turns = (int) session->request_rows.turns().size();
        request_trace.store_tokens = session->request_rows.tokens();
        // Feature capture and parity checks (request probe on only): append
        // this call's block-14 rows and request logits to
        // PFLASH_TRACE_REQUEST_FEATURES as one record: char[4] "RPF2",
        // i32 resume, i32 n_new, i32 hidden, f32 rows[n_new][hidden],
        // f32 request_logits[n_new] (tokens resume .. resume + n_new).
        if (const char * features = std::getenv("PFLASH_TRACE_REQUEST_FEATURES")) {
            if (*features && n_new > 0) {
                std::vector<float> rows((size_t) hidden * (size_t) n_new);
                ggml_backend_tensor_get(act_in, rows.data(), 0, rows.size() * sizeof(float));
                FILE * file = std::fopen(features, "ab");
                const int32_t header[3] = {resume, n_new, hidden};
                const bool written = file &&
                    std::fwrite("RPF2", 1, 4, file) == 4 &&
                    std::fwrite(header, sizeof(header), 1, file) == 1 &&
                    std::fwrite(rows.data(), sizeof(float), rows.size(), file) == rows.size() &&
                    std::fwrite(session->request_raw.data() + resume, sizeof(float),
                                (size_t) n_new, file) == (size_t) n_new;
                if (file) std::fclose(file);
                if (!written) {
                    return fail("PFLASH_TRACE_REQUEST_FEATURES: feature dump write failed");
                }
            }
        }
        const char * dump = std::getenv("PFLASH_TRACE_REQUEST_LOGITS");
        if (dump && std::strcmp(dump, "1") == 0) {
            for (const auto & message : request_trace.set.messages) {
                request_trace.message_logits.emplace_back(
                    session->request_raw.begin() + message.message.begin,
                    session->request_raw.begin() + message.message.end);
            }
        }
    }
    // PFLASH_SELECT_QUERY_IDENT: the identifier query's block-14 rows, read
    // while this call's activations exist. Positions before ``resume`` have
    // no rows (only the request probe keeps a row store): the request then
    // falls back to today's windows.
    const luce::pflash::PFlashIdentQuery ident_mode = experiment.query_ident;
    luce::pflash::PFlashIdentQueryTrace ident_trace;
    std::vector<float> ident_query_rows;   // [hidden, query tokens]
    if (ident_mode != luce::pflash::PFlashIdentQuery::Off) {
        ident_trace.mode = ident_mode;
        ident_trace.turns = experiment.ident_turns;
        ident_trace.ident_tokens = (int) experiment.ident_tokens.size();
        std::vector<int> query = experiment.ident_tokens;
        if (!experiment.ident_error.empty()) {
            ident_trace.fallback = "mapping: " + experiment.ident_error;
        } else if (query.empty()) {
            ident_trace.fallback = "no_identifiers";
        } else {
            if (query.front() < resume || query.back() >= S) {
                ident_trace.fallback = "rows_missing";
            } else {
                ident_query_rows.assign((size_t) hidden * query.size(), 0.0f);
                size_t index = 0;
                for (const auto & span : luce::pflash::pflash_request_token_spans(query)) {
                    const int n = span.end - span.begin;
                    ggml_backend_tensor_get(act_in, ident_query_rows.data() + index * (size_t) hidden,
                                            (size_t) (span.begin - resume) * act_in->nb[1],
                                            (size_t) hidden * (size_t) n * sizeof(float));
                    index += (size_t) n;
                }
                ident_trace.tokens = std::move(query);
            }
        }
    }
    cleanup();
    // The session now covers this prompt, resumable at the checkpoint and,
    // while the query stays put, reusing its rows.
    session->ids = ids;
    session->checkpoint = checkpoint;
    for (const auto & window : windows) {
        auto & stored = session->query_windows;
        stored.erase(std::remove_if(stored.begin(), stored.end(),
            [&window] (const Qwen35ScoringSession::QueryRows & old) {
                return old.begin == window.begin && old.end == window.end;
            }), stored.end());
        stored.push_back({window.begin, window.end, window.rows});
        if (stored.size() > 8) stored.erase(stored.begin());
    }
    session->last_used = ++st.session_clock;

    // Block-15 NoPE Q/K scoring, once per query window: softmax over the
    // keys outside the query window (before it, and after it when those
    // tokens are candidates), then mean over heads and window tokens. No
    // window scores itself. The logits land in one [S, rows, H] buffer for
    // a single softmax. History windows share the query's key set and mix
    // into its mass at their weights.
    const int n_key_chunks = (S + key_chunk - 1) / key_chunk;
    const auto score_window = [&](const ScoredWindow & window,
                                  std::vector<float> & mass) -> bool {
        const int nq = window.end - window.begin;
        ggml_init_params lip{};
        lip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
        lip.no_alloc = true;
        ggml_context * lctx = ggml_init(lip);
        if (!lctx) {
            set_last_error("qwen35 score buffer ctx allocation failed");
            return false;
        }
        ggml_tensor * logits = ggml_new_tensor_3d(lctx, GGML_TYPE_F32, S, nq, H);
        ggml_tensor * mask = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, S, nq);
        ggml_tensor * x_q = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, hidden, nq);
        ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
        if (!lbuf) {
            ggml_free(lctx);
            set_last_error("qwen35 score buffer allocation failed");
            return false;
        }
        ggml_backend_tensor_set(x_q, window.rows.data(), 0,
                                window.rows.size() * sizeof(float));
        {
            // Keys are the context before the query window and, when the
            // tokens after it are candidates too, the context after it. NoPE
            // scoring has no position term, so a later key scores like an
            // earlier one.
            std::vector<float> m((size_t)nq * S, -INFINITY);
            for (int t = 0; t < nq; ++t) {
                float * row = m.data() + (size_t)t * S;
                std::fill_n(row, (size_t)query_start, 0.0f);
                if (experiment.query_suffix_candidates) {
                    std::fill_n(row + query_end, (size_t)(S - query_end), 0.0f);
                }
                if (window.end <= query_start) {
                    std::fill_n(row + window.begin, (size_t)nq, -INFINITY);
                }
            }
            ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
        }
        ggml_init_params sip{};
        sip.mem_size = ggml_tensor_overhead() * (size_t)(64 + 24 * n_key_chunks) +
                       ggml_graph_overhead_custom(4096, false) + 64 * 1024;
        sip.no_alloc = true;
        ggml_context * sctx = ggml_init(sip);
        if (!sctx) {
            ggml_backend_buffer_free(lbuf); ggml_free(lctx);
            set_last_error("qwen35 score graph ctx allocation failed");
            return false;
        }
        ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 4096, false);
        ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
        ggml_tensor * Q = nullptr;
        if (st.head_loaded) {
            Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, nq);
        } else {
            // Native block 15 packs query and gate rows per head; keep the query half.
            ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                               D * 2, H, nq);
            Q = ggml_view_3d(sctx, QG, D, H, nq,
                             ggml_element_size(QG) * D * 2,
                             ggml_element_size(QG) * D * 2 * H, 0);
        }
        Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
        ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, nq, H]
        for (int b = 0; b < S; b += key_chunk) {
            const int n = std::min(key_chunk, S - b);
            ggml_tensor * K = ggml_view_3d(sctx, session->keys, D, Hk, n,
                                           session->keys->nb[1], session->keys->nb[2],
                                           (size_t)b * session->keys->nb[2]);
            K = ggml_cont(sctx, ggml_permute(sctx, K, 0, 2, 1, 3));  // [D, n, Hk]
            ggml_tensor * K_score = K;
            if (H != Hk) {
                const int gqa = H / Hk;
                ggml_tensor * K_4d = ggml_reshape_4d(sctx, K, D, n, 1, Hk);
                ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, D, n, gqa, Hk);
                K_score = ggml_reshape_3d(sctx, ggml_repeat(sctx, K_4d, K_tpl), D, n, H);
            }
            ggml_tensor * part = ggml_mul_mat(sctx, K_score, Q_perm);  // [n, nq, H]
            ggml_tensor * dst = ggml_view_3d(sctx, logits, n, nq, H,
                                             logits->nb[1], logits->nb[2],
                                             (size_t)b * logits->nb[0]);
            ggml_build_forward_expand(sgf, ggml_cpy(sctx, part, dst));
        }
        ggml_tensor * probs = ggml_soft_max_ext(sctx, logits, mask,
                                                1.0f / std::sqrt((float)D), 0.0f);
        ggml_set_output(probs);
        ggml_build_forward_expand(sgf, probs);
        ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        const bool ok = ggml_gallocr_alloc_graph(salloc, sgf) &&
            ggml_backend_graph_compute(w.backend, sgf) == GGML_STATUS_SUCCESS;
        std::vector<float> probs_h;
        if (ok) {
            probs_h.resize((size_t)S * nq * H);
            ggml_backend_tensor_get(probs, probs_h.data(), 0, probs_h.size() * sizeof(float));
        }
        ggml_gallocr_free(salloc);
        ggml_free(sctx);
        ggml_backend_buffer_free(lbuf);
        ggml_free(lctx);
        if (!ok) {
            set_last_error("qwen35 score graph compute failed");
            return false;
        }
        const size_t nonfinite = count_nonfinite_scores(probs_h.data(), probs_h.size());
        if (nonfinite != 0) {
            const std::string message =
                "non-finite Qwen3.5 scoring-head scores: " + std::to_string(nonfinite) +
                "/" + std::to_string(probs_h.size());
            std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
            std::fflush(stderr);
            set_last_error(message);
            return false;
        }
        scoring_head_mean_token_mass(probs_h.data(), S, nq, H, mass);
        return true;
    };
    std::vector<float> token_mass;
    // Today's windows: each scored on its own, mixed at their weights.
    const auto score_windows = [&]() -> bool {
        token_mass.clear();
        double total_weight = 0.0;
        for (const auto & window : windows) {
            std::vector<float> mass;
            if (!score_window(window, mass)) return false;
            if (token_mass.empty()) token_mass.assign(mass.size(), 0.0f);
            for (size_t i = 0; i < mass.size(); ++i) {
                token_mass[i] += (float) window.weight * mass[i];
            }
            total_weight += window.weight;
        }
        for (auto & value : token_mass) value = (float) (value / total_weight);
        return true;
    };
    // Request probe: ONE gathered query set with one common key mask --
    // today's key range (the context before the query window, and after it
    // when those tokens are candidates) minus every query token -- mean over
    // heads and query tokens, as for a window. Queries run in bounded tiles;
    // each tile's probabilities are reduced to one row per key on the
    // device, so only [S] floats per tile come back.
    // The identifier query uses the same scoring with its tokens left among
    // the keys (scored, not pinned: arm viii's key mask).
    const auto score_request_set = [&](const std::vector<int> & tokens,
                                       const std::vector<float> & rows,
                                       bool mask_query_tokens, int & tiles) -> bool {
        constexpr int kTile = 32;
        const int nq = (int) tokens.size();
        std::vector<float> mask_row((size_t) S, -INFINITY);
        std::fill_n(mask_row.begin(), (size_t) query_start, 0.0f);
        if (experiment.query_suffix_candidates) {
            std::fill(mask_row.begin() + query_end, mask_row.end(), 0.0f);
        }
        if (mask_query_tokens) {
            for (const int t : tokens) mask_row[(size_t) t] = -INFINITY;
        }
        std::vector<double> sum((size_t) S, 0.0);
        std::vector<float> part((size_t) S);
        for (int q0 = 0; q0 < nq; q0 += kTile) {
            const int n = std::min(kTile, nq - q0);
            ggml_init_params lip{};
            lip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
            lip.no_alloc = true;
            ggml_context * lctx = ggml_init(lip);
            if (!lctx) {
                set_last_error("qwen35 request score buffer ctx allocation failed");
                return false;
            }
            ggml_tensor * logits = ggml_new_tensor_3d(lctx, GGML_TYPE_F32, S, n, H);
            ggml_tensor * mask = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, S, n);
            ggml_tensor * x_q = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, hidden, n);
            ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
            if (!lbuf) {
                ggml_free(lctx);
                set_last_error("qwen35 request score buffer allocation failed");
                return false;
            }
            ggml_backend_tensor_set(x_q, rows.data() + (size_t) q0 * hidden, 0,
                                    (size_t) hidden * n * sizeof(float));
            for (int t = 0; t < n; ++t) {
                ggml_backend_tensor_set(mask, mask_row.data(), (size_t) t * mask->nb[1],
                                        (size_t) S * sizeof(float));
            }
            ggml_init_params sip{};
            sip.mem_size = ggml_tensor_overhead() * (size_t)(64 + 24 * n_key_chunks) +
                           ggml_graph_overhead_custom(4096, false) + 64 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_backend_buffer_free(lbuf); ggml_free(lctx);
                set_last_error("qwen35 request score graph ctx allocation failed");
                return false;
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 4096, false);
            ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
            ggml_tensor * Q = nullptr;
            if (st.head_loaded) {
                Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, n);
            } else {
                ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                                   D * 2, H, n);
                Q = ggml_view_3d(sctx, QG, D, H, n,
                                 ggml_element_size(QG) * D * 2,
                                 ggml_element_size(QG) * D * 2 * H, 0);
            }
            Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
            ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, n, H]
            for (int b = 0; b < S; b += key_chunk) {
                const int nk = std::min(key_chunk, S - b);
                ggml_tensor * K = ggml_view_3d(sctx, session->keys, D, Hk, nk,
                                               session->keys->nb[1], session->keys->nb[2],
                                               (size_t)b * session->keys->nb[2]);
                K = ggml_cont(sctx, ggml_permute(sctx, K, 0, 2, 1, 3));  // [D, nk, Hk]
                ggml_tensor * K_score = K;
                if (H != Hk) {
                    const int gqa = H / Hk;
                    ggml_tensor * K_4d = ggml_reshape_4d(sctx, K, D, nk, 1, Hk);
                    ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, D, nk, gqa, Hk);
                    K_score = ggml_reshape_3d(sctx, ggml_repeat(sctx, K_4d, K_tpl), D, nk, H);
                }
                ggml_tensor * scores = ggml_mul_mat(sctx, K_score, Q_perm);  // [nk, n, H]
                ggml_tensor * dst = ggml_view_3d(sctx, logits, nk, n, H,
                                                 logits->nb[1], logits->nb[2],
                                                 (size_t)b * logits->nb[0]);
                ggml_build_forward_expand(sgf, ggml_cpy(sctx, scores, dst));
            }
            ggml_tensor * probs = ggml_soft_max_ext(sctx, logits, mask,
                                                    1.0f / std::sqrt((float)D), 0.0f);
            // Sum over (query, head) per key: [S, n*H] -> [n*H, S] -> [1, S].
            ggml_tensor * per_key = ggml_sum_rows(sctx, ggml_cont(sctx,
                ggml_transpose(sctx, ggml_reshape_2d(sctx, probs, S, (int64_t) n * H))));
            ggml_set_output(per_key);
            ggml_build_forward_expand(sgf, per_key);
            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            const bool ok = ggml_gallocr_alloc_graph(salloc, sgf) &&
                ggml_backend_graph_compute(w.backend, sgf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(per_key, part.data(), 0, (size_t) S * sizeof(float));
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
            ggml_backend_buffer_free(lbuf);
            ggml_free(lctx);
            if (!ok) {
                set_last_error("qwen35 request score graph compute failed");
                return false;
            }
            const size_t nonfinite = count_nonfinite_scores(part.data(), part.size());
            if (nonfinite != 0) {
                const std::string message =
                    "non-finite Qwen3.5 scoring-head scores (request probe): " +
                    std::to_string(nonfinite) + "/" + std::to_string(part.size());
                std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
                std::fflush(stderr);
                set_last_error(message);
                return false;
            }
            for (int j = 0; j < S; ++j) sum[(size_t) j] += part[(size_t) j];
            ++tiles;
        }
        const double denominator = (double) H * (double) nq;
        token_mass.assign((size_t) S, 0.0f);
        for (int j = 0; j < S; ++j) token_mass[(size_t) j] = (float) (sum[(size_t) j] / denominator);
        return true;
    };
    bool request_set = request_probe && request_trace.fallback.empty();
    const bool ident_set = ident_mode != luce::pflash::PFlashIdentQuery::Off &&
        ident_trace.fallback.empty();
    if (request_set ? !score_request_set(request_trace.set.tokens, request_query_rows,
                                         /*mask_query_tokens=*/true, request_trace.query_tiles)
        : ident_set ? !score_request_set(ident_trace.tokens, ident_query_rows,
                                         /*mask_query_tokens=*/false, ident_trace.query_tiles)
                    : !score_windows()) {
        session->ids.clear();
        return {};
    }
    if (ident_mode != luce::pflash::PFlashIdentQuery::Off) {
        ident_trace.query_tokens = ident_set ? (int) ident_trace.tokens.size() : 0;
        if (!ident_set) ident_trace.tokens.clear();
        std::fprintf(stderr,
            "[pflash-ident-query] mode=%s %s turns=%d ident_tokens=%d query_tokens=%d tiles=%d\n",
            luce::pflash::pflash_ident_query_name(ident_mode),
            ident_set ? "scored" : ("fallback=" + ident_trace.fallback).c_str(),
            ident_trace.turns, ident_trace.ident_tokens, ident_trace.query_tokens,
            ident_trace.query_tiles);
        std::fflush(stderr);
    }
    const std::vector<float> & probe_raw = session->probe_raw;
    const std::vector<float> & subunit_raw = session->subunit_raw;
    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[qwen35-scorer] forward %.2fs (blocks 0-%d, S=%d, resumed at %d, "
        "%d new) score %.2fs (%zu query windows) total %.2fs head=%s\n",
        std::chrono::duration<double>(t1 - t0).count(), kQwen35HeadBlock - 1, S,
        resume, n_new,
        std::chrono::duration<double>(t2 - t1).count(),
        request_set || ident_set ? (size_t) 1 : windows.size(),
        std::chrono::duration<double>(t2 - t0).count(),
        st.head_loaded ? "trained" : "native-block15");
    std::fflush(stderr);
    pflash_set_scoring_stats({resume, n_new, request_set || ident_set ? 1 : (int) windows.size(),
                              std::chrono::duration<double>(t1 - t0).count()});

    // Candidates: probe segments cut at the pinned spans' edges, or fixed
    // chunks when the probe is off or finds too few boundaries.
    const auto segment = [&](const std::vector<PFlashTokenSpan> & pins,
                             std::vector<PFlashTokenSpan> & segments, bool & density) {
        segments.clear();
        density = experiment.candidate_score == luce::pflash::PFlashCandidateScore::Density;
        if (!use_probe) return;
        // Tap-count smoothing over the raw logits (torch Conv1d, symmetric
        // padding) plus the residual logit, then sigmoid: the boundary score
        // per token.
        const auto smooth = [&](const std::vector<float> & raw,
                                const std::vector<float> & conv_w, float conv_b,
                                std::vector<float> & out) {
            const int taps = (int) conv_w.size();
            const int radius = taps / 2;
            out.assign((size_t) S, 0.0f);
            for (int t = 0; t < S; ++t) {
                float acc = raw[(size_t) t] + conv_b;
                for (int k = 0; k < taps; ++k) {
                    const int u = t + k - radius;
                    if (u >= 0 && u < S) acc += conv_w[(size_t) k] * raw[(size_t) u];
                }
                out[(size_t) t] = 1.0f / (1.0f + std::exp(-acc));
            }
        };
        std::vector<float> boundary;
        smooth(probe_raw, st.probe_conv_w, st.probe_conv_b, boundary);
        // Sub-unit scores feed only the oversize interior argmax below.
        std::vector<float> split_scores;
        if (!subunit_raw.empty()) {
            smooth(subunit_raw, st.probe_sub_conv_w, st.probe_sub_conv_b, split_scores);
        }
        const int query_end = score_query_end < 0 ? S : score_query_end;
        const int query_begin = query_end - std::min(n_lookahead, query_end);
        std::vector<int> forced{query_begin, query_end};
        for (const auto & span : pins) {
            forced.push_back(span.begin);
            forced.push_back(span.end);
        }
        int boundaries_in_context = 0;
        for (int t = 1; t < S; ++t) {
            if (t >= query_begin &&
                (t < query_end || !experiment.query_suffix_candidates)) {
                continue;
            }
            if (boundary[(size_t) t] > st.probe_threshold) ++boundaries_in_context;
        }
        const bool forced_probe =
            experiment.segmentation == luce::pflash::PFlashSegmentation::Probe;
        if (boundaries_in_context >= 4 || forced_probe) {
            if (st.probe_line_aligned && st.probe_vocab) {
                std::vector<std::string> texts((size_t) S);
                for (int t = 0; t < S; ++t) texts[(size_t) t] = st.probe_vocab->token_text(ids[(size_t) t]);
                const auto lines = luce::pflash::pflash_line_flags(texts);
                segments = luce::pflash::pflash_probe_segments(
                    boundary, S, st.probe_threshold, st.probe_min_segment,
                    st.probe_max_segment, forced, split_scores, &lines);
            } else {
                segments = luce::pflash::pflash_probe_segments(
                    boundary, S, st.probe_threshold, st.probe_min_segment,
                    st.probe_max_segment, forced, split_scores);
            }
        }
        if (segments.empty()) {
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context, "
                "falling back to fixed %d-token chunks\n",
                boundaries_in_context, experiment.chunk_size);
        } else {
            if (experiment.candidate_score == luce::pflash::PFlashCandidateScore::Auto) {
                density = true;
            }
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context -> %zu segments "
                "(threshold %.2f, %d-%d tokens), score=%s\n",
                boundaries_in_context, segments.size(), st.probe_threshold,
                st.probe_min_segment, st.probe_max_segment,
                density ? "density" : "sum");
        }
        std::fflush(stderr);
    };

    std::vector<PFlashTokenSpan> segments;
    bool density = false;
    // The pins the selector charges: the caller's, plus the query tokens of
    // a scored request set.
    std::vector<PFlashTokenSpan> pins = required_instruction_spans;
    if (request_set) {
        for (const auto & span : luce::pflash::pflash_request_token_spans(request_trace.set.tokens)) {
            pins.push_back(span);
        }
        // v2 bounded recency pin: the latest human message's last K tokens, pinned (not scored).
        if (!request_trace.set.messages.empty()) {
            request_trace.recency_pin = luce::pflash::pflash_request_recency_pin(
                request_trace.set.messages[0].message, st.request_policy.recency_pin_tokens);
            if (request_trace.recency_pin.begin >= 0) pins.push_back(request_trace.recency_pin);
        }
        std::sort(pins.begin(), pins.end(), [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
            return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
        });
        std::vector<PFlashTokenSpan> merged;
        for (const auto & span : pins) {
            if (!merged.empty() && span.begin <= merged.back().end) {
                merged.back().end = std::max(merged.back().end, span.end);
            } else {
                merged.push_back(span);
            }
        }
        pins.swap(merged);
    }
    segment(pins, segments, density);

    if (request_probe) {
        // Tokens the selector will keep whatever their score, counted the
        // way it charges them (pflash_chunk_is_structurally_required over
        // the candidates); the query pins must fit the reader budget, or
        // the whole request falls back to today's windows and pins.
        const int sel_query_end = score_query_end < 0 ? S : score_query_end;
        const int sel_query_begin = sel_query_end - std::min(n_lookahead, sel_query_end);
        const auto charged = [&](const std::vector<PFlashTokenSpan> & candidates_pins,
                                 const std::vector<PFlashTokenSpan> & candidate_segments) {
            int total = 0;
            const int n_chunks = candidate_segments.empty()
                ? (S + experiment.chunk_size - 1) / experiment.chunk_size
                : (int) candidate_segments.size();
            for (int c = 0; c < n_chunks; ++c) {
                const int begin = candidate_segments.empty()
                    ? c * experiment.chunk_size : candidate_segments[(size_t) c].begin;
                const int end = candidate_segments.empty()
                    ? std::min(S, begin + experiment.chunk_size)
                    : candidate_segments[(size_t) c].end;
                if (luce::pflash::pflash_chunk_is_structurally_required(
                        begin, end, sel_query_begin, sel_query_end, S, candidates_pins,
                        !experiment.query_suffix_candidates)) {
                    total += end - begin;
                }
            }
            return total;
        };
        request_trace.token_budget = (int) std::floor((double) S * (double) keep_ratio);
        if (request_set) {
            request_trace.query_tokens = (int) request_trace.set.tokens.size();
            int already = 0;
            for (const int t : request_trace.set.tokens) {
                bool pinned = t >= sel_query_begin && t < sel_query_end;
                for (const auto & span : required_instruction_spans) {
                    pinned = pinned || (t >= span.begin && t < span.end);
                }
                already += pinned ? 1 : 0;
            }
            request_trace.pinned_new_tokens = request_trace.query_tokens - already;
            if (request_trace.recency_pin.begin >= 0) {
                const auto & q = request_trace.set.tokens;
                for (int t = request_trace.recency_pin.begin; t < request_trace.recency_pin.end; ++t) {
                    bool pinned = (t >= sel_query_begin && t < sel_query_end) ||
                        std::binary_search(q.begin(), q.end(), t);
                    for (const auto & span : required_instruction_spans) {
                        pinned = pinned || (t >= span.begin && t < span.end);
                    }
                    request_trace.recency_pin_new_tokens += pinned ? 0 : 1;
                }
            }
            request_trace.mandatory_tokens = charged(pins, segments);
            std::vector<PFlashTokenSpan> base_segments;
            bool base_density = false;
            segment(required_instruction_spans, base_segments, base_density);
            request_trace.mandatory_tokens_base = charged(required_instruction_spans, base_segments);
            if (request_trace.mandatory_tokens > request_trace.token_budget &&
                request_trace.mandatory_tokens_base <= request_trace.token_budget) {
                request_trace.fallback = "pins_exceed_budget";
                request_trace.query_tokens = 0;
                request_trace.recency_pin = {-1, -1};
                request_trace.recency_pin_new_tokens = 0;
                request_set = false;
                pins = required_instruction_spans;
                segments.swap(base_segments);
                density = base_density;
                if (!score_windows()) {
                    session->ids.clear();
                    return {};
                }
            }
        } else {
            request_trace.mandatory_tokens = charged(pins, segments);
            request_trace.mandatory_tokens_base = request_trace.mandatory_tokens;
        }
        std::string policy_note;
        if (request_trace.policy_evaluated) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), " policy: margin mean %.3f max %.3f, latest share %.3f, runs %d",
                          request_trace.abstain.mean_margin, request_trace.abstain.max_margin,
                          request_trace.abstain.latest_share, request_trace.abstain.runs);
            policy_note = buf;
        }
        if (request_trace.recency_pin.begin >= 0) {
            policy_note += " recency_pin=[" + std::to_string(request_trace.recency_pin.begin) + "," +
                std::to_string(request_trace.recency_pin.end) + ") new " +
                std::to_string(request_trace.recency_pin_new_tokens);
        }
        std::string runs;
        for (const auto & m : request_trace.set.messages) {
            runs += " [" + std::to_string(m.message.begin) + "," + std::to_string(m.message.end) +
                ")x" + std::to_string(m.runs.size()) + "/" + std::to_string(m.request_tokens);
        }
        std::fprintf(stderr,
            "[pflash-request-probe] %s query_tokens=%d (latest %d, previous %d%s) rows new=%d "
            "stored=%d pins_new=%d mandatory=%d (base %d) budget=%d store=%d turns/%d tokens "
            "(evicted %d, invalidated %d) messages:%s%s%s\n",
            request_set ? "scored" : ("fallback=" + request_trace.fallback).c_str(),
            request_trace.query_tokens, request_trace.set.latest_tokens,
            request_trace.set.previous_tokens, request_trace.set.capped ? ", capped" : "",
            request_trace.rows_new, request_trace.rows_stored, request_trace.pinned_new_tokens,
            request_trace.mandatory_tokens, request_trace.mandatory_tokens_base,
            request_trace.token_budget, request_trace.store_turns, request_trace.store_tokens,
            request_trace.store_evicted_turns, request_trace.store_invalidated_turns,
            runs.empty() ? " none" : runs.c_str(), request_trace.stub ? " (stub row)" : "",
            policy_note.c_str());
        std::fflush(stderr);
    }

    if (token_mass_out) {
        // Scoring only (two-scorer selection): return the per-token mass,
        // the probe segments and the ranking rule; the caller selects.
        *token_mass_out = token_mass;
        if (segments_out) *segments_out = segments;
        if (density_out) *density_out = density;
        return ids;
    }

    luce::pflash::PFlashSelectionConfig request_config;
    const bool traced_query = request_probe || ident_mode != luce::pflash::PFlashIdentQuery::Off;
    if (traced_query) {
        request_config = experiment;
        if (request_probe) request_config.request_probe_trace = &request_trace;
        if (ident_mode != luce::pflash::PFlashIdentQuery::Off) {
            request_config.ident_query_trace = &ident_trace;
        }
    }
    return select_pflash_chunks(
        ids, token_mass, keep_ratio, n_lookahead, score_query_end,
        /*pool_kernel=*/1, traced_query ? request_config : experiment, pins,
        /*direct_mass=*/true, /*write_trace=*/true,
        segments.empty() ? nullptr : &segments, density,
        /*other_token_scores=*/nullptr, /*split_fraction=*/0.0,
        st.fold_vocab.get());
}


std::vector<int32_t> qwen35_drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {
    if (!ctx.state) {
        set_last_error("qwen35 drafter state missing");
        return {};
    }
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    // Strict budget selection scores with the block-15 head; the
    // legacy all-layer running-max scorer stays available for legacy
    // selection or when PFLASH_QWEN35_LEGACY_SCORER=1 forces it.
    const char * legacy_scorer = std::getenv("PFLASH_QWEN35_LEGACY_SCORER");
    const bool force_legacy = (legacy_scorer && std::string(legacy_scorer) == "1") ||
        experiment.scorer == luce::pflash::PFlashScorer::Legacy;
    // Only the block-15 head scores keys after the query window; the
    // running-max scorer, alone or in the split, keeps the suffix.
    luce::pflash::PFlashSelectionConfig suffix_kept = experiment;
    suffix_kept.query_suffix_candidates = false;
    if (experiment.selection_active &&
        experiment.scorer == luce::pflash::PFlashScorer::Split) {
        // Two scorers, one budget: the block-15 head ranks (and segments)
        // first, the all-layer running-max scorer fills the remainder.
        std::vector<float> head_mass;
        std::vector<PFlashTokenSpan> head_segments;
        bool head_density = false;
        if (qwen35_strict_score_and_compress(
                *st, ids, keep_ratio, n_lookahead, score_query_end, suffix_kept,
                required_instruction_spans, &head_mass, &head_segments,
                &head_density).empty()) {
            return {};
        }
        std::vector<float> other_scores;
        if (qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                      n_lookahead, pool_kernel, score_query_end,
                                      suffix_kept, required_instruction_spans,
                                      &other_scores).empty()) {
            return {};
        }
        if (other_scores.size() != head_mass.size()) {
            set_last_error("two-scorer selection: score lengths differ");
            return {};
        }
        std::fprintf(stderr,
            "[pflash-select] two-scorer selection: head fraction %.2f, "
            "segments=%s\n", experiment.split_fraction,
            head_segments.empty() ? "fixed" : "probe");
        std::fflush(stderr);
        return select_pflash_chunks(
            ids, head_mass, keep_ratio, n_lookahead, score_query_end,
            /*pool_kernel=*/1, suffix_kept, required_instruction_spans,
            /*direct_mass=*/true, /*write_trace=*/true,
            head_segments.empty() ? nullptr : &head_segments, head_density,
            &other_scores, experiment.split_fraction);
    }
    if (experiment.selection_active && !force_legacy) {
        auto kept = qwen35_strict_score_and_compress(
            *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
            required_instruction_spans);
        // Non-finite head scores (once in ~500 development requests, not
        // reproduced) leave the scoring session forgotten: score the prompt
        // again from scratch, one drafter forward, instead of failing the
        // request.
        if (kept.empty() &&
            std::strncmp(luce_last_error(), "non-finite", 10) == 0) {
            std::fprintf(stderr,
                "[qwen35-scorer] non-finite scores; rescoring from scratch\n");
            std::fflush(stderr);
            kept = qwen35_strict_score_and_compress(
                *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                required_instruction_spans);
        }
        return kept;
    }
    if (st->head_loaded && !experiment.selection_active) {
        set_last_error("Qwen3.5 scoring head requires strict selection");
        return {};
    }
    return qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                     n_lookahead, pool_kernel, score_query_end,
                                     suffix_kept,
                                     required_instruction_spans);
}

} // namespace luce::common
