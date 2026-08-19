#include "dflash2_head.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash::common {

namespace {

// Selector projection graph, built once per (drafter, backend, n_cand, K).
struct SelectorGraph {
    const DraftWeights * dw = nullptr;
    ggml_backend_t backend = nullptr;
    int n_cand = 0;
    int K = 0;
    std::vector<uint8_t> arena;
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf = nullptr;
    ggml_gallocr_t  galloc = nullptr;
    ggml_tensor * inp_hidden = nullptr;
    ggml_tensor * inp_succ = nullptr;
    ggml_tensor * inp_pred = nullptr;
    ggml_tensor * hproj = nullptr;
    ggml_tensor * succ = nullptr;
    ggml_tensor * pred = nullptr;
};

SelectorGraph & selector_graph() {
    static thread_local SelectorGraph g;
    return g;
}

void selector_graph_free(SelectorGraph & g) {
    if (g.galloc) { ggml_gallocr_free(g.galloc); g.galloc = nullptr; }
    if (g.ctx)    { ggml_free(g.ctx); g.ctx = nullptr; }
    g.gf = nullptr;
    g.dw = nullptr;
    g.n_cand = 0;
    g.K = 0;
}

}  // namespace

bool dflash2_select_chain(const DraftWeights & dw,
                          ggml_backend_t backend,
                          DFlashTarget & target,
                          const float * local_hidden,
                          int q_len,
                          int32_t last_tok,
                          std::vector<int32_t> & draft_tok) {
    const DraftSelectorWeights & sel = dw.selector;
    if (!sel.enabled || !sel.hproj || !sel.pred_cb || !sel.succ_cb) return false;
    if (q_len <= 1 || !local_hidden || !backend) return false;
    const int hdim   = dw.n_embd;
    const int rank   = sel.rank;
    const int K      = sel.top_k;
    const int n_cand = q_len - 1;
    if (hdim <= 0 || rank <= 0 || K <= 0) return false;

    // 1. Top-k candidates (log-probs) per block position through the target
    //    lm_head. Position 0 of local_hidden is the seed slot; candidates are
    //    rows 1 .. q_len-1.
    std::vector<float>   cand_lp;
    std::vector<int32_t> cand_ids;
    if (!target.project_hidden_to_topk(local_hidden + (size_t)hdim, n_cand, K, /*temperature=*/1.0f,
                                       cand_lp, cand_ids)) {
        return false;
    }
    if (cand_lp.size() != (size_t)n_cand * K || cand_ids.size() != (size_t)n_cand * K) return false;

    // 2. One graph on the draft backend: hproj(h) for every candidate position,
    //    successor rows for every candidate, predecessor rows for the seed and
    //    every candidate (the path picks its predecessor among them). The
    //    graph shape only depends on (n_cand, K), so it is built once and
    //    reused across steps.
    const int n_rows_pred = 1 + n_cand * K;
    SelectorGraph & g = selector_graph();
    if (!g.ctx || g.dw != &dw || g.backend != backend || g.n_cand != n_cand || g.K != K) {
        selector_graph_free(g);
        const size_t arena_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096;
        g.arena.assign(arena_size, 0);
        ggml_init_params ip{};
        ip.mem_size   = g.arena.size();
        ip.mem_buffer = g.arena.data();
        ip.no_alloc   = true;
        g.ctx = ggml_init(ip);
        if (!g.ctx) return false;
        g.gf = ggml_new_graph(g.ctx);
        g.inp_hidden = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, hdim, n_cand);
        g.inp_succ   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_cand * K);
        g.inp_pred   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_rows_pred);
        ggml_set_input(g.inp_hidden);
        ggml_set_input(g.inp_succ);
        ggml_set_input(g.inp_pred);
        g.hproj = ggml_mul_mat(g.ctx, sel.hproj, g.inp_hidden);      // [rank, n_cand]
        g.succ  = ggml_get_rows(g.ctx, sel.succ_cb, g.inp_succ);      // [rank, n_cand*K] f32
        g.pred  = ggml_get_rows(g.ctx, sel.pred_cb, g.inp_pred);      // [rank, 1+n_cand*K] f32
        ggml_set_output(g.hproj);
        ggml_set_output(g.succ);
        ggml_set_output(g.pred);
        ggml_build_forward_expand(g.gf, g.hproj);
        ggml_build_forward_expand(g.gf, g.succ);
        ggml_build_forward_expand(g.gf, g.pred);
        g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!g.galloc || !ggml_gallocr_alloc_graph(g.galloc, g.gf)) {
            std::fprintf(stderr, "dflash2_select_chain: gallocr_alloc_graph failed\n");
            selector_graph_free(g);
            return false;
        }
        g.dw = &dw; g.backend = backend; g.n_cand = n_cand; g.K = K;
    }

    std::vector<int32_t> pred_ids((size_t)n_rows_pred);
    pred_ids[0] = last_tok;
    std::memcpy(pred_ids.data() + 1, cand_ids.data(), sizeof(int32_t) * (size_t)n_cand * K);
    ggml_backend_tensor_set(g.inp_hidden, local_hidden + (size_t)hdim, 0, sizeof(float) * (size_t)hdim * n_cand);
    ggml_backend_tensor_set(g.inp_succ, cand_ids.data(), 0, sizeof(int32_t) * (size_t)n_cand * K);
    ggml_backend_tensor_set(g.inp_pred, pred_ids.data(), 0, sizeof(int32_t) * (size_t)n_rows_pred);
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dflash2_select_chain: graph_compute failed\n");
        return false;
    }
    std::vector<float> h_hproj((size_t)rank * n_cand);
    std::vector<float> h_succ((size_t)rank * n_cand * K);
    std::vector<float> h_pred((size_t)rank * n_rows_pred);
    ggml_backend_tensor_get_async(backend, g.hproj, h_hproj.data(), 0, sizeof(float) * h_hproj.size());
    ggml_backend_tensor_get_async(backend, g.succ,  h_succ.data(),  0, sizeof(float) * h_succ.size());
    ggml_backend_tensor_get_async(backend, g.pred,  h_pred.data(),  0, sizeof(float) * h_pred.size());
    ggml_backend_synchronize(backend);

    // 3. Path search: greedy over the candidates, conditioned on the previous pick.
    draft_tok.assign((size_t)q_len, last_tok);
    int prev_row = 0;   // row in h_pred: 0 = seed, 1 + i*K + k = candidate k of position i
    for (int i = 0; i < n_cand; ++i) {
        const float * pr = h_pred.data() + (size_t)prev_row * rank;
        const float * hp = h_hproj.data() + (size_t)i * rank;
        float best = -INFINITY;
        int   best_k = 0;
        for (int k = 0; k < K; ++k) {
            const float * sc = h_succ.data() + ((size_t)i * K + k) * rank;
            float dot = 0.0f;
            for (int r = 0; r < rank; ++r) dot += pr[r] * hp[r] * sc[r];
            const float score = cand_lp[(size_t)i * K + k] + dot;
            if (score > best) { best = score; best_k = k; }
        }
        draft_tok[(size_t)i + 1] = cand_ids[(size_t)i * K + best_k];
        prev_row = 1 + i * K + best_k;
    }
    return true;
}

}  // namespace dflash::common
