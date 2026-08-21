#include "dflash2_head.h"

#include "dflash2_selector_validation.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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

DFlash2DepthSignal make_depth_signal(
        const float * log_probs, const std::vector<float> & scores,
        int K, int selected) {
    DFlash2DepthSignal signal;
    if (!log_probs || K <= 0 || selected < 0 || selected >= K ||
        static_cast<int>(scores.size()) != K) {
        return signal;
    }
    signal.selected_log_prob = log_probs[selected];
    signal.lm_top2_margin = K > 1 ? log_probs[0] - log_probs[1]
                                  : std::numeric_limits<float>::infinity();
    float top_k_mass = 0.0f;
    for (int k = 0; k < K; ++k) top_k_mass += std::exp(log_probs[k]);
    signal.top_k_mass = std::clamp(top_k_mass, 0.0f, 1.0f);
    signal.selected_rank = selected;
    signal.agrees_with_lm_top1 = selected == 0;

    float runner_up = -INFINITY;
    for (int k = 0; k < K; ++k) {
        if (k != selected) runner_up = std::max(runner_up, scores[(size_t)k]);
    }
    signal.selector_margin = K > 1
        ? scores[(size_t)selected] - runner_up
        : std::numeric_limits<float>::infinity();

    const float max_score = *std::max_element(scores.begin(), scores.end());
    float z = 0.0f;
    for (float score : scores) z += std::exp(score - max_score);
    if (z > 0.0f && std::isfinite(z)) {
        signal.selector_winner_mass =
            std::exp(scores[(size_t)selected] - max_score) / z;
        float entropy = 0.0f;
        for (float score : scores) {
            const float p = std::exp(score - max_score) / z;
            if (p > 0.0f) entropy -= p * std::log(p);
        }
        signal.selector_entropy = entropy;
    }
    return signal;
}

bool score_depth(const Dflash2TreeScores & scores,
                 const std::vector<int32_t> & prefix,
                 int next_depth,
                 std::vector<float> & out_scores) {
    const int position = next_depth - 1;
    if (position < 0 || position >= scores.n_cand ||
        static_cast<int>(prefix.size()) != position) {
        return false;
    }

    int predecessor_row = 0;
    if (position > 0) {
        predecessor_row = -1;
        const int32_t parent_token = prefix.back();
        for (int candidate = 0; candidate < scores.K; ++candidate) {
            if (scores.ids[(size_t)(position - 1) * scores.K + candidate] ==
                parent_token) {
                predecessor_row = 1 + (position - 1) * scores.K + candidate;
                break;
            }
        }
        if (predecessor_row < 0) predecessor_row = 0;
    }

    const float * predecessor =
        scores.pred.data() + (size_t)predecessor_row * scores.rank;
    const float * projected_hidden =
        scores.hproj.data() + (size_t)position * scores.rank;
    out_scores.resize((size_t)scores.K);
    for (int candidate = 0; candidate < scores.K; ++candidate) {
        const float * successor = scores.succ.data() +
            ((size_t)position * scores.K + candidate) * scores.rank;
        float correction = 0.0f;
        for (int component = 0; component < scores.rank; ++component) {
            correction += predecessor[component] * projected_hidden[component] *
                          successor[component];
        }
        out_scores[(size_t)candidate] =
            scores.lp[(size_t)position * scores.K + candidate] + correction;
    }
    return true;
}

}  // namespace

bool dflash2_score_candidates(const DraftWeights & dw,
                              ggml_backend_t backend,
                              DFlashTarget & target,
                              const float * local_hidden,
                              int q_len,
                              int32_t last_tok,
                              float temperature,
                              Dflash2TreeScores & out) {
    const DraftSelectorWeights & sel = dw.selector;
    if (!sel.enabled || !sel.hproj || !sel.pred_cb || !sel.succ_cb) return false;
    if (q_len <= 1 || !local_hidden || !backend) return false;
    const int hdim   = dw.n_embd;
    const int rank   = sel.rank;
    const int K      = sel.top_k;
    const int n_cand = q_len - 1;
    if (hdim <= 0 || rank <= 0 || K <= 0) return false;
    DFlash2SelectorLayout selector_layout;
    selector_layout.rank = rank;
    selector_layout.top_k = K;
    selector_layout.hproj_rank = sel.hproj->ne[1];
    selector_layout.pred_rank = sel.pred_cb->ne[0];
    selector_layout.pred_vocab = sel.pred_cb->ne[1];
    selector_layout.succ_rank = sel.succ_cb->ne[0];
    selector_layout.succ_vocab = sel.succ_cb->ne[1];
    std::string selector_error;
    if (!validate_dflash2_selector_layout(
            selector_layout, selector_error)) {
        std::fprintf(stderr, "dflash2_score_candidates: %s\n",
                     selector_error.c_str());
        return false;
    }

    // 1. Top-k candidates (log-probs) per block position through the target
    //    lm_head. Position 0 of local_hidden is the seed slot; candidates are
    //    rows 1 .. q_len-1.
    if (!target.project_hidden_to_topk(local_hidden + (size_t)hdim, n_cand, K, temperature,
                                       out.lp, out.ids)) {
        return false;
    }
    if (out.lp.size() != (size_t)n_cand * K || out.ids.size() != (size_t)n_cand * K) return false;

    // 2. One graph on the draft backend: hproj(h) for every candidate position,
    //    successor rows for every candidate, predecessor rows for the seed and
    //    every candidate. Built once per (n_cand, K) and reused across steps.
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
            std::fprintf(stderr, "dflash2_score_candidates: gallocr_alloc_graph failed\n");
            selector_graph_free(g);
            return false;
        }
        g.dw = &dw; g.backend = backend; g.n_cand = n_cand; g.K = K;
    }

    std::vector<int32_t> pred_ids((size_t)n_rows_pred);
    pred_ids[0] = last_tok;
    std::memcpy(pred_ids.data() + 1, out.ids.data(), sizeof(int32_t) * (size_t)n_cand * K);
    ggml_backend_tensor_set(g.inp_hidden, local_hidden + (size_t)hdim, 0, sizeof(float) * (size_t)hdim * n_cand);
    ggml_backend_tensor_set(g.inp_succ, out.ids.data(), 0, sizeof(int32_t) * (size_t)n_cand * K);
    ggml_backend_tensor_set(g.inp_pred, pred_ids.data(), 0, sizeof(int32_t) * (size_t)n_rows_pred);
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dflash2_score_candidates: graph_compute failed\n");
        return false;
    }
    out.hproj.resize((size_t)rank * n_cand);
    out.succ.resize((size_t)rank * n_cand * K);
    out.pred.resize((size_t)rank * n_rows_pred);
    ggml_backend_tensor_get_async(backend, g.hproj, out.hproj.data(), 0, sizeof(float) * out.hproj.size());
    ggml_backend_tensor_get_async(backend, g.succ,  out.succ.data(),  0, sizeof(float) * out.succ.size());
    ggml_backend_tensor_get_async(backend, g.pred,  out.pred.data(),  0, sizeof(float) * out.pred.size());
    ggml_backend_synchronize(backend);
    out.n_cand = n_cand; out.K = K; out.rank = rank; out.seed = last_tok;
    return true;
}

bool Dflash2TreeScores::topk(const std::vector<int32_t> & prefix, int next_depth,
                             std::vector<float> & out_lp, std::vector<int32_t> & out_ids) const {
    const int position = next_depth - 1;
    std::vector<float> scores;
    if (!score_depth(*this, prefix, next_depth, scores)) return false;

    std::vector<std::pair<float, int>> scored((size_t)K);
    for (int candidate = 0; candidate < K; ++candidate) {
        scored[(size_t)candidate] = {
            scores[(size_t)candidate], candidate
        };
    }
    std::sort(scored.begin(), scored.end(),
              [](const std::pair<float,int> & a, const std::pair<float,int> & b) { return a.first > b.first; });
    // The compatibility dot is not on a log-prob scale; renormalize the
    // adjusted scores per position (log-softmax) so the tree builder's
    // cumulative best-first comparison across depths stays meaningful.
    float lse = 0.0f;
    const float mx = scored[0].first;
    for (int k = 0; k < K; ++k) lse += std::exp(scored[(size_t)k].first - mx);
    lse = mx + std::log(lse);
    out_lp.resize((size_t)K);
    out_ids.resize((size_t)K);
    for (int k = 0; k < K; ++k) {
        out_lp[(size_t)k]  = scored[(size_t)k].first - lse;
        out_ids[(size_t)k] =
            ids[(size_t)position * K + scored[(size_t)k].second];
    }
    return true;
}

bool dflash2_select_chain(const DraftWeights & dw,
                          ggml_backend_t backend,
                          DFlashTarget & target,
                          const float * local_hidden,
                          int q_len,
                          int32_t last_tok,
                          std::vector<int32_t> & draft_tok,
                          DFlash2SelectorTrace * trace) {
    Dflash2TreeScores sc;
    if (!dflash2_score_candidates(dw, backend, target, local_hidden, q_len, last_tok,
                                  /*temperature=*/1.0f, sc)) {
        return false;
    }
    // Greedy path over the candidates, conditioned on the previous pick.
    draft_tok.assign((size_t)q_len, last_tok);
    std::vector<int32_t> prefix;
    std::vector<float>   top_lp;
    std::vector<int32_t> top_ids;
    if (trace) {
        trace->depths.clear();
        trace->depths.reserve((size_t)sc.n_cand);
    }
    for (int i = 0; i < sc.n_cand; ++i) {
        if (!sc.topk(prefix, i + 1, top_lp, top_ids)) return false;
        draft_tok[(size_t)i + 1] = top_ids[0];
        if (trace) {
            int selected = -1;
            for (int candidate = 0; candidate < sc.K; ++candidate) {
                if (sc.ids[(size_t)i * sc.K + candidate] == top_ids[0]) {
                    selected = candidate;
                    break;
                }
            }
            std::vector<float> scores;
            if (selected < 0 || !score_depth(sc, prefix, i + 1, scores)) {
                return false;
            }
            trace->depths.push_back(make_depth_signal(
                sc.lp.data() + (size_t)i * sc.K, scores, sc.K, selected));
        }
        prefix.push_back(top_ids[0]);
    }
    return true;
}

}  // namespace dflash::common
