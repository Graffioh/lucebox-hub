// GPU parity test for the SpecLA topology-masked delta-net verify builder
// (src/delta_net_specla.cpp) against the fused sequential ggml_gated_delta_net
// kernel, which advances the recurrent state token by token and is the
// numerical ground truth for the recurrence.
//
// Checks, per shape/topology case:
//   1. per-node outputs of build_delta_net_specla match the fused kernel
//      (chain cases use ggml_gated_delta_net, tree cases the _tree variant);
//   2. host-side DeltaConstruct over the captured factors —
//         S_A = exp(g⁺_A) S0 + Σ_{u ∈ path(A)} exp(g⁺_A − g⁺_u) k_u ⊗ ṽ_u
//      — matches the fused kernel's per-token intermediate state at EVERY
//      possible accepted endpoint A (every prefix of a chain, every node of a
//      tree). This is the correctness contract the factor-based accepted-state
//      commit (SPECLA.md §1-§3) relies on;
//   3. g⁺ equals the host-computed ancestor path sum of g.

#include "delta_net_specla.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"  // ggml_backend_cuda_init; maps to HIP under GGML_USE_HIP

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using dflash::common::build_delta_net_specla;
using dflash::common::fill_specla_masks;

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        failures++; \
    } \
} while (0)

namespace {

struct CaseInputs {
    int S = 0, H = 0, n = 0;
    std::vector<int32_t> parents;             // parents[0] = -1, parents[t] < t
    std::vector<float> q, k, v, g, b, s0;     // layouts as fed to the builders
};

CaseInputs make_inputs(int S, int H, int n, const std::vector<int32_t> & parents,
                       unsigned seed) {
    CaseInputs in;
    in.S = S; in.H = H; in.n = n; in.parents = parents;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    auto fill_unit_heads = [&](std::vector<float> & dst) {
        // [S, H, n]: one l2-normalized S-vector per (head, token), like the
        // post-l2_norm q/k the real block feeds the recurrence.
        dst.resize((size_t)S * H * n);
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < H; h++) {
                float norm2 = 0.0f;
                float * vec = dst.data() + (size_t)t * S * H + (size_t)h * S;
                for (int s = 0; s < S; s++) { vec[s] = nd(rng); norm2 += vec[s] * vec[s]; }
                const float inv = 1.0f / std::sqrt(norm2 + 1e-6f);
                for (int s = 0; s < S; s++) vec[s] *= inv;
            }
        }
    };
    fill_unit_heads(in.q);
    fill_unit_heads(in.k);

    in.v.resize((size_t)S * H * n);
    for (auto & x : in.v) x = 0.5f * nd(rng);
    in.g.resize((size_t)H * n);               // [1, H, n] — log-decay, negative
    for (auto & x : in.g) x = -(0.05f + 1.5f * ud(rng));
    in.b.resize((size_t)H * n);               // [1, H, n] — sigmoid-like in (0,1)
    for (auto & x : in.b) x = 0.1f + 0.8f * ud(rng);
    in.s0.resize((size_t)S * S * H);
    for (auto & x : in.s0) x = 0.1f * nd(rng);
    return in;
}

struct RefOutputs {
    std::vector<float> attn;    // [S*H per token][n] token-major
    std::vector<float> inter;   // [S*S*H per token][n] state after node t
};

struct SpecLAOutputs {
    std::vector<float> out;     // [S, H, n] — same layout as RefOutputs::attn
    std::vector<float> v_new;   // [n, S_v, 1, H]
    std::vector<float> g_ps;    // [n, 1, 1, H]
};

struct GraphEnv {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;

    explicit GraphEnv(size_t n_tensors = 512) {
        ggml_init_params ip{};
        ip.mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, 2048, false);
    }
    ~GraphEnv() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }
    bool alloc_and_run(ggml_backend_t backend) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) return false;
        return true;
    }
};

void set_f32(ggml_tensor * t, const std::vector<float> & host) {
    GGML_ASSERT((size_t)ggml_nelements(t) == host.size());
    ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
}

std::vector<float> get_f32(const ggml_tensor * t, size_t off_elems, size_t n_elems) {
    std::vector<float> out(n_elems);
    ggml_backend_tensor_get(t, out.data(), off_elems * sizeof(float),
                            n_elems * sizeof(float));
    return out;
}

// Reference pass: fused sequential kernel, chain (plain op) or tree variant.
bool run_reference(ggml_backend_t backend, const CaseInputs & in, bool tree_op,
                   RefOutputs & ref) {
    const int S = in.S, H = in.H, n = in.n;
    GraphEnv env;
    ggml_context * ctx = env.ctx;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    for (ggml_tensor * t : {q, k, v, g, b, s}) ggml_set_input(t);

    ggml_tensor * parent_ids = nullptr;
    ggml_tensor * result = nullptr;
    if (tree_op) {
        parent_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(parent_ids);
        result = ggml_gated_delta_net_tree(ctx, q, k, v, g, b, s, parent_ids);
    } else {
        result = ggml_gated_delta_net(ctx, q, k, v, g, b, s);
    }
    // Intermediates deliberately kept (no set_skip_intermediate): they are the
    // per-token ground-truth states DeltaConstruct is validated against.
    ggml_set_output(result);
    ggml_build_forward_expand(env.gf, result);

    if (!env.alloc_and_run(backend)) return false;
    set_f32(q, in.q); set_f32(k, in.k); set_f32(v, in.v);
    set_f32(g, in.g); set_f32(b, in.b); set_f32(s, in.s0);
    if (parent_ids) {
        ggml_backend_tensor_set(parent_ids, in.parents.data(), 0,
                                sizeof(int32_t) * n);
    }
    if (ggml_backend_graph_compute(backend, env.gf) != GGML_STATUS_SUCCESS) return false;

    // Packed result: [ attn: S*H*n | final_state: S*S*H | inter: S*S*H*n ]
    ref.attn  = get_f32(result, 0, (size_t)S * H * n);
    ref.inter = get_f32(result, (size_t)S * H * n + (size_t)S * S * H,
                        (size_t)S * S * H * n);
    return true;
}

bool run_specla(ggml_backend_t backend, const CaseInputs & in, SpecLAOutputs & out) {
    const int S = in.S, H = in.H, n = in.n;
    GraphEnv env;
    ggml_context * ctx = env.ctx;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * m_strict = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * m_incl   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * m_eye    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    for (ggml_tensor * t : {q, k, v, g, b, s, m_strict, m_incl, m_eye}) ggml_set_input(t);

    auto r = build_delta_net_specla(ctx, q, k, v, g, b, s, m_strict, m_incl, m_eye);
    for (ggml_tensor * t : {r.output, r.v_new, r.g_ps}) {
        ggml_set_output(t);
        ggml_build_forward_expand(env.gf, t);
    }

    if (!env.alloc_and_run(backend)) return false;
    set_f32(q, in.q); set_f32(k, in.k); set_f32(v, in.v);
    set_f32(g, in.g); set_f32(b, in.b); set_f32(s, in.s0);
    std::vector<float> ms((size_t)n * n), mi((size_t)n * n), me((size_t)n * n);
    fill_specla_masks(in.parents.data(), n, ms.data(), mi.data(), me.data());
    set_f32(m_strict, ms); set_f32(m_incl, mi); set_f32(m_eye, me);
    if (ggml_backend_graph_compute(backend, env.gf) != GGML_STATUS_SUCCESS) return false;

    out.out   = get_f32(r.output, 0, (size_t)S * H * n);
    out.v_new = get_f32(r.v_new,  0, (size_t)n * S * H);
    out.g_ps  = get_f32(r.g_ps,   0, (size_t)n * H);
    return true;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

void run_case(ggml_backend_t backend, const char * name, const CaseInputs & in,
              bool tree_op, float out_tol, float state_tol) {
    const int S = in.S, H = in.H, n = in.n;

    RefOutputs ref;
    SpecLAOutputs sp;
    if (!run_reference(backend, in, tree_op, ref)) {
        CHECK_MSG(false, "%s: reference compute failed", name);
        return;
    }
    if (!run_specla(backend, in, sp)) {
        CHECK_MSG(false, "%s: specla compute failed", name);
        return;
    }

    // 1. Per-node outputs.
    const float out_diff = max_abs_diff(ref.attn, sp.out);
    CHECK_MSG(out_diff <= out_tol, "%s: output max diff %.3e > %.3e", name, out_diff, out_tol);

    // 3. g⁺ vs host ancestor path sums (g host layout [H, n] per token block).
    float gps_diff = 0.0f;
    std::vector<float> gps_host((size_t)n * H);
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < H; h++) {
            float acc = 0.0f;
            for (int u = t; u >= 0; u = in.parents[u]) acc += in.g[(size_t)u * H + h];
            gps_host[(size_t)t * H + h] = acc;
            // sp.g_ps layout [n, 1, 1, H]: (t, h) at t + h*n
            gps_diff = std::max(gps_diff,
                std::fabs(acc - sp.g_ps[(size_t)h * n + t]));
        }
    }
    CHECK_MSG(gps_diff <= 1e-5f, "%s: g_ps max diff %.3e", name, gps_diff);

    // 2. DeltaConstruct at every accepted endpoint t: reconstruct S_t from
    //    {k, ṽ, g⁺} along root→t and compare with the kernel's state after t.
    float state_diff = 0.0f;
    std::vector<float> s_rec((size_t)S * S * H);
    for (int t = 0; t < n; t++) {
        // path root→t
        std::vector<int> path;
        for (int u = t; u >= 0; u = in.parents[u]) path.push_back(u);
        for (int h = 0; h < H; h++) {
            const float gA = gps_host[(size_t)t * H + h];
            const float decay0 = std::exp(gA);
            for (int c = 0; c < S; c++) {
                for (int sk = 0; sk < S; sk++) {
                    s_rec[(size_t)h * S * S + (size_t)c * S + sk] =
                        decay0 * in.s0[(size_t)h * S * S + (size_t)c * S + sk];
                }
            }
            for (int u : path) {
                const float w = std::exp(gA - gps_host[(size_t)u * H + h]);
                // k host layout [S, H, n]; ṽ layout [n, S_v, 1, H]
                const float * ku = in.k.data() + (size_t)u * S * H + (size_t)h * S;
                for (int c = 0; c < S; c++) {
                    const float wv = w * sp.v_new[(size_t)h * n * S + (size_t)c * n + u];
                    float * dst = s_rec.data() + (size_t)h * S * S + (size_t)c * S;
                    for (int sk = 0; sk < S; sk++) dst[sk] += wv * ku[sk];
                }
            }
        }
        const std::vector<float> s_ref(ref.inter.begin() + (size_t)t * S * S * H,
                                       ref.inter.begin() + (size_t)(t + 1) * S * S * H);
        state_diff = std::max(state_diff, max_abs_diff(s_rec, s_ref));
    }
    CHECK_MSG(state_diff <= state_tol, "%s: DeltaConstruct state max diff %.3e > %.3e",
              name, state_diff, state_tol);

    std::printf("%-28s S=%-3d H=%-2d n=%-3d out=%.3e g+=%.3e state=%.3e\n",
                name, S, H, n, out_diff, gps_diff, state_diff);
}

std::vector<int32_t> chain_parents(int n) {
    std::vector<int32_t> p(n);
    for (int t = 0; t < n; t++) p[t] = t - 1;
    return p;
}

std::vector<int32_t> random_tree_parents(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> p(n);
    p[0] = -1;
    for (int t = 1; t < n; t++) {
        // Bias toward recent nodes so trees have realistic depth.
        std::uniform_int_distribution<int> d(std::max(0, t - 4), t - 1);
        p[t] = d(rng);
    }
    return p;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "test_delta_net_specla: no GPU backend available\n");
        return 77;  // ctest SKIP
    }

    const float kOutTol   = 5e-4f;
    const float kStateTol = 5e-4f;

    // Chain drafts vs the plain fused op.
    run_case(backend, "chain-small",
             make_inputs(64, 4, 8, chain_parents(8), 1), false, kOutTol, kStateTol);
    run_case(backend, "chain-model-shape",
             make_inputs(128, 8, 16, chain_parents(16), 2), false, kOutTol, kStateTol);
    run_case(backend, "chain-n1",
             make_inputs(64, 2, 1, chain_parents(1), 3), false, kOutTol, kStateTol);
    run_case(backend, "chain-odd",
             make_inputs(64, 2, 33, chain_parents(33), 4), false, kOutTol, kStateTol);

    // Tree drafts vs the fused tree op.
    run_case(backend, "tree-chain-shaped",
             make_inputs(64, 4, 8, chain_parents(8), 5), true, kOutTol, kStateTol);
    {
        std::vector<int32_t> star(9, 0);
        star[0] = -1;
        run_case(backend, "tree-star",
                 make_inputs(64, 4, 9, star, 6), true, kOutTol, kStateTol);
    }
    run_case(backend, "tree-random-small",
             make_inputs(64, 4, 15, random_tree_parents(15, 42), 7), true, kOutTol, kStateTol);
    run_case(backend, "tree-model-shape",
             make_inputs(128, 8, 31, random_tree_parents(31, 43), 8), true, kOutTol, kStateTol);

    // Full qwen35-27B delta-net shape: S=128, H_v=48, 16-token verify window.
    run_case(backend, "chain-qwen35-27b",
             make_inputs(128, 48, 16, chain_parents(16), 9), false, kOutTol, kStateTol);
    run_case(backend, "tree-qwen35-27b",
             make_inputs(128, 48, 24, random_tree_parents(24, 44), 10), true, kOutTol, kStateTol);

    ggml_backend_free(backend);
    if (failures) {
        std::fprintf(stderr, "test_delta_net_specla: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_delta_net_specla: all cases passed\n");
    return 0;
}
