// GGML_ROPE_TYPE_TAIL regression test.
//
// DeepSeek4 rotates the last n_rot dims of each head. The tail mode rotates
// them in place and passes the head through, replacing the previous split,
// rotate, concat idiom (two contiguity copies, one rope, one concat). This
// test builds both forms on the HIP backend and on the CPU backend and
// requires bit-identical results, for positive and negative positions and for
// the single-head 2D shape the compressor uses.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint32_t lcg_state = 0x12345678u;
float lcg_uniform() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float) ((lcg_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

struct RopeParams {
    int n_rot = 64;
    int n_ctx_orig = 4096;
    float freq_base = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
};

// Returns false on compute failure. Fills out with the rope result of shape
// [head_dim, n_heads, n_tokens], either through the old idiom or the tail mode.
bool run(ggml_backend_t backend, bool tail_mode, int head_dim, int n_heads, int n_tokens,
         const std::vector<float> & x, const std::vector<int32_t> & pos, const RopeParams & rp,
         std::vector<float> & out) {
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * xt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, n_tokens);
    ggml_tensor * pt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(xt);
    ggml_set_input(pt);
    ggml_tensor * y = nullptr;
    if (tail_mode) {
        y = ggml_rope_ext(ctx, xt, pt, nullptr, rp.n_rot, GGML_ROPE_TYPE_NORMAL | GGML_ROPE_TYPE_TAIL,
                          rp.n_ctx_orig, rp.freq_base, rp.freq_scale, rp.ext_factor, rp.attn_factor,
                          rp.beta_fast, rp.beta_slow);
    } else {
        const int n_nope = head_dim - rp.n_rot;
        ggml_tensor * nope = ggml_view_3d(ctx, xt, n_nope, n_heads, n_tokens, xt->nb[1], xt->nb[2], 0);
        ggml_tensor * tail = ggml_view_3d(ctx, xt, rp.n_rot, n_heads, n_tokens, xt->nb[1], xt->nb[2],
                                          (size_t) n_nope * sizeof(float));
        tail = ggml_cont(ctx, tail);
        tail = ggml_rope_ext(ctx, tail, pt, nullptr, rp.n_rot, GGML_ROPE_TYPE_NORMAL, rp.n_ctx_orig,
                             rp.freq_base, rp.freq_scale, rp.ext_factor, rp.attn_factor, rp.beta_fast,
                             rp.beta_slow);
        y = ggml_concat(ctx, ggml_cont(ctx, nope), tail, 0);
    }
    ggml_set_output(y);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(pt, pos.data(), 0, pos.size() * sizeof(int32_t));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if (ok) {
        out.resize((size_t) head_dim * n_heads * n_tokens);
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

size_t bit_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return a.size() + b.size();
    }
    size_t mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++mm;
        }
    }
    return mm;
}

bool check(ggml_backend_t backend, const char * label, int head_dim, int n_heads, int n_tokens,
           const std::vector<int32_t> & pos, const RopeParams & rp) {
    std::vector<float> x((size_t) head_dim * n_heads * n_tokens);
    lcg_state = 0x51a7u ^ (uint32_t) (head_dim * 7 + n_heads * 13 + n_tokens);
    for (float & v : x) {
        v = lcg_uniform() * 3.0f;
    }
    std::vector<float> old_out, tail_out;
    if (!run(backend, false, head_dim, n_heads, n_tokens, x, pos, rp, old_out) ||
        !run(backend, true, head_dim, n_heads, n_tokens, x, pos, rp, tail_out)) {
        std::printf("FAIL %s: compute failed\n", label);
        return false;
    }
    const size_t mm = bit_mismatches(old_out, tail_out);
    // The head must be a verbatim copy and the tail must differ from the input.
    size_t head_changed = 0, tail_same = 0;
    const int n_nope = head_dim - rp.n_rot;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const size_t base = ((size_t) t * n_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                const bool same = std::memcmp(&tail_out[base + d], &x[base + d], sizeof(float)) == 0;
                if (d < n_nope && !same) ++head_changed;
                if (d >= n_nope && same && pos[t] != 0) ++tail_same;
            }
        }
    }
    std::printf("%s %s: %zu values, %zu bit mismatches vs split-rotate-concat, head changed %zu, tail unrotated %zu\n",
                mm == 0 && head_changed == 0 ? "PASS" : "FAIL", label, tail_out.size(), mm, head_changed, tail_same);
    return mm == 0 && head_changed == 0;
}

} // namespace

int main() {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr, "failed to query HIP device 0\n");
        return 1;
    }
    ggml_backend_t hip = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!hip || !cpu) {
        std::fprintf(stderr, "failed to initialize backends\n");
        return 1;
    }
    const bool previous_graphs = ggml_backend_cuda_set_graphs_disabled_override(true);
    RopeParams rp;
    bool ok = true;
    const std::vector<int32_t> pos4 = {5, 100, 1000, 3500};
    const std::vector<int32_t> neg4 = {-5, -100, -1000, -3500};
    const std::vector<int32_t> pos1 = {127};
    for (auto backend : {hip, cpu}) {
        const char * name = backend == hip ? "hip" : "cpu";
        char label[96];
        std::snprintf(label, sizeof(label), "%s q[512,64,4]", name);
        ok = check(backend, label, 512, 64, 4, pos4, rp) && ok;
        std::snprintf(label, sizeof(label), "%s inverse[512,64,4]", name);
        ok = check(backend, label, 512, 64, 4, neg4, rp) && ok;
        std::snprintf(label, sizeof(label), "%s kv[512,1,1]", name);
        ok = check(backend, label, 512, 1, 1, pos1, rp) && ok;
        RopeParams yarn = rp;
        yarn.freq_scale = 0.25f;
        yarn.ext_factor = 1.0f;
        yarn.attn_factor = 1.1f;
        std::snprintf(label, sizeof(label), "%s yarn[512,8,4]", name);
        ok = check(backend, label, 512, 8, 4, pos4, yarn) && ok;
    }
    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    std::printf("%s\n", ok ? "PASS rope tail mode" : "FAIL rope tail mode");
    return ok ? 0 : 1;
}
