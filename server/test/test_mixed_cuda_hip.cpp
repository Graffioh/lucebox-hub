#include "common/dynamic_backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using dflash::common::PlacementBackend;
using dflash::common::backend_pair_capabilities;
using dflash::common::init_placement_backend;
using dflash::common::placement_backend_of;

namespace {

bool run_scale(ggml_backend_t backend, const char * label) {
    constexpr int64_t n = 4096;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(input);
    ggml_tensor * output = ggml_scale(ctx, input, 2.0f);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> host((size_t)n);
    for (int64_t i = 0; i < n; ++i) host[(size_t)i] = (float)i / 128.0f;
    ggml_backend_tensor_set(input, host.data(), 0, ggml_nbytes(input));
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> result((size_t)n);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    }

    bool ok = status == GGML_STATUS_SUCCESS;
    for (int64_t i = 0; ok && i < n; ++i) {
        ok = std::fabs(result[(size_t)i] - 2.0f * host[(size_t)i]) < 1.0e-5f;
    }
    std::printf("mixed-backend %s scale: %s\n", label, ok ? "ok" : "FAILED");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

bool run_cross_copy(ggml_backend_t src_backend,
                    ggml_backend_t dst_backend,
                    const char * label) {
    constexpr int64_t n = 4096;
    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * src_ctx = ggml_init(params);
    ggml_context * dst_ctx = ggml_init(params);
    if (!src_ctx || !dst_ctx) {
        if (src_ctx) ggml_free(src_ctx);
        if (dst_ctx) ggml_free(dst_ctx);
        return false;
    }

    ggml_tensor * src = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F32, n);
    ggml_tensor * dst = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, n);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors(src_ctx, src_backend);
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors(dst_ctx, dst_backend);
    if (!src_buf || !dst_buf) {
        if (src_buf) ggml_backend_buffer_free(src_buf);
        if (dst_buf) ggml_backend_buffer_free(dst_buf);
        ggml_free(src_ctx);
        ggml_free(dst_ctx);
        return false;
    }

    std::vector<float> input((size_t)n);
    std::vector<float> output((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; ++i) input[(size_t)i] = (float)(i * 17 - 31);
    ggml_backend_tensor_set(src, input.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_copy_async(src_backend, dst_backend, src, dst);
    ggml_backend_synchronize(dst_backend);
    ggml_backend_tensor_get(dst, output.data(), 0, ggml_nbytes(dst));
    const bool ok = input == output;
    std::printf("mixed-backend %s copy: %s\n", label, ok ? "ok" : "FAILED");

    ggml_backend_buffer_free(src_buf);
    ggml_backend_buffer_free(dst_buf);
    ggml_free(src_ctx);
    ggml_free(dst_ctx);
    return ok;
}

bool run_cross_graph(ggml_backend_t first,
                     ggml_backend_t second,
                     const char * label) {
    constexpr int64_t n = 4096;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) {
        if (cpu) ggml_backend_free(cpu);
        if (ctx) ggml_free(ctx);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(input);
    ggml_tensor * first_head = ggml_scale(ctx, input, 2.0f);
    ggml_tensor * second_middle = ggml_scale(ctx, first_head, 3.0f);
    ggml_tensor * first_tail = ggml_scale(ctx, second_middle, 4.0f);
    ggml_set_output(first_tail);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, first_tail);

    ggml_backend_t backends[] = { first, second, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 3, 64, false, true);
    bool ok = sched != nullptr;
    if (ok) {
        ggml_backend_sched_set_tensor_backend(sched, input, first);
        ggml_backend_sched_set_tensor_backend(sched, first_head, first);
        ggml_backend_sched_set_tensor_backend(sched, second_middle, second);
        ggml_backend_sched_set_tensor_backend(sched, first_tail, first);
        ok = ggml_backend_sched_alloc_graph(sched, graph);
    }

    std::vector<float> host((size_t)n);
    std::vector<float> result((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; ++i) host[(size_t)i] = (float)i / 128.0f;
    if (ok) {
        ggml_backend_tensor_set(input, host.data(), 0, ggml_nbytes(input));
        ok = ggml_backend_sched_graph_compute(sched, graph) ==
             GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ggml_backend_tensor_get(
            first_tail, result.data(), 0, ggml_nbytes(first_tail));
    }
    for (int64_t i = 0; ok && i < n; ++i) {
        ok = std::fabs(result[(size_t)i] - 24.0f * host[(size_t)i]) <
             1.0e-4f;
    }
    std::printf("mixed-backend %s graph: %s\n", label,
                ok ? "ok" : "FAILED");

    if (sched) ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    std::string error;
    ggml_backend_t cuda = init_placement_backend(PlacementBackend::Cuda, 0, &error);
    if (!cuda) {
        std::fprintf(stderr, "CUDA initialization failed: %s\n", error.c_str());
        return 1;
    }
    ggml_backend_t hip = init_placement_backend(PlacementBackend::Hip, 0, &error);
    if (!hip) {
        std::fprintf(stderr, "HIP initialization failed: %s\n", error.c_str());
        ggml_backend_free(cuda);
        return 1;
    }

    const auto pair = backend_pair_capabilities(cuda, hip);
    bool ok = placement_backend_of(cuda) == PlacementBackend::Cuda &&
              placement_backend_of(hip) == PlacementBackend::Hip &&
              !pair.same_runtime && !pair.native_gpu_handoff;
    ok = run_scale(cuda, "CUDA") && ok;
    ok = run_scale(hip, "HIP") && ok;
    ok = run_cross_copy(cuda, hip, "CUDA->HIP") && ok;
    ok = run_cross_copy(hip, cuda, "HIP->CUDA") && ok;
    ok = run_cross_graph(cuda, hip, "CUDA->HIP->CUDA") && ok;
    ok = run_cross_graph(hip, cuda, "HIP->CUDA->HIP") && ok;

    ggml_backend_free(hip);
    ggml_backend_free(cuda);
    return ok ? 0 : 1;
}
