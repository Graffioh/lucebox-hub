#include "hipblaslt.cuh"

#if defined(GGML_HIPBLASLT)

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "off") != 0;
}

uint64_t workspace_limit() {
    static const uint64_t limit = [] {
        constexpr uint64_t default_mb = 32;
        const char * value = std::getenv("DFLASH_HIPBLASLT_WORKSPACE_MB");
        if (!value || value[0] == '\0') return default_mb << 20;
        char * end = nullptr;
        const unsigned long long mb = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0') return default_mb << 20;
        return std::min<uint64_t>(mb, 512) << 20;
    }();
    return limit;
}

int heuristic_count() {
    static const int count = [] {
        const char * value = std::getenv("DFLASH_HIPBLASLT_HEURISTICS");
        if (!value || value[0] == '\0') return 8;
        return std::clamp(std::atoi(value), 1, 32);
    }();
    return count;
}

struct GemmKey {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldd;

    bool operator==(const GemmKey & other) const {
        return m == other.m && n == other.n && k == other.k &&
            lda == other.lda && ldb == other.ldb && ldd == other.ldd;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey & key) const {
        size_t h = 0xcbf29ce484222325ULL;
        const int64_t values[] = {
            key.m, key.n, key.k, key.lda, key.ldb, key.ldd,
        };
        for (int64_t value : values) {
            h ^= std::hash<int64_t>{}(value);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

struct GemmPlan {
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t a_layout = nullptr;
    hipblasLtMatrixLayout_t b_layout = nullptr;
    hipblasLtMatrixLayout_t c_layout = nullptr;
    hipblasLtMatrixLayout_t d_layout = nullptr;
    hipblasLtMatmulAlgo_t algorithm{};
    size_t workspace_size = 0;
    bool supported = false;

    ~GemmPlan() {
        if (d_layout) hipblasLtMatrixLayoutDestroy(d_layout);
        if (c_layout) hipblasLtMatrixLayoutDestroy(c_layout);
        if (b_layout) hipblasLtMatrixLayoutDestroy(b_layout);
        if (a_layout) hipblasLtMatrixLayoutDestroy(a_layout);
        if (operation) hipblasLtMatmulDescDestroy(operation);
    }
};

struct HipblasltContext {
    hipblasLtHandle_t handle = nullptr;
    std::unordered_map<GemmKey, std::unique_ptr<GemmPlan>, GemmKeyHash> plans;

    ~HipblasltContext() {
        plans.clear();
        if (handle) hipblasLtDestroy(handle);
    }
};

bool check(hipblasStatus_t status) {
    return status == HIPBLAS_STATUS_SUCCESS;
}

std::unique_ptr<GemmPlan> make_plan(
        hipblasLtHandle_t handle, const GemmKey & key) {
    auto plan = std::make_unique<GemmPlan>();
    if (!check(hipblasLtMatmulDescCreate(
            &plan->operation, HIPBLAS_COMPUTE_16F, HIP_R_16F))) {
        return plan;
    }

    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    const hipblasOperation_t trans_b = HIPBLAS_OP_N;
    if (!check(hipblasLtMatmulDescSetAttribute(
            plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA,
            &trans_a, sizeof(trans_a))) ||
        !check(hipblasLtMatmulDescSetAttribute(
            plan->operation, HIPBLASLT_MATMUL_DESC_TRANSB,
            &trans_b, sizeof(trans_b))) ||
        !check(hipblasLtMatrixLayoutCreate(
            &plan->a_layout, HIP_R_16F, key.k, key.m, key.lda)) ||
        !check(hipblasLtMatrixLayoutCreate(
            &plan->b_layout, HIP_R_16F, key.k, key.n, key.ldb)) ||
        !check(hipblasLtMatrixLayoutCreate(
            &plan->c_layout, HIP_R_16F, key.m, key.n, key.ldd)) ||
        !check(hipblasLtMatrixLayoutCreate(
            &plan->d_layout, HIP_R_16F, key.m, key.n, key.ldd))) {
        return plan;
    }

    hipblasLtMatmulPreference_t preference = nullptr;
    if (!check(hipblasLtMatmulPreferenceCreate(&preference))) return plan;
    const uint64_t max_workspace = workspace_limit();
    const hipblasStatus_t preference_status =
        hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace, sizeof(max_workspace));
    if (!check(preference_status)) {
        hipblasLtMatmulPreferenceDestroy(preference);
        return plan;
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> candidates(
        (size_t)heuristic_count());
    int returned = 0;
    const hipblasStatus_t heuristic_status =
        hipblasLtMatmulAlgoGetHeuristic(
            handle, plan->operation,
            plan->a_layout, plan->b_layout,
            plan->c_layout, plan->d_layout,
            preference, (int)candidates.size(), candidates.data(), &returned);
    hipblasLtMatmulPreferenceDestroy(preference);
    if (!check(heuristic_status)) return plan;

    for (int i = 0; i < returned; ++i) {
        if (!check(candidates[(size_t)i].state) ||
            candidates[(size_t)i].workspaceSize > max_workspace) {
            continue;
        }
        plan->algorithm = candidates[(size_t)i].algo;
        plan->workspace_size = candidates[(size_t)i].workspaceSize;
        plan->supported = true;
        if (env_enabled("DFLASH_HIPBLASLT_LOG")) {
            GGML_LOG_INFO(
                "hipBLASLt: selected shape m=%lld n=%lld k=%lld "
                "workspace=%zu waves=%.2f candidate=%d/%d\n",
                (long long)key.m, (long long)key.n, (long long)key.k,
                plan->workspace_size, candidates[(size_t)i].wavesCount,
                i, returned);
        }
        break;
    }
    return plan;
}

} // namespace

bool ggml_cuda_hipblaslt_gemm_f16(
        ggml_backend_cuda_context & ctx,
        void *& opaque_context,
        int device,
        const half * a, const half * b, half * d,
        int64_t m, int64_t n, int64_t k,
        int64_t lda, int64_t ldb, int64_t ldd,
        cudaStream_t stream) {
    if (!env_enabled("DFLASH_HIPBLASLT")) return false;

    auto * state = static_cast<HipblasltContext *>(opaque_context);
    if (!state) {
        ggml_cuda_set_device(device);
        auto created = std::make_unique<HipblasltContext>();
        if (!check(hipblasLtCreate(&created->handle))) return false;
        state = created.release();
        opaque_context = state;
    }

    const GemmKey key{m, n, k, lda, ldb, ldd};
    auto it = state->plans.find(key);
    if (it == state->plans.end()) {
        it = state->plans.emplace(
            key, make_plan(state->handle, key)).first;
    }
    GemmPlan & plan = *it->second;
    if (!plan.supported) return false;

    ggml_cuda_pool_alloc<char> workspace(ctx.pool(device));
    if (plan.workspace_size > 0) workspace.alloc(plan.workspace_size);

    const half alpha = 1.0f;
    const half beta = 0.0f;
    const hipblasStatus_t status = hipblasLtMatmul(
        state->handle, plan.operation,
        &alpha, a, plan.a_layout,
        b, plan.b_layout,
        &beta, d, plan.c_layout,
        d, plan.d_layout,
        &plan.algorithm, workspace.get(), plan.workspace_size, stream);
    if (!check(status)) {
        plan.supported = false;
        if (env_enabled("DFLASH_HIPBLASLT_LOG")) {
            GGML_LOG_WARN(
                "hipBLASLt: shape m=%lld n=%lld k=%lld failed (%d); "
                "falling back to hipBLAS\n",
                (long long)m, (long long)n, (long long)k, (int)status);
        }
        return false;
    }
    return true;
}

void ggml_cuda_hipblaslt_destroy(void * opaque_context) {
    delete static_cast<HipblasltContext *>(opaque_context);
}

#endif
