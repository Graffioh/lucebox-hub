#pragma once

#include "common.cuh"

#if defined(GGML_HIPBLASLT)

bool ggml_cuda_hipblaslt_gemm_f16(
    ggml_backend_cuda_context & ctx,
    void *& opaque_context,
    int device,
    const half * a, const half * b, half * d,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldd,
    cudaStream_t stream);

void ggml_cuda_hipblaslt_destroy(void * opaque_context);

#endif
