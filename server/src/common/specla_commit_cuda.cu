// Fused SpecLA DeltaConstruct commit — see specla_commit_cuda.h.

#include "specla_commit_cuda.h"

#include <cuda_runtime.h>

namespace dflash::common {

namespace {

constexpr int kBlock = 256;
constexpr int kMaxAccept = 64;  // >= every verify window / ddtree budget in use

// Grid: x over the S_k*S_v state elements of one (head, layer) plane,
//       y over head-layer planes (hl = h + H*l).
// Factor element (·, h, l, t) lives at block index fo = h + H*(l + L*t) —
// identical for fk (stride S_k), fv (stride S_v), and fg (stride 1).
__global__ void specla_commit_kernel(float * const * ssm_ptrs,
                                     const float * __restrict__ fk,
                                     const float * __restrict__ fv,
                                     const float * __restrict__ fg,
                                     const int32_t * __restrict__ idx,
                                     int A, int S_k, int S_v, int H, int L) {
    const int hl = blockIdx.y;
    const int l  = hl / H;
    const int h  = hl % H;

    // Per-plane scalars shared by every thread: the accepted-end gate and the
    // per-token decay weights along the accepted path.
    __shared__ float s_w[kMaxAccept];
    __shared__ float s_gA_exp;
    __shared__ int   s_tok[kMaxAccept];
    if (threadIdx.x < A) {
        s_tok[threadIdx.x] = idx[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int tA = s_tok[A - 1];
        const float gA = fg[(size_t)h + (size_t)H * (l + (size_t)L * tA)];
        s_gA_exp = expf(gA);
        for (int t = 0; t < A; t++) {
            const size_t fo = (size_t)h + (size_t)H * (l + (size_t)L * s_tok[t]);
            s_w[t] = expf(gA - fg[fo]);
        }
    }
    __syncthreads();

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= S_k * S_v) return;
    const int i = e % S_k;   // state row (k-dim, ne0)
    const int c = e / S_k;   // state column

    float * s_plane = ssm_ptrs[l] + (size_t)h * S_k * S_v;
    float acc = s_gA_exp * s_plane[(size_t)c * S_k + i];
    for (int t = 0; t < A; t++) {
        const size_t fo = (size_t)h + (size_t)H * (l + (size_t)L * s_tok[t]);
        acc += s_w[t] * fk[(size_t)fo * S_k + i] * fv[(size_t)fo * S_v + c];
    }
    s_plane[(size_t)c * S_k + i] = acc;
}

}  // namespace

bool specla_commit_fused(float * const * ssm_ptrs_dev,
                         const float * fk,
                         const float * fv,
                         const float * fg,
                         const int32_t * idx_dev,
                         int A, int S_k, int S_v, int H, int n_delta,
                         void * stream) {
    if (!ssm_ptrs_dev || !fk || !fv || !fg || !idx_dev) return false;
    if (A <= 0 || A > kMaxAccept || S_k <= 0 || S_v <= 0 || H <= 0 || n_delta <= 0) {
        return false;
    }
    const int planes = H * n_delta;
    if (planes > 65535) return false;  // grid.y limit

    dim3 block(kBlock);
    dim3 grid(((unsigned)(S_k * S_v) + kBlock - 1) / kBlock, (unsigned)planes);
    specla_commit_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        ssm_ptrs_dev, fk, fv, fg, idx_dev, A, S_k, S_v, H, n_delta);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess;
}

}  // namespace dflash::common
