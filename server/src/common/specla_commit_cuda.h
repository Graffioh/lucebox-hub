// Fused SpecLA DeltaConstruct commit (docs/SPECLA.md).
//
// Advances EVERY delta-net layer's recurrent state along the accepted path in
// one kernel launch:
//
//   S_l ← exp(g⁺_A) · S_l + Σ_{t ∈ path} exp(g⁺_A − g⁺_t) · k_t ⊗ ṽ_t
//
// The ggml-graph implementation of the same math needs ~6 small ops per layer
// (48 layers ⇒ hundreds of kernel launches ⇒ ~5 ms of launch overhead per
// commit on ROCm without HIP graphs). This single launch reads the
// consolidated factor buffers and updates all per-layer states in place.
//
// Compiled for CUDA and (via the hip_compat <cuda_runtime.h> shim with
// LANGUAGE HIP) for ROCm — same shared-.cu pattern as
// geometric_draft_topk_cuda.cu. Returns false on any device error; the caller
// falls back to the ggml-graph commit.

#pragma once

#include <cstdint>

namespace dflash::common {

// ssm_ptrs_dev: DEVICE array of n_delta pointers, one per delta layer's
//               [S_v, S_v, H] f32 state tensor (ne0 = k-dim rows).
// fk/fv/fg:     consolidated factor buffers, f32, token axis outermost:
//               fk [S_k, H, n_delta, T], fv [S_v, H, n_delta, T],
//               fg [H, n_delta, T].
// idx_dev:      DEVICE array of A accepted token indices, deepest last.
// stream:       cudaStream_t / hipStream_t (nullptr = default stream).
bool specla_commit_fused(float * const * ssm_ptrs_dev,
                         const float * fk,
                         const float * fv,
                         const float * fg,
                         const int32_t * idx_dev,
                         int A, int S_k, int S_v, int H, int n_delta,
                         void * stream);

}  // namespace dflash::common
