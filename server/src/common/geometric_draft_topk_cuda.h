// GPU top-K + log-prob extraction for DDTree draft distributions.
//
// Drop-in device-side replacement for extract_draft_topk (ddtree.cpp): instead
// of D2H-ing the full [vocab × n_positions] draft logits and running an OpenMP
// heap top-K + online-logsumexp on the CPU, this runs the whole thing on the
// GPU directly on the logits tensor's device buffer and copies back only the
// [n_positions × K] results.
//
// Semantics match extract_draft_topk exactly:
//   scaled        = logit * (1 / max(1e-3, temperature))
//   log_z         = logsumexp_j(scaled_j)                       (per position)
//   out_log_probs = top-K scaled logits (desc), minus log_z
//   out_token_ids = matching vocab ids; ties broken toward the lower id
//
// Returns false (caller must fall back to the CPU path) when the GPU runtime is
// unavailable, the pointer is not device memory, K is out of range, or any
// device call fails. Compiled on both CUDA and HIP/ROCm builds — this same .cu
// is compiled directly with LANGUAGE HIP on ROCm, the cuda_runtime.h spellings
// mapped by the hip_compat shim; guarded by LUCE_HAVE_DRAFT_TOPK. See
// CMakeLists.txt.

#pragma once

#include <cstdint>

namespace luce::common {

inline constexpr bool geometric_draft_topk_cuda_supports_k(int K) noexcept {
    return (K >= 1 && K <= 8) || K == 12 || K == 16;
}

// d_logits: device pointer to row-major [n_positions][vocab] f32 logits (the
//           position stride is `vocab` floats — pass an offset pointer to skip
//           leading positions). out_* are HOST buffers of size n_positions*K.
bool geometric_extract_draft_topk_cuda(const void * d_logits,
                             int n_positions, int vocab, int K,
                             float * out_log_probs,
                             int32_t * out_token_ids,
                             float temperature);

// Sampled-verify candidates: apply sample_logits' repetition / frequency /
// presence penalties IN PLACE to the device logits (one entry per penalized
// (row, id)), then copy back each row's K largest raw logits, descending,
// ties toward the lower id. With K >= the sampler's top_k this is the
// candidate set sample_logits keeps after its top_k cut.
struct GeometricPenalty {
    int32_t row;
    int32_t id;
    int32_t repeat;   // in the repetition-penalty window
    int32_t count;    // occurrences for the frequency/presence penalty (0 = none)
};

inline constexpr bool geometric_topk_logits_cuda_supports_k(int K) noexcept {
    return K == 8 || K == 16 || K == 20 || K == 24 || K == 32;
}

bool geometric_extract_topk_logits_cuda(void * d_logits, int n_positions, int vocab, int K,
                                        const GeometricPenalty * penalties, int n_penalties,
                                        float rep_pen, float freq_pen, float pres_pen,
                                        float * out_logits, int32_t * out_token_ids);

}  // namespace luce::common
