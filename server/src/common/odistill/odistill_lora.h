// odistill_lora.h — LoRA adapter tensors the draft graph builders consume.
//
// DraftWeights carries `const ODistillLoraWeights * odistill` (nullptr when
// --odistill is off, in which case every graph is built without LoRA nodes —
// zero overhead for existing users). When present, each targeted projection
// W gains y += scale * B @ (A @ x), scale = alpha / rank.
//
// Swap discipline (ODISTILL.md §7 + CUDA/HIP-graph constraint): the tensors
// here are preallocated once at init, zero-filled (delta == 0 is bit-exact
// with the base drafter), and adapter loads OVERWRITE THEIR CONTENTS at a
// draft-block boundary. Their addresses never change, so once-built graphs
// (the KV-cached drafter path, CUDA-graph replay) stay valid; the drafter's
// ctx-KV ring must still be reset after a swap because cached rows embed the
// previous wk/wv delta.
//
// Targeted tensors (ODISTILL.md §5): per layer wq, wk, wv, wo, w_up, w_down,
// plus the feature-fusion fc. w_gate and the (target-owned) LM head are
// excluded. ggml shapes follow the loader convention ne[0]=in_features:
//   A: [in_features, rank]   B: [rank, out_features]

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common::odistill {

struct ODistillLoraPair {
    ggml_tensor * a = nullptr;  // [in, rank]
    ggml_tensor * b = nullptr;  // [rank, out]
};

struct ODistillLoraLayer {
    ODistillLoraPair wq, wk, wv, wo, w_up, w_down;
};

struct ODistillLoraWeights {
    ggml_context *        ctx = nullptr;  // owns tensor metadata
    ggml_backend_buffer_t buf = nullptr;  // owns device bytes (persistent,
                                          // never gallocr-recycled)
    int   rank  = 0;
    float alpha = 0.0f;
    float scale = 0.0f;                   // alpha / rank
    ODistillLoraPair fc;
    std::vector<ODistillLoraLayer> layers;  // size = drafter n_layer
    // Generation currently resident in the tensors (0 = zero-filled base).
    uint64_t generation = 0;
};

// y = W@x [+ scale * B@(A@x)]. The one helper every builder call site uses;
// defined in draft_graph.cpp so graph construction stays in one TU.
ggml_tensor * odistill_lora_mm(ggml_context * ctx,
                             ggml_tensor * w,
                             ggml_tensor * x,
                             const ODistillLoraPair * pair,
                             float scale);

}  // namespace dflash::common::odistill
