// oflash_adapter.h — OFlash LoRA adapter files and device slots (OFLASH.md §6.2).
//
// Three concerns, all engine-side:
//   1. Parse + validate a single-file safetensors adapter produced by the
//      trainer (tensor names/shapes per oflash_format.h, metadata keys
//      oflash.*). The engine refuses adapters whose drafter SHA-256 or rank
//      do not match the loaded drafter GGUF.
//   2. Preallocate the pointer-stable device tensors (OFlashLoraWeights) the
//      draft graphs reference, zero-filled so "no adapter yet" is bit-exact
//      with the base drafter.
//   3. The on-disk profile store <dir>/<hash16>/<profile>/ used for warm
//      start: the engine reads promoted.json to find the adapter to load at
//      init and rewrites it on promote so the next session warm-starts from
//      the last promoted generation.

#pragma once

#include "oflash_lora.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {
struct DraftWeights;  // internal.h
}

namespace dflash::common::oflash {

// Host-side staging of one adapter generation: every targeted tensor as F16
// in ggml element order, keyed by tensor name. Kept for the previous
// generation too, so guard rollback is a re-upload, not a file reload.
struct OFlashAdapterHost {
    uint64_t generation = 0;
    std::string path;                 // file it was loaded from ("" = zeros)
    std::unordered_map<std::string, std::vector<uint16_t>> tensors;
};

// Load + validate an adapter file against the loaded drafter.
// `drafter_sha256` is the full lowercase hex of the drafter GGUF.
// On success fills `out` (F16-converted, ggml element order).
// Returns false with a "[oflash] ..."-style reason in `error`.
bool oflash_adapter_load(const std::string & path,
                         const DraftWeights & dw,
                         int rank,
                         float alpha,
                         const std::string & drafter_sha256,
                         OFlashAdapterHost & out,
                         std::string & error);

// Allocate the zero-filled device tensors for every targeted projection.
// The returned object owns a dedicated ggml context + backend buffer; free
// with oflash_lora_free. Fails (nullptr) on allocation failure.
OFlashLoraWeights * oflash_lora_create(const DraftWeights & dw,
                                       ggml_backend_t backend,
                                       int rank,
                                       float alpha,
                                       std::string & error);

void oflash_lora_free(OFlashLoraWeights * lw);

// Overwrite the slot tensors' contents with `host` (or zeros when
// host.tensors is empty). Pointer-stable: never reallocates.
bool oflash_lora_upload(OFlashLoraWeights & lw,
                        const OFlashAdapterHost & host,
                        std::string & error);

// Expected safetensors names + ggml dims for a drafter, in a stable order.
// Exposed for validation, tests, and to keep the Python exporter honest.
struct OFlashLoraTensorSpec {
    std::string name;   // e.g. "blk.0.attn_q.lora_a"
    int64_t in_dim;     // ggml ne[0]
    int64_t out_dim;    // ggml ne[1]
};
std::vector<OFlashLoraTensorSpec> oflash_lora_expected_tensors(
    const DraftWeights & dw, int rank);

// ── Profile store ───────────────────────────────────────────────────

// <dir>/<first 16 hex of drafter sha>/<profile>; created on demand.
std::string oflash_profile_dir(const std::string & base_dir,
                               const std::string & drafter_sha256,
                               const std::string & profile);

// Read promoted.json → adapter path + generation. False when absent/corrupt.
bool oflash_store_read_promoted(const std::string & profile_dir,
                                std::string & adapter_path,
                                uint64_t & generation);

// Atomically rewrite promoted.json after a guard promote.
bool oflash_store_write_promoted(const std::string & profile_dir,
                                 const std::string & adapter_path,
                                 uint64_t generation);

}  // namespace dflash::common::oflash
