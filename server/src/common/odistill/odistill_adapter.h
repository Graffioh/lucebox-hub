// odistill_adapter.h — ODistill LoRA adapter files and device slots (ODISTILL.md §6.2).
//
// Three concerns, all engine-side:
//   1. Parse + validate a single-file safetensors adapter produced by the
//      trainer (tensor names/shapes per odistill_format.h, metadata keys
//      odistill.*). The engine refuses adapters whose draft or target SHA-256,
//      resolved RoPE/SWA/mask semantics, rank, or alpha do not match the
//      loaded graph.
//   2. Preallocate the pointer-stable device tensors (ODistillLoraWeights) the
//      draft graphs reference, zero-filled so "no adapter yet" is bit-exact
//      with the base drafter.
//   3. The on-disk profile store
//      <dir>/<hash16>/<profile>-sem-<contract-hash>/ used for warm
//      start: the engine reads promoted.json to find the adapter to load at
//      init and rewrites it on promote so the next session warm-starts from
//      the last promoted generation.

#pragma once

#include "odistill_lora.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {
struct DraftWeights;  // internal.h
}

namespace dflash::common::odistill {

// Host-side staging of one adapter generation: every targeted tensor as F16
// in ggml element order, keyed by tensor name. Kept for the previous
// generation too, so guard rollback is a re-upload, not a file reload.
struct ODistillAdapterHost {
    uint64_t generation = 0;
    std::string path;                 // file it was loaded from ("" = zeros)
    std::unordered_map<std::string, std::vector<uint16_t>> tensors;
};

// Load + validate an adapter file against the loaded drafter.
// Both model hashes are full lowercase SHA-256 digests;
// `drafter_semantics` identifies post-load RoPE/SWA overrides and `profile`
// preserves workload-store isolation.
// On success fills `out` (F16-converted, ggml element order).
// Returns false with a "[odistill] ..."-style reason in `error`.
bool odistill_adapter_load(const std::string & path,
                         const DraftWeights & dw,
                         int rank,
                         float alpha,
                         const std::string & drafter_sha256,
                         const std::string & target_sha256,
                         const std::string & drafter_semantics,
                         const std::string & profile,
                         ODistillAdapterHost & out,
                         std::string & error);

// Allocate the zero-filled device tensors for every targeted projection.
// The returned object owns a dedicated ggml context + backend buffer; free
// with odistill_lora_free. Fails (nullptr) on allocation failure.
ODistillLoraWeights * odistill_lora_create(const DraftWeights & dw,
                                       ggml_backend_t backend,
                                       int rank,
                                       float alpha,
                                       std::string & error);

void odistill_lora_free(ODistillLoraWeights * lw);

// Overwrite the slot tensors' contents with `host` (or zeros when
// host.tensors is empty). Pointer-stable: never reallocates.
bool odistill_lora_upload(ODistillLoraWeights & lw,
                        const ODistillAdapterHost & host,
                        std::string & error);

// Expected safetensors names + ggml dims for a drafter, in a stable order.
// Exposed for validation, tests, and to keep the Python exporter honest.
struct ODistillLoraTensorSpec {
    std::string name;   // e.g. "blk.0.attn_q.lora_a"
    int64_t in_dim;     // ggml ne[0]
    int64_t out_dim;    // ggml ne[1]
};
std::vector<ODistillLoraTensorSpec> odistill_lora_expected_tensors(
    const DraftWeights & dw, int rank);

// ── Profile store ───────────────────────────────────────────────────

// <dir>/<first 16 hex of drafter sha>/<semantic profile>; created on demand.
std::string odistill_profile_dir(const std::string & base_dir,
                               const std::string & drafter_sha256,
                               const std::string & profile);

// Read promoted.json → adapter path + generation. False when absent/corrupt.
bool odistill_store_read_promoted(const std::string & profile_dir,
                                std::string & adapter_path,
                                uint64_t & generation);

// Atomically rewrite promoted.json after a guard promote.
bool odistill_store_write_promoted(const std::string & profile_dir,
                                 const std::string & adapter_path,
                                 uint64_t generation);

}  // namespace dflash::common::odistill
