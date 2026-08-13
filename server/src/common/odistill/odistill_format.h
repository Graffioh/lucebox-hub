// odistill_format.h — versioned wire/file formats shared between the engine
// (C++ writer) and the trainer sidecar (Python reader).
//
// Two contracts live here, both little-endian, fixed-width, packed:
//
//   1. The capture ring: a single-producer/single-consumer byte ring in a
//      POSIX shared-memory segment. The engine appends variable-size records
//      (never blocking; drop-on-full), the trainer consumes them.
//   2. The adapter file: a single-file safetensors with a JSON metadata
//      block. Only the metadata KEYS and tensor-name convention are defined
//      here; parsing lives in odistill_adapter.cpp / the Python exporter.
//
// The Python mirror of these structs is optimizations/odistill/src/odistill/
// ring_format.py. Any change here must bump ODISTILL_RING_VERSION and be
// mirrored there; test/test_odistill_unit.cpp writes a fixture ring that the
// Python test suite replays to keep the two sides honest.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace dflash::common::odistill {

// ── Ring header ─────────────────────────────────────────────────────
//
// Byte 0 of the shared segment. `head` is the total number of bytes ever
// written (monotonic, not wrapped); `tail` the total consumed. Both are
// 8-byte-aligned so aligned 64-bit loads/stores are atomic on x86-64 and
// aarch64; the C++ side uses std::atomic_ref semantics (release on head
// store, acquire on tail load), the Python side relies on aligned loads.
// A record at logical offset L lives at byte offset
//   data_offset + (L % capacity).
// Records never wrap mid-record: when the contiguous space to the end of
// the buffer is too small, the writer emits a PAD record covering exactly
// that space first.

// The magic changed with the rename so a legacy consumer cannot
// attach to a new ring accidentally. Keep the version bump explicit too.
inline constexpr uint32_t ODISTILL_RING_MAGIC   = 0x4F444931;  // "ODI1"
inline constexpr uint32_t ODISTILL_RING_VERSION = 3;

// Explicit byte offsets (no alignas tricks — Python mirrors these):
//   0   magic, 4 version, 8 capacity, 16 data_offset
//   64  head   (own cache line, written by engine only)
//   128 tail   (own cache line, written by trainer only)
//   192 dropped_records, 200 dropped_bytes
//   208 drafter_hash, 216 n_capture_layers, 220 hidden, 224 block_size,
//   228 topk, 232 vocab, 236 reserved, 240 target_hash,
//   248 drafter_semantics_hash
//   256 = sizeof (data_offset points here)
struct ODistillRingHeader {
    uint32_t magic;            // ODISTILL_RING_MAGIC
    uint32_t version;          // ODISTILL_RING_VERSION
    uint64_t capacity;         // data bytes (excludes this header)
    uint64_t data_offset;      // byte offset of data[0] from segment start
    uint8_t  pad0_[40];
    uint64_t head;             // logical bytes written (monotonic)
    uint8_t  pad1_[56];
    uint64_t tail;             // logical bytes consumed (monotonic)
    uint8_t  pad2_[56];
    uint64_t dropped_records;  // engine: full-ring drops
    uint64_t dropped_bytes;
    // Immutable stream facts, filled once by the engine before the first
    // record. The trainer validates them against the drafter it loaded.
    uint64_t drafter_hash;     // first 8 bytes of the drafter GGUF's SHA-256,
                               // big-endian (== uint64 of the first 16 hex
                               // chars); full hex lives in adapter metadata
    uint32_t n_capture_layers; // feature slices per row (5)
    uint32_t hidden;           // target hidden size (5120)
    uint32_t block_size;       // draft block q_len (16)
    uint32_t topk;             // K of target_topk rows (0 = not captured)
    uint32_t vocab;            // target vocab size (for id validation)
    uint32_t reserved_;
    uint64_t target_hash;      // first 8 bytes of target GGUF SHA-256
    uint64_t drafter_semantics_hash; // FNV-1a of resolved semantics string
};
static_assert(sizeof(ODistillRingHeader) == 256, "keep layout stable");
static_assert(offsetof(ODistillRingHeader, head) == 64, "layout");
static_assert(offsetof(ODistillRingHeader, tail) == 128, "layout");
static_assert(offsetof(ODistillRingHeader, dropped_records) == 192, "layout");
static_assert(offsetof(ODistillRingHeader, drafter_hash) == 208, "layout");
static_assert(offsetof(ODistillRingHeader, target_hash) == 240, "layout");
static_assert(offsetof(ODistillRingHeader, drafter_semantics_hash) == 248,
              "layout");

// ── Record framing ──────────────────────────────────────────────────
//
// Every record starts with ODistillRecordHeader. `size_bytes` counts the
// whole record including the header, and is always a multiple of 8.
// The writer publishes a record by storing head_new = head + size_bytes
// with release ordering AFTER the payload bytes are in place.

enum ODistillRecordType : uint32_t {
    ODISTILL_REC_PAD     = 0,  // filler to the end of the buffer; skip
    ODISTILL_REC_CONTEXT = 1,  // feature backfill rows, no labels
    ODISTILL_REC_STEP    = 2,  // one verify step: rows + labels + flags
    ODISTILL_REC_SEQ_END = 3,  // request finished; seq_id is retired
};
// A buffer-tail gap can be smaller than sizeof(ODistillRecordHeader); the PAD
// written there carries only its first 8 bytes (type + size_bytes), which is
// all a reader needs to skip it. Readers must special-case
// capacity - (tail % capacity) < sizeof(ODistillRecordHeader).

struct ODistillRecordHeader {
    uint32_t type;        // ODistillRecordType
    uint32_t size_bytes;  // total record size incl. this header, mult. of 8
    uint64_t seq_id;      // engine request counter (monotonic per process)
    // Position of the first feature row this record carries, in target
    // token coordinates. For SEQ_END: the final committed length.
    int32_t  pos;
    int32_t  n_rows;      // feature rows in this record (0 for SEQ_END)
    uint64_t t_mono_ns;   // engine CLOCK_MONOTONIC at capture
};
static_assert(sizeof(ODistillRecordHeader) == 32, "keep layout stable");

// CONTEXT payload, in order:
//   feat        bf16[n_rows][n_capture_layers * hidden]
// Rows are positions [pos, pos + n_rows).
//
// STEP payload, in order:
//   feat         bf16[n_rows][n_capture_layers * hidden]
//                Feature rows for the COMMITTED positions
//                [pos, pos + n_rows) — i.e. the rows that extend the
//                trainer's shadow feature window, exactly the rows the
//                engine itself syncs into the draft feature mirror.
//                (n_rows can exceed n_labels: e.g. a committed bonus token
//                carries features but no draft proposal.)
//   n_labels     i32               labeled positions in this record — the
//                                  full block for chain steps, the accepted
//                                  spine for tree steps
//   topk_k       i32               K for the trailing top-K arrays (0 = none;
//                                  when > 0 it equals header.topk)
//   draft_tok    i32[n_labels]     drafter proposal; [0] is the seed token
//   target_tok   i32[n_labels]     target argmax/sample AFTER consuming
//                                  draft_tok[0..i] — i.e. target_tok[i] is
//                                  the ground truth for draft position i+1
//                                  (drafter row i+1's training label is
//                                  target_tok[i]; row 0 is the known seed
//                                  and carries no training signal). The
//                                  same holds for target_topk_* rows.
//   accept_flags u8[n_labels]      1 = accepted, 0 = rejected
//   (pad to 8-byte multiple)
//   accept_len   i32               committed accepted length incl. the seed
//   bonus_tok    i32               target correction committed at the first
//                                  rejection, -1 when none was committed
//   target_topk_ids i32[n_labels][topk_k]
//   target_topk_lp  f32[n_labels][topk_k]    log-probs, same order
//
// STEP records with n_rows == 0 are legal (budget-clamped steps).

// ── Adapter file conventions ────────────────────────────────────────
//
// Single-file safetensors. Tensor names mirror the drafter GGUF:
//   dflash.fc.lora_a            [rank, n_capture_layers*hidden]
//   dflash.fc.lora_b            [hidden, rank]
//   blk.<i>.attn_q.lora_a       [rank, hidden]
//   blk.<i>.attn_q.lora_b       [q_dim, rank]
//   blk.<i>.attn_k.lora_a       [rank, hidden]
//   blk.<i>.attn_k.lora_b       [kv_dim, rank]
//   blk.<i>.attn_v.lora_a       [rank, hidden]
//   blk.<i>.attn_v.lora_b       [kv_dim, rank]
//   blk.<i>.attn_output.lora_a  [rank, q_dim]
//   blk.<i>.attn_output.lora_b  [hidden, rank]
//   blk.<i>.ffn_up.lora_a       [rank, hidden]
//   blk.<i>.ffn_up.lora_b       [intermediate, rank]
//   blk.<i>.ffn_down.lora_a     [rank, intermediate]
//   blk.<i>.ffn_down.lora_b     [hidden, rank]
// Shapes are given [rows, cols] row-major as safetensors stores them —
// identical to the torch lora_A.weight / lora_B.weight layout, so the
// fastest-varying axis is the projection input dim for A and rank for B
// (which is what the engine's ggml tensors expect in ne[0]).
// dtype: F32 or F16. The forward delta is y += (alpha/rank) * B @ (A @ x).
//
// Required __metadata__ keys (all values are strings, per safetensors):
//   odistill.format        "3"
//   odistill.drafter_sha256  full lowercase hex SHA-256 of the drafter GGUF
//   odistill.target_sha256    full lowercase hex SHA-256 of the target GGUF
//   odistill.drafter_semantics  resolved model contract, currently
//                              v1;rope=<f32-bits>;swa=<tokens>;
//                              pattern=<bits>;mask=<token-id>
//   odistill.rank          decimal, must equal the engine's --odistill-lora-rank
//   odistill.alpha         decimal float
//   odistill.generation    decimal, monotonically increasing per profile
//   odistill.profile       profile name the trainer ran under
// The engine refuses adapters whose draft/target identity, resolved semantics,
// profile, rank, or alpha mismatch. Format 3 intentionally rejects older adapters
// that did not distinguish runtime SWA/RoPE overrides.
// Model hashing reuses read_gguf_metadata(path, /*compute_sha256=*/true)
// (common/gguf_inspect.h), which sidecar-caches each digest next to its GGUF.

// Truncate a lowercase-hex SHA-256 string to the u64 used in the ring header
// and in profile directory names (first 16 hex chars, big-endian). Returns 0
// on malformed input — callers treat 0 as "no hash".
inline uint64_t odistill_hash_from_hex(const char * hex) {
    if (!hex || std::strlen(hex) < 16) return 0;
    uint64_t h = 0;
    for (int i = 0; i < 16; i++) {
        const char c = hex[i];
        uint64_t nib;
        if (c >= '0' && c <= '9')      nib = (uint64_t)(c - '0');
        else if (c >= 'a' && c <= 'f') nib = (uint64_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') nib = (uint64_t)(c - 'A' + 10);
        else return 0;
        h = (h << 4) | nib;
    }
    return h;
}

}  // namespace dflash::common::odistill
