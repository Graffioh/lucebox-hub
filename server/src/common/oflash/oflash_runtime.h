// oflash_runtime.h — the engine-side OFlash orchestrator.
//
// Owned by the (qwen35) backend when --oflash is on. Composes:
//   OFlashRing        capture stream to the trainer (shm, drop-on-full)
//   OFlashLoraWeights pointer-stable adapter tensors wired into DraftWeights
//   OFlashGuard       acceptance guard state machine
//   OFlashSupervisor  trainer sidecar lifecycle + control lines
//   profile store     warm start + promote persistence
//
// Threading: every method except props() is called from the single decode
// thread. props() is called by HTTP threads; a small mutex guards the
// counters it snapshots. The supervisor's own thread never touches ggml.

#pragma once

#include "oflash_config.h"
#include "oflash_guard.h"
#include "oflash_lora.h"
#include "oflash_props.h"
#include "oflash_ring.h"
#include "oflash_supervisor.h"

#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace dflash::common {
struct DraftWeights;
}

namespace dflash::common::oflash {

// Adapter generations are strictly monotonic within one runtime. A queued
// candidate must advance the resident generation, any newer candidate already
// waiting locally, and the announcement high-water mark retained across guard
// rollbacks. Kept pure so refusal behavior has a CPU-only test.
inline constexpr bool oflash_generation_is_newer(uint64_t candidate,
                                                 uint64_t resident,
                                                 uint64_t pending = 0,
                                                 uint64_t high_water = 0) {
    return candidate > resident &&
           (pending == 0 || candidate > pending) &&
           (high_water == 0 || candidate > high_water);
}

// pimpl holder so this header need not include oflash_adapter.h.
struct OFlashAdapterHostHolder;

// One verify step's labels, assembled by the decode loop. Feature rows are
// pulled through the reader so the (device-aware) copy stays in the backend.
struct OFlashStepCapture {
    int pos    = 0;               // committed base position of the step
    int n_rows = 0;               // committed feature rows to capture
    int block  = 0;               // entries in the per-position arrays
    const int32_t * draft_tok  = nullptr;   // [block]
    const int32_t * target_tok = nullptr;   // [block]
    const uint8_t * accept_flags = nullptr; // [block] 1/0/2 per oflash_format.h
    int32_t accept_len = 0;       // guard sample (committed accepted length)
    int32_t bonus_tok  = -1;
    const float   * topk_lp  = nullptr;     // [block * topk] or null
    const int32_t * topk_ids = nullptr;
};

class OFlashRuntime {
public:
    // Copies one bf16 feature row for absolute position `pos` into `dst`
    // (n_capture_layers * hidden uint16 elements). Returns false to abort
    // the record (record is then dropped, serving unaffected).
    using FeatReader = std::function<bool(int pos, uint16_t * dst)>;

    OFlashRuntime() = default;
    ~OFlashRuntime() { shutdown(); }
    OFlashRuntime(const OFlashRuntime &) = delete;
    OFlashRuntime & operator=(const OFlashRuntime &) = delete;

    // Hashes the drafter, creates ring + LoRA slots (wired into dw.oflash),
    // warm-starts from the profile store, spawns the trainer. Any failure
    // logs and leaves the affected sub-feature off. Returns false for an
    // incompatible capture shape or failed LoRA slot allocation; the caller
    // then disables OFlash while leaving inference available.
    bool init(const OFlashConfig & cfg,
              const std::string & target_path,
              const std::string & drafter_path,
              DraftWeights & dw,
              ggml_backend_t draft_backend,
              int target_capture_layers,
              int target_hidden,
              int vocab);

    void shutdown();  // detach dw.oflash BEFORE freeing the draft backend

    // ── Capture (decode thread) ─────────────────────────────────────
    void on_request_begin();
    void on_request_end(int committed);
    bool capture_active() const { return ring_.active(); }
    int  topk() const { return cfg_.topk; }
    // True once per request: the prompt-window CONTEXT record is wanted.
    bool want_context() const { return want_context_; }
    void emit_context(int pos0, int n_rows, const FeatReader & read);
    void emit_step(const OFlashStepCapture & s, const FeatReader & read);

    // ── Adapter lifecycle (decode thread, block boundary) ───────────
    // Applies a pending trainer swap if the guard allows. After ANY tensor
    // change (swap or guard rollback inside emit_step), consume_adapter_dirty
    // returns true once — the caller must then reset the drafter's ctx-KV
    // ring (cached rows embed the old wk/wv delta).
    void maybe_apply_swap();
    bool consume_adapter_dirty() {
        const bool d = adapter_dirty_;
        adapter_dirty_ = false;
        return d;
    }

    // ── Stats (any thread) ──────────────────────────────────────────
    OFlashPropsSnapshot props() const;

private:
    void apply_guard_action(OFlashGuardAction action);
    bool load_and_upload(const std::string & path, uint64_t generation,
                         bool warm_start);

    OFlashConfig cfg_;
    std::string drafter_sha256_;
    std::string profile_dir_;
    std::string ring_name_;
    DraftWeights * dw_ = nullptr;

    OFlashRing ring_;
    OFlashGuard guard_{};
    OFlashSupervisor supervisor_;
    OFlashLoraWeights * lora_ = nullptr;

    // Host staging: current + previous generation for guard rollback.
    OFlashAdapterHostHolder * staging_ = nullptr;  // pimpl (adapter header)

    uint64_t seq_id_ = 0;
    bool want_context_ = false;
    bool adapter_dirty_ = false;
    bool training_disabled_ = false;

    // Swap deferred by the guard (probation/backoff) until can_swap().
    OFlashPendingSwap pending_local_;
    bool has_pending_local_ = false;
    // Never decreases on rollback: an old/rejected generation must not be
    // accepted again merely because the resident adapter moved backwards.
    uint64_t generation_high_water_ = 0;

    // Guard state is only touched by the decode thread; props() serves HTTP
    // threads from this mutex-guarded cache refreshed after every mutation.
    void refresh_props_cache();
    mutable std::mutex stats_mu_;
    OFlashPropsSnapshot props_cache_;
    uint64_t steps_ = 0;
};

}  // namespace dflash::common::oflash
