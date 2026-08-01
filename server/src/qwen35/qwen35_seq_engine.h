// Qwen35SeqEngine — SeqEngine implementation for the paged Qwen3.5/3.6
// backend (--max-concurrency N).
//
// Three layers, each with one job:
//   SeqSlotManager          host bookkeeping — pool-handle lifecycle,
//                           admission arithmetic, per-slot sampler/RNG/
//                           penalty history, the kv-length mirror
//   Qwen35SeqEngine         the device half — chunked slot prefill, the
//                           batched decode forward, sampling, and the
//                           block-table / kv-length uploads
//   Qwen35Backend           the model — weights, cache, step graph, the
//                           paged pool, park/unpark, generate()
//
// The engine borrows the backend's GPU state rather than copying accessors
// for it: it is a friend of Qwen35Backend so that concurrent serving can be
// its own subsystem without widening the backend's public surface.
//
// Single-threaded by the SeqEngine contract — the HTTP scheduler thread is
// the only caller of the engine, the pool, and the device uploads, so there
// is no locking anywhere below here.

#pragma once

#include "common/seq_engine.h"
#include "common/seq_slot_manager.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace dflash::common {

class Qwen35Backend;

class Qwen35SeqEngine final : public SeqEngine {
public:
    // `pool` and `backend` must outlive the engine. `scratch_row` is the
    // first row of the block appended past the pool's index space, used as
    // the K/V write destination of graph-bucket padding rows.
    Qwen35SeqEngine(Qwen35Backend & backend, PagedKvPool & pool,
                    int max_ctx, int64_t scratch_row);

    int slot_count() const override { return slots_.slot_count(); }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler,
                      int n_gen,
                      const ResumeState * resume = nullptr) override;

    bool step(const std::vector<StepInput> & inputs,
              std::vector<StepOutput> & outputs) override;

    bool prefill_pending() const override;

    bool capture_resume_state(int slot, ResumeState & out) const override;

    void retire(int slot) override;

    bool token_is_eos(int32_t token) const override;

private:
    struct PendingPrefill {
        int slot = -1;
        std::vector<int32_t> prompt;
        int progress = 0;
        std::chrono::steady_clock::time_point admitted_at{};
    };

    std::vector<std::optional<PendingPrefill>> pending_prefills_;
    int prefill_cursor_ = 0;

    int32_t sample_prefill_first_token(int slot, int logits_row);

    Qwen35Backend & b_;
    SeqSlotManager  slots_;
    int64_t         scratch_row_ = 0;   // bucket-padding KV write destination
};

}  // namespace dflash::common
