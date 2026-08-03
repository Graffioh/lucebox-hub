// Qwen35SeqEngine — SeqEngine implementation for the paged Qwen3.5/3.6
// backend (--max-concurrency N).
//
// Three layers, each with one job:
//   SeqSlotManager          host bookkeeping — pool-handle lifecycle,
//                           admission arithmetic, per-slot sampler/RNG/
//                           penalty history, the position counters
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
                    int max_ctx, int64_t scratch_row)
        : pending_prefills_((size_t)pool.max_sequences()),
          b_(backend), slots_(pool, max_ctx),
          scratch_row_(scratch_row) {}

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;

    bool step(const std::vector<StepInput> & inputs,
              std::vector<StepOutput> & outputs) override;

    bool prefill_pending() const override;

    void retire(int slot) override;

    bool token_is_eos(int32_t token) const override;

private:
    struct PendingPrefill {
        std::vector<int32_t> prompt;
        int progress = 0;
    };

    // One prompt's share of this step's prefill budget, pool rows already
    // allocated. Selected prompts appear in scan order and their chunks
    // occupy the leading token axis in the same order.
    struct SelectedPrefill {
        int slot = -1;
        int kv_pos = 0;
        int chunk = 0;
        bool commit = false;
        std::vector<int64_t> rows;
    };

    // Indexed by slot. Independent prompts prefill concurrently under one
    // shared per-step token budget (select_prefill_chunks).
    std::vector<std::optional<PendingPrefill>> pending_prefills_;
    // Budget-scan rotation: the starting slot advances one per step so a
    // prompt behind others in slot order cannot be perpetually last.
    int prefill_cursor_ = 0;

    bool upload_block_table_delta(int slot, int first_block,
                                  const int32_t * blocks, size_t count);
    void fail_pending_prefill(int slot, std::vector<StepOutput> & outputs,
                              const char * log_message,
                              const char * client_message);
    std::vector<SelectedPrefill> select_prefill_chunks(
        std::vector<StepOutput> & outputs);
    int32_t sample_graph_row(int slot, int logits_row,
                             const int32_t * cached_argmax = nullptr,
                             std::vector<float> * logits_scratch = nullptr);

    Qwen35Backend & b_;
    SeqSlotManager  slots_;
    int64_t         scratch_row_ = 0;   // bucket-padding KV write destination
};

}  // namespace dflash::common
