// Qwen35SeqEngine — SeqEngine implementation for the paged Qwen3.5/3.6
// backend (--max-concurrency N).
//
// Three layers, each with one job:
//   Qwen35SlotManager          host bookkeeping — pool-handle lifecycle,
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

#include "common/concurrency/seq_engine.h"
#include "qwen35_slot_manager.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

class Qwen35Backend;

class Qwen35SeqEngine final : public SeqEngine {
public:
    // `pool` and `backend` must outlive the engine. `scratch_row` is the
    // first row of the block appended past the pool's index space, used as
    // the K/V write destination of graph-bucket padding rows.
    // `max_prefills` controls how many prefills can advance per step()
    // (each uses its own staging set; must match the count allocated in
    // TargetCache).
    Qwen35SeqEngine(Qwen35Backend & backend, PagedKvPool & pool,
                    int max_ctx, int64_t scratch_row,
                    int max_prefills = 2)
        : max_prefills_(std::max(1, max_prefills)), b_(backend),
          slots_(pool, max_ctx), scratch_row_(scratch_row) {
        prefill_slots_.reserve((size_t)slots_.slot_count());
    }

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;

    StepResult step(const StepPlan & plan) override;
    StepPlanLimits step_plan_limits(int decode_rows) const override {
        const bool mixed = decode_rows > 0;
        const int per_sequence = mixed ? 512 : 2048;
        const int total_cap = mixed ? 2048 : 4096;
        return {
            max_prefills_,
            per_sequence,
            std::min(max_prefills_ * per_sequence, total_cap),
            512,
        };
    }

    void retire(int slot) override;

    bool token_is_eos(int32_t token) const override;


private:
    struct PrefillStage {
        bool ready = false;
        int kv_pos = 0;
        int chunk = 0;
        bool commit = false;
        int staging_idx = 0;
        std::vector<int64_t> rows;
        std::vector<float> embeddings;
    };

    int max_prefills_;

    bool upload_block_table_delta(int slot, int first_block,
                                  const int32_t * blocks, size_t count);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                              const char * log_message,
                              const char * client_message);
    PrefillStage stage_prefill_chunk(int slot, int max_tokens,
                                     int staging_idx,
                                     std::vector<PrefillOutput> & outputs);
    bool run_prefill_graph(const PrefillStage & prefill, int prefill_slot,
                          std::vector<PrefillOutput> & outputs);
    int32_t sample_graph_row(int slot, int logits_row,
                             const int32_t * cached_argmax = nullptr,
                             std::vector<float> * logits_scratch = nullptr);

    Qwen35Backend & b_;
    Qwen35SlotManager slots_;
    int64_t         scratch_row_ = 0;

    // Hoisted per-step buffers (reused across step() calls).
    std::vector<int>         prefill_slots_;
    std::vector<int>         prefill_token_limits_;
    std::vector<int>         output_rows_;
    std::vector<int32_t>     live_tokens_;
    std::vector<int32_t>     live_positions_;
    std::vector<int64_t>     live_physical_rows_;
    std::vector<int32_t>     live_slot_ids_;
    std::vector<int32_t>     dec_tokens_;
    std::vector<int32_t>     dec_pos_;
    std::vector<int64_t>     dec_rows_;
    std::vector<int32_t>     active_slot_ids_;
    std::vector<int32_t>     state_slot_ids_;
    std::vector<int32_t>     seq_lens_;
    std::vector<float>       embed_buf_;
    std::vector<int32_t>     pos_buf_;
    std::vector<int64_t>     rows_buf_;
    std::vector<int32_t>     argmax_buf_;
    std::vector<float>       logits_buf_;
};

}  // namespace dflash::common