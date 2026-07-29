// SeqSlotManager — host-side bookkeeping for concurrent decode slots.
//
// Companion of PagedKvPool: the pool hands out sequence handles and physical
// blocks; this class owns everything else a slot needs between admission and
// retirement — the pool-handle lifecycle (including every error path), the
// admission arithmetic (context clamp, worst-case block reservation, the
// never-fits capacity check), per-slot sampler/RNG/penalty-history state, the
// position counters, and the host mirror of the device kv-length vector.
//
// It deliberately owns NO device state. Admission returns the physical rows
// and the block-table column as plain vectors; the backend decides how they
// reach the GPU. That split keeps this class GPU-free and unit-testable
// (test_seq_slot_manager), the same way PagedKvPool is.
//
// Not thread-safe; the single scheduler thread is the only caller.

#pragma once

#include "paged_kv_pool.h"
#include "sampler.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

// Per-slot request state below the SeqEngine interface. The scheduler holds
// the other half of each slot under the same slot id (SchedSlot in
// server/scheduler.cpp): KV stays here, sockets stay there.
struct SeqSlot {
    bool active = false;
    uint64_t request_id = 0;
    PagedKvSequenceHandle handle;
    int cur_pos = 0;                 // committed KV tokens
    SamplerCfg sampler;              // sole authority on how this slot samples
    std::mt19937_64 rng{0x9E3779B97F4A7C15ull};
    // Penalty history (sample_logits' `history`), recorded as FED rather than
    // as sampled — the scheduler may override a sample (thinking-budget
    // force-close), and penalties must see the stream the model consumed.
    std::vector<int32_t> sample_history;
};

class SeqSlotManager {
public:
    // `max_ctx` is the per-sequence logical bound; slot count comes from the
    // pool's max_sequences. The pool must outlive the manager.
    SeqSlotManager(PagedKvPool & pool, int max_ctx);

    struct AdmitOutcome {
        bool ok = false;
        // Full right now (no free slot / pool blocks held by live sequences)
        // — retrying after a retirement can succeed. Distinct from a hard
        // error (ok=false, busy=false), which can never succeed.
        bool busy = false;
        int slot = -1;
        std::string error;
        // Physical pool row of every prompt token, in logical order — the
        // prefill's set_rows destinations.
        std::vector<int64_t> prompt_rows;
        // The slot's complete block table: reserve() fixes it for the whole
        // lifetime, so one upload at admission covers every block.
        std::vector<int32_t> table_column;
    };

    // Claim a free slot and reserve its worst case (prompt + n_gen - 1 rows;
    // the first generated token comes from the prefill logits without a K/V
    // row), then append the prompt. All-or-nothing: on any failure the pool
    // is left unchanged. Requests too large for the pool hard-fail rather
    // than report busy — no drain could ever admit them. Seeds the slot RNG
    // from sampler.seed only when the sampler actually draws, else
    // nondeterministically; the sampler config alone decides.
    AdmitOutcome admit(uint64_t request_id, int prompt_len, int n_gen,
                       const SamplerCfg & sampler);

    // The prefill's compute succeeded: cur_pos and the kv-length mirror.
    void commit_prefill(int slot, int committed);

    struct StepAppend {
        bool ok = false;
        int64_t physical_row = -1;
        int position = -1;   // logical position the fed token is written at
    };

    // Allocate the next decode token's cache row and log it to sample_history.
    // Pushes the kv-length mirror to position + 1 right away — the row is
    // written and attended in the same step — while cur_pos waits for
    // commit_step(), so a step that fails to compute leaves nothing committed.
    StepAppend append_token(int slot, int32_t fed_token);

    // The batched step's compute succeeded: cur_pos++.
    void commit_step(int slot);

    // Release the slot's blocks and clear its state. Safe on inactive slots
    // and after a failed admission/prefill. Zeroes the kv-length mirror; the
    // caller owns pushing that zero to the device tensor.
    void retire(int slot);

    int slot_count() const { return (int)slots_.size(); }
    int active_count() const;
    bool is_active(int slot) const;
    SeqSlot & slot(int i) { return slots_[(size_t)i]; }
    const SeqSlot & slot(int i) const { return slots_[(size_t)i]; }

    // Host mirror of the device paged_kv_seq_lens tensor, indexed by slot.
    const std::vector<int32_t> & lens_host() const { return lens_host_; }

private:
    PagedKvPool & pool_;
    int max_ctx_ = 0;
    std::vector<SeqSlot> slots_;
    std::vector<int32_t> lens_host_;
};

}  // namespace dflash::common
