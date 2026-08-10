// SeqSlotManager — host-side bookkeeping for concurrent decode slots.
//
// Companion of PagedKvPool: the pool hands out sequence handles and physical
// blocks; this class owns everything else a slot needs between admission and
// retirement — the pool-handle lifecycle (including every error path), the
// admission arithmetic (context clamp, prompt reservation, and rolling decode
// headroom), on-demand block allocation, per-slot sampler/RNG/penalty-history
// state, and the position counters.
//
// It deliberately owns NO device state. Prefill/decode allocation returns
// physical rows and newly appended block-table entries as plain vectors; the
// backend decides how they reach the GPU. That split keeps this class GPU-free
// and unit-testable (test_seq_slot_manager), the same way PagedKvPool is.
//
// Not thread-safe; the single scheduler thread is the only caller.

#pragma once

#include "paged_kv_pool.h"
#include "sampler.h"
#include "seq_admission.h"

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
    // Admitted but not fully prefilled. It is excluded from the decode
    // batch until commit_prefill().
    bool prefilling = false;
    PagedKvSequenceHandle handle;
    int cur_pos = 0;                 // appended KV tokens
    int prompt_len = 0;              // admission-time reserved prefill extent
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

    // Claim a free slot and atomically reserve all K/V blocks needed by the
    // known prompt plus its next logical decode page when that page can exist
    // in both max_ctx and the physical pool. Existing decoders are topped up
    // first, so a younger admission cannot steal their next-page headroom.
    // Prompts larger than the whole pool hard-fail; temporary capacity pressure
    // reports busy. Seeds the slot RNG from sampler.seed only when the sampler
    // actually draws, else nondeterministically.
    SeqAdmissionResult admit(uint64_t request_id,
                             const std::vector<int32_t> & prompt,
                             const SamplerCfg & sampler);

    struct PrefillChunk {
        bool ok = false;
        // The pool is temporarily out of blocks; retrying after another slot
        // retires can succeed. BlocksExhausted leaves the pool unchanged.
        bool busy = false;
        std::vector<int64_t> rows;
        // Delta to patch into the slot's device block-table column.
        std::vector<int32_t> new_blocks;
        int first_new_block = -1;
    };

    // Append `n_tokens` more prompt rows for a prefilling slot. Physical block
    // ids come from the slot's admission reservation, so any append within the
    // admitted prompt is guaranteed not to wait on another sequence.
    PrefillChunk append_prefill(int slot, int n_tokens);

    // Record a finished prefill, expose the slot to decode, and set its
    // committed position.
    void commit_prefill(int slot, int committed);

    struct StepAppend {
        bool ok = false;
        bool busy = false;    // no physical block available right now
        int64_t physical_row = -1;
        int position = -1;   // logical position the fed token is written at
        int32_t new_block = -1;
        int new_block_index = -1;
    };

    // Allocate the next decode token's cache row, report any new block-table
    // entry, and log it to sample_history. cur_pos waits for commit_step().
    StepAppend append_token(int slot, int32_t fed_token);

    // The batched step's compute succeeded: cur_pos++.
    void commit_step(int slot);

    // Release the slot's blocks and clear its state. Safe on inactive slots
    // and after a failed admission/prefill.
    void retire(int slot);

    int slot_count() const { return (int)slots_.size(); }
    int max_context() const { return max_ctx_; }
    int decoding_count() const;
    bool is_active(int slot) const;
    bool is_prefilling(int slot) const;
    SeqSlot & slot(int i) { return slots_[(size_t)i]; }
    const SeqSlot & slot(int i) const { return slots_[(size_t)i]; }

private:
    // Logical extent whose block count includes the sequence's current pages
    // plus one future page, capped at max_ctx.
    uint32_t decode_headroom_capacity(int logical_tokens) const;
    bool capacity_fits_pool(uint32_t token_capacity) const;

    // Atomically preflight and top up every decoding slot as one cohort before
    // a younger sequence may reserve capacity.
    PagedKvStatus protect_decode_headroom();

    PagedKvPool & pool_;
    int max_ctx_ = 0;
    std::vector<SeqSlot> slots_;
};

}  // namespace dflash::common
