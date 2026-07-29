// SeqSlotManager — host-side bookkeeping for concurrent decode slots.
//
// Companion of PagedKvPool: the pool hands out sequence handles and physical
// blocks; this class owns everything else a slot needs between admission and
// retirement — the pool-handle lifecycle (including every error path), the
// admission arithmetic (context clamp and prompt-capacity checks), on-demand
// block allocation, per-slot sampler/RNG/penalty-history state, the position
// counters, and the host mirror of the device kv-length vector.
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
    // Admitted but not fully prefilled. Its device length remains zero and
    // it is excluded from the decode batch until commit_prefill().
    bool prefilling = false;
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
        // Requested output cap clamped to this sequence's logical context.
        // The first generated token comes from the prefill logits without a
        // K/V row, so a max_ctx-sized prompt can still produce one token.
        int n_gen_cap = 0;
        std::string error;
    };

    // Claim a free slot without allocating K/V blocks. Admission gates only
    // on the known prompt size: prompts larger than the whole pool hard-fail,
    // while prompts that fit the pool but not its current free blocks report
    // busy. `n_gen` is validated and clamped for the scheduler, but never
    // sizes memory. Seeds the slot RNG from sampler.seed only when the sampler
    // actually draws, else nondeterministically; the sampler config alone
    // decides.
    AdmitOutcome admit(uint64_t request_id, int prompt_len, int n_gen,
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

    // Allocate `n_tokens` more prompt rows for a prefilling slot. Allocation
    // follows the chunk being scheduled rather than the prompt's total size.
    PrefillChunk append_prefill(int slot, int n_tokens);

    // Record a finished prefill, expose the slot to decode, and set its
    // committed position / kv-length mirror.
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
    // entry, and log it to sample_history. Pushes the kv-length mirror to
    // position + 1 right away — the row is written and attended in the same
    // step — while cur_pos waits for commit_step().
    StepAppend append_token(int slot, int32_t fed_token);

    // The batched step's compute succeeded: cur_pos++.
    void commit_step(int slot);

    // Release the slot's blocks and clear its state. Safe on inactive slots
    // and after a failed admission/prefill. Zeroes the kv-length mirror; the
    // caller owns pushing that zero to the device tensor.
    void retire(int slot);

    int slot_count() const { return (int)slots_.size(); }
    int active_count() const;
    int decoding_count() const;
    bool is_active(int slot) const;
    bool is_prefilling(int slot) const;
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
