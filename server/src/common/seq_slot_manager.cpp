#include "common/seq_slot_manager.h"

#include "common/paged_attention_config.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

SeqSlotManager::SeqSlotManager(PagedKvPool & pool, int max_ctx)
    : pool_(pool), max_ctx_(max_ctx) {
    slots_.assign(pool.max_sequences(), SeqSlot{});
    lens_host_.assign(pool.max_sequences(), 0);
}

int SeqSlotManager::active_count() const {
    int n = 0;
    for (const SeqSlot & s : slots_) n += s.active ? 1 : 0;
    return n;
}

bool SeqSlotManager::is_active(int slot) const {
    return slot >= 0 && slot < (int)slots_.size() &&
           slots_[(size_t)slot].active;
}

SeqSlotManager::AdmitOutcome SeqSlotManager::admit(
        uint64_t request_id, int prompt_len, int n_gen,
        const SamplerCfg & sampler) {
    AdmitOutcome r;
    if (prompt_len < 1)          { r.error = "empty prompt"; return r; }
    if (n_gen < 1)               { r.error = "n_gen must be >= 1"; return r; }
    if (prompt_len > max_ctx_)   { r.error = "prompt exceeds max_ctx"; return r; }

    int slot = -1;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (!slots_[(size_t)i].active) { slot = i; break; }
    }
    if (slot < 0) {
        r.busy = true;
        r.error = "all decode slots are busy";
        return r;
    }

    // Same clamp as generate_impl: the first generated token comes from the
    // prefill logits without a K/V row, so ar_n_gen tokens need only
    // prompt + ar_n_gen - 1 cache rows.
    const int max_ar_n_gen = max_ctx_ - prompt_len + 1;
    const int ar_n_gen = std::min(n_gen, max_ar_n_gen);
    const uint32_t reserve_tokens = (uint32_t)(prompt_len + ar_n_gen - 1);

    // A request larger than the whole pool can NEVER be admitted; waiting
    // for other sequences to drain would stall the queue forever and then
    // fail anyway. Hard-fail it up front instead of reporting busy.
    const uint64_t pool_capacity =
        (uint64_t)pool_.physical_block_count() * pool_.block_size();
    if ((uint64_t)reserve_tokens > pool_capacity) {
        r.error = "request needs " + std::to_string(reserve_tokens) +
                  " KV tokens but the pool holds " +
                  std::to_string(pool_capacity) +
                  "; raise --kv-pool-tokens or lower max_tokens";
        return r;
    }

    PagedKvSequenceHandle handle;
    PagedKvStatus status = pool_.acquire(request_id, handle);
    if (status != PagedKvStatus::Ok) {
        r.busy = (status == PagedKvStatus::SequenceSlotsExhausted);
        r.error = paged_kv_status_string(status);
        return r;
    }
    // Reserve the whole worst case up front: admission is all-or-nothing and
    // decode-time appends can never hit BlocksExhausted. It also fixes the
    // sequence's entire block table NOW, so one table_column upload covers
    // every block the sequence will ever touch.
    status = pool_.reserve(handle, reserve_tokens);
    if (status != PagedKvStatus::Ok) {
        pool_.release(handle);
        // Waiting only helps if someone else holds blocks to free.
        r.busy = (status == PagedKvStatus::BlocksExhausted) &&
                 pool_.active_sequence_count() > 0;
        r.error = paged_kv_status_string(status);
        return r;
    }
    PagedKvAppendResult app = pool_.append(handle, (uint32_t)prompt_len);
    if (!app) {
        pool_.release(handle);
        r.error = paged_kv_status_string(app.status);
        return r;
    }

    PagedKvSequenceSnapshot snap;
    status = pool_.sequence(handle, snap);
    if (status != PagedKvStatus::Ok ||
        snap.block_table.size() > (size_t)paged_block_count(max_ctx_)) {
        pool_.release(handle);
        r.error = "block table snapshot failed";
        return r;
    }

    r.prompt_rows.resize((size_t)prompt_len);
    for (int i = 0; i < prompt_len; i++) {
        r.prompt_rows[(size_t)i] =
            (int64_t)app.write_slots[(size_t)i].physical_token_index;
    }
    r.table_column.resize(snap.block_table.size());
    for (size_t i = 0; i < snap.block_table.size(); i++) {
        r.table_column[i] = (int32_t)snap.block_table[i];
    }

    SeqSlot & s = slots_[(size_t)slot];
    s.active = true;
    s.request_id = request_id;
    s.handle = handle;
    s.cur_pos = 0;
    s.sampler = sampler;
    s.sample_history.clear();
    // Same predicate the engine uses to pick CPU sampling over GPU argmax:
    // a seed only means anything when the sampler actually draws.
    if (sampler.needs_logit_processing() && sampler.seed != 0) {
        s.rng.seed(sampler.seed);
    } else {
        s.rng.seed(std::random_device{}());
    }
    lens_host_[(size_t)slot] = 0;

    r.ok = true;
    r.slot = slot;
    return r;
}

void SeqSlotManager::commit_prefill(int slot, int committed) {
    if (!is_active(slot)) return;
    slots_[(size_t)slot].cur_pos = committed;
    lens_host_[(size_t)slot] = committed;
}

SeqSlotManager::StepAppend SeqSlotManager::append_token(int slot,
                                                        int32_t fed_token) {
    StepAppend out;
    if (!is_active(slot)) return out;
    SeqSlot & s = slots_[(size_t)slot];
    if (s.cur_pos >= max_ctx_) {
        // No context left; the scheduler should have stopped this slot.
        return out;
    }
    PagedKvAppendSpan app = pool_.append_compact(s.handle, 1);
    if (!app || app.token_count != 1 ||
        app.last.logical_position != (uint32_t)s.cur_pos) {
        std::fprintf(stderr, "[parallel] slot %d append failed: %s\n",
                     slot, paged_kv_status_string(app.status));
        return out;
    }
    s.sample_history.push_back(fed_token);
    lens_host_[(size_t)slot] = s.cur_pos + 1;
    out.ok = true;
    out.physical_row = (int64_t)app.last.physical_token_index;
    out.position = s.cur_pos;
    return out;
}

void SeqSlotManager::commit_step(int slot) {
    if (!is_active(slot)) return;
    slots_[(size_t)slot].cur_pos += 1;
}

void SeqSlotManager::retire(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    SeqSlot & s = slots_[(size_t)slot];
    if (!s.active) return;
    const PagedKvStatus status = pool_.release(s.handle);
    if (status != PagedKvStatus::Ok && status != PagedKvStatus::StaleHandle) {
        std::fprintf(stderr, "[parallel] slot %d release failed: %s\n",
                     slot, paged_kv_status_string(status));
    }
    s = SeqSlot{};
    lens_host_[(size_t)slot] = 0;
}

}  // namespace dflash::common
