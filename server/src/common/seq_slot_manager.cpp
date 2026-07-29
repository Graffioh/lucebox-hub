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

int SeqSlotManager::decoding_count() const {
    int n = 0;
    for (const SeqSlot & s : slots_) {
        n += (s.active && !s.prefilling) ? 1 : 0;
    }
    return n;
}

bool SeqSlotManager::is_active(int slot) const {
    return slot >= 0 && slot < (int)slots_.size() &&
           slots_[(size_t)slot].active;
}

bool SeqSlotManager::is_prefilling(int slot) const {
    return is_active(slot) && slots_[(size_t)slot].prefilling;
}

SeqSlotManager::AdmitOutcome SeqSlotManager::admit(
        uint64_t request_id, int prompt_len, int n_gen,
        const SamplerCfg & sampler) {
    AdmitOutcome r;
    if (prompt_len < 1)          { r.error = "empty prompt"; return r; }
    if (n_gen < 1)               { r.error = "n_gen must be >= 1"; return r; }
    if (prompt_len > max_ctx_)   { r.error = "prompt exceeds max_ctx"; return r; }

    // A prompt larger than the whole pool can NEVER be admitted; waiting
    // for other sequences to drain would stall the queue forever and then
    // fail anyway. Hard-fail it up front instead of reporting busy.
    const uint64_t pool_capacity =
        (uint64_t)pool_.physical_block_count() * pool_.block_size();
    if ((uint64_t)prompt_len > pool_capacity) {
        r.error = "prompt needs " + std::to_string(prompt_len) +
                  " KV tokens but the pool holds " +
                  std::to_string(pool_capacity) +
                  "; raise --kv-pool-tokens or shorten the prompt";
        return r;
    }

    int slot = -1;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (!slots_[(size_t)i].active) { slot = i; break; }
    }
    if (slot < 0) {
        r.busy = true;
        r.error = "all decode slots are busy";
        return r;
    }

    // A prompt that fits the whole pool but not the blocks currently free can
    // be admitted later. Gate on the exact prompt, never the speculative
    // output cap.
    const uint32_t need = (uint32_t)paged_block_count(prompt_len);
    if (need > pool_.free_block_count()) {
        r.busy = pool_.active_sequence_count() > 0;
        r.error = "not enough free KV blocks for the prompt";
        return r;
    }

    PagedKvSequenceHandle handle;
    PagedKvStatus status = pool_.acquire(request_id, handle);
    if (status != PagedKvStatus::Ok) {
        r.busy = (status == PagedKvStatus::SequenceSlotsExhausted);
        r.error = paged_kv_status_string(status);
        return r;
    }

    SeqSlot & s = slots_[(size_t)slot];
    s.active = true;
    s.prefilling = true;
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
    r.n_gen_cap = std::min(n_gen, max_ctx_ - prompt_len + 1);
    return r;
}

SeqSlotManager::PrefillChunk SeqSlotManager::append_prefill(
        int slot, int n_tokens) {
    PrefillChunk out;
    if (!is_prefilling(slot) || n_tokens < 1) return out;

    SeqSlot & s = slots_[(size_t)slot];
    PagedKvSequenceSnapshot before;
    if (pool_.sequence(s.handle, before) != PagedKvStatus::Ok ||
        before.kv_seq_len > (uint32_t)max_ctx_ ||
        (uint32_t)n_tokens > (uint32_t)max_ctx_ - before.kv_seq_len) {
        return out;
    }

    PagedKvAppendResult app = pool_.append(s.handle, (uint32_t)n_tokens);
    if (!app) {
        out.busy = app.status == PagedKvStatus::BlocksExhausted;
        return out;
    }

    PagedKvSequenceSnapshot after;
    if (pool_.sequence(s.handle, after) != PagedKvStatus::Ok ||
        after.block_table.size() < before.block_table.size() ||
        after.block_table.size() > (size_t)paged_block_count(max_ctx_)) {
        return out;
    }

    out.rows.reserve(app.write_slots.size());
    for (const PagedKvWriteSlot & write : app.write_slots) {
        out.rows.push_back((int64_t)write.physical_token_index);
    }
    if (after.block_table.size() > before.block_table.size()) {
        out.first_new_block = (int)before.block_table.size();
        out.new_blocks.reserve(after.block_table.size() -
                               before.block_table.size());
        for (size_t i = before.block_table.size();
             i < after.block_table.size(); i++) {
            out.new_blocks.push_back((int32_t)after.block_table[i]);
        }
    }
    out.ok = true;
    return out;
}

void SeqSlotManager::commit_prefill(int slot, int committed) {
    if (!is_active(slot)) return;
    slots_[(size_t)slot].prefilling = false;
    slots_[(size_t)slot].cur_pos = committed;
    lens_host_[(size_t)slot] = committed;
}

SeqSlotManager::StepAppend SeqSlotManager::append_token(int slot,
                                                        int32_t fed_token) {
    StepAppend out;
    if (!is_active(slot) || slots_[(size_t)slot].prefilling) return out;
    SeqSlot & s = slots_[(size_t)slot];
    if (s.cur_pos >= max_ctx_) {
        // No context left; the scheduler should have stopped this slot.
        return out;
    }
    PagedKvAppendSpan app = pool_.append_compact(s.handle, 1);
    if (!app || app.token_count != 1 ||
        app.last.logical_position != (uint32_t)s.cur_pos) {
        out.busy = app.status == PagedKvStatus::BlocksExhausted;
        return out;
    }
    s.sample_history.push_back(fed_token);
    lens_host_[(size_t)slot] = s.cur_pos + 1;
    out.ok = true;
    out.physical_row = (int64_t)app.last.physical_token_index;
    out.position = s.cur_pos;
    if ((uint32_t)s.cur_pos % pool_.block_size() == 0) {
        out.new_block = (int32_t)app.last.physical_block;
        out.new_block_index = s.cur_pos / (int)pool_.block_size();
    }
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
