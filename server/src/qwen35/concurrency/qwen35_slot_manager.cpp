#include "qwen35_slot_manager.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace dflash::common {

Qwen35SlotManager::Qwen35SlotManager(
        PagedKvPool & pool, int max_ctx, int speculative_headroom,
        PagedKvResidencyManager * residency)
    : pool_(pool), max_ctx_(max_ctx),
      headroom_tokens_(std::max<int>(pool.block_size(), speculative_headroom)),
      residency_(residency) {
    slots_.assign(pool.max_sequences(), Qwen35Slot{});
    const char * adaptive = std::getenv("DFLASH_DDTREE_ADAPTIVE");
    speculation_adaptive_ = !(adaptive && std::atoi(adaptive) == 0);
}

int Qwen35SlotManager::decoding_count() const {
    int n = 0;
    for (const Qwen35Slot & s : slots_) {
        n += s.decoding() ? 1 : 0;
    }
    return n;
}

uint32_t Qwen35SlotManager::decode_headroom_capacity(int logical_tokens) const {
    const uint64_t extended =
        static_cast<uint64_t>(std::max(0, logical_tokens)) +
        static_cast<uint64_t>(headroom_tokens_);
    return static_cast<uint32_t>(std::min<uint64_t>(
        static_cast<uint64_t>(max_ctx_), extended));
}

bool Qwen35SlotManager::capacity_fits_pool(uint32_t token_capacity) const {
    const uint64_t blocks = token_capacity == 0 ? 0 :
        1 + (static_cast<uint64_t>(token_capacity) - 1) /
                pool_.block_size();
    return blocks <= pool_.physical_block_count();
}

PagedKvStatus Qwen35SlotManager::protect_decode_headroom() {
    struct TopUp {
        PagedKvSequenceHandle handle;
        uint32_t token_capacity = 0;
    };

    std::vector<TopUp> topups;
    topups.reserve(slots_.size());
    uint64_t total_additional = 0;
    const uint64_t block_size = pool_.block_size();
    for (const Qwen35Slot & slot : slots_) {
        if (!slot.decoding()) continue;
        const uint32_t capacity = decode_headroom_capacity(slot.cur_pos);
        if (!capacity_fits_pool(capacity)) continue;

        uint32_t owned_blocks = 0;
        const PagedKvStatus status =
            pool_.owned_block_count(slot.handle, owned_blocks);
        if (status != PagedKvStatus::Ok) return status;
        const uint64_t target_blocks = capacity == 0 ? 0 :
            1 + (static_cast<uint64_t>(capacity) - 1) / block_size;
        if (target_blocks <= owned_blocks) continue;
        const uint32_t additional =
            static_cast<uint32_t>(target_blocks - owned_blocks);
        total_additional += additional;
        topups.push_back({slot.handle, capacity});
    }

    // Preflight the whole cohort before moving a block, so a failed admission
    // attempt cannot protect only whichever decoder happened to be visited
    // first.
    if (total_additional > pool_.free_block_count()) {
        return PagedKvStatus::BlocksExhausted;
    }
    for (const TopUp & topup : topups) {
        const PagedKvStatus status =
            pool_.reserve_capacity(topup.handle, topup.token_capacity);
        if (status != PagedKvStatus::Ok) return status;
    }
    return PagedKvStatus::Ok;
}

bool Qwen35SlotManager::is_active(int slot) const {
    return slot >= 0 && slot < (int)slots_.size() &&
           slots_[(size_t)slot].active();
}

bool Qwen35SlotManager::is_prefilling(int slot) const {
    return is_active(slot) && slots_[(size_t)slot].prefilling();
}

void Qwen35SlotManager::accumulate_residency_delta(
        Qwen35Slot & slot, const PagedKvResidencyStats & before) {
    if (!residency_) return;
    const PagedKvResidencyStats after = residency_->stats();
    if (after.page_ins >= before.page_ins) {
        slot.kvflash_page_ins_pending += after.page_ins - before.page_ins;
    }
    if (after.page_outs >= before.page_outs) {
        slot.kvflash_page_outs_pending += after.page_outs - before.page_outs;
    }
    if (after.reselects >= before.reselects) {
        slot.kvflash_reselects_pending += after.reselects - before.reselects;
    }
}

bool Qwen35SlotManager::has_prefill_prompt_at_least(int tokens) const {
    if (tokens <= 0) return true;
    return std::any_of(slots_.begin(), slots_.end(),
        [tokens](const Qwen35Slot & slot) {
            return slot.prefilling() && slot.prompt_len >= tokens;
        });
}

SeqEngine::AdmitResult Qwen35SlotManager::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    using AdmitStatus = SeqEngine::AdmitResult::Status;
    SeqEngine::AdmitResult r;
    if (prompt.empty()) {
        r.error = "empty prompt";
        return r;
    }
    if (prompt.size() > static_cast<size_t>(max_ctx_)) {
        r.error = "prompt exceeds max_ctx";
        return r;
    }
    const int prompt_len = static_cast<int>(prompt.size());

    // A prompt larger than the whole pool can NEVER be admitted; waiting
    // for other sequences to drain would stall the queue forever and then
    // fail anyway. Hard-fail it up front instead of reporting busy.
    const uint64_t pool_capacity =
        (uint64_t)pool_.physical_block_count() * pool_.block_size();
    if (!residency_ && (uint64_t)prompt_len > pool_capacity) {
        r.error = "prompt needs " + std::to_string(prompt_len) +
                  " KV tokens but the pool holds " +
                  std::to_string(pool_capacity) +
                  "; raise --kv-pool-tokens or shorten the prompt";
        return r;
    }

    // Retirement keeps failed copy-stream ownership quarantined. Retry those
    // barriers before deciding whether a sequence slot is actually available.
    for (int i = 0; i < (int)slots_.size(); ++i) {
        if (slots_[(size_t)i].retiring()) retire(i);
    }

    int slot = -1;
    for (int i = 0; i < (int)slots_.size(); i++) {
        if (slots_[(size_t)i].phase == Qwen35SlotPhase::free) { slot = i; break; }
    }
    if (slot < 0) {
        r.status = AdmitStatus::busy;
        r.error = "all decode slots are busy";
        return r;
    }

    // A newly freed block belongs to any older decoder missing its rolling
    // next-page reserve before it can belong to this admission.
    const PagedKvStatus headroom_status = residency_
        ? PagedKvStatus::Ok : protect_decode_headroom();
    if (headroom_status != PagedKvStatus::Ok) {
        r.status = headroom_status == PagedKvStatus::BlocksExhausted
            ? AdmitStatus::busy : AdmitStatus::failed;
        r.error = r.status == AdmitStatus::busy
            ? "existing decoders need the available KV headroom"
            : paged_kv_status_string(headroom_status);
        return r;
    }

    PagedKvSequenceHandle handle;
    uint32_t reservation_capacity = residency_ ? 0 :
        decode_headroom_capacity(prompt_len);
    if (!capacity_fits_pool(reservation_capacity)) {
        // The prompt itself fits, but this physical pool can never hold its
        // following page. Preserve useful prompt-only behavior and report
        // decode exhaustion later if the sequence reaches that boundary.
        reservation_capacity = static_cast<uint32_t>(prompt_len);
    }
    const PagedKvStatus status = residency_
        ? pool_.acquire(request_id, handle)
        : pool_.acquire_reserved(request_id, reservation_capacity, handle);
    if (status != PagedKvStatus::Ok) {
        r.status = status == PagedKvStatus::SequenceSlotsExhausted ||
                           status == PagedKvStatus::BlocksExhausted
            ? AdmitStatus::busy : AdmitStatus::failed;
        r.error = status == PagedKvStatus::BlocksExhausted
            ? "not enough unreserved KV blocks for the prompt and decode headroom"
            : paged_kv_status_string(status);
        return r;
    }

    if (residency_) {
        const PagedKvResidencyStatus resident_status =
            residency_->register_sequence(handle);
        if (resident_status != PagedKvResidencyStatus::Ok) {
            (void)pool_.release(handle);
            r.error = std::string("KV residency registration failed: ") +
                paged_kv_residency_status_string(resident_status);
            return r;
        }
    }

    Qwen35Slot & s = slots_[(size_t)slot];
    s.phase = Qwen35SlotPhase::prefill;
    s.request_id = request_id;
    s.ddtree_sampled_steps = 0;
    s.speculation.reset(speculation_adaptive_);
    s.handle = handle;
    s.cur_pos = 0;
    s.prompt_len = prompt_len;
    s.sampler = sampler;
    s.sample_history = prompt;
    // Same predicate the engine uses to pick CPU sampling over GPU argmax:
    // a seed only means anything when the sampler actually draws.
    if (sampler.needs_logit_processing() && sampler.seed != 0) {
        s.rng.seed(sampler.seed);
    } else {
        s.rng.seed(std::random_device{}());
    }

    r.status = AdmitStatus::admitted;
    r.slot = slot;
    return r;
}

bool Qwen35SlotManager::ddtree_speculation_allowed(int slot) const {
    return is_active(slot) &&
        slots_[(size_t)slot].speculation.wants_speculation();
}

SpeculationGoodputTransition Qwen35SlotManager::record_speculation_sample(
        int slot, double emitted_tokens, double elapsed_us) {
    if (!is_active(slot)) return SpeculationGoodputTransition::none;
    Qwen35Slot & s = slots_[(size_t)slot];
    ++s.ddtree_sampled_steps;
    return s.speculation.observe_speculation(emitted_tokens, elapsed_us);
}

SpeculationGoodputTransition Qwen35SlotManager::record_ar_sample(
        int slot, double elapsed_us) {
    if (!is_active(slot)) return SpeculationGoodputTransition::none;
    Qwen35Slot & s = slots_[(size_t)slot];
    return s.speculation.observe_autoregressive(elapsed_us);
}

Qwen35SlotManager::PrefillChunk Qwen35SlotManager::append_prefill(
        int slot, int n_tokens) {
    PrefillChunk out;
    if (!is_prefilling(slot) || n_tokens < 1) return out;

    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.cur_pos > s.prompt_len ||
        n_tokens > s.prompt_len - s.cur_pos) {
        return out;
    }

    PagedKvAppendResult app;
    if (residency_) {
        const PagedKvResidencyStats before = residency_->stats();
        PagedKvResidentAppendResult resident = residency_->append(
            s.handle, (uint32_t)n_tokens);
        accumulate_residency_delta(s, before);
        if (!resident) {
            std::fprintf(stderr,
                "[parallel-kvflash] prefill append failed for slot %d: %s\n",
                slot, paged_kv_residency_status_string(resident.status));
            return out;
        }
        app = std::move(resident.pool_result);
    } else {
        app = pool_.append(s.handle, (uint32_t)n_tokens);
    }
    if (!app) {
        // Admission reserved the whole prompt. Treat exhaustion here as a
        // broken invariant, not a retryable condition: retrying a batch of
        // all-prefill slots without any decoder able to retire would livelock.
        if (app.status == PagedKvStatus::BlocksExhausted) {
            std::fprintf(stderr,
                "[parallel] reserved prefill capacity missing for slot %d\n",
                slot);
        }
        return out;
    }

    out.rows.reserve(app.write_slots.size());
    for (const PagedKvWriteSlot & write : app.write_slots) {
        out.rows.push_back((int64_t)write.physical_token_index);
        if (write.block_offset == 0) {
            if (out.first_new_block < 0) {
                out.first_new_block =
                    (int)(write.logical_position / pool_.block_size());
            }
            out.new_blocks.push_back((int32_t)write.physical_block);
        }
    }
    if (residency_) {
        PagedKvSequenceSnapshot snapshot;
        if (pool_.sequence(s.handle, snapshot) != PagedKvStatus::Ok) {
            return out;
        }
        out.full_block_table.reserve(snapshot.block_table.size());
        for (uint32_t block : snapshot.block_table) {
            out.full_block_table.push_back(
                block == PAGED_KV_COLD_BLOCK ? -1 : (int32_t)block);
        }
    }
    s.cur_pos += n_tokens;
    out.ok = true;
    return out;
}

void Qwen35SlotManager::commit_prefill(int slot) {
    if (!is_prefilling(slot)) return;
    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.cur_pos != s.prompt_len) return;
    s.phase = Qwen35SlotPhase::decode;
}

Qwen35SlotManager::StepAppend Qwen35SlotManager::append_tokens(
        int slot, const int32_t * fed_tokens, int n_tokens) {
    StepAppend out;
    if (!is_active(slot) || !slots_[(size_t)slot].decoding() ||
        !fed_tokens || n_tokens < 1) {
        return out;
    }
    Qwen35Slot & s = slots_[(size_t)slot];
    if (!s.staged_tokens.empty() || s.cur_pos > max_ctx_ ||
        n_tokens > max_ctx_ - s.cur_pos) {
        return out;
    }

    PagedKvAppendResult app;
    if (residency_) {
        const PagedKvResidencyStats before = residency_->stats();
        PagedKvResidentAppendResult resident = residency_->append(
            s.handle, static_cast<uint32_t>(n_tokens));
        accumulate_residency_delta(s, before);
        if (!resident) {
            out.busy = resident.status ==
                    PagedKvResidencyStatus::PoolExhausted ||
                resident.status == PagedKvResidencyStatus::NoEvictableBlock;
            return out;
        }
        app = std::move(resident.pool_result);
    } else {
        app = pool_.append(s.handle, static_cast<uint32_t>(n_tokens));
    }
    if (!app || app.token_count != static_cast<uint32_t>(n_tokens)) {
        out.busy = app.status == PagedKvStatus::BlocksExhausted;
        return out;
    }
    if (app.write_slots.size() != static_cast<size_t>(n_tokens) ||
        app.write_slots.front().logical_position !=
            static_cast<uint32_t>(s.cur_pos) ||
        app.write_slots.back().logical_position !=
            static_cast<uint32_t>(s.cur_pos + n_tokens - 1)) {
        // Pool success guarantees this shape. Retain staged ownership so a
        // fatal caller retires the sequence rather than double-appending.
        s.staged_tokens.assign(fed_tokens, fed_tokens + n_tokens);
        return out;
    }

    out.physical_rows.reserve(app.write_slots.size());
    for (const PagedKvWriteSlot & write : app.write_slots) {
        out.physical_rows.push_back(
            static_cast<int64_t>(write.physical_token_index));
        if (write.block_offset == 0) {
            if (out.first_new_block < 0) {
                out.first_new_block = static_cast<int>(
                    write.logical_position / pool_.block_size());
            }
            out.new_blocks.push_back(
                static_cast<int32_t>(write.physical_block));
        }
    }
    s.staged_tokens.assign(fed_tokens, fed_tokens + n_tokens);
    if (residency_) {
        PagedKvSequenceSnapshot snapshot;
        if (pool_.sequence(s.handle, snapshot) != PagedKvStatus::Ok) {
            return out;
        }
        out.full_block_table.reserve(snapshot.block_table.size());
        for (uint32_t block : snapshot.block_table) {
            out.full_block_table.push_back(
                block == PAGED_KV_COLD_BLOCK ? -1 : (int32_t)block);
        }
    }

    out.ok = true;
    out.count = n_tokens;
    out.position = s.cur_pos;
    if (n_tokens == 1) {
        out.physical_row = out.physical_rows.front();
        if (!out.new_blocks.empty()) {
            out.new_block = out.new_blocks.front();
            out.new_block_index = out.first_new_block;
        }
    }
    return out;
}

Qwen35SlotManager::StepAppend Qwen35SlotManager::append_token(
        int slot, int32_t fed_token) {
    return append_tokens(slot, &fed_token, 1);
}

void Qwen35SlotManager::commit_step(int slot) {
    if (!is_active(slot)) return;
    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.staged_tokens.empty()) return;
    s.sample_history.insert(
        s.sample_history.end(), s.staged_tokens.begin(),
        s.staged_tokens.end());
    s.cur_pos += static_cast<int>(s.staged_tokens.size());
    s.staged_tokens.clear();
}

bool Qwen35SlotManager::block_table_snapshot(
        int slot, std::vector<int32_t> & out) const {
    out.clear();
    if (!is_active(slot)) return false;
    PagedKvSequenceSnapshot snapshot;
    if (pool_.sequence(slots_[(size_t)slot].handle, snapshot) !=
        PagedKvStatus::Ok) {
        return false;
    }
    out.reserve(snapshot.block_table.size());
    for (uint32_t block : snapshot.block_table) {
        out.push_back(block == PAGED_KV_COLD_BLOCK ? -1 : (int32_t)block);
    }
    return true;
}

bool Qwen35SlotManager::commit_residency_writes(int slot) {
    if (!residency_) return true;
    if (!is_active(slot)) return false;
    return residency_->commit_pending_writes(slots_[(size_t)slot].handle) ==
        PagedKvResidencyStatus::Ok;
}

bool Qwen35SlotManager::reselect_residency(
        int slot, const std::vector<float> * scores, std::string * error) {
    if (!residency_) return true;
    if (!is_active(slot)) {
        if (error) *error = "inactive KVFlash slot";
        return false;
    }
    Qwen35Slot & s = slots_[(size_t)slot];
    const PagedKvResidencyStats before = residency_->stats();
    const std::vector<float> no_scores;
    PagedKvResidencyStatus status = residency_->set_scores(
        s.handle, scores ? *scores : no_scores);
    if (status == PagedKvResidencyStatus::Ok) {
        status = residency_->reselect(s.handle);
    }
    accumulate_residency_delta(s, before);
    if (status != PagedKvResidencyStatus::Ok) {
        if (error) {
            *error = std::string("KVFlash reselect failed: ") +
                paged_kv_residency_status_string(status);
        }
        return false;
    }
    return true;
}

void Qwen35SlotManager::take_residency_telemetry(
        int slot, SeqEngine::DecodeOutput & out) {
    if (!residency_ || !is_active(slot)) return;
    Qwen35Slot & s = slots_[(size_t)slot];
    out.kvflash_page_ins = s.kvflash_page_ins_pending;
    out.kvflash_page_outs = s.kvflash_page_outs_pending;
    out.kvflash_reselects = s.kvflash_reselects_pending;
    uint32_t resident = 0;
    if (pool_.resident_block_count(s.handle, resident) == PagedKvStatus::Ok) {
        out.kvflash_resident_blocks = resident;
    }
    s.kvflash_page_ins_pending = 0;
    s.kvflash_page_outs_pending = 0;
    s.kvflash_reselects_pending = 0;
}

void Qwen35SlotManager::retire(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    Qwen35Slot & s = slots_[(size_t)slot];
    if (s.phase == Qwen35SlotPhase::free) return;
    if (residency_) {
        const PagedKvResidencyStatus resident_status =
            residency_->forget_sequence(s.handle);
        if (resident_status != PagedKvResidencyStatus::Ok &&
            resident_status != PagedKvResidencyStatus::StaleHandle &&
            resident_status != PagedKvResidencyStatus::SequenceNotRegistered) {
            std::fprintf(stderr,
                "[parallel-kvflash] slot %d residency release failed: %s\n",
                slot, paged_kv_residency_status_string(resident_status));
            // A failed copy-stream barrier leaves physical pages in flight.
            // Keep the slot and its pool handle intact so a later retirement
            // can retry forget_sequence without recycling those pages.
            s.phase = Qwen35SlotPhase::retiring;
            return;
        }
    }
    const PagedKvStatus status = pool_.release(s.handle);
    if (status != PagedKvStatus::Ok && status != PagedKvStatus::StaleHandle) {
        std::fprintf(stderr, "[parallel] slot %d release failed: %s\n",
                     slot, paged_kv_status_string(status));
    }
    s = Qwen35Slot{};
}

}  // namespace dflash::common
