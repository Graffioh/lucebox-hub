// Host-side unit test for SeqSlotManager (concurrent decode slots).
//
// Mirrors test_paged_kv_pool: no model, ggml, or GPU required. Covers the
// prompt-only admission checks, output-cap clamp, on-demand prefill/decode
// allocation, block-table deltas, exhaustion atomicity, the pool-handle
// lifecycle across retire, and the kv-length mirror.

#include "common/seq_slot_manager.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

static int g_checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,     \
                         #cond);                                             \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static SamplerCfg greedy_sampler() {
    return SamplerCfg{};
}

int main() {
    // 8 blocks x 16 tokens = 128 pool tokens, 2 slots, per-seq max_ctx 64.
    {
        PagedKvPool pool(/*physical_block_count=*/8, /*max_sequences=*/2);
        SeqSlotManager mgr(pool, /*max_ctx=*/64);
        CHECK(mgr.slot_count() == 2);
        CHECK(mgr.active_count() == 0);

        // Invalid asks are hard errors, not busy.
        CHECK(!mgr.admit(1, 0, 8, greedy_sampler()).ok);
        CHECK(!mgr.admit(1, 8, 0, greedy_sampler()).ok);
        CHECK(!mgr.admit(1, 65, 8, greedy_sampler()).ok);
        CHECK(mgr.active_count() == 0);
        CHECK(pool.active_sequence_count() == 0);

        // Admission claims identity only; it does not allocate the prompt or
        // the speculative output budget.
        auto a = mgr.admit(1, /*prompt_len=*/20, /*n_gen=*/8,
                           greedy_sampler());
        CHECK(a.ok && !a.busy);
        CHECK(a.slot == 0);
        CHECK(a.n_gen_cap == 8);
        CHECK(pool.free_block_count() == 8);
        CHECK(mgr.is_active(0));
        CHECK(mgr.lens_host()[0] == 0);   // set by commit_prefill

        // Prompt allocation follows the chunks actually scheduled. The first
        // ten rows open block 0; the next ten consume its tail and open block
        // 1, returning only that block-table delta.
        auto p0 = mgr.append_prefill(a.slot, 10);
        CHECK(p0.ok && !p0.busy);
        CHECK(p0.rows.size() == 10 && p0.rows.front() == 0 &&
              p0.rows.back() == 9);
        CHECK(p0.first_new_block == 0 && p0.new_blocks.size() == 1 &&
              p0.new_blocks[0] == 0);
        auto p1 = mgr.append_prefill(a.slot, 10);
        CHECK(p1.ok && !p1.busy);
        CHECK(p1.rows.size() == 10 && p1.rows.front() == 10 &&
              p1.rows.back() == 19);
        CHECK(p1.first_new_block == 1 && p1.new_blocks.size() == 1 &&
              p1.new_blocks[0] == 1);
        mgr.commit_prefill(0, 20);
        CHECK(mgr.slot(0).cur_pos == 20);
        CHECK(mgr.lens_host()[0] == 20);

        // Decode appends: row allocation + sample_history + lens mirror;
        // cur_pos advances separately after the step's compute.
        auto st = mgr.append_token(0, /*fed_token=*/42);
        CHECK(st.ok);
        CHECK(st.position == 20);
        CHECK(st.physical_row == 20);     // tail of the prompt's last block
        CHECK(st.new_block < 0 && st.new_block_index < 0);
        CHECK(mgr.lens_host()[0] == 21);
        CHECK(mgr.slot(0).cur_pos == 20);
        CHECK(mgr.slot(0).sample_history.size() == 1 &&
              mgr.slot(0).sample_history[0] == 42);
        mgr.commit_step(0);
        CHECK(mgr.slot(0).cur_pos == 21);

        // Second admission lands in slot 1 with non-identity rows.
        auto b = mgr.admit(2, 20, 8, greedy_sampler());
        CHECK(b.ok && b.slot == 1);
        auto pb = mgr.append_prefill(b.slot, 20);
        CHECK(pb.ok && pb.rows.front() != 0);
        mgr.commit_prefill(1, 20);

        // Third admission: no free slot -> busy.
        auto c = mgr.admit(3, 8, 8, greedy_sampler());
        CHECK(!c.ok && c.busy);

        // Retire frees the slot AND the pool blocks; the mirror zeroes.
        const uint32_t free_before = pool.free_block_count();
        mgr.retire(0);
        CHECK(!mgr.is_active(0));
        CHECK(mgr.lens_host()[0] == 0);
        CHECK(pool.free_block_count() > free_before);
        CHECK(mgr.active_count() == 1);

        // The freed slot admits again.
        auto d = mgr.admit(3, 8, 8, greedy_sampler());
        CHECK(d.ok && d.slot == 0);

        // Inactive-slot calls are safe no-ops.
        mgr.retire(0);
        mgr.retire(0);
        CHECK(!mgr.append_prefill(0, 1).ok);
        CHECK(!mgr.append_token(0, 1).ok);
        mgr.commit_step(0);
        mgr.retire(-1);
        mgr.retire(99);
    }

    // The output cap is not a memory reservation. Even a cap much larger than
    // the physical pool admits when the known prompt fits.
    {
        PagedKvPool pool(/*physical_block_count=*/1,
                         /*max_sequences=*/2);
        SeqSlotManager mgr(pool, /*max_ctx=*/128);
        auto a = mgr.admit(1, /*prompt_len=*/8, /*n_gen=*/128,
                           greedy_sampler());
        CHECK(a.ok && a.n_gen_cap == 121);
        CHECK(pool.free_block_count() == 1);
        CHECK(mgr.append_prefill(a.slot, 8).ok);
    }

    // Busy-vs-never-fits classification against a small pool.
    {
        // 4 blocks x 16 = 64 pool tokens, 2 slots, max_ctx 64.
        PagedKvPool pool(4, 2);
        SeqSlotManager mgr(pool, 64);

        // The context clamp is returned to the scheduler, not used as a block
        // reservation. A 60-token prompt can produce at most five tokens.
        auto exact = mgr.admit(1, 60, 40, greedy_sampler());
        CHECK(exact.ok && exact.n_gen_cap == 5);
        CHECK(pool.free_block_count() == 4);
        CHECK(mgr.append_prefill(exact.slot, 60).ok);
        mgr.retire(exact.slot);

        // Occupy most of the pool, then a second request that WOULD fit an
        // empty pool reports busy (blocks held by a live sequence).
        auto big = mgr.admit(2, 48, 1, greedy_sampler());  // 3 blocks
        CHECK(big.ok);
        CHECK(mgr.append_prefill(big.slot, 48).ok);
        auto blocked = mgr.admit(3, 32, 1, greedy_sampler());  // 2 blocks
        CHECK(!blocked.ok && blocked.busy);
        mgr.retire(big.slot);
        auto now_fits = mgr.admit(3, 32, 1, greedy_sampler());
        CHECK(now_fits.ok);
    }

    // Never-fits: a prompt beyond the WHOLE pool is a hard error, not
    // busy — waiting for a drain could never help.
    {
        // 4 blocks x 16 = 64 pool tokens, but max_ctx allows asking for more.
        PagedKvPool pool(4, 2);
        SeqSlotManager mgr(pool, /*max_ctx=*/128);
        auto never = mgr.admit(1, 100, 10, greedy_sampler());
        CHECK(!never.ok && !never.busy);   // prompt 100 > pool 64
        CHECK(pool.active_sequence_count() == 0);

        // Impossibility wins over temporary slot pressure: do not queue an
        // oversized prompt merely because every sequence slot is occupied.
        auto live = mgr.admit(2, 16, 1, greedy_sampler());
        CHECK(live.ok);
        auto live2 = mgr.admit(3, 16, 1, greedy_sampler());
        CHECK(live2.ok);
        auto still_never = mgr.admit(4, 100, 10, greedy_sampler());
        CHECK(!still_never.ok && !still_never.busy);
    }

    // Chunk exhaustion is retryable and all-or-nothing: neither length nor
    // block table changes when a whole chunk cannot be allocated.
    {
        PagedKvPool pool(/*physical_block_count=*/2,
                         /*max_sequences=*/3);
        SeqSlotManager mgr(pool, /*max_ctx=*/64);
        auto a = mgr.admit(1, 16, 8, greedy_sampler());
        CHECK(a.ok);
        CHECK(mgr.append_prefill(a.slot, 16).ok);  // block 0

        // One block remains, so a known two-block prompt waits rather than
        // acquiring a slot it cannot currently prefill.
        auto later = mgr.admit(2, 32, 8, greedy_sampler());
        CHECK(!later.ok && later.busy);
        CHECK(pool.active_sequence_count() == 1);

        auto b = mgr.admit(2, 16, 8, greedy_sampler());
        CHECK(b.ok);
        CHECK(mgr.append_prefill(b.slot, 16).ok);  // block 1
        CHECK(pool.free_block_count() == 0);

        PagedKvSequenceSnapshot before;
        CHECK(pool.sequence(mgr.slot(a.slot).handle, before) ==
              PagedKvStatus::Ok);
        auto blocked = mgr.append_prefill(a.slot, 16);
        CHECK(!blocked.ok && blocked.busy);
        PagedKvSequenceSnapshot after;
        CHECK(pool.sequence(mgr.slot(a.slot).handle, after) ==
              PagedKvStatus::Ok);
        CHECK(after.kv_seq_len == before.kv_seq_len);
        CHECK(after.block_table == before.block_table);
        CHECK(pool.free_block_count() == 0);

        // Decode exposes the same temporary exhaustion distinctly and does
        // not append penalty history or advance the length mirror.
        mgr.commit_prefill(a.slot, 16);
        auto decode_blocked = mgr.append_token(a.slot, 77);
        CHECK(!decode_blocked.ok && decode_blocked.busy);
        CHECK(mgr.slot(a.slot).sample_history.empty());
        CHECK(mgr.lens_host()[(size_t)a.slot] == 16);
    }

    // Context exhaustion: append_token refuses past max_ctx.
    {
        PagedKvPool pool(4, 1);
        SeqSlotManager mgr(pool, /*max_ctx=*/17);
        auto a = mgr.admit(1, 16, 2, greedy_sampler());
        CHECK(a.ok);
        CHECK(mgr.append_prefill(a.slot, 16).ok);
        mgr.commit_prefill(0, 16);
        auto s1 = mgr.append_token(0, 7);   // position 16 (== max_ctx-1)
        CHECK(s1.ok && s1.position == 16);
        CHECK(s1.new_block == 1 && s1.new_block_index == 1);
        mgr.commit_step(0);
        auto s2 = mgr.append_token(0, 8);   // cur_pos == max_ctx -> refuse
        CHECK(!s2.ok);
    }

    // Prefilling lifecycle: an admitted slot stays out of the decode batch
    // until commit_prefill() makes its first sampled token available.
    {
        PagedKvPool pool(8, 2);
        SeqSlotManager mgr(pool, 64);
        auto a = mgr.admit(1, 20, 8, greedy_sampler());
        CHECK(a.ok);
        CHECK(mgr.is_prefilling(a.slot));
        CHECK(mgr.active_count() == 1);
        CHECK(mgr.decoding_count() == 0);
        CHECK(!mgr.append_token(a.slot, 42).ok);
        CHECK(mgr.lens_host()[(size_t)a.slot] == 0);

        CHECK(mgr.append_prefill(a.slot, 20).ok);
        mgr.commit_prefill(a.slot, 20);
        CHECK(!mgr.is_prefilling(a.slot));
        CHECK(mgr.decoding_count() == 1);
        CHECK(mgr.append_token(a.slot, 42).ok);
        CHECK(!mgr.append_prefill(a.slot, 1).ok);

        // A second admission can prefill while the first slot decodes.
        auto b = mgr.admit(2, 20, 8, greedy_sampler());
        CHECK(b.ok);
        CHECK(mgr.active_count() == 2);
        CHECK(mgr.decoding_count() == 1);

        // Retiring during prefill clears the state before slot reuse.
        mgr.retire(b.slot);
        CHECK(!mgr.is_prefilling(b.slot));
        auto c = mgr.admit(3, 8, 4, greedy_sampler());
        CHECK(c.ok && c.slot == b.slot);
        CHECK(mgr.is_prefilling(c.slot));
    }

    // Seeded sampling RNG is deterministic per admission. The sampler alone
    // decides: a seed is honoured exactly when needs_logit_processing() says
    // the slot actually draws, so there is no way to ask for seeded sampling
    // and be given argmax (or the reverse).
    {
        PagedKvPool pool(4, 1);
        SeqSlotManager mgr(pool, 64);
        SamplerCfg cfg = greedy_sampler();
        cfg.temp = 0.7f;            // needs_logit_processing() -> true
        cfg.seed = 1234;
        CHECK(cfg.needs_logit_processing());
        auto a = mgr.admit(1, 4, 4, cfg);
        CHECK(a.ok);
        const uint64_t first = mgr.slot(0).rng();
        mgr.retire(0);
        auto b = mgr.admit(2, 4, 4, cfg);
        CHECK(b.ok);
        CHECK(mgr.slot(0).rng() == first);

        // A different seed is a different stream.
        mgr.retire(0);
        cfg.seed = 5678;
        auto c = mgr.admit(3, 4, 4, cfg);
        CHECK(c.ok);
        CHECK(mgr.slot(0).rng() != first);
    }

    // A greedy sampler never draws, so its seed is irrelevant and admission
    // must not depend on one being present.
    {
        PagedKvPool pool(4, 1);
        SeqSlotManager mgr(pool, 64);
        SamplerCfg cfg = greedy_sampler();
        cfg.seed = 1234;
        CHECK(!cfg.needs_logit_processing());
        auto a = mgr.admit(1, 4, 4, cfg);
        CHECK(a.ok);
        CHECK(!mgr.slot(0).sampler.needs_logit_processing());
    }

    std::printf("OK test_seq_slot_manager (%d checks)\n", g_checks);
    return 0;
}
