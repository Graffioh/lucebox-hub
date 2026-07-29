// Host-side unit test for SeqSlotManager (concurrent decode slots).
//
// Mirrors test_paged_kv_pool: no model, ggml, or GPU required. Covers the
// admission arithmetic (context clamp, worst-case reservation, never-fits
// capacity check), the busy-vs-hard-error classification, the pool-handle
// lifecycle across retire, per-step appends, and the kv-length mirror.

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

        // Fresh pool: prompt rows are identity-mapped from block 0.
        auto a = mgr.admit(1, /*prompt_len=*/20, /*n_gen=*/8,
                           greedy_sampler());
        CHECK(a.ok && !a.busy);
        CHECK(a.slot == 0);
        CHECK(a.prompt_rows.size() == 20);
        CHECK(a.prompt_rows[0] == 0 && a.prompt_rows[19] == 19);
        // reserve covers prompt + n_gen - 1 = 27 tokens -> 2 blocks.
        CHECK(a.table_column.size() == 2);
        CHECK(mgr.is_active(0));
        CHECK(mgr.lens_host()[0] == 0);   // set by commit_prefill
        mgr.commit_prefill(0, 20);
        CHECK(mgr.slot(0).cur_pos == 20);
        CHECK(mgr.lens_host()[0] == 20);

        // Decode appends: row allocation + sample_history + lens mirror;
        // cur_pos advances separately after the step's compute.
        auto st = mgr.append_token(0, /*fed_token=*/42);
        CHECK(st.ok);
        CHECK(st.position == 20);
        CHECK(st.physical_row == 20);     // still inside the reserved run
        CHECK(mgr.lens_host()[0] == 21);
        CHECK(mgr.slot(0).cur_pos == 20);
        CHECK(mgr.slot(0).sample_history.size() == 1 &&
              mgr.slot(0).sample_history[0] == 42);
        mgr.commit_step(0);
        CHECK(mgr.slot(0).cur_pos == 21);

        // Second admission lands in slot 1 with non-identity rows.
        auto b = mgr.admit(2, 20, 8, greedy_sampler());
        CHECK(b.ok && b.slot == 1);
        CHECK(b.prompt_rows[0] != 0);
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
        CHECK(!mgr.append_token(0, 1).ok);
        mgr.commit_step(0);
        mgr.retire(-1);
        mgr.retire(99);
    }

    // Busy-vs-never-fits classification against a small pool.
    {
        // 4 blocks x 16 = 64 pool tokens, 2 slots, max_ctx 64.
        PagedKvPool pool(4, 2);
        SeqSlotManager mgr(pool, 64);

        // The context clamp caps the reservation: ar_n_gen = min(40, 64-60+1)
        // = 5, so the worst case is exactly the 64-token pool — admitted.
        auto exact = mgr.admit(1, 60, 40, greedy_sampler());
        CHECK(exact.ok);
        mgr.retire(exact.slot);

        // Occupy most of the pool, then a second request that WOULD fit an
        // empty pool reports busy (blocks held by a live sequence).
        auto big = mgr.admit(2, 48, 1, greedy_sampler());  // 3 blocks
        CHECK(big.ok);
        auto blocked = mgr.admit(3, 32, 1, greedy_sampler());  // 2 blocks
        CHECK(!blocked.ok && blocked.busy);
        mgr.retire(big.slot);
        auto now_fits = mgr.admit(3, 32, 1, greedy_sampler());
        CHECK(now_fits.ok);
    }

    // Never-fits: worst case beyond the WHOLE pool is a hard error, not
    // busy — waiting for a drain could never help.
    {
        // 4 blocks x 16 = 64 pool tokens, but max_ctx allows asking for more.
        PagedKvPool pool(4, 2);
        SeqSlotManager mgr(pool, /*max_ctx=*/128);
        auto never = mgr.admit(1, 100, 10, greedy_sampler());
        CHECK(!never.ok && !never.busy);   // reserve 109 > pool 64
        CHECK(pool.active_sequence_count() == 0);
    }

    // Context exhaustion: append_token refuses past max_ctx.
    {
        PagedKvPool pool(4, 1);
        SeqSlotManager mgr(pool, /*max_ctx=*/17);
        auto a = mgr.admit(1, 16, 2, greedy_sampler());
        CHECK(a.ok);
        mgr.commit_prefill(0, 16);
        auto s1 = mgr.append_token(0, 7);   // position 16 (== max_ctx-1)
        CHECK(s1.ok && s1.position == 16);
        mgr.commit_step(0);
        auto s2 = mgr.append_token(0, 8);   // cur_pos == max_ctx -> refuse
        CHECK(!s2.ok);
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
