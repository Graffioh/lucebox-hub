// Executable conformance check for the SeqEngine contract.
//
// The concurrent scheduler (server/scheduler.cpp) drives a backend through
// common/seq_engine.h and nothing else, so that header's prose is the whole
// agreement between the two sides. This turns it into code: hand the checker
// any SeqEngine whose slots are all free and it returns the list of contract
// violations — empty means the engine is safe to pass to scheduler_loop().
//
// It exists so the second, third, … model's engine is held to the same rules
// as the Qwen3.5 one without either side importing the other's internals: the
// checker only ever calls slot_count/admit/step/retire/token_is_eos, and it
// knows nothing about block tables, graphs, or recurrent state.
//
// No ggml and no GPU, so a host-only test can use it (test_seq_engine_contract
// runs it against an in-memory fake). A device-backed engine can be checked
// with the same call — the checker allocates nothing device-side; it just
// costs a handful of real prefills.
//
// Entry condition: every slot free. On return every slot is free again.

#pragma once

#include "common/seq_engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

inline std::vector<std::string> check_seq_engine_contract(SeqEngine & engine) {
    std::vector<std::string> violations;
    auto require = [&violations](bool ok, const char * msg) {
        if (!ok) violations.emplace_back(msg);
    };

    const int n_slots = engine.slot_count();
    require(n_slots >= 1, "slot_count() must be at least 1");
    if (n_slots < 1) return violations;

    const SamplerCfg greedy{};
    // The sampler is the engine's only signal for how a slot draws, so admit
    // through both of its branches: greedy takes the GPU-argmax path, seeded
    // sampling takes the CPU sample_logits path. An engine that reads the two
    // out of step fails here rather than in production.
    SamplerCfg seeded{};
    seeded.temp = 0.7f;             // needs_logit_processing() -> true
    seeded.seed = 20260729;
    const std::vector<int32_t> prompt{11, 12, 13, 14};
    const int n_gen = 16;

    // ── 1. Admission hands out distinct, in-range slots ──────────────────
    // The scheduler indexes its own per-request array by the returned slot, so
    // two live requests sharing a slot id would cross their token streams.
    std::vector<int>     live;      // slot ids, in admission order
    std::vector<int32_t> pending;   // parallel to `live`: token to feed next
    std::vector<bool>    taken((size_t)n_slots, false);
    bool filled_all = true;
    for (int i = 0; i < n_slots; i++) {
        const SeqEngine::AdmitResult r = engine.admit(
            (uint64_t)(i + 1), prompt, (i % 2) ? seeded : greedy, n_gen);
        if (!r.ok) {
            // A tight KV pool may legitimately go busy before the last slot;
            // only the first admission into an idle engine must succeed.
            require(i > 0, "admit() must succeed on an idle engine");
            require(r.busy,
                    "a failed admit() with a free slot must report busy=true");
            filled_all = false;
            break;
        }
        require(r.slot >= 0 && r.slot < n_slots,
                "admit() returned a slot outside [0, slot_count())");
        if (r.slot < 0 || r.slot >= n_slots) { filled_all = false; break; }
        require(!taken[(size_t)r.slot], "admit() reused a slot that is live");
        taken[(size_t)r.slot] = true;
        require(r.first_token >= 0,
                "a successful admit() must return a pending first token");
        require(r.prefill_s >= 0.0, "prefill_s must not be negative");
        live.push_back(r.slot);
        pending.push_back(r.first_token);
    }
    require(!live.empty(), "no request could be admitted");
    if (live.empty()) return violations;

    // ── 2. A full engine is busy, never a hard error ─────────────────────
    // The scheduler re-queues a busy job and retries it after the next
    // retire; a hard error fails a request that only needed to wait.
    if (filled_all) {
        const SeqEngine::AdmitResult over =
            engine.admit(9001, prompt, greedy, n_gen);
        require(!over.ok, "admit() must fail when every slot is live");
        if (!over.ok) {
            require(over.busy,
                    "a full engine must report busy=true, not a hard error");
            require(over.slot < 0, "a failed admit() must not claim a slot");
        }
    }

    // ── 3. A batched step answers every live slot, exactly once ──────────
    for (int iter = 0; iter < 3; iter++) {
        std::vector<SeqEngine::StepInput> inputs;
        inputs.reserve(live.size());
        for (size_t i = 0; i < live.size(); i++) {
            inputs.push_back({live[i], pending[i]});
        }
        std::vector<SeqEngine::StepOutput> outputs;
        if (!engine.step(inputs, outputs)) {
            require(false, "step() over every live slot must succeed");
            break;
        }
        if (outputs.size() != inputs.size()) {
            require(false, "step() must return one output per input");
            break;
        }

        std::vector<bool> answered((size_t)n_slots, false);
        bool shape_ok = true;
        for (const SeqEngine::StepOutput & o : outputs) {
            if (o.slot < 0 || o.slot >= n_slots) {
                require(false, "step() returned an output for an unknown slot");
                shape_ok = false;
                continue;
            }
            require(!answered[(size_t)o.slot],
                    "step() returned two outputs for one slot");
            answered[(size_t)o.slot] = true;
            if (o.failed) continue;
            require(o.token >= 0,
                    "a non-failed step() output must carry a token");
        }
        for (const int s : live) {
            if (!answered[(size_t)s]) {
                require(false, "step() left a live slot without an output");
                shape_ok = false;
                break;
            }
        }
        if (!shape_ok) break;

        // Feed each sampled token back as the next step's input — the handoff
        // that lets the scheduler override a token (thinking-budget close)
        // before it is committed to the cache.
        bool ended = false;
        for (const SeqEngine::StepOutput & o : outputs) {
            if (o.failed || engine.token_is_eos(o.token)) {
                ended = true;
                continue;
            }
            for (size_t i = 0; i < live.size(); i++) {
                if (live[i] == o.slot) { pending[i] = o.token; break; }
            }
        }
        if (ended) break;   // the scheduler would retire that slot here
    }

    // ── 4. A short batch is refused, not silently advanced ───────────────
    // One forward covers every row, so an omitted live slot would have its
    // backend-owned state advanced by whatever filler the engine substitutes.
    if (live.size() >= 2) {
        const std::vector<SeqEngine::StepInput> partial{{live[0], pending[0]}};
        std::vector<SeqEngine::StepOutput> outputs;
        require(!engine.step(partial, outputs),
                "step() must reject a batch that omits a live slot");
    }

    // ── 5. An empty batch is a no-op that still clears `outputs` ─────────
    {
        std::vector<SeqEngine::StepOutput> outputs(1);
        require(engine.step({}, outputs), "step() with no inputs must succeed");
        require(outputs.empty(), "step() must clear `outputs` before filling it");
    }

    // ── 6. retire() frees the slot and is safe to repeat ─────────────────
    {
        const int freed = live.front();
        engine.retire(freed);
        engine.retire(freed);   // must be a no-op on an already-free slot
        live.erase(live.begin());
        pending.erase(pending.begin());

        const SeqEngine::AdmitResult again =
            engine.admit(9002, prompt, greedy, n_gen);
        require(again.ok, "admit() must succeed after a retire() freed a slot");
        if (again.ok) engine.retire(again.slot);
    }

    // ── 7. Retiring everything returns the engine to idle ────────────────
    for (const int s : live) engine.retire(s);
    live.clear();
    {
        const SeqEngine::AdmitResult fresh =
            engine.admit(9003, prompt, greedy, n_gen);
        require(fresh.ok, "admit() must succeed once every slot has retired");
        if (fresh.ok) engine.retire(fresh.slot);
    }

    return violations;
}

}  // namespace dflash::common
