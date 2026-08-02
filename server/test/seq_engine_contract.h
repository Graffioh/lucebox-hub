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
// checker only ever calls slot_count/admit/step/prefill_pending/retire/
// token_is_eos, and it knows nothing about block tables, graphs, or recurrent
// state.
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
    require(engine.max_context() >= (int)prompt.size(),
            "max_context() must fit the contract-check prompt");
    if (engine.max_context() < (int)prompt.size()) return violations;

    std::vector<int>     live;      // decoding slot ids, in admission order
    std::vector<int32_t> pending;   // token to feed each decoding slot next
    std::vector<bool>    taken((size_t)n_slots, false);

    // Common shape checks for both prefill and decode outputs. `answered` is
    // supplied for a batched decode, where every slot must appear once.
    auto validate_output = [&](const SeqEngine::StepOutput & output,
                               std::vector<bool> * answered = nullptr) {
        if (output.slot < 0 || output.slot >= n_slots) {
            require(false, "step() returned an output for an unknown slot");
            return false;
        }
        if (answered) {
            require(!(*answered)[(size_t)output.slot],
                    "step() returned two outputs for one slot");
            if ((*answered)[(size_t)output.slot]) return false;
            (*answered)[(size_t)output.slot] = true;
        }
        if (!output.failed) {
            require(output.token >= 0,
                    "a non-failed step() output must carry a token");
        }
        return true;
    };

    // Advance the one pending prefill to completion. PR #1 deliberately sends
    // no decode inputs while doing this; the claim-only contract is final even
    // though the baseline scheduling policy remains blocking.
    auto complete_prefill = [&](int prefill_slot) {
        for (int iteration = 0; iteration < 1024; iteration++) {
            std::vector<SeqEngine::StepOutput> outputs;
            if (!engine.step({}, outputs)) {
                require(false, "step() must advance a pending prefill");
                return false;
            }

            bool completed = false;
            int32_t first_token = -1;
            for (const SeqEngine::StepOutput & o : outputs) {
                if (!validate_output(o)) continue;
                require(!o.failed,
                        "a contract-check prefill step must not fail a slot");
                if (o.failed) continue;
                if (o.prefill_done) {
                    require(o.slot == prefill_slot,
                            "prefill_done reported the wrong slot");
                    require(!completed,
                            "step() reported prefill completion twice");
                    completed = true;
                    first_token = o.token;
                    continue;
                }

                require(false,
                        "blocking prefill step returned a decode output");
            }

            if (completed) {
                require(!engine.prefill_pending(),
                        "prefill_pending() stayed true after completion");
                live.push_back(prefill_slot);
                pending.push_back(first_token);
                return true;
            }
            require(engine.prefill_pending(),
                    "prefill_pending() cleared before prefill_done");
        }
        require(false, "pending prefill did not complete");
        return false;
    };

    // ── 1. Admission is non-computing and completes through step() ───────
    SeqEngine::AdmitResult first =
        engine.admit(1, prompt, greedy);
    require(first.ok, "admit() must succeed on an idle engine");
    if (!first.ok) return violations;
    require(first.slot >= 0 && first.slot < n_slots,
            "admit() returned a slot outside [0, slot_count())");
    if (first.slot < 0 || first.slot >= n_slots) return violations;
    taken[(size_t)first.slot] = true;
    require(engine.prefill_pending(),
            "successful admit() must leave a prefill pending");

    // A single staging set permits only one admission prefill at a time.
    const SeqEngine::AdmitResult while_pending =
        engine.admit(9000, prompt, greedy);
    require(!while_pending.ok && while_pending.busy,
            "admit() while a prefill is pending must report busy=true");
    if (!complete_prefill(first.slot)) {
        engine.retire(first.slot);
        return violations;
    }

    // ── 2. Fill the remaining slots one claim-only admission at a time ───
    bool filled_all = true;
    for (int i = 1; i < n_slots; i++) {
        const SeqEngine::AdmitResult r = engine.admit(
            (uint64_t)(i + 1), prompt, (i % 2) ? seeded : greedy);
        if (!r.ok) {
            require(r.busy,
                    "a failed admit() with a free slot must report busy=true");
            filled_all = false;
            break;
        }
        require(r.slot >= 0 && r.slot < n_slots,
                "admit() returned a slot outside [0, slot_count())");
        if (r.slot < 0 || r.slot >= n_slots) {
            filled_all = false;
            break;
        }
        require(!taken[(size_t)r.slot], "admit() reused a slot that is live");
        if (taken[(size_t)r.slot]) {
            engine.retire(r.slot);
            for (const int slot : live) engine.retire(slot);
            return violations;
        }
        taken[(size_t)r.slot] = true;
        require(engine.prefill_pending(),
                "successful admit() must leave a prefill pending");
        if (!complete_prefill(r.slot)) {
            filled_all = false;
            break;
        }
    }
    require(!live.empty(), "no request completed admission");

    // ── 3. A full engine is busy, never a hard error ─────────────────────
    // The scheduler re-queues a busy job and retries it after the next
    // retire; a hard error fails a request that only needed to wait.
    if (filled_all) {
        const SeqEngine::AdmitResult over =
            engine.admit(9001, prompt, greedy);
        require(!over.ok, "admit() must fail when every slot is live");
        if (!over.ok) {
            require(over.busy,
                    "a full engine must report busy=true, not a hard error");
            require(over.slot < 0, "a failed admit() must not claim a slot");
        }
    }

    // ── 4. A batched decode step answers every live slot exactly once ────
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
            if (!validate_output(o, &answered)) {
                shape_ok = false;
                continue;
            }
            if (o.failed) continue;
            require(!o.prefill_done,
                    "decode-only step reported a prefill completion");
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

    // ── 5. A short batch is refused, not silently advanced ───────────────
    // One forward covers every row, so an omitted live slot would have its
    // backend-owned state advanced by whatever filler the engine substitutes.
    if (live.size() >= 2) {
        const std::vector<SeqEngine::StepInput> partial{{live[0], pending[0]}};
        std::vector<SeqEngine::StepOutput> outputs;
        require(!engine.step(partial, outputs),
                "step() must reject a batch that omits a live slot");
    }

    // ── 6. An empty batch cannot omit decoding slots ─────────────────────
    if (!live.empty()) {
        std::vector<SeqEngine::StepOutput> outputs(1);
        require(!engine.step({}, outputs),
                "step() must reject an empty batch while slots are decoding");
    }

    // ── 7. retire() frees decoding and prefilling slots safely ───────────
    {
        const int freed = live.front();
        engine.retire(freed);
        engine.retire(freed);   // must be a no-op on an already-free slot
        live.erase(live.begin());
        pending.erase(pending.begin());

        const SeqEngine::AdmitResult again =
            engine.admit(9002, prompt, greedy);
        require(again.ok, "admit() must succeed after a retire() freed a slot");
        if (again.ok) {
            require(engine.prefill_pending(),
                    "replacement admission must start a prefill");
            engine.retire(again.slot);
            require(!engine.prefill_pending(),
                    "retiring a prefilling slot must clear pending state");
            engine.retire(again.slot);
        }
    }

    // ── 8. Retiring everything returns the engine to idle ────────────────
    for (const int s : live) engine.retire(s);
    live.clear();
    {
        std::vector<SeqEngine::StepOutput> outputs(1);
        require(engine.step({}, outputs),
                "step() with no work must succeed");
        require(outputs.empty(),
                "idle step() must clear `outputs` before returning");
    }
    {
        const SeqEngine::AdmitResult fresh =
            engine.admit(9003, prompt, greedy);
        require(fresh.ok, "admit() must succeed once every slot has retired");
        if (fresh.ok) {
            engine.retire(fresh.slot);
            require(!engine.prefill_pending(),
                    "idle engine must not retain a pending prefill");
        }
    }

    return violations;
}

}  // namespace dflash::common
