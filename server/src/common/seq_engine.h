// SeqEngine — concurrent serving over decode slots (iteration-level
// scheduling), the boundary between the HTTP scheduler and a backend that can
// hold several live sequences at once.
//
// A backend qualifies when it can keep several independent sequences in a
// paged KV cache and execute a batched decode step. Any additional
// per-sequence model state is owned by the concrete engine, not by this
// interface. admit() claims a slot and queues its prompt without compute.
// Each step() then advances that prefill by one chunk alongside the live
// decode batch. Once prefilling completes, the scheduler advances the slot
// one token per step(), feeding each sampled token back as the next step's
// input — which is what lets it override a token (thinking-budget
// force-close) before it is committed to the cache.
//
// The split of duties is deliberate and is the reason this interface exists
// apart from ModelBackend:
//   scheduler   policy — who gets admitted, when a slot stops, what reaches
//               the client, how a slow reader is handled
//   engine      mechanism — KV blocks, backend-owned per-sequence state, the
//               batched forward, sampling
// Nothing above this interface knows about block tables, and nothing below it
// knows about sockets.
//
// Threading contract: every call comes from ONE thread — the same one that
// calls ModelBackend::generate() — so implementations need no locking.
//
// ── Adding a backend ────────────────────────────────────────────────────
// Implement this interface next to the model (qwen35/qwen35_seq_engine.* is
// the worked example) and return it from ModelBackend::seq_engine(). Nothing
// else in the server changes: scheduler_loop() takes over the worker thread
// as soon as seq_engine() returns non-null.
//
// The model-independent half already exists in common/ and is meant to be
// reused as-is: PagedKvPool (block allocator) and SeqSlotManager (slot
// lifecycle, admission arithmetic, per-slot sampler/RNG/penalty history, the
// kv-length mirror) are GPU-free and unit-tested. A new engine supplies only
// the device half — its prefill, its batched forward, its metadata uploads.
//
// What must never reach this interface, or anything above it: block tables,
// graph shapes, and per-sequence model state (recurrent/SSM/conv tensors,
// cache layout). A model that seems to need SeqEngine widened to serve
// concurrently is the signal that the split has slipped — the model-specific
// part belongs inside the engine, not in the contract.
//
// test/seq_engine_contract.h drives an engine through exactly the call
// sequence the scheduler makes and returns the violations it finds; run a new
// engine through it before wiring it up.

#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "sampler.h"
#include "seq_admission.h"

namespace dflash::common {

class SeqEngine {
public:
    virtual ~SeqEngine() = default;

    // Number of decode slots served concurrently. Fixed for the engine's
    // lifetime: the scheduler sizes its own per-slot array from this once and
    // then indexes it by the slot id admit() returns.
    virtual int slot_count() const = 0;
    // Per-sequence logical context bound. The scheduler owns generation
    // policy and clamps its output cap against this value after admission.
    virtual int max_context() const = 0;

    using AdmitResult = SeqAdmissionResult;

    // Sampling state that must survive recompute preemption. Model K/V and
    // recurrent state are rebuilt from the expanded prompt; the RNG position
    // cannot be reconstructed from that prompt alone. sample_history is the
    // scheduler's to fill: penalties must cover every emitted token, and only
    // the scheduler knows the sampled token still pending in its step loop.
    struct ResumeState {
        std::vector<int32_t> sample_history;
        std::mt19937_64 rng{0x9E3779B97F4A7C15ull};
    };

    // Admit one request into a free slot and queue its prompt for chunked
    // prefill. No K/V blocks or compute are consumed here — subsequent step()
    // calls allocate and advance one prefill chunk alongside the decode batch.
    // At most one admission may prefill at a time.
    //
    // `sampler` is the only source of truth for how the slot samples:
    // sampler.needs_logit_processing() selects CPU sampling over GPU argmax
    // AND decides whether sampler.seed is honoured. There is deliberately no
    // separate do_sample flag — a second copy of that one fact is something
    // an engine can disagree with, and the failure mode is silent (a seeded
    // request sampling nondeterministically, with no error anywhere).
    virtual AdmitResult admit(uint64_t request_id,
                              const std::vector<int32_t> & prompt,
                              const SamplerCfg & sampler,
                              const ResumeState * resume = nullptr) = 0;

    struct StepInput {
        int     slot  = -1;
        int32_t token = -1;   // token to commit at this slot's next position
    };
    struct StepOutput {
        int     slot   = -1;
        int32_t token  = -1;  // newly sampled token (pending until next step)
        bool    failed = false;
        // No input was committed: the scheduler must free a victim and retry
        // this exact batch. This is a pressure signal, not a request failure.
        bool    pool_exhausted = false;
        // Present when failed=true so the scheduler can report an honest
        // per-request error instead of silently truncating generation.
        std::string error;
        // This step completed the slot's admission prefill. `token` is the
        // request's first sampled token, still pending like every decode
        // output.
        bool    prefill_done = false;
    };

    // One scheduler iteration: commit each input token and advance every
    // decoding slot, while also advancing the pending admission's prefill by
    // one chunk when present. Every decoding slot must appear in `inputs`;
    // the prefilling slot must not. A completed prefill contributes one extra
    // output with prefill_done=true. On batch allocation pressure, no input is
    // mutated and one pool_exhausted output identifies the blocked decode
    // slot. Returns false on a whole-batch failure (outputs may be partial).
    virtual bool step(const std::vector<StepInput> & inputs,
                      std::vector<StepOutput> & outputs) = 0;

    virtual bool prefill_pending() const = 0;

    // Copy the host-only RNG position before recompute preemption. Leaves
    // out.sample_history untouched — see ResumeState.
    virtual bool capture_resume_state(int slot, ResumeState & out) const = 0;

    // Release a slot's KV blocks and mark it free. Safe on failed slots.
    virtual void retire(int slot) = 0;

    // EOS check for scheduler-side stop decisions.
    virtual bool token_is_eos(int32_t token) const = 0;
};

}  // namespace dflash::common
