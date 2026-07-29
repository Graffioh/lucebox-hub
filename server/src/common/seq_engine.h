// SeqEngine — concurrent serving over decode slots (iteration-level
// scheduling), the boundary between the HTTP scheduler and a backend that can
// hold several live sequences at once.
//
// A backend qualifies when it can keep several independent sequences in a
// paged KV cache and execute a batched decode step. Any additional
// per-sequence model state is owned by the concrete engine, not by this
// interface. The scheduler admits one request per slot (blocking prefill),
// then advances every live slot one token per step(), feeding each sampled
// token back as the next step's input — which is what lets the scheduler
// override a token (thinking-budget force-close) before it is committed to
// the cache.
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
#include <string>
#include <vector>

#include "sampler.h"

namespace dflash::common {

class SeqEngine {
public:
    virtual ~SeqEngine() = default;

    // Number of decode slots served concurrently. Fixed for the engine's
    // lifetime: the scheduler sizes its own per-slot array from this once and
    // then indexes it by the slot id admit() returns.
    virtual int slot_count() const = 0;

    struct AdmitResult {
        bool ok = false;
        // Full: no free slot or not enough free KV blocks — retrying after a
        // retire can succeed. Distinct from a hard error (ok=false,
        // busy=false), which never can.
        bool busy = false;
        int  slot = -1;
        // First generated token, sampled from the prefill logits. It is
        // PENDING: its KV row is written by the first step() that feeds it
        // back (mirrors the AR loop's first-token handoff).
        int32_t first_token = -1;
        double  prefill_s = 0.0;
        std::string error;
    };

    // Admit one request into a free slot: allocate its KV blocks (reserving
    // room for prompt + n_gen), run the full chunked prefill, and sample the
    // pending first token with the request's sampler. Blocking; decode of the
    // other slots pauses for the duration.
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
                              int n_gen) = 0;

    struct StepInput {
        int     slot  = -1;
        int32_t token = -1;   // token to commit at this slot's next position
    };
    struct StepOutput {
        int     slot   = -1;
        int32_t token  = -1;  // newly sampled token (pending until next step)
        bool    failed = false;
    };

    // One batched decode iteration: commit each input token to its slot's KV,
    // run one forward over all slots, sample one new token per live slot.
    // Every live slot must appear in `inputs`. The engine owns the mapping
    // from slots to batch rows and must not silently advance an active slot
    // with a dummy input. Returns false on a whole-batch failure (outputs may
    // be partial).
    virtual bool step(const std::vector<StepInput> & inputs,
                      std::vector<StepOutput> & outputs) = 0;

    // Release a slot's KV blocks and mark it free. Safe on failed slots.
    virtual void retire(int slot) = 0;

    // EOS check for scheduler-side stop decisions.
    virtual bool token_is_eos(int32_t token) const = 0;
};

}  // namespace dflash::common
