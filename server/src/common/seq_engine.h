// SeqEngine — concurrent serving over decode slots (iteration-level
// scheduling), the boundary between the HTTP scheduler and a backend that can
// hold several live sequences at once.
//
// A backend qualifies when it can keep several independent sequences in a
// paged KV cache and execute a batched decode step. Any additional
// per-sequence model state is owned by the concrete engine, not by this
// interface. admit() claims a slot and queues its prompt without compute.
// Each step() then advances a scheduler-selected cohort of prompt slices
// alongside the complete live decode batch. Once a prefill completes, the
// scheduler advances that slot one token per step(), feeding each sampled
// token back as the next step's input — which is what lets it override a token
// (thinking-budget force-close) before it is committed to the cache.
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
#include "seq_admission.h"
#include "seq_batch_plan.h"

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

    // Admit one request into a free slot and queue its prompt for chunked
    // prefill. No model compute is performed here. Implementations may reserve
    // persistent capacity atomically so that every admitted prompt can finish;
    // subsequent step() calls advance scheduler-selected prompt slices.
    //
    // `sampler` is the only source of truth for how the slot samples:
    // sampler.needs_logit_processing() selects CPU sampling over GPU argmax
    // AND decides whether sampler.seed is honoured. There is deliberately no
    // separate do_sample flag — a second copy of that one fact is something
    // an engine can disagree with, and the failure mode is silent (a seeded
    // request sampling nondeterministically, with no error anywhere).
    virtual AdmitResult admit(uint64_t request_id,
                              const std::vector<int32_t> & prompt,
                              const SamplerCfg & sampler) = 0;

    struct StepInput {
        int     slot  = -1;
        int32_t token = -1;   // token to commit at this slot's next position
    };
    struct StepOutput {
        int     slot   = -1;
        int32_t token  = -1;  // newly sampled token (pending until next step)
        bool    failed = false;
        // Present when failed=true so the scheduler can report an honest
        // per-request error instead of silently truncating generation.
        std::string error;
        // This step completed the slot's admission prefill. `token` is the
        // request's first sampled token, still pending like every decode
        // output.
        bool    prefill_done = false;
    };

    // One scheduler iteration owns both kinds of logical work. `decode` must
    // contain every currently decoding slot exactly once; `prefills` is the
    // bounded subset of pending prompt work selected by scheduler policy.
    // Device graph shapes, staging indices, cache blocks, and model state stay
    // behind the engine boundary.
    struct StepPlan {
        std::vector<StepInput>    decode;
        std::vector<PrefillSlice> prefills;
    };

    struct PrefillProgress {
        int slot   = -1;
        int tokens = 0;
    };

    struct StepResult {
        enum class Status {
            idle,
            progressed,
            resource_blocked,
            failed,
        };

        Status                       status = Status::idle;
        std::vector<StepOutput>      outputs;
        std::vector<PrefillProgress> prefill_progress;
        // Whole-step diagnostic. A failed result must explain the terminal
        // validation/execution error; resource_blocked may explain which
        // retryable resource prevented an otherwise valid plan from running.
        // A failed result carries no usable outputs/progress. Validation
        // failures occur before mutation; a device/build/compute failure may
        // leave backend state partially advanced, so the caller must retire
        // every live sequence before invoking step() again. Only
        // resource_blocked promises an atomic, unchanged state.
        std::string error;

        bool ok() const { return status != Status::failed; }
        bool made_progress() const { return status == Status::progressed; }
    };

    // Hard per-step capabilities at the requested live decode width. Global
    // token budgets and fairness remain scheduler policy; an engine may
    // advertise different sequence, per-sequence, and total-token limits for
    // idle, mixed, or larger decode buckets.
    virtual StepPlanLimits step_plan_limits(int decode_rows) const = 0;

    // A valid progressed result returns one output for every decode input.
    // Every selected prefill returns positive PrefillProgress (never exceeding
    // its slice), unless it instead terminates with a failed prefill_done
    // output. A successful completion contributes both progress and an output
    // with prefill_done=true. Invalid plans return failed without advancing
    // state. Runtime failures are terminal for the
    // live cohort and may follow partial backend mutation, but expose no
    // consumable payload. resource_blocked is retryable, atomic, and likewise
    // reports no partial progress.
    virtual StepResult step(const StepPlan & plan) = 0;

    virtual bool prefill_pending() const = 0;

    // Release a slot's KV blocks and mark it free. Safe on failed slots.
    virtual void retire(int slot) = 0;

    // EOS check for scheduler-side stop decisions.
    virtual bool token_is_eos(int32_t token) const = 0;
};

}  // namespace dflash::common
