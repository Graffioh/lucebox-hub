# LuceEngine component design

Status: Draft

Source snapshot: `upstream/main` at `298031aa`

## Summary

Lucebox currently exposes two local generation paths to `HttpServer`:

- A worker runs a complete request through `ModelBackend::generate()`.
- A scheduler advances concurrent requests through `SeqEngine`.

This split let continuous batching land without rewriting every model family.
It now makes the HTTP layer responsible for model scheduling and gives
request-level generation two unrelated shapes.

The target has one request boundary:

```text
HTTP or daemon adapter
  -> LuceEngine::generate()
       -> request lifecycle and scheduling
       -> model execution capability
       -> model-family state
       -> GGML
  <- token events and one terminal result
  -> protocol formatting
  -> client
```

The boundary is structural. C++ access modifiers do not define it.
`LuceEngine` owns request scheduling, model access, cancellation, and
completion. `HttpServer` owns HTTP and sockets. Model-family code owns weights,
caches, graphs, and GGML objects.

There is one request-level function named `generate()`. The final design does
not retain `ModelBackend::generate()` below `LuceEngine::generate()`, and it
does not present `ModelBackend` and `SeqEngine` as peer engines.

## Goals

- Give HTTP and daemon generation one operation with one lifecycle.
- Keep scheduling mode out of request callers.
- Define each component by the decisions and state it owns.
- Keep HTTP, JSON, SSE, and socket types out of engine and model execution.
- Keep GGML, graph, cache-layout, and device types out of HTTP and engine
  request types.
- Preserve the concurrency invariants introduced by PR #594.
- Preserve serial generation behavior, including speculative empty-output
  retry, snapshot restore, cache behavior, and cancellation.
- Make request data safe to retain after `generate()` returns.
- Give shutdown and model control one serialization point.
- Migrate in steps that leave one usable request path after each change.

## Non-goals

- Copying the multi-process topology of vLLM or SGLang.
- Moving normal local generation behind IPC.
- Making every model family implement continuous batching.
- Forcing complete-request and batch-step execution into the same low-level
  method.
- Passing `ParsedRequest` or `ServerConfig` into model-family code.
- Adding a request-wide `GenerationPlan` type.
- Renaming existing types without moving ownership.
- Splitting every optional backend capability before its callers move.

## Why the current split exists

`ModelBackend` and `SeqEngine` were created for different callers and at
different times.

Commit `3d1dcad85` introduced `ModelBackend` as the shared adapter for the
line-oriented daemon. Its whole-request operation matched the daemon command:
prefill, decode, stream tokens, and return a result. The same interface also
collected park, snapshot, compression, and command hooks.

Commits `35784218` and `9daa25c2`, followed by
[PR #594](https://github.com/Luce-Org/lucebox/pull/594), added concurrent Qwen
serving in two parts. `SeqEngine` first defined model-side slot execution.
`scheduler_loop()` then added admission, fair prefill, cancellation, and
non-blocking client delivery. `HttpServer::run()` selected the scheduler only
when a backend returned a sequence engine.

That history explains the asymmetry. It does not require the HTTP server to
keep choosing between the two paths.

The useful boundary from PR #594 remains:

- Request coordination owns admission order, fairness, stop decisions,
  cancellation, client progress, and retirement.
- Model execution owns slot allocation, KV blocks, recurrent state, graph
  shapes, batched forward execution, and sampling.

The new structure moves request coordination into `LuceEngine`. It does not
move it into GGML or model-family code.

## Current flow

### Startup

`server_main.cpp` constructs a backend, a tokenizer, and `HttpServer`.
`HttpServer::run()` then checks `ModelBackend::seq_engine()` once:

```text
seq_engine() == nullptr  -> worker_loop()
seq_engine() != nullptr  -> scheduler_loop()
```

The upstream proxy forces the worker path even when the backend has a
`SeqEngine`.

### Complete-request execution

```text
HttpServer::process_job()
  -> prompt preparation and cache selection
  -> GenerateRequest
  -> ModelBackend::generate() or restore_and_generate()
  -> model-specific prefill and decode
  -> GGML graph execution
  -> DaemonIO token callbacks
  -> GenerateResult
  -> cache and response finalization
```

`process_job()` owns HTTP state and model-serving policy in one function. It
coordinates PFlash, FlowKV, prefix snapshots, draft residency, generation
limits, status, token delivery, and response completion.

### Batch execution

```text
HttpServer::scheduler_loop()
  -> SchedSlot
  -> SeqEngine::admit()
  -> build StepPlan
  -> SeqEngine::step()
  -> apply token and stop policy
  -> buffer client output
  -> SeqEngine::retire()
```

`SchedSlot` contains two kinds of state. Request progress, slot identity, token
history, and cancellation belong to generation. Sockets, `SseEmitter`, and
`ClientSendBuffer` belong to HTTP response delivery.

`Qwen35SeqEngine` is not a peer of `Qwen35Backend`. The backend owns the
sequence engine, paged KV pool, weights, graphs, and GGML backends. The
sequence engine borrows those resources to run batched steps.

## Target ownership

### Composition and lifetime

The configuration design in
[PR #688](https://github.com/Luce-Org/lucebox/pull/688) and its implementation
follow-up own launch-time input, validation, and model construction. The engine
design begins after backend construction succeeds:

```text
BackendArgs + BackendAdmissionContext
  -> BackendPlan
  -> create_backend(plan)
  -> BackendRuntime
  -> LuceEngine
  -> HttpServer
```

`BackendPlan` remains launch-time data. It is not a request or scheduler plan.
`BackendRuntime` owns the validated plan and the model-family resources whose
configuration points into it.

`LuceEngine` takes ownership of `BackendRuntime`. `HttpServer` borrows
`LuceEngine` for the duration of server execution. This ownership gives the
shutdown order a concrete shape:

1. `HttpServer` stops accepting work and cancels its live generations.
2. `LuceEngine` rejects new requests, completes or retires active work, and
   joins its execution thread.
3. `BackendRuntime` releases model and device resources as part of
   `LuceEngine` destruction.

`ServerConfig` remains the HTTP and request-default configuration. Construct
`LuceEngine` with only scheduler, cache, and output-channel values that the
engine owns. Do not pass the full `ServerConfig` into backend preparation,
backend construction, or model code.

### Caller view

HTTP prepares an owned model request and receives a generation handle:

```cpp
GenerateRequest request = make_generate_request(parsed_request);
Generation generation = engine.generate(std::move(request));

for (;;) {
    GenerateEvent event = generation.next();
    if (const auto * tokens = std::get_if<TokenBatch>(&event)) {
        response.write(tokens->tokens);
        continue;
    }
    if (const auto * progress = std::get_if<GenerationProgress>(&event)) {
        status.update(*progress);
        continue;
    }

    response.finish(std::get<GenerateCompleted>(event).result);
    break;
}
```

The daemon uses the same operation and maps events to its descriptor protocol.
Neither caller checks concurrency, engine slots, or model type.

Upstream forwarding remains an HTTP transport operation. It does not pretend
to be local generation.

### Request and completion

The existing names remain where their meaning still fits:

```cpp
struct GenerateRequest {
    std::vector<int32_t> prompt;
    int n_gen;
    SamplerCfg sampler;
    BudgetHook budget_hook;
    std::vector<std::string> stop_sequences;
    std::optional<SnapshotId> restore_from;
    std::optional<SnapshotCapture> capture;

    // These become owned values instead of pointers into an HTTP stack frame.
    std::vector<int32_t> hint_tokens;
    std::vector<int32_t> stall_tool_prefix_tokens;
    std::vector<int32_t> stall_action_suffix_tokens;
    std::vector<int32_t> stall_skip_tokens;
};

struct TokenBatch {
    std::vector<int32_t> tokens;
};

struct GenerationProgress {
    GenerationPhase phase;
    int processed_tokens;
};

struct GenerateCompleted {
    GenerateResult result;
};

using GenerateEvent =
    std::variant<TokenBatch, GenerationProgress, GenerateCompleted>;
```

`GenerateRequest` contains model-ready, transport-free input. It does not
contain a file descriptor, JSON value, HTTP format, response identifier, or
socket callback.

The request owns normalized stop strings. `LuceEngine` receives the model's
token decoder as a construction dependency and keeps one incremental matcher
per request. A stop string may cross token boundaries. It is not reduced to a
list of independently tokenized suffixes.

The first migration keeps the existing complete token vector in
`GenerateResult`. The generation operation owns that vector. Token events are
bounded in-flight copies used for streaming. This preserves daemon, cache, and
telemetry callers while making the memory cost explicit.

### `Generation`

```cpp
class Generation {
public:
    GenerateEvent next();
    void cancel(GenerationCancelReason reason);
    ~Generation();
};

class LuceEngine {
public:
    LuceEngine(std::unique_ptr<BackendRuntime> runtime,
               const TokenDecoder & decoder,
               LuceEngineConfig config);

    Generation generate(GenerateRequest request);
    ControlResult control(ControlCommand command);
    void shutdown();
};
```

`Generation` is move-only. It owns a reference to channel and terminal state.
Dropping an unfinished handle cancels its request.

Each request has a bounded token channel and a separate terminal cell. Model
execution never waits for a socket. When a consumer stops draining and the
channel fills, `LuceEngine` cancels only that request with a typed output
backpressure result. The terminal cell remains writable even when the token
channel is full.

The request queue is bounded. When the queue is full, or shutdown has started,
`generate()` returns an already-completed `Generation` with a typed rejection.
It does not block or throw. HTTP calls `generate()` before committing streaming
headers, so it can map an immediate rejection to a normal HTTP status.

`shutdown()` is idempotent. It wakes every waiting `Generation`, publishes one
terminal result for each request, retires model state on the engine thread,
and joins before returning. A `Generation` handle may outlive `LuceEngine`, but
after shutdown it contains only detached channel and terminal state. It cannot
retain a backend, executor, observer, or engine pointer.

### `LuceEngine`

`LuceEngine` owns:

- The bounded request queue and engine thread.
- Request IDs and operation state.
- Selection of serial or continuous-batch coordination at construction.
- Admission order, fair prefill selection, generation limits, cancellation,
  retry decisions, and retirement.
- Incremental stop matching and thinking-budget token decisions before the
  token returns to model KV.
- Typed terminal completion and output-channel backpressure.
- Prefix-cache policy and coordination with physical snapshots.
- Typed model controls serialized with live generation.
- Backend lifetime and orderly shutdown.

`LuceEngine` does not own HTTP routes, API response formats, sockets, GGML
graphs, block tables, or architecture-specific cache layouts.

The serial and batch coordinators are implementation components inside this
ownership boundary. They are not alternate entry points. Both consume the
same engine request state and publish through the same `Generation` channel.

### Model execution capabilities

Complete-request and batch-step execution remain different below
`LuceEngine`. Their contracts use names on the same axis:

```cpp
class SingleRequestExecutor {
public:
    virtual GenerateResult execute(
        const GenerateRequest & request,
        ExecutionHooks & hooks) = 0;
};

class BatchExecutor {
public:
    virtual SlotClaim claim(const GenerateRequest & request) = 0;
    virtual StepPlanLimits step_limits(int decode_rows) const = 0;
    virtual BatchStepResult step(const BatchStepPlan & plan) = 0;
    virtual void retire(SlotId slot) = 0;
    virtual bool token_is_eos(int32_t token) const = 0;
};
```

`ExecutionHooks` is transport-free. It supplies cancellation, progress, and a
synchronous pre-commit token decision. For each sampled token, the
single-request executor asks the hook whether to accept, replace, or stop
before it writes that token to reusable KV state. Accepted tokens then enter
the bounded `Generation` channel. Progress reports are model-neutral and may
be coalesced when a consumer falls behind.

The batch coordinator performs the same decision directly because
`BatchExecutor::step()` already returns pending tokens before the next step
commits them. This keeps one owner for stop matching and force-close policy
without making a complete-request backend expose a batch-step lifecycle.

A model backend supplies exactly one execution capability. The type selected
at construction is fixed for the engine lifetime.

`ModelBackend::generate()` does not exist in the final structure. Existing
`generate_impl()` bodies move to `SingleRequestExecutor::execute()`. Existing
`SeqEngine` implementations move to `BatchExecutor` after the scheduler sits
behind `LuceEngine`.

This is not a universal low-level engine interface. It records the two real
model execution units without making HTTP understand either one.

### Model-family and GGML boundary

`BackendRuntime` and the model-family backend own:

- Loaded weights and model configuration.
- Target and draft GGML backends.
- KV, recurrent, and speculative state.
- Graph construction and tensor binding.
- Model-specific prefill, decode, verification, and sampling.
- Scratch buffers and device placement.
- Physical snapshot capture and restore.
- Role-specific IPC clients.

No `ggml_*` type, graph tensor, allocator, block table, or model-specific state
crosses into `LuceEngine`, `Generation`, or `HttpServer`.

`ModelBackend` remains the migration name for this resource owner. Do not
rename it until generation and daemon concerns have moved out and its final
responsibility can be judged from the remaining code.

### Stop conditions and token commitment

Stop conditions that affect model progress belong to generation policy, not
wire formatting. `LuceEngine` owns EOS, generation limits, normalized stop
sequences, cancellation, and thinking-budget force-close.

The engine must decide whether to continue before a sampled token is fed back
into KV. This preserves the current batch invariant that a force-close or stop
decision can replace or reject a pending token before the next step commits
it. API-specific reasoning, tool-call, and event formatting remain in
`SseEmitter` or a later transport-neutral semantic response component.

The incremental stop matcher and budget hook live in engine request state. The
single-request executor calls `ExecutionHooks` before commit. The batch
coordinator applies the same decision to each pending token before it builds
the next step. HTTP never decides whether model execution continues. It only
formats token events already accepted by generation policy.

### Retry ownership

`LuceEngine` owns the common empty speculative-output decision now implemented
by `ModelBackend::generate()` and `restore_and_generate()`.

The single-request coordinator preserves the current retry and result-merge
behavior. The token policy withholds suppressed EOS-only output, so an attempt
marked `empty_visible_output` cannot leak token events before the retry. The
coordinator retries once with autoregressive decode and merges timing,
restored-prefix, speculation, budget-close, and degeneration metadata exactly
as the current wrapper does.

A batch retry is allowed only when the executor reports that no visible token
was published and the attempt can be reset without reusing partially mutated
state. Otherwise the engine returns a typed failure. Retiring and re-admitting
a slot is not sufficient proof of rollback.

### Cache and snapshot boundary

Prefix policy and physical model state have different owners:

```text
LuceEngine cache component
  -> prefix keys, selection, eviction, request association
  -> opaque SnapshotId and backend-neutral metadata

model-family snapshot component
  -> physical images, release, import, and export
  -> GGML buffers and architecture-specific representation
```

The model backend supplies a snapshot capability beside its execution
capability:

```cpp
class SnapshotStore {
public:
    virtual SnapshotResult release(SnapshotId id) = 0;
    virtual SnapshotExport export_snapshot(SnapshotId id) = 0;
    virtual SnapshotResult import_snapshot(SnapshotImport image) = 0;
};
```

`LuceEngine` is the only coordinator of this capability and invokes it at an
engine safe point. The selected execution capability applies snapshots to its
own state. `SingleRequestExecutor::execute()` handles `restore_from` before
prefill. `BatchExecutor::claim()` binds `restore_from` to the claimed slot as
one atomic admission operation. A batch executor that cannot restore returns a
typed unsupported result without claiming a slot.

`GenerateRequest::capture` asks the selected executor to capture its request
state at the specified position and return an opaque `SnapshotId`. The
executor and the model-family snapshot component share the physical
representation below the engine boundary. Unsupported operations never fall
back to a second generation path.

`DiskPrefixCache` currently reads `ModelBackend::SnapshotRef`, including raw
GGML handles. The target replaces that crossing with an owned snapshot
transaction or stream. A failed import must leave no adopted partial state.

Model-cache validity must depend on engine facts such as accepted tokens and
terminal state. API or socket visibility must not enter model execution.
Client-visible conversation memory can remain a response-layer concern when
its validity depends on what reached the client.

### Control and daemon boundary

Park, unpark, explicit snapshots, compression, bootstrap, and daemon-specific
commands are control operations. They do not become `GenerateRequest` modes or
extra `generate()` functions.

The daemon adapter parses its line protocol into typed commands. `LuceEngine`
serializes those commands through the engine thread so they cannot race model
execution. Each command declares one lifecycle rule: reject while busy, queue
behind active work, drain active work, or cancel active work. Model-family code
implements the mechanism after the engine establishes the safe state.

`DaemonIO`, string commands, and file descriptors stop at the daemon adapter.
Model execution receives `ExecutionHooks` and typed control values. Model
progress enters those hooks and becomes coalesced
`GenerationProgress` events. This preserves live status without passing a
daemon or HTTP observer into model-family code.

### Response and transport boundary

`HttpServer` owns:

- HTTP framing, routes, and request validation.
- API aliases, chat rendering, and tokenization.
- Sockets, disconnect detection, CORS, and heartbeats.
- SSE and non-streaming response formatting.
- Translation of typed engine failures into HTTP or stream errors.

Each client thread consumes its `Generation` and remains the only code that
writes that socket. This removes `SocketHandle`, `ServerJob`, `SseEmitter`, and
`ClientSendBuffer` from engine scheduler state.

Streaming headers may already be committed when generation fails. The HTTP
adapter maps the same typed terminal error to either a normal HTTP error or the
matching stream error based on its own connection state. This keeps the
terminal boundary in
[PR #689](https://github.com/Luce-Org/lucebox/pull/689) intact.

Keep the name `SseEmitter` until semantic accumulation is physically moved out
of it. A later extraction should separate response meaning from SSE framing,
but that change is not required to introduce `LuceEngine`.

### IPC boundary

`BackendIpcProcess` remains the owner of process launch, pipes, shared payload
setup, framing, and shutdown. Role-specific clients continue to own their
payload protocols.

IPC stays below the model execution capability. `LuceEngine` receives typed
execution or control outcomes. It does not receive pipe descriptors or decode
subprocess strings.

## Final dependency direction

```text
server_main
  -> BackendPlan
  -> BackendRuntime
  -> LuceEngine
       -> request coordinator
            -> SingleRequestExecutor
            or BatchExecutor
                 -> model-family resources
                 -> optional role-specific IPC
                 -> GGML
  -> HttpServer
       -> Generation
       -> response formatting
       -> client socket
```

The model backend and batch executor are not alternative top-level paths.
`BackendRuntime` owns the model resources. `LuceEngine` owns one request
lifecycle and uses the execution capability supplied by those resources.

## Source organization

The target directory structure follows owned state:

| Path | Responsibility |
|---|---|
| `server/src/engine/luce_engine.{h,cpp}` | Request submission, `Generation`, lifecycle, cancellation, and shutdown |
| `server/src/engine/coordinator.{h,cpp}` | Serial and continuous-batch request coordination and shared policy |
| `server/src/engine/generation_channel.h` | Bounded token delivery and one terminal result |
| `server/src/engine/stop_matcher.{h,cpp}` | Incremental text stops and pre-commit token decisions |
| `server/src/engine/prefix_cache.{h,cpp}` | Prefix policy and opaque snapshot association |
| `server/src/common/generation_executor.h` | `SingleRequestExecutor` and `BatchExecutor` contracts |
| `server/src/server/http_server.{h,cpp}` | Listener, client lifetime, routes, and HTTP framing |
| `server/src/server/response_writer.{h,cpp}` | `SseEmitter`, JSON responses, and typed error mapping |
| `server/src/common/daemon_loop.{h,cpp}` | Legacy command parsing and protocol output during migration |
| `server/src/<model-family>/` | Execution capability and physical model state |

Do not move files only to match this table. Move a file when the state and
decisions listed here move with it.

## Migration

### 1. Add owned request and generation output

Make `GenerateRequest` own its optional token vectors. Add `Generation` with a
bounded request queue, bounded token channel, typed terminal result, idempotent
cancellation, immediate rejection, and shutdown tests.

### 2. Put both current loops behind `LuceEngine`

Move the job queue and worker ownership out of `HttpServer`. Wrap the current
worker and scheduler without changing model mechanics. Switch HTTP and daemon
generation callers to `LuceEngine::generate()`.

At this point callers have one request operation, but transitional lower
interfaces still exist.

### 3. Split scheduler state from transport state

Move request IDs, admission, prefill progress, pending tokens, cancellation,
generation limits, retry state, and retirement into engine-owned state. Keep
sockets, `SseEmitter`, heartbeats, and wire formatting in HTTP-owned state.

Delete `ClientSendBuffer` from scheduler state after client threads consume
bounded `Generation` channels directly.

### 4. Rename lower execution operations by role

Move existing whole-request implementations to
`SingleRequestExecutor::execute()`. Move `SeqEngine` implementations to
`BatchExecutor` only after the scheduler no longer lives in `HttpServer`.

Update HTTP support runs, cache staging, and daemon callers in the same wave.
Delete `ModelBackend::generate()`, `restore_and_generate()`, `generate_impl()`,
and `seq_engine()` when their final callers migrate.

### 5. Move shared generation policy once

Move generation limits, stop decisions, thinking-budget substitution,
speculative empty-output retry, cancellation reasons, and terminal completion
into `LuceEngine`. Preserve the current serial behavior with focused tests.
Add `ExecutionHooks` before moving the serial stop and force-close decisions.

### 6. Move cache and control ownership

Separate prefix policy from physical snapshots. Replace raw snapshot handles
with `SnapshotStore` and an owned transaction. Parse daemon commands at the
adapter and serialize typed controls through `LuceEngine::control()`.

### 7. Delete compatibility structure

Remove the HTTP execution-mode branch, the old worker and scheduler entry
points, `DaemonIO` from HTTP generation, transport fields in scheduler slots,
and temporary adapters. The migration is incomplete while any caller can
bypass `LuceEngine::generate()` for local generation.

## Required invariants

The migration must preserve these facts from PR #594:

- One engine thread is the only caller that mutates model execution state.
- Admission claims capacity without running model compute.
- Request and slot identity remain stable until retirement.
- Each decode step covers every selected live row exactly once.
- Prefill work is bounded, fair, and may progress beside active decode.
- A pending token may be substituted before the executor commits it to KV.
- Each selected row returns one explicit outcome.
- A possibly mutating step failure retires the affected cohort before another
  step.
- Temporary capacity pressure preserves FIFO order.
- Permanent capacity failure returns a typed terminal result.
- A slow or disconnected consumer cannot block progress for other requests.
- Retirement releases all model-owned request state and is safe after failure.

The migration must also preserve these complete-request facts:

- Empty speculative output retries through autoregressive decode exactly once.
- Restored-prefix and timing metadata survive retry.
- Cancellation is checked during prefill and decode.
- Prefix restore and cache staging cannot race live model work.
- Draft residency changes occur only at an engine safe point.
- Streaming and non-streaming calls receive one terminal success or failure.

## Verification

Each implementation step must include checks at the boundary it changes:

- Model-free tests for `Generation` ownership, cancellation, bounded output,
  immediate overload rejection, terminal delivery, shutdown, and a handle
  destroyed after its engine.
- Existing `SeqEngine` contract tests against `BatchExecutor` after migration.
- Serial empty-output retry and result-merge tests.
- Streaming and non-streaming success and failure for every API format.
- Disconnects during admission, prefill, decode, and final output.
- A slow client while other batch slots continue.
- Stop and force-close decisions before the next KV commit.
- Cohort retirement after a partially mutating batch failure.
- Prefix restore, disk snapshot failure cleanup, and cache staging.
- Park, unpark, and shutdown while requests are queued or active.
- Coalesced progress delivery without an HTTP or daemon observer below the
  engine boundary.
- Static dependency checks that keep HTTP types out of engine and model code,
  and GGML types out of HTTP and engine request headers.

## Lessons from other engines

The useful comparison is component ownership, not process count.

- [vLLM](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)
  keeps HTTP input and output processing separate from an engine core that
  owns scheduling, KV management, and worker coordination. Lucebox can use the
  ownership split without adopting ZMQ or one worker process per GPU.
- [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py)
  uses distinct typed messages between API, tokenization, scheduling, and
  model execution. Lucebox needs fewer stages, but it benefits from the same
  rule that wire objects stop at the API boundary.
- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md)
  separates HTTP routes, task queues, inference slots, task results, and
  response readers. Lucebox differs because several model families still own
  complete-request execution, so the lower capability boundary must admit
  both honest execution shapes.

## Alternatives rejected

### Keep both paths under a facade

`LuceEngine::generate()` calling another request-level
`ModelBackend::generate()` would add a pass-through layer. It would hide the
name conflict without moving retry, cache, cancellation, or lifecycle
ownership.

### Force every backend into `admit()`, `step()`, and `retire()`

A single step protocol looks uniform, but it makes every complete-request
backend expose resumable state only to satisfy the abstraction. The target
keeps one request interface and two honest model execution capabilities.

### Rename `ModelBackend` to `LuceEngine`

The current type owns daemon commands, resource controls, complete-request
execution, snapshots, and optional batch execution. Renaming it would not move
the HTTP scheduler or split its responsibilities.

### Keep scheduling in `HttpServer`

This would preserve duplicate request state, cancellation, shutdown, and
completion behavior. HTTP would still depend on model execution mode.

### Put sockets in `LuceEngine`

Socket ownership would couple scheduling to HTTP and retain separate blocking
and non-blocking send paths. The existing client thread can consume a bounded
`Generation` without exposing transport to the engine.

### Add mandatory IPC

An in-process boundary is sufficient. IPC adds serialization and subprocess
lifecycle but does not improve source ownership.

## Synthesis decision

Two target shapes were compared.

The selected design keeps `SingleRequestExecutor` and `BatchExecutor` as
separate model capabilities under one `LuceEngine`. This matches the existing
model state machines and avoids making serial backends imitate continuous
batching.

The rejected design used one scheduler and one `admit/step/retire` contract for
every model, with serial represented as capacity one. Its final diagram was
smaller, but the migration required every model family to become resumable and
moved model differences into a wider shared step protocol.

The selected design takes four details from that alternative: an explicit
split of `SchedSlot` state, a bounded generation channel consumed by client
threads, the PR #594 invariant checklist, and a migration that first moves
both loops behind `LuceEngine` before renaming lower contracts.

## Tradeoffs

- We accept two model-execution capabilities in exchange for one honest
  request lifecycle and no fake universal step method.
- We accept one engine thread and per-request channels in serial mode in
  exchange for the same cancellation, control, output, and shutdown behavior
  for every model.
- We accept owned copies of optional request token vectors in exchange for
  safe asynchronous lifetime.
- We cancel a request whose output channel remains full so one client cannot
  stop shared model progress.
- We move backend ownership into `LuceEngine` so model controls cannot bypass
  the engine's single-writer rule.

## Open questions

- Which park and compression controls reject, queue, drain, or cancel active
  requests?
- Should snapshot export materialize one owned image or stream chunks to the
  disk cache?
- Which cache policies depend on client-visible output and therefore must stay
  above model-state caching?
- Can every batched speculative implementation prove safe retry after an empty
  output, or should some return a typed non-retryable failure?
- Should `server/docs/ARCHITECTURE.md` become the short operational overview
  that links to this target design?

## Next implementation step

Add an owned `GenerateRequest` and a model-free `Generation` channel test. The
test must prove bounded publication, one terminal result, idempotent
cancellation, overload behavior, and shutdown wakeup before model code moves.
