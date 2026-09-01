# Serving and engine components

Status: Draft

Source snapshot: `upstream/main` at `298031aa`

## Summary

Lucebox has two local generation paths behind one HTTP server:

- The classic worker runs one complete request through `ModelBackend`.
- The concurrent scheduler advances several requests through `SeqEngine`.

Both paths are valid. A full-request backend call and an iteration-level engine
step solve different problems, so this design does not combine them behind a
new universal interface. It instead defines the ownership boundaries between
the HTTP edge, serving coordination, model execution, architecture runtime,
GGML, auxiliary processes, and the client response.

The first implementation work should preserve the established types and split
the large `http_server.cpp` translation unit by responsibility. Later changes
can share narrowly defined response and error state without forcing the two
execution paths into the same lifecycle.

## Goals

- Make a request traceable from the socket to GGML and back to the client.
- State which component owns transport, request policy, scheduling, model
  state, tensor execution, and response formatting.
- Keep JSON and HTTP concerns out of model and GGML code.
- Preserve the distinction between whole-request and iteration-level
  execution.
- Reduce the amount of server state a reader must hold in mind at once.
- Reuse current names when they already describe their responsibility.
- Introduce new types only when they remove duplicated state or make failure
  handling explicit.
- Provide an incremental migration that can be reviewed and verified in small
  changes.

## Non-goals

- Replacing `ModelBackend` and `SeqEngine` with one engine interface.
- Copying the process topology of vLLM or SGLang.
- Moving the normal local request path behind IPC.
- Renaming every request, slot, callback, or result type.
- Splitting every optional `ModelBackend` capability at once.
- Changing scheduling policy, cache behavior, or API output in the first file
  split.
- Defining a request-wide `GenerationPlan` type.
- Hiding architecture-specific graph and cache state behind generic maps or
  untyped payloads.

## Current request paths

### Startup

`server_main.cpp` resolves configuration, creates the selected backend and
tokenizer, constructs `HttpServer`, and calls `HttpServer::run()`.

At startup, `HttpServer::run()` selects the execution loop:

- If `ModelBackend::seq_engine()` returns `nullptr`, it starts `worker_loop()`.
- If the backend exposes a `SeqEngine`, it starts `scheduler_loop()`.
- The upstream proxy path remains an HTTP concern and bypasses local model
  execution for the forwarded request.

Today, Qwen 3.5 is the worked concurrent implementation under
`server/src/qwen35/concurrency/`. Other model families use the classic path.

### Relationship to inference configuration

The configuration design in
[PR #688](https://github.com/Luce-Org/lucebox/pull/688) owns startup input,
validation, and backend construction. This design begins after the backend and
server configuration have been resolved.

The two changes should meet at existing constructor boundaries. The component
work should not move startup parsing into `HttpServer`, and the configuration
work should not introduce request-lifecycle types. The initial server file
split uses `ServerConfig` and `ModelBackend` as they exist, so it can be stacked
or merged independently of the configuration refactor.

### Common HTTP edge

The request enters through these existing components:

```text
client socket
  -> HttpServer::handle_client()
  -> HttpServer::route_request()
  -> request parsing, chat rendering, and tokenization
  -> ParsedRequest
  -> ServerJob
  -> server queue
```

`ParsedRequest` is the normalized, model-ready representation of one supported
HTTP request. It contains prompt tokens, sampling settings, output limits,
streaming mode, tools, stop sequences, and the API fields needed to construct
the response.

`ServerJob` connects that request to its client socket and completion wait. The
client thread owns the job on its stack. The worker or scheduler borrows it
until it signals completion.

### Classic execution

The classic worker owns the complete lifecycle of one request:

```text
ServerJob
  -> HttpServer::process_job()
  -> prepare prompt and cache state
  -> GenerateRequest
  -> ModelBackend::generate() or restore_and_generate()
  -> architecture-specific prefill and decode
  -> GGML graph execution
  -> GenerateResult
```

`process_job()` also coordinates FlowKV, PFlash, prefix snapshots, draft
residency, agent-turn memory, status reporting, and final response delivery.
These are serving policies around generation. They are not responsibilities of
GGML.

`ModelBackend::generate()` represents one complete prefill and decode cycle.
The backend selects autoregressive, speculative, or other model-specific
execution. `GenerateRequest` and `GenerateResult` are the typed boundary for
that call.

`DaemonIO` carries the legacy daemon stream descriptor together with token,
cancellation, and observation callbacks. The HTTP server uses the callbacks;
the stdin daemon protocol still uses the descriptor. This mixed role should be
reduced only after the two callers can be migrated independently.

### Concurrent execution

The concurrent path keeps request policy in the scheduler and model state in
the engine:

```text
ServerJob
  -> HttpServer::scheduler_loop()
  -> SchedSlot
  -> SeqEngine::admit()
  -> SeqEngine::step()
  -> architecture-specific batched forward and sampling
  -> GGML graph execution
  -> SeqEngine::StepResult
```

The scheduler owns admission order, fairness, output caps, stop conditions,
thinking-budget token substitution, client backpressure, and retirement. Its
`SchedSlot` contains only server-side request progress and response state.

The engine owns slot allocation, prompt progress, paged KV blocks, recurrent
state, graph shapes, tensor inputs, batched forward execution, and sampling.
Those details remain inside the architecture implementation. The scheduler
sees only slot identifiers and the existing `SeqEngine` inputs and outputs.

`SeqEngine::StepPlan` is intentionally limited to one scheduler iteration. It
is not a request-wide configuration object and should remain named for the
single step it describes.

### Response path

Both local execution paths use the same response components:

```text
token or terminal result
  -> SseEmitter
  -> API-specific events or complete JSON
  -> direct socket write in the classic worker
     or ClientSendBuffer in the concurrent scheduler
  -> client
```

`SseEmitter` currently owns two concerns:

1. Semantic response state, including reasoning, content, tool calls, stop
   sequences, and finish reason.
2. OpenAI, Anthropic, and Responses API event formatting.

This combination keeps behavior consistent today, but it makes non-streaming
and streaming response construction harder to reason about. A later extraction
can move the first concern into `ResponseState` while keeping `SseEmitter` as
the established wire-format component.

`ClientSendBuffer` is specific to the shared concurrent loop. It prevents a
slow reader from blocking other active sequences. The classic worker can write
directly because it serves only one generation at a time.

### Auxiliary process execution

IPC is an optional branch inside backend and optimization implementations. It
is not the normal boundary between `HttpServer` and `ModelBackend`.

```text
ModelBackend or architecture runtime
  -> role-specific IPC client
  -> BackendIpcProcess
  -> draft, compression, shard, or expert subprocess
```

`BackendIpcProcess` owns process launch, pipes, shared payload setup, status
transport, scratch paths, and shutdown. Role-specific clients such as
`DFlashDraftIpcClient`, `PFlashDrafterIpcClient`, and
`TargetShardIpcSession` own their payload protocols.

This distinction matters for both naming and failure handling. A transport
failure belongs to the IPC session. A generation or model failure belongs to a
typed engine result. The server should not need to decode subprocess strings.

## Target ownership

The dependency direction should remain one way:

```text
HTTP and API edge
  -> serving request state
    -> classic worker or concurrent scheduler
      -> ModelBackend or SeqEngine
        -> architecture runtime
          -> GGML

architecture runtime
  -> optional role-specific IPC client
    -> BackendIpcProcess

GGML result
  -> typed backend or engine result
    -> response state
      -> API events or response JSON
        -> socket or ClientSendBuffer
```

### HTTP and API edge

Owned by `server/src/server/`.

Responsibilities:

- Accept sockets and parse HTTP framing.
- Route supported endpoints.
- Validate JSON and map API aliases into one internal representation.
- Apply chat templates and tokenize prompts.
- Construct API-specific success and error responses.
- Detect client disconnects and manage streaming headers.

The edge may depend on tokenizer, chat-template, and API-format code. Model and
GGML code must not depend on HTTP status codes, SSE frames, or request JSON.

### Serving request state

Keep these current names:

- `ParsedRequest` for the normalized request accepted by local serving.
- `ServerJob` for the socket-bound queued unit and its completion wait.
- `GenerationInputs` for the classic worker's private aggregate.
- `SchedSlot` for one request's server-side concurrent state.

These names match their scope. Promoting `GenerationInputs` into a shared
engine contract would be a mistake because the classic and concurrent paths do
not consume the same unit of work.

The main improvement is ownership, not renaming. Fields should move out of
these structures only when another component becomes their clear owner.

### Serving coordination

The classic worker and concurrent scheduler should remain separate loops.
They can share pure policy functions and response construction, but not a
synthetic lifecycle interface.

The classic worker owns:

- Whole-request prompt preparation and cache restore.
- Request-scoped draft residency.
- One call to `ModelBackend::generate()` or `restore_and_generate()`.
- Whole-request cache finalization and performance reporting.

The concurrent scheduler owns:

- Admission and fairness.
- Prefill slice selection.
- Batched decode iteration order.
- Per-slot stop and retirement decisions.
- Non-blocking response delivery.

Small shared functions should describe the value they resolve, for example
`resolve_generation_cap()`, rather than collecting unrelated decisions into a
new request-wide object.

### Model execution

Keep the two current contracts:

- `ModelBackend` owns backend lifetime, whole-request generation, snapshots,
  and optional capabilities.
- `SeqEngine` is the optional interface for backends that can keep several
  live sequences and execute scheduler-selected work together.

`ModelBackend::seq_engine()` is enough to select the concurrent path. A third
base class named `Engine`, `ServingEngine`, or similar would add indirection
without removing either existing contract.

`ModelBackend` is broad, but its optional methods should be extracted only when
there is a concrete caller and more than one useful implementation. Until
then, grouped methods and capability checks are easier to follow than a set of
one-method interfaces.

Concurrent failures should eventually reuse `GenerateError` and
`GenerateErrorCode` rather than add another string-based error family. The
exact migration can update `AdmitResult`, `DecodeOutput`, `PrefillOutput`, and
`StepResult` independently while preserving each result's current scope.

### Architecture runtime and GGML

Owned by model-family directories such as `qwen35/`, `gemma4/`, `laguna/`,
and `deepseek4/`, together with genuinely model-neutral helpers in `common/`.

Responsibilities:

- Load weights and choose device placement.
- Own model-specific KV, recurrent, and speculative state.
- Build graphs and bind tensor inputs.
- Execute GGML backends and read outputs.
- Implement model-specific prefill, decode, and sampling mechanisms.

The engine boundary should expose tokens, sampling configuration, progress,
timings, and typed failures. It should not expose graph tensors, allocator
handles, block-table layouts, or architecture-specific state to the server.

Code belongs in `common/` only when at least two model families can use the
same semantics. Similar graph code is not automatically the same component if
the model invariants differ.

### Response state and transport

Keep these current names:

- `SseEmitter` for API event construction and SSE framing.
- `ClientSendBuffer` for non-blocking concurrent socket output.

Introduce `ResponseState` only when semantic accumulation is physically moved
out of `SseEmitter`. It should own content, reasoning, tool-call parsing, stop
matching, and finish reason. It should not know about SSE syntax, HTTP status,
or sockets.

The draft terminal-error change in
[PR #689](https://github.com/Luce-Org/lucebox/pull/689) is the first part of
this boundary. It gives the API layer one response error representation while
preserving `GenerateError` as the backend-facing failure.

### IPC transport

Keep `BackendIpcProcess` as the process and transport owner. Keep payload
semantics in role-specific clients.

An IPC client should either complete one protocol transaction or invalidate
the session. This prevents a partial read or write from being mistaken for the
next response. A small shared helper for marking a process unusable is
preferable to a generic request envelope shared by unrelated IPC modes.

## Source organization

The first structural change should split `http_server.cpp` without changing
the `HttpServer` class or request behavior:

| File | Responsibility |
|---|---|
| `http_server.cpp` | Server lifetime, accept loop, socket I/O, job queue, and disconnect monitoring |
| `http_routes.cpp` | Endpoint routing, request validation, chat rendering, tokenization, and `ParsedRequest` construction |
| `generation_worker.cpp` | Classic `worker_loop()`, `process_job()`, prompt preparation, cache lifecycle, and backend call |
| `scheduler.cpp` | Concurrent admission, iteration policy, slot lifecycle, and buffered delivery |
| `sse_emitter.cpp` | Existing semantic stream state and API event formatting until `ResponseState` is extracted |

This split changes file ownership, not public interfaces. Existing
`HttpServer` member functions can be defined across the translation units.
Tests and the server target should compile the same source set.

After the split, helpers that are used by only one file should move into that
file's anonymous namespace. Helpers shared by classic and concurrent serving
should have narrow typed signatures in a server-local header.

## Migration order

### 1. Make terminal results explicit

Land the API-facing terminal error boundary from PR #689. Both execution paths
must branch on success or failure before success finalization, cache updates,
or HTTP 200 responses.

### 2. Split the server translation unit

Create `http_routes.cpp` and `generation_worker.cpp`, then move existing
functions with no behavior or naming changes. Update both the server target and
model-free test target together.

This change creates reviewable component boundaries before adding new types.

### 3. Extract semantic response state

Move content, reasoning, tool-call, stop-sequence, and finish-reason state from
`SseEmitter` into `ResponseState`.

Both streaming and non-streaming builders should consume the same state.
`SseEmitter` should remain responsible for API events and SSE framing. This
keeps the familiar name and removes the current accumulation ambiguity.

### 4. Share only common request policy

Extract pure helpers for values that classic and concurrent serving must
derive identically, beginning with generation cap, thinking budget, EOS
classification, and terminal error mapping.

Do not create a shared request executor. Each path should call the helper at
the point where it owns the relevant decision.

### 5. Type concurrent engine failures

Replace free-form `SeqEngine` failure strings with `GenerateError` values.
Keep admission, per-row, and whole-step results distinct. The scheduler can
then map both classic and concurrent failures through the same response error
function.

### 6. Narrow `DaemonIO`

Move the legacy file-descriptor behavior behind the daemon caller. Let HTTP
generation pass only the existing token callback, cancellation probe, and
inference observer responsibilities needed by the backend.

Migrate all callers in the same change before deleting unused fields. Do not
add a compatibility wrapper that preserves both shapes indefinitely.

### 7. Harden IPC transactions

Centralize the rule that a framing, payload, or subprocess failure closes the
affected `BackendIpcProcess`. Keep the error returned to generation typed and
specific to the role-specific client.

### 8. Revisit optional backend capabilities

After call sites are narrow and covered by tests, measure whether snapshots,
compression, parking, or remote draft support benefit from separate capability
interfaces. Extract only the groups that reduce real coupling.

## Naming rules

New names should reveal both scope and owner:

- Use `Request` and `Response` for API or whole-request data.
- Use `Admission`, `Slot`, `Prefill`, `Decode`, and `Step` for scheduler and
  concurrent engine data.
- Use `Backend` for model-family lifetime and capability ownership.
- Use `Ipc` plus the remote role for subprocess clients and sessions.
- Use `State` only for data that persists across calls.
- Use `Result` for a completed operation with success or failure.

Avoid names that hide the unit of work. In particular, do not introduce
`GenerationPlan`, `ApiRequest`, `ServingRequest`, `OwnedGenerationPlan`,
`GenerationCallbacks`, `RequestSlotState`, or `ApiStreamEncoder` as renames of
the current types. The established names are clearer when their ownership is
documented and their files are smaller.

## Verification

Each migration step should leave both execution paths usable.

Required checks:

- Build `dflash_server` for the enabled CUDA or HIP configuration.
- Build and run the model-free server unit tests.
- Run `test_seq_engine_contract` for each concurrent engine.
- Exercise streaming and non-streaming success responses for all supported API
  formats.
- Exercise classic and concurrent backend failures and verify non-200 or SSE
  error termination.
- Exercise client disconnects during prefill and decode.
- Exercise a slow concurrent reader and verify other slots continue.
- Exercise prefix restore, request-scoped draft residency, and upstream proxy
  paths after moving the classic worker.
- Check that model-family and `common/` sources do not construct HTTP JSON or
  SSE frames.
- Check that server sources do not depend on architecture-specific graph or KV
  structures.

The file split should produce no wire-format or scheduling changes. A useful
review technique is to compare the moved function bodies before and after the
split and keep functional edits in later commits.

## Alternatives rejected

### One universal engine interface

`ModelBackend::generate()` completes a request. `SeqEngine::step()` advances a
selected batch by one iteration. A common interface would either expose both
lifecycle models or reduce one to callbacks around the other. Neither result
simplifies the caller.

### An internal event bus

The request path is direct and performance-sensitive. Typed calls and results
make ownership and failure propagation visible. An event bus would obscure
ordering, lifetime, and cancellation without adding a current deployment
benefit.

### IPC between HTTP and every backend

vLLM and SGLang use process boundaries to support their deployment and
scheduling designs. Lucebox currently has a direct in-process local path plus
targeted subprocesses for heterogeneous execution. Mandatory IPC would add
serialization and lifecycle work without resolving the current source
organization problem.

### A broad rename pass

The current names mostly identify real units: parsed request, server job,
backend request and result, scheduler slot, engine step, SSE emitter, send
buffer, and IPC process. Moving responsibilities first gives any future rename
concrete evidence and keeps review focused.

### Splitting all `ModelBackend` capabilities now

Small interfaces are useful when they let callers depend on less. Creating one
interface per optional method before narrowing callers would increase the
number of types without changing ownership.

### JSON below the server layer

JSON is part of the external API contract. Passing it into backend or engine
code couples model execution to current endpoints and makes non-HTTP callers
harder to support.

## Related designs

These projects support the ownership direction in this document, but Lucebox
should not copy their process layouts.

- [vLLM architecture overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)
  separates API input and output processing from an engine core that owns
  scheduling, KV cache management, and worker coordination. Its API and engine
  core communicate across ZMQ because vLLM chooses a multi-process topology.
- [SGLang manager I/O structures](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py)
  distinguish external generation input from tokenized scheduler input. The
  useful lesson is the typed boundary between stages, not the number of
  manager processes.
- [llama.cpp server developer guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md)
  separates HTTP context, routes, tasks, queues, inference slots, task results,
  and response readers. It also keeps JSON formatting and chat templates in
  the HTTP layer and passes native C++ types to inference slots.

Lucebox differs in two important ways. It supports a classic whole-request
backend path beside continuous batching, and it uses auxiliary IPC inside
specific heterogeneous execution features. The target structure must make
those differences explicit.

## Open questions

- Should `ResponseState` be extracted immediately after the file split, or
  should terminal error parity land in both paths first?
- Which serving policies must reach parity before another model family exposes
  `SeqEngine`?
- Can the daemon protocol stop sharing `DaemonIO` with the HTTP server without
  affecting external scripts?
- Should all concurrent failures use `GenerateError`, or is a smaller subset
  sufficient for `SeqEngine`?
- When this design is implemented, should `server/docs/ARCHITECTURE.md` become
  an operational overview that links here for component ownership?
