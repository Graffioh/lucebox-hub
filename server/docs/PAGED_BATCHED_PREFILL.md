# Paged Multi-Sequence Prefill

## Status

This document defines the target prefill design for concurrent Qwen3.5/3.6
and DeepSeek V4 serving. It is a design, not a description of the current
implementation.

Today the two backends prove different parts of the serving model:

- Qwen processes one prompt at a time in efficient multi-token chunks while
  existing sequences continue decoding. Its prefill K/V, SSM, and convolution
  staging state is shared, so a second prompt cannot safely enter prefill.
- DeepSeek4 keeps persistent state per slot and can advance several prompts in
  one graph. Its exact reference path advances only one token per prompt per
  iteration and gathers chronological MLA history, which is correct but not a
  throughput implementation.

The target combines both properties: several prompts make progress together,
and every prompt advances by a useful multi-token chunk.

## Goals

1. Preserve exact model semantics for paged attention, recurrent state, and
   sampling.
2. Prefill several independent sequences in one bounded token batch.
3. Keep decode latency bounded while admission traffic is heavy.
4. Allocate persistent memory from concurrency and context limits, and scratch
   memory from a fixed per-iteration token budget.
5. Submit MoE work from the packed token batch so both expert owners receive
   useful batches under asymmetric execution.
6. Keep block tables, graph shapes, and model-specific state below
   `SeqEngine`.

## Non-goals

- Moving expert weights between devices on every request.
- Reserving scratch for `max_concurrency * max_prompt_tokens`.
- Making the HTTP scheduler aware of MLA, SSM, convolution, or block tables.
- Combining decode and prefill into one graph when separate shape-specialized
  graphs produce better latency or throughput.
- Publishing partially updated sequence state after a failed iteration.

## Required invariants

These invariants take precedence over batching opportunities:

1. **Slot isolation:** every mutable model state is indexed by slot or belongs
   to an iteration-local workspace.
2. **Ordered state:** tokens for one sequence update attention, compressor,
   SSM, and convolution state in token order.
3. **Causal visibility:** token at absolute position `p` observes its committed
   prefix and current-chunk tokens through and including `p`, never positions
   greater than `p` or another sequence. For DS4, raw visibility is
   `[max(0, p - 127), p]`. For compression ratio `r`, visible compressed rows
   are exactly `[0, floor((p + 1) / r))`; a row emitted when
   `p % r == r - 1` is therefore visible to token `p`.
4. **Publish after success:** sequence length, committed recurrent state, and
   scheduler-visible progress advance only after device execution succeeds.
5. **Capacity safety:** every admitted request reserves enough pages for its
   prompt and declared output cap. Prefill and decode consume only that slot's
   reservation.
6. **Bounded work:** one iteration cannot exceed its configured token, page,
   sequence, or scratch-memory budget.
7. **Decode priority:** sustained admissions cannot indefinitely delay active
   decode lanes.

## Execution model

The engine keeps a queue of pending prefill slots. At each scheduler step it
constructs a `PrefillPlan` using a fixed token budget and a fairness policy.
The plan is internal to the model backend; it does not become part of the
public `SeqEngine` contract.

```text
PrefillPlan
  total_tokens
  sequence_count
  slots[]             slot for each sequence segment
  token_offsets[]     offset in the packed token array
  token_counts[]      tokens assigned to each slot
  positions[]         absolute position for every packed token
  page_tables[]       compact block-table view for each slot
  commit_metadata[]   old and candidate lengths/state generations
```

Tokens are packed by sequence and accompanied by segment offsets, equivalent
to `cu_seqlens` in variable-length attention:

```text
[ A0 A1 ... A95 | B0 B1 ... B63 | C0 C1 ... C31 ]
  <--- A segment --->  <-- B segment -->  <- C segment ->
```

The initial policy should be deterministic and simple:

1. Reserve the decode work required by all active decoding slots.
2. Visit pending prefills in round-robin order.
3. Give each selected slot at least a minimum quantum.
4. Continue filling chunks, up to the per-sequence chunk limit, until the
   iteration token budget is exhausted.
5. Move the round-robin cursor after every iteration.

Chunk size is a tuning result, not an API promise. Use graph buckets and start
with measured values such as 64, 128, 256, and 512 tokens. Very short prompt
tails use the smallest fitting bucket. Padding must predicate off every
persistent K/V, compressor, SSM, convolution, sampler, and metadata write.
Operations that require a valid address receive disjoint scratch rows/state or
a kernel-guaranteed write-suppression path; padding rows must not race by
writing one shared scratch location.

## Scheduling decode and prefill

The scheduler still calls one `SeqEngine::step()` per iteration. The engine
may implement that step in either of two ways:

- **Mixed graph:** pack decode lanes and prefill segments into one graph when
  the model kernels share a profitable token axis.
- **Two shape-specialized graphs:** run a decode graph and a bounded prefill
  graph within the same scheduler iteration when this improves graph reuse or
  kernel efficiency.

Correctness must not depend on which execution form is selected. Both consume
the same plan and publish completed work only after their respective device
operations succeed.

With the current synchronous `SeqEngine::step()` contract, decode output is not
published until the entire step returns. Running the decode graph first does
not by itself reduce externally observed inter-token latency. When decoders are
active, every prefill slice must therefore fit a configured or profiled wall
time on the decode critical path. Token count is only one planning bound;
history span, MoE owner workload, transfers, and the selected graph bucket
must fit the same time budget.

The prefill budget should be reduced when decode latency approaches that hard
limit and increased when the server is admission-bound with spare device time.
Use explicit thresholds and hysteresis initially, not a self-tuning controller.
Longer overlap or device preemption requires a future asynchronous publication
and priority-stream design and is not provided by the synchronous contract.

## Page allocation and admission

Admission computes the maximum cache-resident length from prompt length and
the request's declared output cap. Its full page requirement is added to a
reservation ledger. Pages may still be allocated lazily as prefill and decode
execute, but the following condition must always hold:

```text
free_pages >= unallocated_reserved_pages_for_all_live_requests
```

Each slot consumes its own reservation when either prefill or decode crosses a
page boundary. Admission succeeds only if existing unallocated reservations
plus the new request's reservation fit. Retiring or failing a slot releases
both allocated pages and its remaining reservation. The generic admission
metadata may carry maximum cache-resident length; it must not expose page size
or block tables to the HTTP scheduler.

The plan builder must split a chunk at context and page-table limits. A chunk
may cross physical page boundaries; kernels receive the slot's block table and
translate each logical token position independently. No gathered chronological
cache should be materialized for the optimized path.

## Transactional state publication

Append-only Qwen K/V and DS4 compressed rows may be written before publication
only while their block-table extent and valid-row count remain unpublished.
DS4's 128-row raw ring is different: new positions eventually alias committed
rows. A chunk reads the committed raw ring plus iteration-local projected raw
rows and must not overwrite ring rows still needed by that chunk or by a retry.

The atomic logical commit set is:

- allocator length and block-table high-water mark;
- valid raw and compressed row counts;
- recurrent, convolution, compressor, and raw-ring state identity;
- scheduler-visible prompt progress.

The baseline failure model is fail-and-retire: candidate state is validated
before an in-place commit, and any execution error after a slot's persistent
state may have changed irrevocably fails and retires that slot. If retry after
device failure is required, candidate workspace or alternate state generations
must be used and swapped only after success. Alternate generations are an
option, not a requirement; their potentially large memory cost must be charged
to persistent-state and expert-placement planning.

The current allocator advances logical length before device execution. The new
path must either keep that length candidate-only until commit or guarantee the
fail-and-retire behavior. Slot reuse clears all persistent state and device
metadata before the slot is admitted again.

## DeepSeek4 kernel design

The optimized DS4 path replaces chronological gathering with direct paged MLA
access.

For every packed token, the kernel receives its slot, absolute position, and
block table. It reads:

- committed slot-local rows from the 128-row raw ring;
- projected raw rows from the current sequence segment;
- paged ratio-4 or ratio-128 attention-compressor rows;
- the separate paged ratio-4 indexer-compressor rows;
- the sequence-local rolling state for both compressors.

Ratio-4 maintains two aligned but numerically different compressed streams:
attention values and indexer keys. Top-k selection uses only indexer rows
visible to the query and applies the selected logical row indices to the
corresponding attention-compressor rows.

Within a segment, compressor updates require ordered segmented state-machine
evaluation. Segments execute independently and should not be serialized on the
host. A parallel prefix/scan implementation is allowed only after proving that
its operator and floating-point evaluation preserve the declared oracle; the
compressor pooling and ratio-4 previous/current-half rotation must not be
assumed associative. Emitted compressed entries are written directly to their
paged destinations.

Attention uses the block table directly and applies both masks:

1. sequence isolation from segment offsets and slot IDs;
2. causal masking inside the current chunk.

Two explicit attention modes have different correctness contracts:

- **Dense-exact paged:** attend every causally visible compressed row and match
  the gathered `Explicit` reference within the declared numerical tolerance.
- **Sparse-indexed paged:** run per-query ratio-4 top-k over only the causally
  visible indexer prefix and match a tokenwise sparse reference. Because sparse
  pruning changes the algorithm, compare its quality and generation behavior
  separately against dense-exact output rather than requiring numerical
  equivalence to it.

A fast sparse kernel that lets early tokens observe compressed entries emitted
by later tokens is invalid even if aggregate output looks plausible. The
gathered dense-exact graph remains a correctness oracle for small contexts. It
should not remain an automatic production fallback after the direct kernel has
passed equivalence tests; reference execution should be requested explicitly.

## Qwen kernel design

Qwen first eliminates shared staging K/V by using ragged paged prefill attention
keyed by each query's slot and absolute position. Admission resets that slot's
recurrent state before first use.

Convolution and Gated DeltaNet require different state handling:

- Convolution gathers the slot's last `kernel_size - 1` inputs, executes a
  boundary-isolated causal convolution over its segment, and retains the final
  `kernel_size - 1` inputs as candidate state. It is not modeled as a scan.
- Gated DeltaNet executes the exact ordered recurrence independently for each
  sequence. The correctness baseline may be token-serial within a segment
  while different segments execute in parallel. A chunked or parallel scan is
  accepted only after all token outputs and final state match the exact kernel.

Packed projections, paged attention, FFN/MoE, and independent sequence
recurrences can still be batched, removing admission head-of-line blocking.
Within-sequence recurrence remains a potential serial bottleneck until an exact
parallel formulation is proven. This also removes the need to reset one
cache-wide staging area on every admission.

## MoE and asymmetric execution

Routing runs over the complete packed token batch. Route records contain at
least `(token_row, expert_id, weight)` and are partitioned by static expert
ownership. Primary and secondary owners execute their route batches
concurrently, and results are joined into the original packed token rows.

The execution contract preserves complete MoE semantics:

- skip an owner submission when it has no routes;
- execute the shared expert exactly once on its designated owner;
- preserve canonical route-order reduction when summing owner partials would
  change the declared floating-point association;
- use fixed-capacity masked route arrays for captured graphs or include compact
  owner-route capacity in the graph shape contract.

Prefill batching should improve expert utilization, but concurrency alone does
not determine the ideal ownership split. Placement depends on route frequency,
per-device throughput at each route-batch width, transfer cost, and primary
memory reserved for paged state and graph workspaces.

The first performant implementation keeps ownership static for the server
lifetime. A future startup advisor selects placement from profiles covering
decode-lane buckets, enabled prefill-token buckets, and mixed decode/prefill
steps. Profiles include per-owner and per-expert route histograms, zero-route
owners, transfer mode, and join mode. Optimize critical-path time—the slower
owner plus required transfer and join—not equal route counts or the goal that
both owners always receive work. Runtime adaptation should use coarse
rebalancing or hot expert replication at safe boundaries; it must not migrate
expert weights per request.

## Graph and memory planning

Use a bounded set of captured graph buckets keyed by quantities that materially
change allocation or launch structure. The shape contract is non-exhaustively:

```text
execution_form
prefill_token_bucket
prefill_sequence_bucket
decode_bucket
attention_history_or_launch_bound_bucket
DS4_compressed_span_bucket
MoE_owner_route_capacity_bucket
device_topology_and_join_mode
```

Ratio phase, positions, slot IDs, page IDs, and exact lengths remain runtime
inputs only when the selected kernels have fixed topology for them. Do not key
graphs by exact prompt lengths or slot IDs by default. Runtime metadata fills
unused rows with masked padding and supplies the current block tables,
positions, and segment offsets. Bound graph-cache cardinality and apply an LRU
or equivalent memory budget so the bucket Cartesian product cannot retain
unbounded workspace.

Memory planning has three independent components:

1. **Persistent per-slot state:** page tables, lengths, recurrent/compressor
   state generations, and sampler state.
2. **Paged token storage:** sized from `--kv-pool-tokens`.
3. **Iteration workspace:** sized from the largest enabled graph bucket, not
   from total context capacity.

Automatic expert placement subtracts all three, plus the normal safety margin,
before assigning primary-device experts. Integer arithmetic in planners is
checked before multiplication and addition.

## Failure handling

- A planning or capacity failure affects only the request that cannot be
  admitted.
- A segment-local validation failure retires that slot and releases its page
  reservation.
- A whole-graph execution failure marks every participating slot failed; no
  candidate recurrent generation or sequence length is published.
- Secondary-device failure must not silently rerun only part of an MoE layer
  on the primary with different expert residency. Either execute a validated
  whole-layer fallback or fail the participating step.
- Cancellation removes future segments immediately and retires the slot after
  in-flight device work reaches a safe boundary.

## Telemetry

Record per iteration:

- prefill tokens and sequences;
- decode lanes;
- selected graph bucket and padding ratio;
- queue wait, prefill compute, decode compute, and device-join time;
- pages free, allocated, and reserved;
- route counts and execution time per expert owner;
- TTFT and inter-token latency distributions.

Log summaries, not one line per token. These measurements are required both
for chunk-budget tuning and for a future concurrency-aware expert-placement
advisor.

## Implementation sequence

Each stage must preserve a working exact path and include an end-to-end server
generation before performance claims.

1. Introduce the packed ragged plan and deterministic round-robin token budget
   without changing model kernels.
2. Make Qwen recurrent and convolution staging slot-local and transactional.
3. Execute several Qwen prompt chunks in one packed graph; retain existing
   single-prefill output equivalence tests.
4. Implement DS4 segmented compressor/indexer updates against direct pages.
5. Implement direct block-table-aware DS4 MLA prefill and remove chronological
   gathering from the optimized path.
6. Add shape buckets and graph capture only after uncaptured correctness is
   established.
7. Tune decode/prefill budgets and asymmetric placement from decode-lane,
   prefill-token, and mixed-workload profiles at concurrency 1/2/4/8/16.

## Acceptance criteria

### Correctness

- Dense-exact single-sequence logits match the existing exact implementation
  within the backend's declared numerical tolerance.
- Sparse-indexed logits match a tokenwise sparse oracle; separate quality and
  generation regressions compare sparse-indexed behavior with dense-exact.
- Packed multi-sequence logits match separate single-sequence runs for prompt
  lengths around every page and chunk boundary.
- Results are unchanged by slot assignment, admission order, and unrelated
  sequence lengths.
- Cancellation, graph failure, and immediate slot reuse do not expose stale
  K/V, compressor, SSM, or convolution state.
- Reservation accounting returns to its initial value after every success and
  failure test.

### Performance

- Measure prompt throughput and TTFT at concurrency 1, 2, 4, 8, and 16 with
  short, medium, and long prompts.
- Define and enforce p95 and p99 decode inter-token-latency limits under
  sustained admissions and the largest enabled prefill bucket.
- Compare mixed and separate graph execution before choosing the default.
- Compare against the current Qwen chunked path, the DS4 exact reference path,
  and the same llama.cpp build and model quantization.
- Include warm-up policy, device placement, model, context, output cap, and
  power profile with every result.

The design is complete when increasing concurrent admissions improves total
prompt throughput without corrupting slot state or violating the declared
decode-latency limits. A favorable aggregate number alone is not sufficient.
