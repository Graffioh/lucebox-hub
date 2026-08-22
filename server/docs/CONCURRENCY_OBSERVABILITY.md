# Concurrency observability

Use the concurrency profiler to find scheduler, prefill, speculative decode,
padding, and KV pressure bottlenecks. It captures high-level serving phases.
It does not replace a kernel profiler.

## Start a capture

Set `DFLASH_PROF=concurrency` before starting `dflash_server`.

```bash
DFLASH_PROF=concurrency \
DFLASH_PROF_OUT=/tmp/lucebox-profile.jsonl \
./dflash_server <normal server arguments>
```

Stop the server normally to write the JSONL file. Capture is bounded by the
`DFLASH_PROF_MAX_*` variables documented in
[ENVIRONMENT.md](ENVIRONMENT.md). The footer reports dropped records when a
bound is reached.

Set `DFLASH_PROF_CHECKPOINT_EVERY` to preserve an in-progress capture if the
server later crashes or hangs. Each checkpoint replaces the prior JSONL file
atomically and ends with `"complete": false`. Clean shutdown replaces it with
the final capture and `"complete": true`. Checkpoint writing runs on the
scheduler thread, so leave it disabled for overhead measurements. A signal
handler does not attempt to flush C++ streams because that is not
async-signal-safe.

The metadata record includes the configured Git SHA, model and draft paths,
architecture, backend, maximum concurrency, DDTree budget, draft block-size
override, selected Qwen concurrency environment values, and both wall-clock
and steady-clock anchors. The anchors correlate service-round timestamps with
benchmark logs and external traces. Each Qwen step also records the effective
speculative tree width resolved from the drafter.

Round retention is keep-first. Once `DFLASH_PROF_MAX_ROUNDS` is reached, later
rounds increment `dropped_steps`. The metadata record reports the retention
policy, `max_rounds`, and `step_record_bytes`, so the reserved memory and any
early-run bias are visible in every capture. Raise the limit for long benchmark
runs. Periodic checkpoints protect in-progress data but do not change the
retention policy.

## Inspect a live server

The server exposes two read-only routes:

- `/observability` serves the built-in Lucebox dashboard.
- `/observability/snapshot` returns one low-cardinality JSON snapshot.

The dashboard derives ratios in the browser. The prefill ratio compares
executed prompt tokens with the scheduler's offered service budget. The server
publishes raw counts so the live and offline views use the same facts.

When capture is disabled, the routes remain available and report that state.
The inference path passes a null profile pointer. Phase scopes do not read a
clock, allocate, lock, format, or write in this state.

## Build an offline report

Generate a Markdown summary and a Perfetto trace from the same JSONL capture.

```bash
python3 harness/benchmarks/concurrency/profile_report.py \
  /tmp/lucebox-profile.jsonl \
  --markdown /tmp/lucebox-profile.md \
  --perfetto /tmp/lucebox-profile.perfetto.json \
  --json-summary /tmp/lucebox-profile.summary.json \
  --folded /tmp/lucebox-profile.folded \
  --folded-per-token /tmp/lucebox-profile.per-token.folded
```

Open the Perfetto JSON at [ui.perfetto.dev](https://ui.perfetto.dev). It shows
round phase spans, request queue/prefill/decode spans, and token-ready bursts.
The JSON summary has a versioned schema for benchmark diffs. It includes run
context, latency percentiles, phase totals, padding ratios, concurrency
cohorts, suppression decisions, and acceptance by speculative position.

The folded files use `path;C=<live slots>;phase` stacks. Identical stacks
merge across rounds. Pass `--stack cohort,path,phase` to put the concurrency
cohort first. The flag accepts any permutation of `path`, `cohort`, and
`phase`.

`--folded` writes host wall nanoseconds. The `unattributed` phase covers time
inside a round that has no phase span. The `idle;inter_round` stack covers
positive host-clock gaps between retained rounds. Neither value proves that
the device was idle.

`--folded-per-token` divides each path and cohort's phase totals by the durable
decode tokens that the scheduler consumed for that group. The report omits a
group when it has no durable decode tokens. Inter-round gaps have no honest
path or cohort owner, so the per-token file omits them.

### Build a LuceGraph report

Write one self-contained HTML file that opens from `file://`.

```bash
python3 harness/benchmarks/concurrency/profile_report.py \
  /tmp/lucebox-profile.jsonl \
  --html /tmp/lucegraph.html \
  --device gfx1201
```

The LuceGraph report contains three linked views. The phase-budget wall
compares concurrency cohorts with per-durable-token, per-serviced-token,
per-round, and wall-share normalization. The request waterfall separates
queue, prefill, first-decode, and decode time on a labeled time axis, with
queue, TTFT, end-to-end, and inter-token p50/p95 in the panel head. The
speculation strip shows the funnel, per-lane suppression decisions, and
acceptance by draft position starting at position 1 (position 0 is the root).

Durable-token normalization bills every phase to committed decode tokens, so
prefill-heavy packed rounds look expensive per token. Serviced-token
normalization divides by durable plus executed prefill tokens and is the
honest view when prompt work dominates. Each cohort label shows its round
count and in-round durable tok/s; the header chip shows aggregate durable
tok/s over the whole capture window.

The wall uses the exclusive buckets from `step_phase_buckets`. It includes
unattributed time, overlapping spans, and inter-round host gaps. Select a wall
segment to inspect its zero-inclusive per-round p50 and p95 values. The report
embeds aggregates instead of raw step records.

A capture with several observed `live_slots` values shows a mixed-run badge.
Admission and tail drain can create these cohorts even when the configured
concurrency is fixed. Use separate fixed-C captures for cohort comparisons
that exclude this bias.

Pass `--baseline` to compare two captures.

```bash
python3 harness/benchmarks/concurrency/profile_report.py \
  current.jsonl --html lucegraph-diff.html --device gfx1201 \
  --baseline baseline.jsonl --baseline-device gfx1151
```

`device_specs.json` is the source of device and model facts. The capture's
`arch` field names the model adapter, not the GPU architecture, so the
report never infers a device from capture metadata. Omit `--device` or pass
an unknown key to render neutral segments with a visible notice.

The analytic roofline classifier applies only to `target_compute` and
`draft_compute`. Segment details show arithmetic intensity, machine
balance, and headroom. Recheck any fact whose note starts with `VERIFY`.

Merged "all" cohorts never recompute the roofline on a blended row shape.
They inherit the per-path classification; when packed and speculative
disagree, the segment renders as `mixed paths` and the tooltip lists both
per-path classes. The baseline delta table contains per-path rows only,
sorted by absolute per-durable-token delta, and shows the top 20 movers with
an expand control.

The v1 capture does not contain model FLOP counts, weight bytes, or device
machine balance. Folded stacks also have no portable color metadata. Use a
separate device profile for compute-bound or bandwidth-bound classification.

## Read the speculation funnel

The profiler keeps these stages separate:

1. An eligible lane can use the configured speculative path.
2. A reserved lane is selected for this service round.
3. An attempted lane enters draft preparation.
4. Proposed draft children come from the drafter. The root token is excluded.
5. Verified draft children run through the target model.
6. Accepted draft children pass verification.
7. Durable draft children finish KV and recurrent-state promotion.
8. Scheduler-consumed draft children reach request generation state.

The separately reported pending token is sampled after the accepted path. It
is not a draft child and does not inflate acceptance.

Per-lane decisions explain why an eligible-looking decode did not speculate.
Examples include prompt work in the same round, unsupported sampling, caller
policy, insufficient context, unavailable features, and draft preparation
failure.

## Understand phase timing

Phase spans use the host steady clock around existing high-level operations.
Instrumentation does not add device synchronization. A target compute or
readback span therefore reflects the synchronization behavior already present
in that code path.

`target_compute` measures the host call that submits a graph. Device work can
continue after that scope and finish while the host is in `argmax_readback`.
Do not treat the host span as GPU busy time. The kernel, queue, and memory-copy
tracks in a native rocprof trace are authoritative for device timing and idle
gaps.

## Drill into one target round

Use the built-in profile to find an expensive service round, then use its
`round_id` to locate the same Qwen graph submission in a native rocprof trace.
For example, start a local C=4 server under the repository wrapper:

```bash
PROFILED_SERVER_BIN=/absolute/path/to/dflash_server \
ROCPROF_OUTPUT_DIR=/tmp/lucebox-c4-rocprof \
ROCPROF_START_SECONDS=180 \
ROCPROF_DURATION_SECONDS=60 \
harness/benchmarks/concurrency/rocprof_server_wrapper.sh \
  <normal server arguments, including --max-concurrency 4>
```

In another terminal, wait for the server to become healthy and run the normal
C=4 workload during the collection window. Adjust the start delay to cover
model loading and warmup on the target machine. The wrapper enables the
high-level concurrency capture and Qwen ROCTX markers, and writes local
artifacts under `ROCPROF_OUTPUT_DIR`.

Build the high-level report after stopping the server normally:

```bash
python3 harness/benchmarks/concurrency/profile_report.py \
  /tmp/lucebox-c4-rocprof/profile.jsonl \
  --markdown /tmp/lucebox-c4-rocprof/profile.md \
  --perfetto /tmp/lucebox-c4-rocprof/profile.perfetto.json \
  --json-summary /tmp/lucebox-c4-rocprof/profile.summary.json
```

Choose a `round_id` whose `target_compute` phase or cohort is interesting.
Open the generated `.pftrace` artifact at
[ui.perfetto.dev](https://ui.perfetto.dev), then search for
`qwen35.graph_compute round_id=<round_id>`. The marker also carries the packed
or speculative path and the relevant live, bucket, row, prefill, and KV-length
shape.

Use the native tracks to answer the device-level question:

- Raw kernel names and durations provide evidence about which operations
  dominate. Names are backend implementation details, not a stable
  attention/GDN/expert classification.
- HIP runtime calls beside dispatches expose host launch overhead and launch
  queues.
- Kernel tracks inside and between round markers show real device busy and
  idle gaps.
- Memory-copy tracks show transfers on the critical path.
- Graph launch, capture, instantiate, and update APIs distinguish replay from
  capture for that traced round.

The built-in `profile.perfetto.json` and rocprof `.pftrace` artifact remain
separate by design. The former explains serving policy and request lifecycle;
the latter owns the device timeline. `round_id` joins them without translating
independent clocks or importing profiler-version-specific CSV schemas.

`GGML_CUDA_GRAPH_STATS=1` is an optional aggregate cross-check. Set it before
the wrapper, and optionally set `GGML_CUDA_GRAPH_STATS_EVERY=1` for a short
diagnostic run. The counters are cumulative per graph key and cannot identify
an exact round; use the HIP graph APIs in the native trace for that. Stats are
emitted only by a graph-enabled build when the exercised path has a graph key.
The wrapper does not force graph stats on.

Kernel tracing does not measure occupancy or memory bandwidth. Counter names
and compatible groups depend on the target GPU, so discover and validate them
there instead of hard-coding a repository default:

```bash
rocprofv3-avail --device 0 list --pmc
rocprofv3-avail --device 0 pmc-check <counter-name> [<counter-name> ...]
```

Run a separate pass with the validated counters and write it to a different
local output directory:

```bash
rocprofv3 \
  --pmc <validated-counter-names> \
  --output-format csv \
  --output-directory /tmp/lucebox-c4-pmc \
  --output-file counters \
  -- <same server command>
```

Do not combine this counter pass with the timing trace. Hardware counters and
full tracing can each perturb execution; use these runs to diagnose a chosen
operation, not as throughput benchmark results.

## Extend another model

The scheduler owns request IDs, lifecycle times, planned lanes, and the final
consumption count. A model adapter receives an optional `StepProfile *` and
fills only facts it owns, such as executed rows, padding, phase spans, KV
pressure, and speculation progress.

The contract is model-neutral and fixed-capacity. A future non-batched C=1
adapter can populate the same record without changing the report, metrics, or
dashboard.

The detailed engine instrumentation currently lives in the Qwen35-family
adapter used by Qwen3.8. Another sequence engine can accept the optional
profile pointer and leave it untouched, but its capture will contain only the
scheduler-owned lifecycle and planning fields until that adapter records its
own execution facts.
