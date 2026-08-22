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
  --perfetto /tmp/lucebox-profile.perfetto.json
```

Open the Perfetto JSON at [ui.perfetto.dev](https://ui.perfetto.dev). It shows
round phase spans, request queue/prefill/decode spans, and token-ready bursts.

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

Use the built-in profile to choose the next experiment. Use ROCTX and rocprof
afterward when the question becomes kernel scheduling, memory bandwidth, or a
specific device operation.

## Extend another model

The scheduler owns request IDs, lifecycle times, planned lanes, and the final
consumption count. A model adapter receives an optional `StepProfile *` and
fills only facts it owns, such as executed rows, padding, phase spans, KV
pressure, and speculation progress.

The contract is model-neutral and fixed-capacity. A future non-batched C=1
adapter can populate the same record without changing the report, metrics, or
dashboard.
