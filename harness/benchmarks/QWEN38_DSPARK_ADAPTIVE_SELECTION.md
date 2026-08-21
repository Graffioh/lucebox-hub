# Qwen3.8 DFlash2 adaptive measurement

This protocol exists to make PR #626 easy to measure and profile. Performance
is an observation, not a refactor acceptance gate.

The primary workload is a fixed backlog of 29 production-shaped chat prompts:

- prompt lengths range from 297 to 638 model tokens;
- 29 requests start together behind a barrier;
- the server exposes 16 slots with a 4096-token per-slot context cap;
- every request produces exactly 256 tokens with greedy decoding and
  `ignore-eos`;
- AR, forced speculation, and adaptive mode each use a fresh server process.

The committed input is
`harness/benchmarks/prompts/qwen38_dflash2_fixed_c29.jsonl`. The older six-row
`qwen38_dspark_adaptive_selection.jsonl` file is a historical dense screening
fixture. Do not use it as the primary concurrent measurement.

## Run the measurement

Build `dflash_server` for the GPU that will execute it. The runner reads the
HIP code objects from the binary and compares them with the GPU reported by
the running server. A gfx1151 binary cannot silently produce a gfx1201 result.

Run the three-replicate measurement suite:

```bash
MODEL=/path/Qwen3.8-27B-PR625-IQ4_XS.gguf \
DRAFT_MODEL=/path/dflash2-q8_0.gguf \
SERVER_BIN=/path/build-gfx1201/dflash_server \
OUT_ROOT=/path/results/qwen38-measurement \
harness/benchmarks/concurrency/run_qwen38_dflash2_measurement.sh
```

The suite rotates variant order across replicates:

1. AR, forced speculation, adaptive;
2. adaptive, AR, forced speculation;
3. forced speculation, adaptive, AR.

This separates adaptive behavior from warm-system and thermal order effects.
Set `REPLICATES` to a value of at least two for a shorter diagnostic run. Keep
the default three for a result used in review.

For a quick smoke or an individual replicate, run
`run_qwen38_dflash2_fixed.sh` directly and override `CLIENTS`, `SLOTS`, and
`MAX_TOKENS`. The defaults are the production-shaped configuration above.

## What the runner proves

Each variant performs these steps in order:

1. Reject ambient backend and GPU tuning variables.
2. Verify that the server binary contains a HIP code object.
3. Start one isolated server process and wait for health.
4. Match the active GPU architecture from `server.log` to the binary.
5. Send one short request to prove decode health.
6. Run a discarded 29-request warmup.
7. Run the measured 29-request fixed backlog.
8. Stop the server and record startup-observed runtime metadata.

AR does not receive speculative environment variables. Forced speculation and
adaptive mode receive the same depth-8 DFlash2 configuration.

Every case directory contains:

- `server-command.txt`: exact executable, arguments, and launch environment;
- `server-metadata.json`: binary, model, prompt, configuration, and Git hashes;
- `gpu-identity.txt`: binary code objects and the active GPU architecture;
- `server.log`: startup, scheduler, selector, and request telemetry;
- `smoke.json` and `warmup.json`: discarded health and warmup work;
- `bench.json`: request-level timings and aggregate goodput;
- `runtime-metadata.json`: startup-observed pool dimensions and proof markers.

Adaptive logs also record
`context_method=synthetic-zero-kv-zero-features-v1`. This identifies the
synthetic context used by the startup cost profile. The comparator rejects an
adaptive result when this marker is missing or ambiguous.

The suite root adds `comparison.json` and `comparison.md`. The comparator
rejects missing proof files, failed requests, incomplete fixed-token work, and
any workload fingerprint mismatch across variants or replicates.

## Read the report

`measurement_status: valid` means the workload and proof invariants passed. It
does not mean adaptive mode is faster.

Use these fields to choose later optimization work:

- per-variant goodput values, means, sample CV, and relative range;
- paired adaptive/AR and forced-speculation/AR ratios;
- speculative accepted tokens and speculative steps;
- accepted tokens per speculative step;
- target forwards and speculative-service AR steps;
- selector route and reason counts;
- mean profiled and predicted speculative cost.

`performance_observation` is one of `adaptive_above_ar`,
`adaptive_below_ar`, or `adaptive_variable`. The default 5% relative-range
threshold is a screening heuristic. Use more replicates or a formal statistical
design before publishing a small difference.

## Reference validation

The suite was validated on 2026-08-21 with a Radeon AI PRO R9700 (`gfx1201`),
the updated PR #625 target and DFlash2 drafter, Q8_0 K/V cache, and depth 8.
The final validation intentionally stopped after two complete replicates. The
partial third replicate is excluded. All 174 measured requests completed.

| Variant | Replicate goodput tok/s | Mean | Relative range |
| :-- | :-- | --: | --: |
| AR | 176.81, 177.80 | 177.30 | 0.56% |
| Forced speculation | 127.12, 127.80 | 127.46 | 0.53% |
| Adaptive | 155.07, 156.18 | 155.62 | 0.71% |

Adaptive averaged 0.878 times AR goodput. The adaptive runs accepted 1,288 and
981 speculative tokens over 537 and 401 speculative steps. This is a stable
baseline for the next profiling pass. It is not a performance acceptance gate
for the refactor.

The throughput artifacts in
`.harness-runs/pr626-qwen38-measurement-739cf84a/` were produced before the
profile-context marker was added. Use them for the two-replicate throughput and
route-count evidence above. Before this docs-only disclosure, the same server
and harness tree was smoke-tested at `cc6b7a7d` in
`.harness-runs/pr626-qwen38-final-stack-smoke-lint/`, where adaptive and forced
speculation logs both record
`context_method=synthetic-zero-kv-zero-features-v1`. Re-run the measurement
with the current scripts before treating profile-context enforcement as part of
a full throughput run.
