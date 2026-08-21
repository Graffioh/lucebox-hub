# PR #626 cleanup and measurement handoff

Updated 2026-08-21.

## Final scope

PR #626 is a refactor and measurement-foundation change. Its current adaptive
performance is not an acceptance criterion. The useful end state is:

- updated PR #625 Qwen3.8 and DFlash2 support as the stacked base;
- dead `[spec-activation]` measurement code removed;
- one production-shaped benchmark path for AR, forced speculation, and
  adaptive mode;
- enough runtime proof and profiling detail to identify the next optimization;
- a reviewable, stacked commit history.

Worktree: `/home/berto/lucebox-hub/.codex-pr626`

Branch: `codex/qwen38-dspark-adaptive`

Stacked base: `upstream/qwen38-dspark` at `739cf84a`

Safety backup: `codex/qwen38-dspark-adaptive-pre625-20260821`

## What was fixed

The first re-baseline failure had two independent causes:

1. The saved server binary contained gfx1151 HIP code while the active GPU was
   gfx1201. The runner now checks the binary code objects against the GPU named
   by the running server before sending warmup work.
2. The merged GDN raw-gate path reused `op_params[2]`, which SpecLA also owns.
   The updated protocol uses packed loader-owned `[dt_bias | A]` data in
   `src[9]`, marks raw gates in `op_params[10]`, and leaves `op_params[2]` to
   SpecLA. Dense plain chains may use raw fusion; compact, ragged, tree,
   SpecLA, and unsupported backends use the materialized path.

Updated PR #625 was integrated locally through commit `739cf84a`. The resolution
keeps its Qwen3.8 checkpoint-width, grouped GDN, selector-tree, chunked GDN,
RoPE, and RDNA4 kernel changes while preserving PR #626 adaptive concurrency,
direct mixed execution, active-slot metadata, and transition journal.

## Measurement entry point

Run:

```bash
MODEL=/path/Qwen3.8-27B-PR625-IQ4_XS.gguf \
DRAFT_MODEL=/path/dflash2-q8_0.gguf \
SERVER_BIN=/path/build-gfx1201/dflash_server \
OUT_ROOT=/path/results/qwen38-measurement \
harness/benchmarks/concurrency/run_qwen38_dflash2_measurement.sh
```

Defaults:

- three order-rotated replicates;
- C29 fixed prompt set behind a simultaneous-start barrier;
- 16 server slots;
- 4096-token context cap per slot;
- 256 forced output tokens;
- Q8_0 K/V cache;
- DFlash2 checkpoint depth 8;
- fresh process for every variant.

The comparator validates matching work, models, binary, GPU, token counts, and
request success before reporting throughput. It also extracts measured-window
`[concurrency-metrics]` and `[spec-epoch]` telemetry. Adaptive runs identify
the startup profile context as
`synthetic-zero-kv-zero-features-v1`; the comparator rejects a missing or
ambiguous method.

## Validation record

Build:

- Release gfx1201 `dflash_server`: passed.

Tests:

- concurrency harness: 102 passed;
- selector validation, benefit model, speculation gate, cost profile, SpecLA,
  paged-KV transfer layout, batched GDN, and GDN transition journal: 9 passed;
- direct AR, forced-speculation, and adaptive runtime smokes: passed.

The final validation stopped after two complete replicates. The partial third
replicate is excluded.

| Variant | Goodput tok/s | Mean | Relative range |
| :-- | :-- | --: | --: |
| AR | 176.81, 177.80 | 177.30 | 0.56% |
| Forced speculation | 127.12, 127.80 | 127.46 | 0.53% |
| Adaptive | 155.07, 156.18 | 155.62 | 0.71% |

All 174 measured requests completed. The measurement is valid. Adaptive
averaged 0.878 times AR goodput. This is the current system observation, not a
refactor failure. Adaptive accepted 1,288 and 981 speculative tokens over 537
and 401 speculative steps, giving the next optimization pass a concrete
routing and cost-profile target.

Local evidence is under
`.harness-runs/pr626-qwen38-measurement-739cf84a/`. It is intentionally
untracked. Those throughput artifacts were generated before the
profile-context marker landed. Before this docs-only disclosure, the same
server and harness tree was smoke-tested at `cc6b7a7d` in
`.harness-runs/pr626-qwen38-final-stack-smoke-lint/`, where adaptive and forced
speculation logs record
`context_method=synthetic-zero-kv-zero-features-v1`. Re-run the full
measurement with the current scripts before treating that marker check as part
of a throughput run.

## Preserved side work

- Planner/population experiments remain parked on
  `codex/qwen38-planner-wip`.
- Qwen3.6 feature-matrix work remains on `codex/qwen36-feature-matrix`.
- The untracked C16 mixed prompt file remains a historical diagnostic and is
  not part of the primary measurement.

## Next engineering work

Use the committed measurement suite to change one policy or cost-model input
at a time. Compare route counts, speculative accepted tokens, target forwards,
predicted cost, and goodput across the same rotated replicates. Do not revive
the deleted request-sticky `[spec-activation]` harness or gate the refactor on
the current performance number.
