# OFlash results

**Bounded hardware qualification is measured; workload-scale performance is not.** The
2026-08-13 run covers official Q4_K_M capture, detached training, live export/swap/promotion and
warm start on the intended R9700 + Strix Halo pair. M0-M4 still define the larger held-out and
sustained methodology. Every number must arrive with the exact command that produced it
(CONTRIBUTING.md: numbers without methodology don't get merged). Sections mirror the
[`R9700 and Strix Halo qualification ladder`](../../server/docs/OFLASH.md#r9700-and-strix-halo-qualification-ladder).

The planned OFlash measurements use one box: **AMD Radeon AI PRO R9700** (gfx1201, 32 GB GDDR6)
serving the target and **AMD Ryzen AI Max "Strix Halo" iGPU** (gfx1151, 128 GB unified memory)
running the trainer. Target: **Qwen3.6-27B Q4_K_M**. Selected drafter for the next qualification
and M0 run: **Lucebox/dflash-draft-3.6 Q4_K_M** (DFlash block-diffusion drafter, 16-token blocks),
with the `--draft-swa 2048` CLI equivalent of the published SWA setting. Future
static-vs-adaptive pairs must use the same seeds, power limit and warmup.

Primary metric: decode tok/s as a function of decode-step index and wall-clock session time,
static vs adaptive. Secondary: mean accepted length (AL), acceptance rate α per draft position,
swap/rollback counts, TTFT (must be unchanged), trainer interference on target tok/s, peak memory
on both devices.

## Published reference configurations

These results establish that the model paths run on each device and that the two AMD devices can
work concurrently. They are not OFlash measurements and must not be combined into a synthetic
dual-GPU throughput result.

| Published test | Exact relevant setup | Result |
|---|---|---|
| [R9700 Qwen3.6 server README baseline](../../server/README.md#amd-hip-backend-strix-halo-rx-7900-xtx) | `gfx1201`, ROCm 7.1.1, Q4_K_M target + **Lucebox Q4_K_M draft**, DDTree 22, HumanEval 10 prompts, `n_gen=256` | 54.65 tok/s mean, AL 7.14, 36.9–93.0 tok/s range |
| [Strix Halo Qwen3.6](https://www.lucebox.com/blog/amd) | `gfx1151`, ROCm 7.2.2, Q4_K_M target + **Lucebox Q8_0 draft**, SWA 2048, DDTree 22, fast rollback, 10 prompts, `n_gen=128` | 26.85 tok/s, AL 5.58, 34.9% acceptance |
| [R9700 + Strix Halo](https://www.lucebox.com/blog/deepseek-v4-asymmetric-parallelism) | ROCm 7.2.4, one-process asymmetric DeepSeek-V4 MoE inference | concurrent heterogeneous HIP execution; not Qwen and not training |

The published Qwen tests are two separate single-GPU runs. The R9700 number used the selected
Lucebox Q4_K_M draft; the Strix number used Lucebox Q8_0 plus the published SWA setting. Neither
predicts online-training behavior. The
[Lucebox model card](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF) recommends Q4_K_M for
fast inference and Q8_0 for parity/debug. Q4's smaller serving footprint does not shrink the dense
FP16 trainer mirror, and its quantized-engine transfer quality remains a measurement.

## 0. 2026-08-12 hardware qualification

Measured on the intended box with ROCm 7.2.2: R9700 `gfx1201` (31.86 GiB), Strix Halo `gfx1151`
(96 GiB GPU-accessible KFD heap), and 128 GiB host RAM.

- Pinned target and Spiritbuun drafter downloads matched their published SHA-256 values:
  `41ae55b347988dca8352ed4c85f3d8ee3804a23cc89aaea165c071d61ec3cca0` for the
  16,817,244,064-byte Q4_K_M target and
  `29ba8b816eedea674e8bdabbd29db8da69539117c76da40e40d2207c0fb224db` for the
  1,849,481,856-byte Q8_0 drafter.
- The release `gfx1201` build passed all 373 C++ server tests; the trainer passed all 40 Python
  tests with AMD PyTorch 2.9.1 / ROCm 7.2.1.
- R9700 flash-prefill numerical checks passed (maximum error 0.00027); the 8K kernel smoke averaged
  14.6 ms over five iterations.
- A Strix FP16 GEMM/backward smoke passed on `gfx1151` with a 96 GiB-visible heap.
- The command below loaded both GGUFs, created the 256 MiB capture ring, and reached the listen
  socket. The target loader reported 14.99 GiB of GPU tensors plus a 682 MiB CPU-only embedding.

```bash
server/build/dflash_server /models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft /models/dflash-draft-3.6-q8_0.gguf \
  --target-device hip:0 --draft-device hip:0 \
  --host 127.0.0.1 --port 18080 --max-ctx 4096 \
  --chunk 128 --cache-type-k q4_0 --cache-type-v q4_0 \
  --prefix-cache-slots 0 --prefill-cache-slots 0 \
  --ddtree --ddtree-budget 22 \
  --oflash --oflash-ring-mb 256 --oflash-topk 8
```

The request/capture and full-mirror training phases were not run: a concurrent persistent
DeepSeek-V4 process repeatedly reclaimed both GPUs during the qualification window, reaching about
32 GiB R9700 VRAM and 127 GiB across KFD allocations. OFlash was stopped instead of competing past
the documented reserve. An exclusive GPU window is therefore the next prerequisite; the successful
model boot is a capacity result, not an online-distillation or performance result.

This historical boot predates the selection of the official Lucebox Q4_K_M drafter. It is retained
as evidence for the machine and code path, not relabeled as a Q4 test. The next M0 run must use the
pinned 1,055,917,280-byte Lucebox artifact with SHA-256
`e2500e90165a0f8e7b52c9882c29ed1fa391c60b300ff11b817bf10e31fa092e`, RoPE 1,000,000 and the
resolved 2,048-token `[S,S,S,S,F]` layout.

## 1. M0 — acceptance-vs-step baseline curves

**Bounded smoke measured on 2026-08-13; full curves not yet measured.**

Capture-only mode (no trainer): per-step AL and α-per-position over long sessions, plus the capture
overhead itself (target tok/s with vs without `--oflash`; budget < 1%). This baseline is what every
later claim is measured against.

```bash
server/build/dflash_server qwen36-27b-Q4_K_M.gguf \
    --draft dflash-draft-3.6-q4_k_m.gguf --draft-swa 2048 \
    --target-device hip:0 --draft-device hip:0 \
    --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
    --ddtree --ddtree-budget 22 \
    --oflash                                               # no trainer bin: capture-only

python3 harness/benchmarks/generation_benchmark.py run \
    --name oflash-m0-he --url http://127.0.0.1:8080/v1 --model qwen36 \
    --prompts harness/benchmarks/prompts/bench_he.jsonl --json-out m0_he.json
# repeat with bench_math.jsonl, bench_gsm.jsonl, and one real long agentic
# coding session via harness/clients/run_claude_code.sh
```

Per-step AL/α come from the engine's rolling stats (`/props` + logs); extending the harness client
to per-step timing output is part of this milestone. Deliverable: acceptance-vs-decode-step curves
for HumanEval / Math500 / GSM8K and the agentic session, plus the capture-overhead delta.

The bounded Q4 run used target SHA-256
`41ae55b347988dca8352ed4c85f3d8ee3804a23cc89aaea165c071d61ec3cca0`
(16,817,244,064 bytes) and official Lucebox draft SHA-256
`e2500e90165a0f8e7b52c9882c29ed1fa391c60b300ff11b817bf10e31fa092e`
(1,055,917,280 bytes). Three fixed temperature-zero prompts produced 8, 25 and 111 tokens with
acceptance rates 0.15625, 0.31250 and 0.51042. OFlash-off and capture-only outputs were
byte-identical, `records_dropped=0`, and the R9700 allocation was about 19.25 GiB.

For three warmed repeats of the 111-token code prompt, OFlash-off measured 90.7/90.7/90.7 tok/s
and capture-only measured 87.8/87.7/87.7 tok/s: a 3.27% capture regression. This misses the
aspirational <1% M0 budget but remains below the 10% safety stop. The M-RoPE fix discovered during
M2 changes target prefill/verify numerics, so those pre-fix throughput numbers are retained as
capture-path evidence, not a final post-fix benchmark baseline.

## 2. M1 — detached trainer/static load: does the recipe move α?

**Detached trainer qualification passed; held-out static-load α gate remains open.**

Keep the M0 capture-only server running, attach a direct trainer to its live bounded ring, then
re-run the M0 benchmarks with the resulting adapter loaded statically (promoted at startup, trainer
off). The ring is not a persisted replay file, and exactly one consumer may attach. This proves the
loss/rank/λ recipe moves α before any live swap machinery is trusted. It also quantifies the
FP16-mirror → Q4_K_M transfer gap: α in the PyTorch mirror vs α on the quantized engine, same
adapter.

```bash
# Copy the ring name and exact profile directory from the capture-only server log.
OFLASH_PROFILE_DIR=/path/from/oflash-profile-dir-log
OFLASH_RING_NAME=/lucebox-oflash-12345
optimizations/oflash/bin/oflash-trainer \
    server/models/oflash/dflash-draft-3.6-q4_k_m.gguf \
    --ring-name "$OFLASH_RING_NAME" \
    --out-dir "$OFLASH_PROFILE_DIR" --profile default \
    --rank 16 --alpha 32 --device 1 \
    --drafter-sha256 e2500e90165a0f8e7b52c9882c29ed1fa391c60b300ff11b817bf10e31fa092e \
    --target-sha256 41ae55b347988dca8352ed4c85f3d8ee3804a23cc89aaea165c071d61ec3cca0 \
    --resolved-rope-theta 1000000 --resolved-swa-window 2048 \
    --resolved-swa-pattern 1,1,1,1,0 --resolved-mask-token-id 248070 \
    --target server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf \
    --batch-rows 64 --train-ctx 64 --reservoir-rows 2048 \
    --export-every 1 --keep-generations 4

# Stop the trainer/server after an export, mark that immutable generation as
# promoted in "$OFLASH_PROFILE_DIR/promoted.json", then restart the M0 server command.
# promoted.json: {"adapter":"adapter-gen<N>.safetensors","generation":<N>}
```

Go/no-go: **≥ +0.05 α** on a held-out slice of the same distribution. Report: α before/after per
benchmark, adapter generation used, mirror-vs-engine α gap.

Use `harness/benchmarks/oflash_benchmark.py` for this gate. Its default three-fold split uses the
repository's HumanEval, GSM8K and Math500 fixtures, trains three independent adapters, and ensures
that every prompt is held out exactly once. Base and adapted evaluation both run capture-only with
the trainer stopped, so their comparison does not mix adapter benefit with optimizer interference.
`harness/client_test_runner.py bench --oflash-phase ...` rejects a live trainer or generation change
during held-out evaluation and records per-request acceptance and server decode timing. Treat this
30-prompt cross-validation as a bounded signal; follow it with the full 164-task HumanEval+ quality
gate and larger domain-specific held-out sets before making a general performance claim.

The disposable Strix trainer passed the accelerator preflight, loaded the dense FP16 mirror in
22.4 seconds and ran 20 optimizer/export steps. Loss EMA moved from 2.5621 at generation 1 to
1.8563 at generation 20; the final counters were 3,801 rows, 2,341 labels, zero backlog and zero
drops. The generation-20 adapter was 13.1 MB and retained the complete format-v2 model/semantic
identity. Strix KFD/GTT allocation held around 8.55 GiB, host `MemAvailable` around 98 GiB, and
swap did not grow. This qualifies execution and export, not the held-out ≥+0.05 α milestone.

The first deterministic held-out fold was run on 2026-08-13 with the official Q4_K_M target and
draft, 512-token request caps and one fixed 18-prompt adaptation epoch. The trainer produced 45
exports; the live guard promoted three, rolled back six and selected generation 33. Frozen
evaluation then restarted from generation 33 with the trainer disabled. Across the 12 held-out
HumanEval/GSM8K/Math500 cases, mean acceptance changed from 0.4937 to 0.4905
(`-0.0032`, paired-bootstrap 95% CI `[-0.0129, +0.0094]`), mean decode throughput changed from
84.10 to 83.45 tok/s (mean per-case speedup `0.995x`, 95% CI `[0.973x, 1.023x]`), and task score
remained 7/12. Only 8/12 responses were byte-identical, so the comparison correctly failed its
output-parity gate. This fold therefore does **not** pass the M1 `+0.05` acceptance criterion, and
the remaining folds should not be presented as confirmatory until the deterministic parity failure
is understood. The earlier repeated-prompt gain remains a bounded overfit/smoke result.

The run completed without a GPU or host-memory safety event: target allocation was about
19.3 GiB on the R9700, trainer allocation about 8.6 GiB on Strix Halo, host `MemAvailable` stayed
near 98 GiB, swap did not grow, the capture backlog drained to zero and `records_dropped` remained
zero. The bounded trainer settings were `--batch-rows 64 --train-ctx 64 --reservoir-rows 2048
--export-every 1 --keep-generations 4`; the capture ring was 512 MiB.

## 3. M2 — within-session AL climb (online loop)

**Bounded fixed-prompt loop passed; domain-shifted and adversarial workloads remain open.**

Full loop: ring → live trainer → swap → guard, on a domain-shifted workload the drafter is weak on
(non-English prose, or a codebase in an under-represented language). Also the adversarial case:
rapidly switching domains must show no adapter-driven acceptance regression with the guard on.
Capture and concurrent-training overhead remain separate from the restored static weights.

```bash
server/build/dflash_server qwen36-27b-Q4_K_M.gguf \
    --draft dflash-draft-3.6-q4_k_m.gguf --draft-swa 2048 \
    --target-device hip:0 --draft-device hip:0 \
    --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
    --ddtree --ddtree-budget 22 \
    --oflash --oflash-trainer-bin optimizations/oflash/bin/oflash-trainer
# domain-shifted workload through the same per-step harness client as M0
```

Deliverable: AL vs decode-step within one session (static vs adaptive, same seeds), swap/rollback
counts, and the adversarial-switching run.

The first integrated attempt exposed a pre-existing Qwen3.5 target bug: multi-token prefill and
chain verification supplied text M-RoPE positions token-major even though ggml consumes four
axis-major rows. A changed draft proposal path could therefore change greedy output. The run was
stopped at the >10% throughput threshold; no OOM or GPU fault occurred. This PR fixes both dense
and MoE production writers and adds a CPU layout regression test. It also fixes promotion/rollback
logs to retain the actual pre-transition AL operands rather than printing cleared windows.

After the fix, target-only AR, zero-adapter DDTree and promoted-adapter DDTree produced the same
111-token code response. A clean integrated run then exported and hot-swapped generation 1 after
eight optimizer steps. On three warmed identical repeats, capture-only was
84.4/84.4/84.5 tok/s at acceptance 0.5052 (AL 9.08); generation 1 was
103.8/104.1/104.4 tok/s at acceptance 0.6250 and promoted with `AL 9.08 -> 11.06`.
`records_dropped=0`, backlog returned to zero, R9700 allocation was about 19.25 GiB, host
`MemAvailable` remained about 98 GiB, and swap did not grow. Two later cycles over smoke, long-list
and code prompts remained stable. This is encouraging single-prompt adaptation evidence, not the
held-out/domain-shifted M2 deliverable.

The final patch built the HIP `dflash_server`, `test_server_unit` and `test_kvflash` targets. All
377 C++ server unit tests and all 66 Python trainer tests passed. The C++ suite was run outside the
restricted development sandbox because its Unix-socket half-close test requires `shutdown(2)`.

## 4. M3 — warm start across sessions

**Not yet measured.**

Persistence: run session 1 (M2 setup), restart the server, measure session-2 AL at t=0 vs
session-1 AL at t=0 on the same workload. Also verifies generation GC (`--keep-generations 4`) and
that a quarantined (rolled-back) generation is never warm-started.

```bash
# session 1, then restart the same command; the engine warm-starts from
# ~/.lucebox/oflash/<hash16>/default-sem-<contract-hash>/promoted.json
```

Deliverable: session-2 speedup at t=0, adapter store contents before/after GC.

## 5. M4 — hardware interference study

**Not yet measured.**

Same benchmarks with the trainer pinned to the iGPU vs CPU vs disabled; measure target tok/s delta,
package power, and thermals under sustained load (the box shares a ~500 W budget and a memory
controller between iGPU and CPU). This decides the default `--oflash-device` and is the publishable
systems measurement.

```bash
# three runs of the M2 command, varying:
#   --oflash-device 1        (iGPU)
#   --oflash-device cpu
#   (--oflash omitted)       (baseline)
```

Deliverable: target tok/s under each config, sustained power draw, thermal steady-state, trainer
steps/s achieved on each device.

## What this doesn't measure

- **Output quality.** Verification is exact-match against the target; the qualification also
  compares deterministic target-only, static-draft and adapted-draft outputs. It does not claim an
  independent quality gain from drafter training.
- **Other model families.** Qwen3.6 DFlash only; gemma4/laguna drafters are out of scope until the
  capture hook is wired there.
- **Multi-tenant / batched serving, tensor-parallel targets.** Single-box, single-stream only.
- **Cross-hardware generality.** All numbers will be from this R9700 + Strix Halo box; other
  iGPU/dGPU combinations are extrapolation, not measurement.
- **Long-horizon adapter drift.** Weeks-scale profile evolution and reservoir adequacy beyond the
  M3 two-session test are unmeasured.
