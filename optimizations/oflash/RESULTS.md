# OFlash results

**Nothing below is measured yet.** This file is the methodology skeleton the measurements will land
in; every number that arrives must arrive with the exact command that produced it
(CONTRIBUTING.md: numbers without methodology don't get merged). Sections mirror the milestones in
[`server/docs/OFLASH.md`](../../server/docs/OFLASH.md) §9.

All measurements on one box: **AMD Radeon AI PRO R9700** (gfx1201, 32 GB GDDR6) serving the target,
**AMD Ryzen AI Max "Strix Halo" iGPU** (gfx1151, 128 GB unified memory) running the trainer.
Target **Qwen3.6-27B Q4_K_M**; drafter **dflash-draft-3.6 q4_k_m** (EAGLE-style DFlash block
drafter, 16-token blocks). Single-stream decode, same seeds for static-vs-adaptive pairs, same
power limit, same warmup.

Primary metric: decode tok/s as a function of decode-step index and wall-clock session time,
static vs adaptive. Secondary: mean accepted length (AL), acceptance rate α per draft position,
swap/rollback counts, TTFT (must be unchanged), trainer interference on target tok/s, peak memory
on both devices.

## 1. M0 — acceptance-vs-step baseline curves

**Not yet measured.**

Capture-only mode (no trainer): per-step AL and α-per-position over long sessions, plus the capture
overhead itself (target tok/s with vs without `--oflash`; budget < 1%). This baseline is what every
later claim is measured against.

```bash
server/build/dflash_server qwen36-27b-Q4_K_M.gguf \
    --draft dflash-draft-3.6-q4_k_m.gguf --oflash        # no trainer bin: capture-only

python3 harness/benchmarks/generation_benchmark.py run \
    --name oflash-m0-he --url http://127.0.0.1:8080/v1 --model qwen36 \
    --prompts harness/benchmarks/prompts/bench_he.jsonl --json-out m0_he.json
# repeat with bench_math.jsonl, bench_gsm.jsonl, and one real long agentic
# coding session via harness/clients/run_claude_code.sh
```

Per-step AL/α come from the engine's rolling stats (`/props` + logs); extending the harness client
to per-step timing output is part of this milestone. Deliverable: acceptance-vs-decode-step curves
for HumanEval / Math500 / GSM8K and the agentic session, plus the capture-overhead delta.

## 2. M1 — offline replay: does the recipe move α?

**Not yet measured.**

Train the LoRA on M0-captured data offline, then re-run the M0 benchmarks with the adapter loaded
statically (promoted at startup, trainer off). Proves the loss/rank/λ recipe moves α on our drafter
before any online machinery is trusted. This is also where the bf16-mirror → Q4_K_M transfer gap is
quantified: α in the PyTorch mirror vs α on the quantized engine, same adapter.

```bash
# replay the captured ring into the trainer (same args the engine would pass)
optimizations/oflash/bin/oflash-trainer dflash-draft-3.6-q4_k_m.gguf \
    --ring-name /lucebox-oflash-<pid> \
    --out-dir ~/.lucebox/oflash/<hash16>/default --profile default \
    --rank 16 --alpha 32 --device 1 --drafter-sha256 <sha256> \
    --target qwen36-27b-Q4_K_M.gguf

# then re-run the M0 commands with the exported adapter promoted
```

Go/no-go: **≥ +0.05 α** on a held-out slice of the same distribution. Report: α before/after per
benchmark, adapter generation used, mirror-vs-engine α gap.

## 3. M2 — within-session AL climb (online loop)

**Not yet measured.**

Full loop: ring → live trainer → swap → guard, on a domain-shifted workload the drafter is weak on
(non-English prose, or a codebase in an under-represented language). Also the adversarial case:
rapidly switching domains must show **zero regressions** with the guard on (worst case = static
drafter speed).

```bash
server/build/dflash_server qwen36-27b-Q4_K_M.gguf \
    --draft dflash-draft-3.6-q4_k_m.gguf \
    --oflash --oflash-trainer-bin optimizations/oflash/bin/oflash-trainer
# domain-shifted workload through the same per-step harness client as M0
```

Deliverable: AL vs decode-step within one session (static vs adaptive, same seeds), swap/rollback
counts, and the adversarial-switching run.

## 4. M3 — warm start across sessions

**Not yet measured.**

Persistence: run session 1 (M2 setup), restart the server, measure session-2 AL at t=0 vs
session-1 AL at t=0 on the same workload. Also verifies generation GC (`--keep-generations 4`) and
that a quarantined (rolled-back) generation is never warm-started.

```bash
# session 1, then restart the same command; the engine warm-starts from
# ~/.lucebox/oflash/<hash16>/default/promoted.json
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

- **Output quality.** Verification is exact-match against the target, so output is identical by
  construction with or without an adapter; no quality benchmark is run and none is needed.
- **Other model families.** qwen35 DFlash only; gemma4/laguna drafters are out of scope until the
  capture hook is wired there.
- **Multi-tenant / batched serving, tensor-parallel targets.** Single-box, single-stream only.
- **Cross-hardware generality.** All numbers will be from this R9700 + Strix Halo box; other
  iGPU/dGPU combinations are extrapolation, not measurement.
- **Long-horizon adapter drift.** Weeks-scale profile evolution and reservoir adequacy beyond the
  M3 two-session test are unmeasured.
