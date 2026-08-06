<p align="left">
  <a href="../../README.md">← lucebox-hub</a>
</p>

<h1 align="center">Luce OFlash</h1>

<p align="center">
  <strong>The drafter learns your workload while the box serves it.</strong><br/>
  Speculative decode speed is capped by the drafter's acceptance rate, and a statically distilled
  drafter degrades off-distribution.<br/>
  OFlash captures what the verify step already computes, fine-tunes a LoRA on the idle iGPU, and
  hot-swaps it into the serving drafter.<br/>
  Correctness-safe by construction: exact-match verification means an adapter can change speed,
  never output.<br/><br/>
  <a href="https://lucebox.com">lucebox.com</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a>
</p>

---

```
Qwen3.6-27B Q4_K_M target · dflash-draft-3.6 q4_k_m drafter
R9700 (gfx1201) serves · Strix Halo iGPU (gfx1151) trains

  measurement plan                            acceptance α    decode tok/s
  M0  static drafter (baseline)                 pending         pending
  M1  + offline replay LoRA (static load)       pending         pending
  M2  + online loop (within-session climb)      pending         pending

  No results yet — this card is the plan, not a claim. Numbers land in
  RESULTS.md first, each with the command that produced it (CONTRIBUTING.md:
  numbers without methodology don't get merged).
  reproduce: server/build/dflash_server target.gguf --draft drafter.gguf --oflash   # M0 capture-only
```

> Every verify step already computes the exact training data the drafter is missing — target hidden
> states and target tokens at every draft position — and throws it away. OFlash keeps it: a capture
> hook streams verify-step records into a lock-free shared-memory ring, a sidecar trains a LoRA on a
> mirror of the drafter, and the engine swaps the adapter in at draft-block boundaries behind an
> acceptance guard that auto-rolls-back regressions. Decode is bandwidth-bound on the dGPU while the
> iGPU sits idle, so the training compute is (to be measured, not assumed) roughly free.

## What OFlash is

OFlash is the **online drafter adaptation layer** for the DFlash speculative-decode path in
[`../../server/`](../../server/) (qwen35 backend first). Design doc:
[`server/docs/OFLASH.md`](../../server/docs/OFLASH.md). It has two halves:

1. **Engine side** (`server/src/common/oflash/`): capture hook in the decode loop, SPSC shm ring,
   LoRA tensors in the draft graph (content-overwrite swaps, safe under HIP/CUDA-graph replay),
   acceptance guard state machine, sidecar supervisor, per-profile adapter store.
2. **Trainer side** (this package): attaches to the ring, rebuilds the drafter as a torch mirror by
   dequantizing the GGUF, trains a LoRA (hybrid CE + top-K KL loss, rejection rows up-weighted),
   and exports safetensors adapters the engine hot-swaps.

The only contracts between the two are the ring layout and the adapter file format, both defined in
[`oflash_format.h`](../../server/src/common/oflash/oflash_format.h) and mirrored in
[`ring_format.py`](src/oflash/ring_format.py) / [`adapter_export.py`](src/oflash/adapter_export.py),
cross-checked by golden-bytes tests on both sides.

A bad adapter can only cost speed, never correctness: this engine verifies drafts by exact match
against the target, and the guard bounds the speed downside at "static drafter".

## How it works

```
        dGPU (R9700)                          iGPU / CPU (Strix Halo)
┌───────────────────────────────┐      ┌─────────────────────────────────┐
│ Qwen35Backend decode loop     │      │ OFlash trainer sidecar (Python) │
│                               │      │                                 │
│ verify_batch / verify_tree ───┼──┐   │  shm ring buffer (consumer)     │
│   target logits per node      │  │   │        │                        │
│   target_hidden_cat features  │  └──▶│  replay buffer + rejection      │
│   accept/reject flags         │ shm  │  weighting                      │
│                               │ ring │        │                        │
│ draft_graph (ggml)            │      │  drafter mirror + LoRA          │
│   base GGUF weights (frozen)  │      │  fwd+bwd, AdamW on LoRA only    │
│   + LoRA slot A/B  ◀──────────┼──────┼─ adapter export (safetensors)   │
│                               │ swap │        │                        │
│ acceptance guard ─────────────┼──────┼─▶ accept / rollback signal      │
└───────────────────────────────┘      └─────────────────────────────────┘
```

Per verify step the engine emits one record: bf16 feature rows (the same `target_hidden_cat` slices
the drafter conditions on), draft + target tokens per position, accept flags, and the target's
top-K logprobs. The push never blocks — if the ring is full the record is dropped and a counter
bumped. The trainer accumulates rows, steps AdamW on LoRA params only, and every few steps exports
an adapter and signals `swap_ready`; the engine overwrites the preallocated LoRA tensor contents at
a draft-block boundary, resets the drafter KV cache, and puts the new generation on probation. If
rolling acceptance length regresses, the guard rolls back and backs off. If the sidecar dies, the
engine keeps serving on the last promoted adapter — this feature must never take down inference.

## Quick start

```bash
# 1. Engine: build the server as usual (server/README.md) — OFlash is a runtime flag.
cd server && cmake -B build && cmake --build build -j

# 2. Trainer env (this directory). Installs numpy/safetensors + torch/gguf extras.
#    For iGPU training install the ROCm torch build; CPU torch works at lower cadence.
cd optimizations/oflash && uv sync --extra train

# 3. Serve with online adaptation: the engine spawns bin/oflash-trainer itself.
server/build/dflash_server qwen36-27b-Q4_K_M.gguf \
    --draft dflash-draft-3.6-q4_k_m.gguf \
    --oflash --oflash-trainer-bin optimizations/oflash/bin/oflash-trainer

# Capture-only (M0 telemetry): omit the trainer bin. Ring + acceptance stats,
# no training, no Python needed.
server/build/dflash_server qwen36-27b-Q4_K_M.gguf \
    --draft dflash-draft-3.6-q4_k_m.gguf --oflash
```

The trainer also needs the **target** GGUF (the drafter shares the target's LM head, and noise
embeddings come from its `token_embd`). Configure it once via any of:
`$OFLASH_TARGET_GGUF`, a `--target` argument (wrap `bin/oflash-trainer`), or
`{"target_gguf": "/path/to/target.gguf"}` in `<oflash-dir>/<drafter-hash>/<profile>/trainer.json`.

Adapters persist under `~/.lucebox/oflash/<drafter-hash>/<profile>/`; the next session warm-starts
from the promoted generation.

## Engine knobs

All flags on `dflash_server` (qwen35 + local `--draft` only; off by default):

| Flag | Purpose |
|---|---|
| `--oflash` | enable online drafter adaptation |
| `--oflash-device <cpu\|N>` | trainer sidecar device: `cpu` or a HIP ordinal (default: `1`, the iGPU) |
| `--oflash-profile <name>` | adapter profile (default: `default`) |
| `--oflash-lora-rank <N>` | LoRA rank (default: 16) |
| `--oflash-alpha <F>` | LoRA alpha (default: 32) |
| `--oflash-dir <path>` | adapter store (default: `~/.lucebox/oflash`) |
| `--oflash-ring-mb <N>` | capture ring size in MiB (default: 2048) |
| `--oflash-topk <K>` | target top-K captured per position (default: 32; 0 = skip) |
| `--oflash-trainer-bin <path>` | trainer sidecar executable; empty = capture-only (M0 telemetry) mode |

## Trainer knobs

`bin/oflash-trainer <drafter.gguf> ...` — the engine passes the wiring arguments itself
(`--ring-name`, `--out-dir`, `--profile`, `--rank`, `--alpha`, `--device`, `--drafter-sha256`,
`--start-generation`, `--stream-fd`); the training knobs below are yours to tune
(OFLASH.md §5 defaults):

| Flag | Default | Purpose |
|---|---|---|
| `--target <path>` | trainer.json / `$OFLASH_TARGET_GGUF` | target GGUF for lm_head + token_embd |
| `--lr <F>` | `1e-4` | AdamW learning rate (constant; non-stationary stream) |
| `--kl-lambda <F>` | `0.5` | weight of the top-K KL term vs CE |
| `--reject-weight <F>` | `3.0` | loss up-weight on rejection-adjacent rows |
| `--batch-rows <N>` | `512` | fresh rows accumulated per training step |
| `--export-every <N>` | `8` | training steps between adapter exports |
| `--train-ctx <N>` | `512` | feature-window length per training sample |
| `--reservoir-rows <N>` | `50000` | replay reservoir size (forgetting knob) |
| `--keep-generations <N>` | `4` | adapter generations kept on disk per profile |

## What's ours, what isn't

Online drafter adaptation is established literature; we did not invent the algorithm:

- [**OSD — Online Speculative Decoding**](https://arxiv.org/abs/2310.07177) (ICML 2024): distill
  the drafter online from target rejections; the data-mix/forgetting analysis our replay reservoir
  answers.
- [**OmniDraft**](https://arxiv.org/abs/2507.02659) (NeurIPS 2025): online cross-vocab drafter
  adaptation with the hybrid CE + KL loss shape we use.
- **OnlineSPEC** (ICML 2026): concurrent online-adaptation work.

What's ours is the production engine integration and the hardware split, per
[`OFLASH.md`](../../server/docs/OFLASH.md):

- Capture in the verify path of a serving engine — lock-free shm ring, drop-on-full, zero copies of
  full-vocab logits (top-K only), never blocking decode.
- LoRA hot-swap that survives HIP/CUDA-graph replay: preallocated adapter tensors, content
  overwrites at draft-block boundaries, drafter-KV reset, host-side generation staging for instant
  rollback.
- The acceptance guard: probation windows, auto-rollback, exponential backoff, session kill-switch.
- The **iGPU-trains / dGPU-serves split** on Strix Halo + R9700 class machines: unified memory
  carries features to the trainer without PCIe copies while the dGPU keeps serving.

## Scope and limits

- **qwen35 DFlash path + local `--draft` only.** No remote-IPC drafters, no gemma4/laguna wiring
  yet (the capture contract is target-agnostic; that's later work).
- **Single box, low concurrency.** No multi-tenant or batched-serving concerns, no tensor-parallel
  targets.
- **Correctness-safe, speed-only risk.** Verification is exact-match against the target, so output
  is byte-identical with any adapter; the guard bounds the downside at static-drafter speed.
- **bf16-mirror → Q4_K_M transfer is measured, not assumed.** The trainer fits a LoRA on a
  dequantized mirror while the engine serves the quantized base; M1 evaluates the adapter on the
  engine, not in PyTorch.
- **ROCm torch on gfx1151 is less battle-tested** than dGPU targets; `--oflash-device cpu` is the
  supported fallback (the trainer is cadence-tolerant by design).

## Citation

```bibtex
@software{luce_oflash_2026,
  title  = {Luce OFlash: online drafter adaptation for speculative decoding on iGPU+dGPU boxes},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/optimizations/oflash},
  year   = {2026}
}

@inproceedings{osd_2024,
  title     = {Online Speculative Decoding},
  author    = {Liu, Xiaoxuan and Hu, Lanxiang and Bailis, Peter and others},
  booktitle = {ICML},
  year      = {2024}
}
```

---

Apache 2.0 · [Lucebox](https://lucebox.com)
