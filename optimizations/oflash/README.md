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
  Exact-match verification is the correctness contract; target-only, static-draft and
  adapted-draft outputs matched in the bounded qualification.<br/><br/>
  <a href="https://lucebox.com">lucebox.com</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a>
</p>

---

```
Qwen3.6-27B Q4_K_M target · Lucebox dflash-draft-3.6 Q4_K_M drafter
R9700 (gfx1201) serves · Strix Halo iGPU (gfx1151) trains

  bounded code-prompt qualification           acceptance α    decode tok/s
  M0  static drafter, capture-only                0.505            84.4
  M2  first promoted online adapter               0.625           104.1

  These are three warmed repeats of one deterministic 111-token prompt, not a
  workload benchmark. Detached training, live swap, promotion and warm start
  passed; full held-out M0-M4 results remain open. Methodology is in RESULTS.md.
  reproduce: server/build/dflash_server target.gguf --draft drafter.gguf --draft-swa 2048 --oflash \
               --ddtree --ddtree-budget 22 --cache-type-k q4_0 --cache-type-v q4_0
```

> Every verify step already computes the exact training data the drafter is missing — target hidden
> states and target tokens at every draft position — and throws it away. OFlash keeps it: a capture
> hook streams verify-step records into a lock-free shared-memory ring, a sidecar trains a LoRA on a
> mirror of the drafter, and the engine swaps the adapter in at draft-block boundaries behind an
> acceptance guard that auto-rolls-back acceptance regressions. A bounded simultaneous run passed;
> sustained serving/training interference, power and thermal budgets remain to be measured. The
> split is a capacity strategy, not a claim that training is free.

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

An adapter changes draft proposals, while the engine rejects every proposed token that does not
exactly match target verification. Preserving target output also requires proposal-independent
target positions and recurrent state. The bounded qualification caught and fixed a multi-token
M-RoPE layout bug, then matched target-only, static-draft and adapted-draft output on the fixed
prompt. The guard rejects accepted-length regressions; capture/trainer interference is measured
separately and remains subject to the documented stop thresholds.

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

The ring lives in host RAM. Capturing a row uses the backend tensor-read path, so features cross
from R9700 VRAM to host memory before Python reads them; selected windows are then copied to the
Strix device. Strix unified memory avoids a separate discrete iGPU VRAM pool, but it does **not**
make the R9700-to-host leg copy-free.

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
# From the repository root: build for the R9700 serving GPU.
cmake -S server -B server/build -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1201 \
  -DDFLASH27B_HIP_SM80_EQUIV=ON
cmake --build server/build --target dflash_server -j
```

Provision the trainer in an explicit `.venv`; do not use an unpinned `uv sync` for the production
HIP sidecar. These Ubuntu 24.04 / Python 3.12 pins are AMD's official Ryzen Linux ROCm 7.2.1,
PyTorch 2.9.1 set. Use AMD's
[Ryzen PyTorch guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html)
to select a different OS/Python ABI.

```bash
cd optimizations/oflash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install 'numpy==1.26.4' 'safetensors>=0.4' 'gguf>=0.10'
python -m pip install --no-deps -e .
cd ../..
```

Pin the exact target and official Lucebox Q4_K_M drafter. The drafter is the
artifact used by Lucebox's published R9700 result; its older GGUF metadata
requires the explicit 2,048-token SWA override shown below.

```bash
mkdir -p server/models/oflash
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf \
  --revision e41d24e0e6909d1e15f79445cd9ac27ced27724a \
  --local-dir server/models/oflash
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF \
  dflash-draft-3.6-q4_k_m.gguf \
  --revision fca10ba135c9d834988e15b7d7f5aee8ebc562a7 \
  --local-dir server/models/oflash

sha256sum server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf \
  server/models/oflash/dflash-draft-3.6-q4_k_m.gguf
# 41ae55b347988dca8352ed4c85f3d8ee3804a23cc89aaea165c071d61ec3cca0  server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf
# e2500e90165a0f8e7b52c9882c29ed1fa391c60b300ff11b817bf10e31fa092e  server/models/oflash/dflash-draft-3.6-q4_k_m.gguf
```

Confirm device ordering with `rocm-smi --showproductname`; on the audited box the R9700 is ordinal
0 and Strix is ordinal 1. Exercise FP16 forward/backward and MIOpen on Strix before a large mirror
allocation:

```bash
HIP_VISIBLE_DEVICES=1 optimizations/oflash/.venv/bin/python - <<'PY'
import torch
import torch.nn.functional as F
assert torch.version.hip and torch.cuda.is_available()
print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))
x = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.float16,
                requires_grad=True)
w = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16,
                requires_grad=True)
F.conv2d(x, w, padding=1).float().mean().backward()
torch.cuda.synchronize()
print("Strix FP16 backward passed; free/total bytes:", torch.cuda.mem_get_info())
PY
HIP_VISIBLE_DEVICES=1 optimizations/oflash/.venv/bin/python -m torch.utils.collect_env
```

`collect_env` must report ROCm and a MIOpen runtime (3.5.1 for this ROCm 7.2 set). The tested wheels
did not provide the MIOpen shared library on the audited host. If MIOpen is absent or the
convolution fails with a MIOpen link error, install
`miopen-hip` from the matching ROCm 7.2 system repository and rerun the test:
`sudo apt-get install miopen-hip`. That fallback is a host package, not something to pip-install. A
gfx-specific kernel database is optional and affects warm-up only.

Get an exclusive window on both GPUs before loading either model. Visibility variables only filter
the devices a process can see; they do not reserve the R9700 or Strix against another ROCm process.
Check `rocm-smi --showpids --showmeminfo vram` and the host stop thresholds below immediately before
each stage. Keep both devices visible in the server process: explicit `hip:0` flags place inference,
then the child scopes itself to ordinal 1 before importing PyTorch.

The first server run must be capture-only, with an intentionally smaller ring and explicit Q4_0 KV:

```bash
server/build/dflash_server \
  server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/oflash/dflash-draft-3.6-q4_k_m.gguf --draft-swa 2048 \
  --target-device hip:0 --draft-device hip:0 \
  --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
  --ddtree --ddtree-budget 22 \
  --oflash --oflash-ring-mb 256 --oflash-topk 8
```

The normal ring default is 512 MiB; 256 MiB is only the first-smoke override. Follow the staged
trainer test and stop thresholds in
[`server/docs/OFLASH.md`](../../server/docs/OFLASH.md#r9700-and-strix-halo-qualification-ladder)
before enabling the integrated sidecar. The engine passes the target path to an integrated trainer;
`--target`, `$OFLASH_TARGET_GGUF`, or `trainer.json` is needed only for direct trainer invocation.

Use the official Lucebox Q4_K_M drafter above. Its
[model card](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF) calls Q4_K_M the recommended
fast draft and Q8_0 the parity/debug option. The legacy file declares RoPE 1,000,000 but no SWA
pattern; `--draft-swa 2048` resolves it to `[S,S,S,S,F]`, matching the published setup. The engine
passes those effective values to the trainer and namespaces/validates adapters by both exact GGUF
hashes plus the resolved RoPE, SWA pattern/window and mask token. Park/unpark preserves the same
semantics.

Keep the published reference configurations separate:

- the [R9700 server README baseline](../../server/README.md#amd-hip-backend-strix-halo-rx-7900-xtx) used the
  Lucebox **Q4_K_M** draft on ROCm 7.1.1: 54.65 tok/s, AL 7.14, 10 HumanEval prompts and
  `n_gen=256` at DDTree budget 22;
- the [Strix Halo result](https://www.lucebox.com/blog/amd) used the Lucebox **Q8_0** draft on
  ROCm 7.2.2 with an explicit 2048 SWA override: 26.85 tok/s, AL 5.58 and 34.9% acceptance over
  10 prompts with `n_gen=128`;
- the Q4_K_M file selected here is the R9700 reference artifact. Strix used Q8_0, so its number is
  capacity/performance evidence rather than an OFlash baseline.

Neither published number is an expected online-training result: both were single-GPU inference
tests. Q4 saves about 0.78 GB over Lucebox Q8_0 on disk and while serving, but the trainer
dequantizes either artifact into a dense FP16 mirror, so it does not provide a comparable Strix
training-memory saving. Measure the quantized-engine/FP16-mirror transfer gap before promotion.

Adapters persist under a semantic namespace such as
`~/.lucebox/oflash/<drafter-hash>/<profile>-sem-<contract-hash>/`; the next session warm-starts
only from a promoted generation with the identical draft, target and resolved model contract. The
server logs the exact directory; the contract hash includes the complete target SHA-256, rank and
exact float32 alpha.

## Engine knobs

All flags on `dflash_server` (qwen35 + local `--draft`, single-device target,
persistent draft residency, and no PFlash compression; off by default):

| Flag | Purpose |
|---|---|
| `--oflash` | enable online drafter adaptation |
| `--oflash-device <cpu\|N>` | trainer sidecar device: `cpu` or a HIP ordinal (default: `1`, the iGPU) |
| `--oflash-dtype <type>` | mirror dtype: `auto`, `fp16`, `bf16`, or `fp32` (default: `auto`; GPU = fp16, CPU = fp32) |
| `--oflash-profile <name>` | adapter profile (default: `default`) |
| `--oflash-lora-rank <N>` | LoRA rank (default: 16) |
| `--oflash-alpha <F>` | LoRA alpha (default: 32) |
| `--oflash-dir <path>` | adapter store (default: `~/.lucebox/oflash`) |
| `--oflash-ring-mb <N>` | capture ring size in MiB (default: 512) |
| `--oflash-topk <K>` | target top-K captured per position (default: 8; 0 = skip) |
| `--oflash-trainer-bin <path>` | trainer sidecar executable; empty = capture-only (M0 telemetry) mode |

## Trainer knobs

`bin/oflash-trainer <drafter.gguf> ...` — the engine passes the wiring arguments itself
(`--ring-name`, `--out-dir`, `--profile`, `--rank`, `--alpha`, `--device`, both model SHA-256
values, resolved RoPE/SWA/mask semantics, `--start-generation`, `--stream-fd`); the training knobs below
are yours to tune (see the matching options in
[`OFLASH.md`](../../server/docs/OFLASH.md#usage)):

| Flag | Default | Purpose |
|---|---|---|
| `--target <path>` | trainer.json / `$OFLASH_TARGET_GGUF` | target GGUF for lm_head + token_embd |
| `--dtype <type>` | `auto` | `fp16` on a GPU and `fp32` on CPU; explicit `bf16`/`fp32` are opt-ins |
| `--lr <F>` | `1e-4` | AdamW learning rate (constant; non-stationary stream) |
| `--kl-lambda <F>` | `0.5` | weight of the top-K KL term vs CE |
| `--reject-weight <F>` | `3.0` | loss up-weight on rejection-adjacent rows |
| `--batch-rows <N>` | `128` | fresh rows accumulated per training step |
| `--export-every <N>` | `8` | training steps between adapter exports |
| `--train-ctx <N>` | `128` | feature-window length per training sample (minimum: 64) |
| `--reservoir-rows <N>` | `10000` | replay reservoir size (forgetting knob) |
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
- The **iGPU-trains / dGPU-serves split** on Strix Halo + R9700 class machines: a bounded host ring
  decouples the processes. Capture still copies features from discrete R9700 VRAM into host memory;
  Strix then reads/copies selected windows into its unified-memory allocation.

## Scope and limits

- **qwen35 DFlash path + local `--draft` only.** No remote-IPC drafters, no gemma4/laguna wiring
  yet (the capture contract is target-agnostic; that's later work).
- **Persistent, single-device decode drafter.** OFlash is rejected with PFlash compression,
  request-scoped/`--lazy-draft` residency, layer-split, or tensor-parallel targets; those paths do
  not preserve the adapted drafter and its pointer-stable graph today.
- **Single box, low concurrency.** No multi-tenant or batched-serving concerns, no tensor-parallel
  targets.
- **Exact-match verification is the correctness contract.** The bounded target-only/static/adapted
  parity check passed after the M-RoPE fix; broader cross-prompt and cross-proposal parity remains
  part of qualification. The guard protects accepted length, while capture and trainer
  interference must still pass the separate throughput stop thresholds.
- **FP16-mirror → Q4_K_M transfer must be measured, not assumed.** `auto` uses FP16 for the GPU mirror
  (FP32 on CPU) while the engine serves the quantized base; M1 evaluates the adapter in the engine,
  not only in PyTorch.
- **There is no implicit GPU-to-CPU fallback.** If the requested HIP device or accelerator preflight
  fails, the sidecar sheds the mirror and drains the ring in capture-only mode. Use
  `--oflash-device cpu` explicitly if CPU training is desired, with the same host-RAM guards.

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
