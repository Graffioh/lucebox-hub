# OFlash: online drafter adaptation

Status: implemented behind `--oflash`. A bounded 2026-08-13 qualification ran
Qwen3.6-27B Q4_K_M serving on R9700 concurrently with FP16 LoRA training on
Strix Halo, including export, live swap, promotion and warm start. Sustained,
multi-workload performance, power and thermal qualification still follow the
staged ladder below.

OFlash teaches the DFlash drafter from the target model's real verification
results while the server is running. The R9700 keeps serving requests; a
Python sidecar on the Strix Halo (or CPU) trains a small LoRA adapter in the
background.

The goal is simple: a drafter adapted to the current workload should propose
more tokens that the target accepts, reducing the number of expensive target
forwards and increasing decode speed.

## In brief

```text
R9700: target verifies draft tokens
             │
             ├─ features, target tokens and acceptance data
             ▼
      non-blocking shared-memory ring
             │
             ▼
Strix Halo: Python trainer updates a drafter LoRA
             │
             └─ adapter file → guarded hot-swap on the R9700
```

The target already computes the supervision OFlash needs. OFlash captures a
small part of it instead of discarding it, trains only LoRA parameters and
periodically tries the new adapter in the serving drafter.

## Correctness contract and validation

The target verifies every proposed token by exact match. The adapter can change
which tokens the drafter proposes, but it cannot make an unverified token enter
the output. Preserving target output also requires target positions and
recurrent state to be independent of the proposal path. The bounded
qualification exposed and fixed a token-major M-RoPE layout in multi-token
prefill/verification, then matched target-only, static-draft and adapted-draft
output on the fixed prompt. Broader cross-prompt and cross-proposal parity is
still part of qualification rather than a universal byte-identity claim.

Each adapter first enters a probation window. If mean accepted length gets
worse, the engine restores the previous adapter and waits longer before the
next attempt. Adapter swaps happen only between draft blocks and reset the
drafter KV cache, so one block never mixes two adapter generations.

## Runtime flow

1. The decode loop records the features seen by the drafter, target tokens,
   accept/reject flags and optional target top-K log-probabilities.
2. Records enter a fixed-size shared-memory ring. Producing a record never
   blocks decoding; a full ring drops the record and increments a counter.
3. The sidecar dequantizes the drafter into a mirror and trains a rank-16 LoRA
   with a mixture of cross-entropy and top-K KL loss. `auto` uses FP16 on a
   GPU and FP32 on CPU.
4. A replay reservoir mixes recent and older examples to reduce forgetting
   when the workload changes.
5. The sidecar exports a safetensors adapter. The engine validates both GGUF
   hashes, resolved RoPE/SWA/mask semantics, rank and tensor shapes before
   loading it.
6. The engine overwrites preallocated, pointer-stable LoRA buffers at a safe
   draft-block boundary. This remains compatible with graph replay.
7. The acceptance guard promotes the generation or rolls it back.

If the trainer crashes or cannot start, inference continues with the last good
adapter. If no trainer executable is configured, OFlash runs in capture-only
mode.

## Hardware split

The intended setup is:

- R9700: target model, serving drafter, verification and token generation;
- Strix Halo: background LoRA training;
- host shared memory: bounded transfer of captured training records;
- disk: bounded candidate generations plus promoted-generation metadata.

The capture path reads features from discrete R9700 VRAM into the host
`/dev/shm` ring. Python reads them in host RAM and copies selected windows to
the Strix allocation. Strix unified memory avoids another discrete VRAM pool;
it does not eliminate the R9700-to-host PCIe transfer.

On the audited machine the R9700 exposes 31.86 GiB VRAM, the Strix KFD heap is
96 GiB GPU-accessible, and system RAM is 128 GiB. During the bounded integrated
run the R9700 allocation was about 19.25 GiB, the Strix trainer's KFD/GTT
allocation about 8.55 GiB, and host `MemAvailable` stayed near 98 GiB. Swap did
not grow; there was no OOM, GPU fault/reset or capture drop. A 10,000-row replay
reservoir is about 0.48 GiB and the default ring is 0.5 GiB. `rocm-smi` may show
only the Strix dedicated segment, so host `MemAvailable`, KFD/GTT accounting and
swap are the useful OOM guards. Exact measurements are in
[`RESULTS.md`](../../optimizations/oflash/RESULTS.md).

Training is asynchronous, but it is not assumed to be free: power, thermal,
memory-controller and PCIe interference must be measured. There is no silent
GPU-to-CPU fallback. A missing/failed HIP device leaves the sidecar draining
the ring in capture-only mode; request CPU training explicitly with
`--oflash-device cpu` and keep the same host-memory guards.

### What Lucebox has already tested

The published evidence is useful, but it covers three different configurations:

- [R9700 Qwen3.6](../README.md#amd-hip-backend-strix-halo-rx-7900-xtx): Q4_K_M target plus the
  Lucebox Q4_K_M draft, ROCm 7.1.1 and DDTree budget 22. The 10-prompt HumanEval run at
  `n_gen=256` measured 54.65 tok/s mean and AL 7.14.
- [Strix Halo Qwen3.6](https://www.lucebox.com/blog/amd): the same target plus the Lucebox Q8_0
  draft, ROCm 7.2.2, SWA 2048, DDTree budget 22 and fast rollback. The 10-prompt `n_gen=128` run
  measured 26.85 tok/s, AL 5.58 and 34.9% acceptance.
- [R9700 + Strix Halo together](https://www.lucebox.com/blog/deepseek-v4-asymmetric-parallelism):
  ROCm 7.2.4 ran asymmetric DeepSeek-V4 MoE inference concurrently across both GPUs. It explicitly
  treats 32 GB R9700 memory and 128 GB Strix memory as separate domains, not a flat 160 GB pool.

The Qwen measurements were separate single-GPU runs. The simultaneous test was not Qwen and did
not run a PyTorch optimizer. Together they are strong component-level feasibility evidence, not a
measurement of the OFlash split.

## Trainer environment

Use a dedicated environment with AMD's official Ubuntu 24.04 / Python 3.12
ROCm 7.2.1, PyTorch 2.9.1 wheel set. Do not let an unpinned `uv sync` choose a
generic CUDA or CPU torch build for the production sidecar.

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

The exact AMD wheel URLs and Strix FP16/MIOpen smoke are in the
[trainer README](../../optimizations/oflash/README.md#quick-start). Require
`python -m torch.utils.collect_env` to report ROCm and a MIOpen runtime; the
tested wheels did not provide the MIOpen shared library on the audited host.
If it is absent, or the MIOpen smoke fails to link, install `miopen-hip` from
the matching ROCm 7.2 host repository and
retest. That fallback is a host package, not something to pip-install. Its
architecture-specific kernel database is optional and affects warm-up, not
correctness.

## Usage

Download the target and official Q4_K_M drafter at pinned revisions and verify them:

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

Capture verification data without starting a trainer:

```bash
server/build/dflash_server server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/oflash/dflash-draft-3.6-q4_k_m.gguf --draft-swa 2048 \
  --target-device hip:0 --draft-device hip:0 \
  --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
  --ddtree --ddtree-budget 22 \
  --oflash --oflash-ring-mb 256 --oflash-topk 8
```

Run online adaptation on HIP device 1:

```bash
server/build/dflash_server server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/oflash/dflash-draft-3.6-q4_k_m.gguf --draft-swa 2048 \
  --target-device hip:0 --draft-device hip:0 \
  --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
  --ddtree --ddtree-budget 22 \
  --oflash \
  --oflash-device 1 --oflash-dtype auto \
  --oflash-ring-mb 512 --oflash-topk 8 \
  --oflash-trainer-bin optimizations/oflash/bin/oflash-trainer
```

The integrated engine passes the target GGUF to the trainer because the mirror
shares its output head and token embeddings. `OFLASH_TARGET_GGUF`, `--target`,
or `trainer.json` is only needed for a direct trainer invocation.

The commands use the official Lucebox Q4_K_M conversion recommended by its
[model card](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF) and used for the published
R9700 result. This legacy GGUF declares RoPE 1,000,000 without an embedded SWA pattern, so the
published 2,048-token `[S,S,S,S,F]` behavior requires `--draft-swa 2048`. OFlash passes the
post-override values to the Python mirror. It preserves RoPE 1,000,000 across park/unpark rather
than silently copying the target's 10,000,000 base.

Adapter format v2 binds each file to the exact draft and target GGUF SHA-256 values plus the
complete resolved RoPE/SWA/mask contract. Engine-only draft debug overrides disable OFlash rather
than allow the trainer and
serving graph to diverge. Lucebox Q8_0 remains useful for parity/debug, but Q4 is the selected
serving artifact here. The trainer dequantizes either one into dense FP16, so Q4's roughly 0.78 GB
serving/file saving does not materially shrink the Strix training mirror.

Adapters are stored under
`~/.lucebox/oflash/<drafter-hash>/<profile>-sem-<contract-hash>/` and the next session starts from
the last promoted generation only when both models and the resolved draft contract are identical.
The contract hash includes the full target hash, rank and exact float32 alpha; the server logs the
exact profile directory.

## R9700 and Strix Halo qualification ladder

Do not jump directly to a long-context online run. First confirm with
`rocm-smi --showproductname` that this box still maps R9700 to ordinal 0 and
Strix to ordinal 1. Obtain an exclusive window on both devices: HIP visibility
does not reserve memory against another ROCm process. Do not start while
`rocm-smi --showpids` reports an unrelated KFD client. Keep both devices
visible in the parent and use the explicit `hip:0` placement above; the child
selects Strix before importing PyTorch. In separate terminals watch the R9700 row, host memory,
kernel faults and OFlash counters:

```bash
watch -n 2 rocm-smi --showuse --showmeminfo vram
watch -n 2 'grep -E "MemAvailable|SwapTotal|SwapFree" /proc/meminfo'
sudo journalctl -kf
watch -n 5 'curl -fsS http://127.0.0.1:8080/props | jq .oflash'
```

1. **Capture only.** Run the first command in Usage at `--max-ctx 4096`, with
   Q4_0 for both K and V, a 256 MiB ring and no trainer executable. Send a
   small fixed prompt set. Require stable server allocation,
   `records_written` increasing, no drops for that bounded run, and identical
   outputs versus OFlash off.
2. **Isolate the Strix allocation.** Keep that capture-only server running,
   copy its `/lucebox-oflash-<pid>` ring name from the log, and attach exactly
   one disposable trainer. Use minimum context and a small reservoir; this
   qualifies the accelerator preflight, mirror load and at least 60 seconds
   of training without allowing a serving hot-swap:

   ```bash
   OFLASH_RING_NAME=/lucebox-oflash-12345  # paste the name from the server log
   optimizations/oflash/bin/oflash-trainer \
     server/models/oflash/dflash-draft-3.6-q4_k_m.gguf \
     --ring-name="$OFLASH_RING_NAME" \
     --target=server/models/oflash/Qwen3.6-27B-Q4_K_M.gguf \
     --out-dir=/tmp/oflash-smoke --profile=smoke \
     --device=1 --dtype=auto \
     --drafter-sha256=e2500e90165a0f8e7b52c9882c29ed1fa391c60b300ff11b817bf10e31fa092e \
     --target-sha256=41ae55b347988dca8352ed4c85f3d8ee3804a23cc89aaea165c071d61ec3cca0 \
     --resolved-rope-theta=1000000 \
     --resolved-swa-window=2048 \
     --resolved-swa-pattern=1,1,1,1,0 \
     --resolved-mask-token-id=248070 \
     --batch-rows=64 --train-ctx=64 --reservoir-rows=2048
   ```

   Require `accelerator preflight passed`, `mirror loaded`, and no automatic
   `training disabled` message. Stop this direct trainer before continuing;
   two consumers must never attach to the same ring. Direct mode hashes both
   files before allocating the mirror even when expected digests are supplied;
   the explicit values make a wrong path fail closed instead of relabeling it.
3. **Integrated defaults.** Restart with the online command in Usage. Hold
   `--max-ctx 4096` for ten minutes under the same request stream. Defaults are
   ring 512 MiB, top-K 8, batch 128, train context 128, reservoir 10,000,
   deterministic replay seed 0 and `auto`/FP16 on Strix. Require
   `/props.oflash.trainer_alive=true`,
   `/props.oflash.training_disabled=false`, increasing `steps`, and bounded
   backlog before accepting a promotion. A self-disabled trainer remains alive
   to drain the ring, so `trainer_alive` alone is not a training-health signal.
4. **Raise context one step at a time.** Repeat the fixed test at 16K and then
   32K, retaining Q4_0 K/V. Do not qualify 128K merely from the 32K result;
   test every intended production context and concurrency separately.

Stop the trainer/server immediately at any of these thresholds:

- R9700 used VRAM reaches **28.5 GiB** (leaves about 3.36 GiB of this card's
  observed 31.86 GiB for compute/drafter reserve), or any allocation OOM;
- host `MemAvailable` falls below **16 GiB**;
- used swap grows by more than **1 GiB** from the pre-run baseline, or faster
  than **256 MiB/min for two consecutive minutes**;
- any `amdgpu` reset, GPU page fault, OOM kill, illegal-memory-access error, or
  repeated trainer respawn appears;
- after a consumer is attached, the 60-second delta satisfies
  `dropped / (written + dropped) > 1%`, drops rise for three consecutive
  samples, or backlog remains above 75% of ring capacity for 60 seconds;
- three identical warmed workload repeats show more than 10% worse decode
  throughput or time-to-first-token than capture-only.

The trainer reduces a 128-row context to 64 after a train-step OOM and disables
training after another OOM at the minimum. That containment is not permission
to continue past a stop threshold. Diagnose first; select CPU only by changing
to `--oflash-device cpu` explicitly.

| Flag | Default | Meaning |
|---|---:|---|
| `--oflash` | off | Enable capture and online adaptation support |
| `--oflash-device <cpu\|N>` | `1` | Device used by the trainer sidecar |
| `--oflash-dtype <type>` | `auto` | Mirror dtype; auto is FP16 on GPU and FP32 on CPU |
| `--oflash-profile <name>` | `default` | Independent persistent adapter profile |
| `--oflash-lora-rank <N>` | `16` | Fixed LoRA rank |
| `--oflash-alpha <F>` | `32` | LoRA scaling value |
| `--oflash-dir <path>` | `~/.lucebox/oflash` | Adapter and profile store |
| `--oflash-ring-mb <N>` | `512` | Shared-memory ring capacity in MiB |
| `--oflash-topk <K>` | `8` | Target top-K values captured; `0` disables them |
| `--oflash-trainer-bin <path>` | empty | Trainer executable; empty means capture-only |

Trainer defaults are `--batch-rows 128`, `--train-ctx 128`,
`--reservoir-rows 10000`, and `--dtype auto`. The engine launcher uses these
defaults; pass non-default trainer flags only through a deliberate wrapper or
direct qualification invocation.

## Current limits

- `qwen35` with a local DFlash draft model only.
- The drafter must remain resident for the whole request.
- The target must use one device; layer split and tensor parallel are rejected.
- PFlash compression, lazy/request-scoped drafting and remote drafters are not
  supported.
- GPU training uses a dequantized FP16 mirror by default while serving may use
  a quantized GGUF. Adapter transfer to the quantized engine must be validated
  empirically.
- The feature is off by default and is aimed at a single-box, low-concurrency
  deployment.

The ring contains token IDs and activations derived from user requests. They
remain on the box and are not written to disk by default; persisted adapters
contain weights and metadata, not raw records.

## What to measure

Compare the same workload with OFlash disabled, capture-only and training
enabled. Record:

- accepted length and decode tokens/s over session time;
- capture overhead and dropped-ring records;
- adapter promotions, rollbacks and time to first improvement;
- time to first token and end-to-end latency;
- R9700 throughput while the Strix or CPU trainer is active;
- power, temperatures and memory use on both devices;
- whether the FP16-trained LoRA improves the quantized serving drafter.

The bounded fixed-prompt run improved acceptance and decode speed without OOM,
GPU fault or capture drops. Held-out workload curves, sustained interference,
power and thermal measurements still do not exist, so OFlash remains an
implemented experiment rather than a general performance claim.

## Implementation map

- `server/src/common/oflash/` contains the ring, formats, supervisor, adapter
  loading and acceptance guard.
- `server/src/draft/draft_graph.cpp` applies the optional LoRA to the drafter.
- `server/src/qwen35/` captures verification records and controls safe swaps.
- `optimizations/oflash/` contains the Python trainer and its runtime package.
- `server/test/` and `optimizations/oflash/tests/` cover engine and trainer
  contracts without requiring the production GPU pair.
