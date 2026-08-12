# OFlash: online drafter adaptation

Status: implemented behind `--oflash`. Capacity analysis says the intended
R9700 + Strix Halo split is feasible, but simultaneous serving/training and
its performance still require the staged qualification below.

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

## Why output correctness is preserved

The target still verifies every proposed token by exact match. The adapter can
change which tokens the drafter proposes, but it cannot make an unverified
token enter the output. A poor adapter can therefore reduce speed, not change
the target model's accepted output.

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
5. The sidecar exports a safetensors adapter. The engine validates its model
   identity, rank and tensor shapes before loading it.
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
- disk: only promoted adapter generations and metadata.

The capture path reads features from discrete R9700 VRAM into the host
`/dev/shm` ring. Python reads them in host RAM and copies selected windows to
the Strix allocation. Strix unified memory avoids another discrete VRAM pool;
it does not eliminate the R9700-to-host PCIe transfer.

On the audited machine the R9700 exposes 31.86 GiB VRAM, the Strix KFD heap is
96 GiB GPU-accessible, and system RAM is 128 GiB. A conservative upper-bound
estimate for the current Qwen3.6 draft mirror is about 8 GiB on Strix plus a
similar temporary host staging peak; 10,000 raw replay rows are about 0.48 GiB
and the default ring is 0.5 GiB. These are capacity estimates, not a successful
training result. `rocm-smi` may show only the Strix dedicated segment, so host
`MemAvailable` and swap are the useful OOM guards.

Training is asynchronous, but it is not assumed to be free: power, thermal,
memory-controller and PCIe interference must be measured. There is no silent
GPU-to-CPU fallback. A missing/failed HIP device leaves the sidecar draining
the ring in capture-only mode; request CPU training explicitly with
`--oflash-device cpu` and keep the same host-memory guards.

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
  torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
python -m pip install 'numpy==1.26.4' 'safetensors>=0.4' 'gguf>=0.10'
python -m pip install --no-deps -e .
cd ../..
```

The exact AMD wheel URLs and Strix FP16/MIOpen smoke are in the
[trainer README](../../optimizations/oflash/README.md#quick-start). Require
`python -m torch.utils.collect_env` to report ROCm and a MIOpen runtime; the
wheels may already provide it. If it does not, or the MIOpen smoke fails to
link, install `miopen-hip` from the matching ROCm 7.2 host repository and
retest. That fallback is a host package, not something to pip-install. Its
architecture-specific kernel database is optional and affects warm-up, not
correctness.

## Usage

Capture verification data without starting a trainer:

```bash
server/build/dflash_server /models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft /models/dflash-draft-3.6-q8_0.gguf \
  --target-device hip:0 --draft-device hip:0 \
  --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
  --oflash --oflash-ring-mb 256 --oflash-topk 8
```

Run online adaptation on HIP device 1:

```bash
server/build/dflash_server /models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft /models/dflash-draft-3.6-q8_0.gguf \
  --target-device hip:0 --draft-device hip:0 \
  --max-ctx 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
  --oflash \
  --oflash-device 1 --oflash-dtype auto \
  --oflash-ring-mb 512 --oflash-topk 8 \
  --oflash-trainer-bin optimizations/oflash/bin/oflash-trainer
```

The integrated engine passes the target GGUF to the trainer because the mirror
shares its output head and token embeddings. `OFLASH_TARGET_GGUF`, `--target`,
or `trainer.json` is only needed for a direct trainer invocation.

Use the Q8_0 drafter shown above: its published model card reports F16-like
acceptance, whereas Q4_K_M saves less than 1 GiB but loses roughly 17
acceptance points. This Q8_0 file embeds the 2048-token `[S,S,S,S,F]`
sliding-window layout. Do not add `--draft-swa`; OFlash rejects an override
that would make the engine attention layout differ from the Python mirror.

Adapters are stored under
`~/.lucebox/oflash/<drafter-hash>/<profile>/` and the next session starts from
the last promoted generation.

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
   optimizations/oflash/bin/oflash-trainer \
     /models/dflash-draft-3.6-q8_0.gguf \
     --ring-name=/lucebox-oflash-<pid> \
     --target=/models/Qwen3.6-27B-Q4_K_M.gguf \
     --out-dir=/tmp/oflash-smoke --profile=smoke \
     --device=1 --dtype=auto \
     --batch-rows=64 --train-ctx=64 --reservoir-rows=2048
   ```

   Require `accelerator preflight passed`, `mirror loaded`, and no automatic
   `training disabled` message. Stop this direct trainer before continuing;
   two consumers must never attach to the same ring.
3. **Integrated defaults.** Restart with the online command in Usage. Hold
   `--max-ctx 4096` for ten minutes under the same request stream. Defaults are
   ring 512 MiB, top-K 8, batch 128, train context 128, reservoir 10,000 and
   `auto`/FP16 on Strix. Require `/props.oflash.trainer_alive=true`,
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

The first go/no-go result is whether acceptance and decode speed improve
without meaningful serving interference. Until those measurements exist,
OFlash is an implemented experiment, not a performance claim.

## Implementation map

- `server/src/common/oflash/` contains the ring, formats, supervisor, adapter
  loading and acceptance guard.
- `server/src/draft/draft_graph.cpp` applies the optional LoRA to the drafter.
- `server/src/qwen35/` captures verification records and controls safe swaps.
- `optimizations/oflash/` contains the Python trainer and its runtime package.
- `server/test/` and `optimizations/oflash/tests/` cover engine and trainer
  contracts without requiring the production GPU pair.
