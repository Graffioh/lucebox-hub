# OFlash: online drafter adaptation

Status: implemented behind `--oflash`; performance on the R9700 + Strix
Halo setup still needs to be measured.

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
3. The sidecar builds a bf16 mirror by dequantizing the drafter GGUF and trains
   a rank-16 LoRA with a mixture of cross-entropy and top-K KL loss.
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
- shared memory: bounded transfer of captured training records;
- disk: only promoted adapter generations and metadata.

Training is asynchronous, but it is not assumed to be free. The real machine
must be tested for power, thermal and memory-controller interference. CPU
training is available as a slower fallback if ROCm PyTorch on the Strix is not
stable enough.

## Usage

Capture verification data without starting a trainer:

```bash
server/build/dflash_server target.gguf \
  --draft dflash-draft.gguf \
  --oflash
```

Run online adaptation on HIP device 1:

```bash
OFLASH_TARGET_GGUF=/models/target.gguf \
server/build/dflash_server /models/target.gguf \
  --draft /models/dflash-draft.gguf \
  --oflash \
  --oflash-device 1 \
  --oflash-trainer-bin optimizations/oflash/bin/oflash-trainer
```

The trainer needs the target GGUF because the drafter shares its output head
and token embeddings. Instead of `OFLASH_TARGET_GGUF`, set `target_gguf` in
`<oflash-dir>/<drafter-hash>/<profile>/trainer.json`.

Adapters are stored under
`~/.lucebox/oflash/<drafter-hash>/<profile>/` and the next session starts from
the last promoted generation.

| Flag | Default | Meaning |
|---|---:|---|
| `--oflash` | off | Enable capture and online adaptation support |
| `--oflash-device <cpu\|N>` | `1` | Device used by the trainer sidecar |
| `--oflash-profile <name>` | `default` | Independent persistent adapter profile |
| `--oflash-lora-rank <N>` | `16` | Fixed LoRA rank |
| `--oflash-alpha <F>` | `32` | LoRA scaling value |
| `--oflash-dir <path>` | `~/.lucebox/oflash` | Adapter and profile store |
| `--oflash-ring-mb <N>` | `2048` | Shared-memory ring capacity in MiB |
| `--oflash-topk <K>` | `32` | Target top-K values captured; `0` disables them |
| `--oflash-trainer-bin <path>` | empty | Trainer executable; empty means capture-only |

## Current limits

- `qwen35` with a local DFlash draft model only.
- The drafter must remain resident for the whole request.
- The target must use one device; layer split and tensor parallel are rejected.
- PFlash compression, lazy/request-scoped drafting and remote drafters are not
  supported.
- Training uses a dequantized bf16 mirror while serving may use a quantized
  GGUF. Adapter transfer to the quantized engine must be validated empirically.
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
- whether the bf16-trained LoRA improves the quantized serving drafter.

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
