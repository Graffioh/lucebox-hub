# Qwen3.8-27B PR #625 baseline

Establish this dense, single-request baseline before measuring PR #626's
concurrent paged path. The two paths intentionally do not share cache or
attention settings.

## Models

Sources:

- target: `bartowski/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-IQ4_XS.gguf`
- drafter: `RadixArk/Qwen3.8-27B-DSpark`, `model.safetensors`

Prepare the permanent local pair with:

```bash
TARGET_SOURCE=/path/Qwen3.8-27B-IQ4_XS.gguf \
DRAFT_SOURCE=/path/RadixArk-Qwen3.8-27B-DSpark/model.safetensors \
LLAMA_QUANTIZE=/path/llama-quantize \
server/scripts/prepare_qwen38_pr625_models.sh
```

This produces and validates:

- target: pure IQ4_XS body, Q5_K `output.weight`, Q6_K `attn_v` and
  `ssm_out`
- drafter: no YaRN, Q8_0, capture layers `4,16,28,40,52`, mask token
  `248077`. PR #625 does not spell out the DSpark quantization command, but
  Q8_0 reproduces its reported compute regime: this setup measured a 1.78x
  speculative/plain step-time ratio on the R9700, versus the PR's ~1.8x.
  The unquantized F16 drafter measured 4.20x and is therefore not the
  benchmark artifact. Set `DRAFT_SCHEME=f16` or `DRAFT_SCHEME=q4-mix` only
  for an explicit ablation.

## Build on Radeon AI PRO R9700

Use ROCm 7.2, `gfx1201`, Release, and HIP graphs:

```bash
cmake -S server -B server/build-pr625-r9700 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1201 \
  -DDFLASH27B_HIP_SM80_EQUIV=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DDFLASH27B_FA_ALL_QUANTS=ON \
  -DDFLASH27B_SERVER=ON -DDFLASH27B_TESTS=OFF
cmake --build server/build-pr625-r9700 -j
```

## Dense single-request launch

Use `HIP_VISIBLE_DEVICES=0` when the R9700 is the first physical GPU. Inside
the process it remains `hip:0`.

```bash
HIP_VISIBLE_DEVICES=0 \
DFLASH_SINGLE_CHAIN_CHECKPOINT_F32=1 \
DFLASH_FAST_ROLLBACK_THRESHOLD=1 \
LUCE_Q8_MEMO=1 \
DFLASH_KV_ROTATE=0 \
server/build-pr625-r9700/dflash_server \
  models/.lucebox/qwen38-pr625/Qwen3.8-27B-PR625-IQ4_XS.gguf \
  --draft models/.lucebox/qwen38-pr625/Qwen3.8-27B-DSpark-RadixArk-no-yarn-q8_0.gguf \
  --target-device hip:0 --draft-device hip:0 \
  --fa-window 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
  --max-ctx 8192 --prefix-cache-slots 0 --prefill-cache-slots 0 \
  --decode-mode speculation --host 127.0.0.1 --port 18140
```

Do not pass `--paged-attention` or `--max-concurrency` for this baseline. For
the AR control, start a fresh process **without `--draft`** and use
`--decode-mode ar`; the dense backend otherwise has a loaded drafter and can
enter its original speculative loop. All target, cache, and attention settings
stay identical. Use greedy 300-token generations with
`harness/benchmarks/prompts/qwen38_pr625.jsonl`, and reject a measurement that
ends before the 300-token cap.

PR #625 reported R9700 decode throughput of 34.3/34.4 tok/s for AR and
45.6/32.4 tok/s for DSpark on its code/prose prompts. Exact numeric parity
requires the original unpublished prompts; the structural check is that code
benefits while prose can remain below AR.
