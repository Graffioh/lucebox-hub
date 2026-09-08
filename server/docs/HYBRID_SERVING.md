# Batched Qwen with single-request DeepSeek4

One `dflash_server` listener can run Qwen with four sequence slots on R9700
and DeepSeek4 with its normal single-request worker on Strix Halo. DS4 can
use fused decode and DSpark through that existing path. Its capacity is one;
Qwen's capacity comes from its sequence engine. Each keeps its own tokenizer,
template, model defaults, queue, and worker.

This extends [model routing](MODEL_ROUTING.md). It does not implement an
agent orchestrator: a client sends four explicitly named `qwen` requests,
collects their candidate answers, then sends one explicitly named `ds4`
request containing those candidates. Use different candidate instructions to
explore different approaches. Four identical deterministic requests do not
provide useful diversity. `model: "auto"` continues to mean load balancing,
not proposal generation or synthesis.

## Launch

Build the server for both GPUs as described in the routing guide. Set the
four model paths and physical GPU indices below. This example maps R9700 to
`hip:0` and Strix Halo to `hip:1` in the shared process. Both the DS4 target
and its DSpark draft stay on Strix.

```bash
listener=(--host 127.0.0.1 --port 8080)
qwen=("$QWEN_MODEL" --model-name qwen --target-device hip:0
      --draft "$QWEN_DRAFT" --draft-device hip:0
      --max-ctx 4096 --max-concurrency 4 --kv-pool-tokens 16384
      --cache-type-k q8_0 --cache-type-v q8_0 --fa-window 0)
ds4=("$DS4_MODEL" --model-name ds4 --target-device hip:1
     --max-ctx 8192 --max-concurrency 1 --default-max-tokens 2048
     --prefix-cache-slots 0 --prefill-cache-slots 0 --disk-prefix-cache off
     --ds4-fused-decode --ds4-expert-top-k 6 --ds4-prefill exact)

ROCR_VISIBLE_DEVICES="$R9700_GPU,$STRIX_GPU" \
DFLASH_DS4_SPEC=1 \
DFLASH_DS4_SPEC_Q=4 \
DFLASH_DS4_FUSED_VERIFY=1 \
DFLASH_DS4_DRAFT="$DS4_DRAFT" \
DFLASH_DS4_DRAFT_GPU=1 \
DFLASH_DS4_DRAFT_CONTEXT_KV_CACHE=1 \
  ./build-hip/dflash_server "${qwen[@]}" "${listener[@]}" \
  --next-model "${ds4[@]}"
```

`DS4_MODEL` can name the public
`DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf`; use a compatible DSpark
artifact for `DS4_DRAFT`. The
[0731 article](https://www.lucebox.com/blog/deepseek-v4-flash-0731)
describes the six-expert single-request recipe and its measured scope.
Its reported rates are workload-dependent, not a guarantee for a synthesis
prompt or this two-model deployment.

Do not pass `--paged-attention` to the DS4 block: even at concurrency one,
that selects a different execution path and does not enable normal DSpark.
Increasing DS4's `--max-concurrency` also selects paged serving, which rejects
DSpark and fused decode. Leave Qwen's normal DFlash `--draft` in its own block;
DSpark currently uses the existing DS4-specific environment settings.

Environment variables remain process-wide. In particular, the DSpark GPU
index refers to the same visible device list as both target blocks; it is
**1** here, whereas a Strix-only process would normally use **0**. Do not
inherit a Strix-only visibility mask, expert-split settings, or conflicting
kernel controls from a separate server launch. The existing DSpark path also
sets its shared MMVQ-width default during loading. All model initialization
finishes before either worker starts; this interface does not provide
arbitrary per-model environment isolation.

## Inspect and use

```bash
curl -s http://127.0.0.1:8080/props | jq '.models[] | {
  id, execution_mode, capacity, in_flight, target_device
}'
```

Expect `qwen` / `batched` / capacity 4 and `ds4` / `single-request` /
capacity 1. A second overlapping request pinned to `ds4` returns HTTP 503;
it does not queue behind a long synthesis or spill onto Qwen. The same
reservation covers parsing, execution, cancellation, and output draining.

For latency measurement, separate the Qwen proposal batch, DS4 prefill,
time to the first visible final-answer token, and final streaming speed.
DSpark's verification width of four checks speculative token positions within
one request. It is not four concurrent synthesis requests. End-to-end latency
still includes waiting for proposals and reading their combined input.

## Validation status

The implementation builds for HIP `gfx1151` and `gfx1201`. All 459 host
server tests pass, including mixed-worker admission, independent progress,
client cancellation and capacity reuse, and shutdown. Six CLI checks cover
invalid capacities, listener placement, duplicate names, shared-policy flags,
and unsupported single-request model families.

Hardware qualification of Qwen C4 plus DS4 C1 with DSpark is pending. The
example above is a launch recipe, not a measured performance claim. In
particular, the hybrid's model residency, DSpark activation, generated-output
comparisons, and time to the first final-answer token still need a GPU run.
