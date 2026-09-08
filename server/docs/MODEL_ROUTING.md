# Multiple models behind one Lucebox endpoint

`--next-model` loads another model in the same `dflash_server` process. There
is one listening socket. Each model keeps its own tokenizer, chat template,
model-card defaults, and existing execution path on an independent worker
thread: a continuous-batch scheduler or the DeepSeek4 single-request worker.

This is the first load-routing implementation. It does not combine models in
one GPU batch, share KV state, move an active request between models, or perform
quality collaboration.

For Qwen batching with DS4 + DSpark at concurrency one, use the
[hybrid recipe](HYBRID_SERVING.md).

## Qwen on R9700 plus DS4 on Strix Halo

Build with both `gfx1201` and `gfx1151` code objects using the project's
architecture option (the generic CMake architecture setting is overridden):

```bash
cmake -S . -B build-hip -DDFLASH27B_GPU_BACKEND=hip \
  '-DDFLASH27B_HIP_ARCHITECTURES=gfx1201;gfx1151' -DGGML_HIP_NO_VMM=ON
cmake --build build-hip --target dflash_server -j "$(nproc)"
```

Run those commands from `server/` in a HIP-configured build environment.
Set `R9700_GPU` and
`STRIX_GPU` to the appropriate device indices in your ROCm environment; ensure
any inherited device-visibility settings agree with that mapping. In this
example the visible devices become `hip:0` (R9700) and `hip:1` (Strix Halo).
Replace the model paths with your actual target and compatible draft files.

```bash
ROCR_VISIBLE_DEVICES="$R9700_GPU,$STRIX_GPU" \
./build-hip/dflash_server /models/Qwen3.8-27B-IQ4_XS.gguf \
  --host 127.0.0.1 --port 8080 \
  --model-name qwen --target-device hip:0 \
  --draft /models/Qwen3.8-27B-DFlash2-Q8_0.gguf --draft-device hip:0 \
  --max-ctx 4096 --max-concurrency 2 --kv-pool-tokens 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --next-model /models/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
  --model-name ds4 --target-device hip:1 \
  --max-ctx 4096 --max-concurrency 2 --ds4-prefill exact
```

Both Qwen target and draft stay on R9700; DS4 runs entirely on Strix. Do not
inherit the DS4 dual-GPU expert-split environment when using this placement.
Model residency, cache budgets, and device placement still need qualification
with the real model files before increasing context or concurrency.

Options after `--next-model` belong to that model. Repeat it to register more
models; the routing table derives each model's capacity from its sequence
engine, or uses capacity one for a single-request worker. Give each model a unique, nonempty `--model-name`, excluding the
reserved name `auto`. Host, port, and CORS configuration belong to the first
model's listener. All models load before any worker starts or the endpoint
opens. Every block is parsed and checked for CLI errors and duplicate names
before loading starts; errors identify the offending block. A load failure
exits and releases models already loaded.

### Keep startup settings together

For a launch script, Bash arrays make the shared listener and each model's
settings easy to edit without mixing their scope:

```bash
listener=(--host 127.0.0.1 --port 8080)
qwen=("$QWEN_MODEL" --model-name qwen --target-device hip:0
      --draft "$QWEN_DRAFT" --draft-device hip:0
      --max-ctx 4096 --max-concurrency 2 --kv-pool-tokens 8192
      --cache-type-k q8_0 --cache-type-v q8_0)
ds4=("$DS4_MODEL" --model-name ds4 --target-device hip:1
     --max-ctx 4096 --max-concurrency 2 --ds4-prefill exact)
ROCR_VISIBLE_DEVICES="$R9700_GPU,$STRIX_GPU" \
  ./build-hip/dflash_server "${qwen[@]}" "${listener[@]}" \
  --next-model "${ds4[@]}"
```

Set the three model-path variables and the two GPU indices first. All CLI
options in a model block start from that model's own defaults; they do not
inherit the previous block. This includes context, concurrency, draft options,
sampling defaults, templates, and admission coalescing. Omitted values are
resolved against the corresponding model card where supported.

Qwen's `--cache-type-k/v` overrides are stored in its backend configuration and
apply to both KV budgeting and allocation. They override environment defaults
without changing them. DS4 uses its own fixed cache layout: those two flags do
not alter DS4's cache and emit a warning if supplied. Its old Q4 launch flags
should be omitted.

Environment variables remain **process-wide**. Set ROCm visibility and shared
kernel controls once, before the executable. There is no per-model `env` block:
arbitrary environment isolation cannot be promised within one process, and
some kernel controls are cached or read while requests run. DS4's automatic
16-column MMVQ setting belongs to its GPU backend context, so it does not
change Qwen's drafter dispatch.
Explicit environment overrides still need to be compatible with the chosen
shared process. Do not copy two separate launch environments blindly.

Inspect the effective settings and current occupancy after startup:

```bash
curl -s http://127.0.0.1:8080/props | jq '.models[] | {
  id, execution_mode, target_device, draft_device, capacity, in_flight, max_context,
  props
}'
```

Each response's `model` tells clients which model answered. Use an explicit
model name for stable behavior across a conversation; use `auto` when either
model is acceptable.

## Request selection and admission

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain a binary search."}],"max_tokens":128}'
```

- `model: "auto"` selects the model with the lowest reserved-request count
  divided by its slot capacity, among models with a free reservation. Equal
  occupancy selects the first model in launch order.
- Explicit `model: "qwen"` or `model: "ds4"` stays on that model. Omitting
  `model` uses the first model, matching the existing default-model behavior.
- With two slots on each model, four overlapping requests reserve two slots
  each. There is no wait for a full batch: a single request runs immediately
  subject to its existing scheduler's admission window.
- When all eligible reservations are occupied, the server returns HTTP 503.
  An explicitly selected full model returns 503 even if another is idle.
  Unknown model names return 404. This version adds no global waiting queue
  and performs no retries or fallback to another model.
- Reservations cover model-specific parsing, pending admission, generation, retirement, and
  final output draining. They are released exactly once when the selected
  request handler returns. A disconnect does not free capacity while the
  engine still owns the request. Slow output draining can conservatively hold
  a reservation after its engine slot becomes free.
- Slot count is an admission ceiling, not a promise that the KV pool can admit
  every prompt immediately. Existing scheduler behavior still handles pool-full
  admission and client cancellation within the bounded reservation count.

Selection precedes model-specific request parsing and tokenization. Responses
identify the actual selected model, including streaming events. Existing chat
completions, Responses, and Anthropic Messages parsing and output formatting
are reused. Anthropic `count_tokens` requires an explicit model and does not
consume a generation reservation; `auto` is rejected because tokenizers differ.

Automatic selection currently considers occupancy only. Context overflow and
unsupported inputs are handled by the selected model's existing validation;
there is no capability-aware rerouting. Use explicit model selection when a
request depends on a particular model, context bound, template, or tool format.
There is no conversation affinity: separate `auto` requests may use different
models, even within one conversation.

## Introspection and lifecycle

`/v1/models` lists every model plus `auto`, including the existing Codex discovery
schema when `client_version` is supplied. The automatic alias advertises the
minimum configured context across the registered models. Its other discovery
defaults come from the first model; actual generation defaults are selected
per model after routing.

`/props` retains the first/default model's existing properties and adds
`routing: "least-occupied"`, `automatic_model: "auto"`, and a `models` array.
Each entry includes its name, execution mode (`batched` or `single-request`),
slot capacity, current reservation count,
device placement, model properties, and scheduler status. `/status/json` adds
the same routing snapshot to the default model's status. The single-model HTML
status page and status-event stream are unavailable in multi-model mode; use
the JSON snapshot to inspect both workers.

SIGINT/SIGTERM stops admission and signals every worker. Batch schedulers
retire their requests; the single-request worker polls cancellation at the
backend's existing work boundaries. The listener waits for workers and request handlers
before model state is destroyed. HTTP reads are interruptible even when a
client has sent only part of an upload.

## Initial supported scope and validation

Models can expose a local paged sequence engine or use DeepSeek4's normal
single-request path. See [hybrid serving](HYBRID_SERVING.md) for batched Qwen
with single-request DS4 + DSpark. The CLI rejects
sharded targets, remote drafts, PFlash/compression forwarding, request-scoped
drafts, and expert-routing collection in this mode. It also rejects per-model
CLI switches that mutate process-wide policy, including SpecLA, KVFlash, Spark,
peer access, and rollback overrides. Qwen KV-type CLI overrides are per-model;
shared environment defaults and kernel settings remain process-wide.
Existing single-model launches retain their CLI behavior.

The implementation was compiled for both HIP `gfx1151` and `gfx1201`, verified
in the actual compile commands. The model-free server suite passes all 456
tests. HTTP integration tests cover routing, distinct tokenizers/templates and
sampling defaults, malformed input, overload, cancellation, response protocols,
and handler lifetimes. A GPU regression verifies independent backend dispatch
settings and preserves the tested ROCmFP4 arithmetic through 16 columns.

The real Qwen3.8-27B IQ4_XS + Q8_0 DFlash2 pair on R9700 and monolithic
DeepSeek-V4-Flash ROCMFP2 on Strix passed the 2+2 workload with 4,096-token
context limits:

- Explicit and automatic C4 routing both reached two active reservations per model.
- Each model's generated outputs matched between its C2 run and mixed C4 run
  on the two comparison prompts. DS4 also matched its previous environment-based
  dispatch control after moving that default into its own backend context.
- The repository canonical concurrency benchmark completed 12/12 mixed
  code/prose requests, each with 128 completion tokens, across three C4 waves.
- TCP-reset cancellation and subsequent slot reuse passed on each model.
- Shutdown with a request active on each model and an incomplete HTTP upload
  closed every connection and exited successfully.

These are short-prompt serving checks on this exact pair. They do not qualify
long-context quality, additional models/devices, or a performance improvement.
Arbitrary per-model environment variables remain unsupported.
