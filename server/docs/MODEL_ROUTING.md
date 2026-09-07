# Multiple models behind one Lucebox endpoint

`--next-model` loads another model in the same `dflash_server` process. There
is one listening socket. Each model keeps its own tokenizer, chat template,
model-card defaults, sequence engine, and existing continuous-batch scheduler
on an independent worker thread.

This is the first load-routing implementation. It does not combine models in
one GPU batch, share KV state, move an active request between models, or perform
quality collaboration.

## Qwen on R9700 plus DS4 on Strix Halo

Build with both `gfx1201` and `gfx1151` code objects. Set `R9700_GPU` and
`STRIX_GPU` to the appropriate device indices in your ROCm environment; ensure
any inherited device-visibility settings agree with that mapping. In this
example the visible devices become `hip:0` (R9700) and `hip:1` (Strix Halo).
Replace the model paths with your actual target and compatible draft files.

```bash
HIP_VISIBLE_DEVICES="$R9700_GPU,$STRIX_GPU" \
DFLASH27B_KV_K=q4_0 DFLASH27B_KV_V=q4_0 \
./build-hip/dflash_server /models/Qwen3.8-27B-IQ4_XS.gguf \
  --host 127.0.0.1 --port 8080 \
  --model-name qwen --target-device hip:0 \
  --draft /models/Qwen3.8-27B-DFlash2-Q8_0.gguf --draft-device hip:0 \
  --max-ctx 4096 --max-concurrency 2 \
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
engine. Give each model a unique, nonempty `--model-name`, excluding the
reserved name `auto`. Host, port, and CORS configuration belong to the first
model's listener. All models load before any worker starts or the endpoint
opens. A load failure exits and releases models already loaded.

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
- Reservations cover parsing, pending admission, generation, retirement, and
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
Each entry includes its name, slot capacity, current reservation count,
device placement, model properties, and scheduler status. `/status/json` adds
the same routing snapshot to the default model's status. The single-model HTML
status page and status-event stream are unavailable in multi-model mode; use
the JSON snapshot to inspect both workers.

SIGINT/SIGTERM stops admission and signals every scheduler. Each scheduler
retires its requests; the listener waits for workers and request handlers
before model state is destroyed. HTTP reads are interruptible even when a
client has sent only part of an upload.

## Initial supported scope and validation

Each model must expose a local paged sequence engine. The initial CLI rejects
sharded targets, remote drafts, PFlash/compression forwarding, request-scoped
drafts, and expert-routing collection in this mode. It also rejects per-model
CLI switches that mutate process-wide policy, including KV-type overrides,
SpecLA, KVFlash, Spark, peer access, and rollback overrides. Shared KV and kernel
environment settings remain process-wide; they are not isolated per model.
Existing single-model launches retain their CLI behavior.

The implementation was compiled for HIP `gfx1151;gfx1201`. Host-only HTTP
integration tests exercise the real listener and scheduler with two controlled
sequence engines: 2+2 admission and overload, independent progress, per-model
tokenization/templates/sampling defaults, malformed input and failed admission,
reset/disconnect retirement, streaming, discovery, and shutdown with an
incomplete upload. The full model-free server suite also passes.

**Real-model simultaneous two-GPU execution is not yet qualified.** GPU execution
was deferred because a separate benchmark was running. Before promoting this
feature, run the exact Qwen/DFlash2 + monolithic DS4 pair together, confirm both
workers advance at C=4 with a 2+2 split, and check mixed prompts, cancellation,
slot reuse, output validity, and shutdown. Host tests do not establish GPU
runtime isolation, numerical correctness, or a performance improvement.
