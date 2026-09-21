# DS4V image serving

**Status: experimental.** The request path works end to end, but the vision
tower has not met its numerical gate and no image-chat acceptance run exists.
See [Verification status](#verification-status) before relying on image output.

The DS4V integration accepts JPEG and PNG images through OpenAI chat
completions when the matching projector is supplied with `--mmproj`. Without
`--mmproj` nothing in the text serving path changes.

The server must be built with hipBLASLt available (the `hipblaslt-dev` package
on ROCm). CMake reports `hipBLASLt found: building the DS4V vision ops`; a build
without it refuses `--mmproj` at startup.

## Supported configuration

Image input needs Linux HIP, a DeepSeek4 decoder whose GGUF carries the image
router biases, `--ds4-prefill sparse`, and `--mmproj` pointing at the
[exported projector](ds4v-mmproj.md). Two layouts work:

- **One GPU holding the whole model** (for example a Strix Halo): nothing else
  to set. The projector is loaded after the weights and must fit beside them.
- **Two GPUs splitting the experts in process** (for example R9700 + Strix
  Halo): `DFLASH_DS4_MOE_TP=1`, `DFLASH_DS4_MOE_TP_INPROC=1`, and
  `DFLASH_DS4_MOE_TP_GPU` selecting the second device, with `--target-device`
  on the first. Device ordinals must match the host's actual topology.

Layer splitting, remote expert IPC, all-on-secondary placement, experts kept
on the CPU, dense prefill, concurrent sequence scheduling, and upstream
forwarding do not support images. `/props` reports the effective capability in
`capabilities.image_input_supported` after backend initialization. Without
`--mmproj`, text serving follows its existing path and image requests are
passed through as they were before.

Published llama.cpp conversions of the decoder load directly (image router
bias named `blk.N.exp_probs_b_vl.bias`, no `deepseek4.vocab_size` key). Split
GGUF files and llama.cpp's `clip` projector files are not read yet.

## Request contract

Use `POST /v1/chat/completions` with user-message content parts in display order:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "max_tokens": 128
}
```

Only base64 JPEG/PNG data URLs are supported. Remote URLs, images outside user
content arrays, and image parts through other API formats are rejected.
Requests permit at most four images, 16 MiB encoded bytes per image, and
32 MiB combined encoded bytes. Decoder pixel and aspect limits also apply.
The reserved DS4 image marker cannot be supplied as ordinary text.

One image request may be outstanding per backend. Its admission lease remains
with the immutable payload through queueing and generation; another image
request is rejected until that payload is released. This bounds simultaneous
preprocessing and prepared-image memory. Text requests retain the normal queue.

The server expands image markers after final rendering and tokenization.
Expanded image tokens count toward context and usage. Image blocks remain
whole during prefill, image rows use their learned routing bias, and raw
attention is bidirectional within each image's visible span. The projector's
tile permutation is applied once when assembling rows with named sentinel
embeddings. All chunks are capped at 1,024 tokens while a projector is loaded.

Image requests use autoregressive decoding and bypass token-only prefix,
disk, and agent-turn caches, prompt compression, and speculative capture.
Their image payload survives request copies and retry paths. Failed or cancelled
multi-image encoding publishes no partial embedding matrices.

## Memory and verification

The projector is validated and loaded before expert placement. Admission counts
actual selected owner tensor sizes, allocation alignment, MIX tables, copy
staging, future KV, and explicit execution reserves. Host and integrated-device
charges share one physical-memory budget. Before image decoding, the server
checks host availability; before encoding, it synchronizes and releases
disposable decoder, owner, and draft graphs and checks live device/host
availability again. KV, saved snapshots, and draft weights remain reflected in
that live measurement. Reservations are conservative policy, not a guarantee
against unrelated concurrent allocations.

## Verification status

Covered by unit tests in the main build: image transport and request policy,
prompt expansion and ownership, embedding assembly and cancellation, image
spans and the expert budget, plus the decoder loader and image-batch admission
tests in `test_deepseek4_unit`.

Not yet established:

- The vision tower misses the fixed 0.9995 feature-cosine gate against the
  reference implementation: 0.99906 on the Radeon RX 7900 XT it was developed
  on, 0.99823 on CPU. Embeddings pass; features do not.
- No image-chat quality run, and no measurement of paired-GPU memory peaks or
  throughput with a projector loaded.
- No other HIP device has been tried.

## Code layout

Shared by every model:

| Piece | Where |
| --- | --- |
| Reading images out of a request, limits, redaction | `server/src/server/image_input.*` |
| JPEG and PNG decoding to RGB | `server/src/common/vision/image_decode.*`, codecs in `server/cmake/ImageCodecs.cmake` |
| Image positions in a prompt, batches that keep an image whole | `server/src/common/vision/image_spans.h` |
| The backend contract | `supports_images`, `image_placeholder`, `prepare_images` in `server/src/common/model_backend.h`, and `GenerateRequest::images` |

DS4V only, all under `server/src/deepseek4/`: resizing and patching
(`deepseek4_vision_preprocess`), the vision tower (`deepseek4_vision`), marker
expansion and embedding assembly (`deepseek4_image_prompt`,
`deepseek4_image_assembly`), attention visibility and expert routing for image
rows (`deepseek4_image_policy`), and memory admission
(`deepseek4_image_admission`).

Another model needs its own preprocessing, tower and prompt expansion, and its
backend implements the three contract methods. Nothing in the HTTP server or in
`common/vision` names a model.
