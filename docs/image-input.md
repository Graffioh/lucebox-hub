# Image input

The server accepts JPEG and PNG images through OpenAI chat completions when a
model is started with its vision projector, `--mmproj <file>`. Without
`--mmproj` nothing in the text serving path changes.

| Model | Projector file | Runs on |
| --- | --- | --- |
| Qwen3.5 / Qwen3.8 dense | the `mmproj-*.gguf` published next to the model (llama.cpp `clip` format, type `qwen3vl_merger`) | one GPU, any backend |
| DeepSeek V4 Flash Vision (DS4V) | [exported with our tool](ds4v-mmproj.md) | HIP: one GPU, or two GPUs splitting the experts |

**Status: experimental.** Both models answer image questions correctly end to
end; see each model's verification notes for what has and has not been
measured.

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
content arrays, and image parts through other API formats are rejected. A
request carries at most four images, 16 MiB encoded each and 32 MiB combined.
Decoder pixel and aspect limits also apply. A model's image marker cannot be supplied
as ordinary text.

The server expands image markers after final rendering and tokenization, and
the expanded image tokens count toward context and usage. Image requests use
plain autoregressive decoding and bypass the token-keyed prefix, disk and
agent-turn caches and prompt compression: tokens alone do not identify an
image. Text requests on the same server keep speculative decoding and caching.

Layer or tensor splitting across GPUs, remote target shards, concurrent
sequence scheduling (`--max-concurrency`) and upstream forwarding do not
support images. `/props` reports the effective capability in
`capabilities.image_input_supported` after backend initialization.

## Qwen3.5 / Qwen3.8

```
luce_server Qwen3.8-27B-UD-IQ4_XS.gguf --target-device hip:0 \
  --draft qwen38-dflash2-q8_0.gguf --draft-device hip:0 \
  --mmproj mmproj-Qwen3.8-27B-BF16.gguf
```

The projector is read directly from the published file. Projectors with
deepstack branches (Qwen3-VL) are refused. An image is resized the way the
model was trained (bicubic, both sides to a multiple of 32 pixels) and costs
one token per 32x32 pixels, between 64 and 1,024 tokens; larger images are
scaled down to the cap.

Image tokens take two-dimensional rotary positions, so positions run behind
token counts after an image. Prefill handles that in its normal chunk loop;
decoding carries the offset for the rest of the request.

### Verification status

Covered by `test_qwen35_image`: target sizes against the model's reference
resize rule, the tower's patch order and position table sampling, marker
expansion, rotary positions, and image rows that straddle prefill chunks.

Measured on an R9700 alone with Qwen3.8-27B UD-IQ4_XS, the DFlash2 drafter and
the published BF16 projector, thinking off:

- 220 seeded questions from `lmms-lab/ai2d` and `lmms-lab/ChartQA` with
  lmms-eval prompts: AI2D 85/100, ChartQA relaxed accuracy 55/60 (augmented)
  and 43/60 (human), no errors. Image prompts average 448 tokens and prefill in
  0.71 s (largest 1,068 tokens, 1.8 s); decode runs at 31 to 35 tok/s.
- llama.cpp (HIP build, `-fa on`, same GGUF, projector and image cap) answers
  the same on every test image; on a 1,012-token image prompt it prefills in
  1.65 s to our 1.68 s, on a 323-token one in 0.61 s to our 0.50 s.
- An image inside a 6,271-token prompt, two images in one request, and a
  follow-up turn after an image all answer correctly.
- Text requests are byte-identical to a build without image support, at the
  same speed, with or without a projector loaded (five prompts up to 19.6K
  tokens). The projector adds 0.9 GiB of VRAM; the peak during image requests
  was 21.6 GiB against 20.8 GiB for text.

- The same requests answer correctly on a Strix Halo alone, where a
  1,012-token image prompt prefills in 4.6 s and decodes at 14 tok/s.

Not yet established: a comparison against the reference implementation on the
same questions, and CUDA. The tower uses only standard ggml
operators, so nothing in it is HIP specific.

## DS4V

The server must be built with hipBLASLt available (the `hipblaslt-dev` package
on ROCm). CMake reports `hipBLASLt found: building the DS4V vision ops`; a build
without it refuses `--mmproj` for this model at startup.

Image input needs Linux HIP, a DeepSeek4 decoder whose GGUF carries the image
router biases, `--ds4-prefill sparse`, and `--mmproj` pointing at the
[exported projector](ds4v-mmproj.md). Two layouts work:

- **One GPU holding the whole model** (for example a Strix Halo): nothing else
  to set. The projector is loaded after the weights and must fit beside them.
- **Two GPUs splitting the experts in process** (for example R9700 + Strix
  Halo): `LUCE_DS4_MOE_TP=1`, `LUCE_DS4_MOE_TP_INPROC=1`, and
  `LUCE_DS4_MOE_TP_GPU` selecting the second device, with `--target-device`
  on the first. Device ordinals must match the host's actual topology.

Remote expert IPC, all-on-secondary placement, experts kept on the CPU and
dense prefill do not support images.

Published llama.cpp conversions of the decoder load directly (image router
bias named `blk.N.exp_probs_b_vl.bias`, no `deepseek4.vocab_size` key). Split
GGUF files and llama.cpp's `clip` projector files are not read yet.

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

The image payload survives request copies and retry paths. Failed or cancelled
multi-image encoding publishes no partial embedding matrices.

### Memory

The projector is validated and loaded before expert placement. Admission counts
actual selected owner tensor sizes, allocation alignment, MIX tables, copy
staging, future KV, and explicit execution reserves. Host and integrated-device
charges share one physical-memory budget. Before image decoding, the server
checks host availability; before encoding, it synchronizes and releases
disposable decoder, owner, and draft graphs and checks live device/host
availability again. KV, saved snapshots, and draft weights remain reflected in
that live measurement. Reservations are conservative policy, not a guarantee
against unrelated concurrent allocations.

### Verification status

Covered by unit tests in the main build: image transport and request policy,
prompt expansion and ownership, embedding assembly and cancellation, image
spans and the expert budget, plus the decoder loader and image-batch admission
tests in `test_deepseek4_unit`.

Measured with the public `DeepSeek-V4-Flash-Vision-Exp` Q2_K_S decoder and
the exported projector, on a Strix Halo alone and on R9700 + Strix Halo, 220
seeded questions from `lmms-lab/ai2d` and `lmms-lab/ChartQA` with lmms-eval
prompts: AI2D 85/100, ChartQA relaxed accuracy 55/60 (augmented) and 43/60
(human). Both layouts score the same and give word-identical answers on 213 of
220 questions. An image request prefills in about 4 s and decodes at about
23 tok/s.

Not yet established:

- The vision tower misses the fixed 0.9995 feature-cosine gate against the
  reference implementation: 0.99906 on the Radeon RX 7900 XT it was developed
  on, 0.99823 on CPU. Embeddings pass; features do not.
- No comparison against the reference implementation on the same questions.

## Code layout

Shared by every model:

| Piece | Where |
| --- | --- |
| Reading images out of a request, limits, redaction | `server/src/server/image_input.*` |
| JPEG and PNG decoding to RGB | `server/src/common/vision/image_decode.*`, codecs in `server/cmake/ImageCodecs.cmake` |
| Bicubic resizing that matches Pillow byte for byte | `server/src/common/vision/image_resize.*` |
| Reading a published `clip`-format projector file | `server/src/common/vision/mmproj_file.*` |
| Image positions in a prompt, batches that keep an image whole | `server/src/common/vision/image_spans.h` |
| The backend contract | `supports_images`, `image_placeholder`, `prepare_images` in `server/src/common/model_backend.h`; `GenerateRequest::images` in `server/src/common/generation_types.h` |

DS4V only, all under `server/src/deepseek4/`: resizing and patching
(`deepseek4_vision_preprocess`), the vision tower (`deepseek4_vision`), marker
expansion and embedding assembly (`deepseek4_image_prompt`,
`deepseek4_image_assembly`), attention visibility and expert routing for image
rows (`deepseek4_image_policy`), and memory admission
(`deepseek4_image_admission`).

Qwen3.5 / Qwen3.8 only, all under `server/src/qwen35/`: the vision tower and
its preprocessing (`qwen35_vision`), marker expansion and rotary positions
(`qwen35_image_prompt`), what a request carries (`qwen35_image_request.h`), and
the backend's three contract methods (`qwen35_backend_images.cpp`). Prefill and
decode changes are a few lines in `qwen35_backend.cpp`.

Another model needs its own preprocessing, tower and prompt expansion, and its
backend implements the three contract methods. Nothing in the HTTP server or in
`common/vision` names a model.
