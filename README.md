![Lucebox logo](assets/lucebox-logo.png)

# Lucebox

**Speculative inference for heterogeneous consumer hardware.**

Lucebox is a local C++ inference server built around two things: speculative
inference and heterogeneous execution. It can put the target, drafter,
mixture-of-experts (MoE) weights, and KV cache on different parts of the same
machine, including NVIDIA GPUs, AMD GPUs or APUs, the CPU, and system memory.
It exposes OpenAI- and Anthropic-compatible endpoints.

[lucebox.com](https://lucebox.com) ·
[HuggingFace](https://huggingface.co/Lucebox) ·
[Discord](https://discord.gg/yHfswqZmJQ) ·
[Blog](https://lucebox.com/blog) ·
[Tutorials](#tutorials-and-results)

[Apache 2.0](LICENSE) ·
[CUDA 12+](https://developer.nvidia.com/cuda-toolkit) ·
[HIP 7+](https://rocm.docs.amd.com/projects/HIP/en/latest/) ·
[C++17](https://isocpp.org)

[Quick start](#quick-start) · [Optimizations](#optimizations) ·
[Supported models](#supported-models-and-drafters) ·
[Recommended setups](docs/RECOMMENDED_SETUPS.md) ·
[Hardware support](#hardware-support) · [Docker](#docker) ·
[Client harnesses](#client-harnesses) · [Documentation](#documentation)

## Optimizations

These are separate paths, not one universal fast mode. Each link includes the
setup, limits, and measurements.

| Optimization | Best for | What it does |
|---|---|---|
| [DFlash, DFlash2, and DSpark](server/) | Decode | Drafts multiple tokens and verifies them with the target model. The target and drafter can share a device or run on separate devices. |
| [PFlash](optimizations/pflash/) | Long-context prefill | Uses a small drafter to select prompt content before target prefill and is validated with long-context retrieval tests. |
| [Luce Spark](optimizations/spark/) | MoE models | Places hot experts on the GPU and cold experts in CPU or host memory, with bounded caching and workload-driven placement. |
| [KVFlash](optimizations/kvflash/) | Long-context decode | Keeps a bounded working set of KV state on the GPU and recalls cold chunks from host memory. |
| [Tensor, layer, draft, and expert parallelism](server/docs/MIXED_BACKEND.md) | Multi-device systems | Uses tensor parallelism on homogeneous NVIDIA GPUs; layer, drafter, and expert placement can cross CUDA and HIP. |
| [Paged attention](optimizations/paged_attention/) | Concurrent serving | Shares a block-based KV pool across active sequences. |
| [Megakernel](optimizations/megakernel/) | Qwen 3.5 0.8B | Fuses 24 layers into one persistent CUDA dispatch for prefill and decode. |

## Supported Models and Drafters

| Target | Acceleration paths | Drafter or helper | Result or documentation |
|---|---|---|---|
| **Qwen 3.8 27B** | DFlash2, tensor parallelism | [Qwen3.8 27B DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | **6.1×** decode on R9700; **3.5×** versus llama.cpp speculating with the same drafter |
| [**DeepSeek V4 Flash 0731 ROCmFPX**](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3) | DSpark, heterogeneous expert parallelism | [DSpark drafter](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF) | **2×** |
| **Laguna XS 2.1 33B** | DFlash, PFlash, Spark, KVFlash | [Laguna DFlash](https://huggingface.co/Lucebox/Laguna-XS-2.1-DFlash-GGUF) | **8.2×** prefill and **1.7×** decode at 256K |
| **Gemma 4 26B-A4B / 31B IT** | DFlash, KVFlash | [26B drafter](https://huggingface.co/Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF) · [31B drafter](https://huggingface.co/Lucebox/gemma-4-31B-it-DFlash-GGUF) | **1.31×** / **3.2×** |
| **Qwen 3.5 / 3.6 27B** | DFlash, PFlash, tensor/layer split, KVFlash, paged attention | Model-matched DFlash draft; [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for prefill | [Server documentation](server/) |
| **Qwen 3.6 35B-A3B** | Spark, KVFlash | [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for KV scoring | [Spark](optimizations/spark/) · [KVFlash](optimizations/kvflash/) |
| **Qwen 3.5 0.8B** | Megakernel | None | [Megakernel documentation](optimizations/megakernel/) |

Speedups are measured against the vendored llama.cpp with flash attention
enabled and matching KV quantization. When both phases are benchmarked, the
combined result is the geometric mean of time-to-first-token (TTFT) and decode
speedups; otherwise it is the stated phase's speedup. A result applies only to
its documented model, quantization, workload, and hardware. It does not apply
across the support matrix.

Qwen 3.5 and 3.6 remain supported for existing setups. Current Qwen
performance claims in this README use Qwen 3.8.

## Hardware Support

| Backend | Measured devices | Additional support |
|---|---|---|
| **NVIDIA CUDA 12+** | RTX 2080 Ti, RTX 3090/A-series, RTX 4090, RTX 5090, DGX Spark/GB10 | RTX 5090 needs CUDA 12.8+ and GB10 needs 12.9+; V100/P40 have fallback paths; Jetson AGX Thor builds with CUDA 13 |
| **AMD ROCm/HIP 6+** | Ryzen AI MAX+ 395/Strix Halo, RX 7900 XT/XTX, Radeon AI PRO R9700 | Build for the GPU's exact `gfx` architecture; R9700 uses `gfx1201` and ROCm 6.4+ |

### Lucebox single-device results

The Lucebox machine pairs an R9700 with Strix Halo. Each device is useful on
its own before the engine starts placing work across both.

| Device | Model | Result | Details |
|---|---|---|---|
| **R9700** | Qwen 3.8 27B + DFlash2 | **204.1 tok/s** HumanEval average; peaks above 230 | Block size 16 on code/math; greedy output stayed byte-identical. [Run notes](server/README.md#amd-hip-backend-strix-halo-rx-7900-xtx) |
| **Strix Halo** | DeepSeek V4 + DSpark | **28.26 tok/s** with model-default top-6; **32.12 tok/s** with top-4 on the uniform artifact | Both profiles scored 10/10 on the same GSM/Math set. Top-4 is an explicit approximation and is not for adaptive artifacts. [DS4 notes](server/docs/DS4.md) |

The current NVIDIA result is the Qwen 3.8 tensor-parallel run below.

### Multi-device evidence

| Hardware combination | Placement | Measured evidence | Status |
|---|---|---|---|
| **2× RTX 3090 + NVLink** | Qwen 3.8 target tensor parallel + DFlash2 | **79.7 tok/s**, **2.16×** autoregressive decode | [Merged validation](https://github.com/Luce-Org/lucebox/pull/637) |
| **RX 7900 XT + Strix Halo** | DeepSeek V4 heterogeneous experts, exact top-6, fixed DSpark q=4 | Controlled decode at **45.0 to 47.7 tok/s**, 100% acceptance and byte-identical output; **111.2 tok/s** prefill at 132,981 tokens | [Qualified](https://github.com/Luce-Org/lucebox/pull/604) |
| **R9700 + Strix Halo** | In-process HIP expert parallelism | **51.1 tok/s** decode and **415.52 tok/s** sparse prefill with explicit top-4/sparse approximation policies | [Opt-in burn-in](https://github.com/Luce-Org/lucebox/pull/505) |
| **RTX 3090 + Strix Halo** | Single-process CUDA + HIP expert parallelism | Exact-mode output identity validated; tuned top-4 mode reached **48.059 tok/s** median | [Opt-in burn-in](https://github.com/Luce-Org/lucebox/pull/570) |

Do not compare throughput across rows: the models, prompts, quantizations, and
accuracy policies differ. Start with the
[recommended setups](docs/RECOMMENDED_SETUPS.md).

## Quick Start

This starts Qwen 3.8 27B with its DFlash2 drafter on NVIDIA CUDA. You need
CMake 3.18+, CUDA 12+, Python 3, and the Hugging Face `hf` CLI.

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox
cd lucebox

cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release
cmake --build server/build --target dflash_server -j

hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ4_XS.gguf \
  --local-dir server/models/
hf download incoai/Qwen3.8-27B-DFlash2 \
  --local-dir server/models/dflash2-hf

mkdir -p server/models/draft
python3 server/scripts/convert_dflash_to_gguf.py \
  server/models/dflash2-hf/model.safetensors \
  server/models/draft/qwen38-dflash2-f16.gguf
python3 server/scripts/quantize_dflash_draft.py \
  server/models/draft/qwen38-dflash2-f16.gguf \
  server/models/draft/qwen38-dflash2-q8_0.gguf --scheme q8_0

./server/build/dflash_server server/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft server/models/draft/qwen38-dflash2-q8_0.gguf \
  --draft-block-size 16 \
  --max-ctx 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --port 8000
```

Make an OpenAI-compatible request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dflash",
    "messages": [{"role": "user", "content": "Write quicksort in Python."}],
    "temperature": 0
  }'
```

For AMD, multi-GPU, model-specific, and lower-memory configurations, use the
[server guide](server/README.md), [mixed-backend guide](server/docs/MIXED_BACKEND.md),
and [recommended setups](docs/RECOMMENDED_SETUPS.md).

## Docker

The images track `main` and do not require a local CUDA or ROCm toolkit. Put
the target GGUF and converted drafter from the quick start under
`server/models/`, then mount that directory into the container.

| Host | Image |
|---|---|
| NVIDIA, CUDA 12+ driver | `ghcr.io/luce-org/lucebox-hub:cuda12` |
| AMD, ROCm 6+ driver | `ghcr.io/luce-org/lucebox-hub:rocm` |

NVIDIA:

```bash
docker run --rm --gpus all -p 8000:8080 \
  -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:cuda12
```

AMD:

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -p 8000:8080 \
  -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:rocm
```

The API is available at `http://127.0.0.1:8000`. See the
[Docker guide](https://lucebox.com/blog/docker) for model layout and container
settings.

## Client Harnesses

[`harness/`](harness/) starts the native server, runs a real client against it,
saves the logs, and stops the server.

| Client | Launcher |
|---|---|
| Claude Code | [`run_claude_code.sh`](harness/clients/run_claude_code.sh) |
| Codex | [`run_codex.sh`](harness/clients/run_codex.sh) |
| OpenCode | [`run_opencode.sh`](harness/clients/run_opencode.sh) |
| Hermes | [`run_hermes.sh`](harness/clients/run_hermes.sh) |
| Pi | [`run_pi.sh`](harness/clients/run_pi.sh) |
| OpenClaw | [`run_openclaw.sh`](harness/clients/run_openclaw.sh) |
| Open WebUI | [`run_openwebui.sh`](harness/clients/run_openwebui.sh) |

Point a launcher at the Qwen 3.8 files from the quick start:

```bash
DFLASH_SERVER_BIN=server/build/dflash_server \
DFLASH_TARGET=server/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
DFLASH_DRAFT=server/models/draft/qwen38-dflash2-q8_0.gguf \
MAX_CTX=32768 \
harness/clients/run_codex.sh
```

For a model without a drafter, set `DRAFT=none`. Missing client command-line
tools are installed under `.harness-work/`; set `AUTO_INSTALL_CLIENTS=0` to
require preinstalled tools instead.

To benchmark a server that is already running:

```bash
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:8000 \
  --suite he,agent \
  --n-sample 3
```

See the [harness guide](harness/README.md) for protocol probes, backend
comparisons, and GPU sweeps.

## Documentation

| Topic | Guide |
|---|---|
| Build, runtime flags, and per-GPU setup | [Server guide](server/README.md) |
| OpenAI Chat Completions, Responses, and Anthropic Messages | [API reference](server/docs/API.md) |
| CUDA/HIP placement and multi-device execution | [Mixed-backend guide](server/docs/MIXED_BACKEND.md) |
| Model and hardware starting profiles | [Recommended setups](docs/RECOMMENDED_SETUPS.md) |
| Environment variables | [Environment reference](server/docs/ENVIRONMENT.md) |
| Server internals | [Architecture](server/docs/ARCHITECTURE.md) |
| Regression tests and client integrations | [Harness guide](harness/README.md) |

## Tutorials and Results

- Results: [DFlash](server/RESULTS.md) · [Megakernel](optimizations/megakernel/RESULTS.md) · [PFlash](optimizations/pflash/) · [Spark](optimizations/spark/RESULTS.md) · [KVFlash](optimizations/kvflash/RESULTS.md)
- Videos: [Spark](https://www.youtube.com/watch?v=LB1aVj9lNhg) · [DFlash](https://www.youtube.com/watch?v=vbPGvvSB8IQ) · [PFlash](https://www.youtube.com/watch?v=NWeKUL9Bc6Y) · [KVFlash](https://www.youtube.com/watch?v=8rTVCRWvRDo) · [Megakernel](https://www.youtube.com/watch?v=e6jY4goVIu0) · [Turboquant](https://www.youtube.com/watch?v=uTOOrfhrnBk) · [Harness setup](https://www.youtube.com/watch?v=PysoxVGfvRE)

## The Lucebox Machine

We build the engine and the Lucebox machine together. The current machine pairs
a Radeon AI PRO R9700 with Strix Halo: both run useful workloads alone, and the
engine can place work across them when a model benefits from both.

[![Lucebox local AI PC](assets/lucebox.png)](https://lucebox.com)

See the [hardware and current benchmarks at lucebox.com](https://lucebox.com).

## Contributing

We especially welcome work on:

- CUDA and HIP kernels
- Speculative decode and prefill
- New consumer GPU and APU profiles
- Heterogeneous placement and scheduling
- Reproducible performance benchmarks and client integrations

Open an [issue](https://github.com/Luce-Org/lucebox/issues) or join the
[Discord](https://discord.gg/yHfswqZmJQ) to coordinate larger changes.

## Citation

```bibtex
@software{lucebox_2026,
  title  = {Lucebox: Speculative inference for heterogeneous consumer hardware},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox},
  year   = {2026}
}
```

[Website](https://lucebox.com) · [Hugging Face](https://huggingface.co/Lucebox) ·
[Blog](https://lucebox.com/blog) · [Discord](https://discord.gg/yHfswqZmJQ) ·
[Apache 2.0](LICENSE)
