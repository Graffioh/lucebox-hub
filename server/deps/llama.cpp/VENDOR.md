# Vendored llama.cpp/ggml snapshot

This directory contains the ggml-only subset used by Lucebox Hub.

- Source repository: https://github.com/Luce-Org/lucebox-ggml
- Source base branch: `luce-dflash`
- Source base commit: `6fbe72d67069136bbd370be703e1d4f441b5e942`
- Included merged PR: `#35` (`0fe65d9354b7c5da52a7741d2e37ba85f0d0c925`)
- Included test PR: `#37` (`0699be81480428f01b9b7ac49a09a2d51c77f8df`)
- Included upstream backport: `llama.cpp #22298` (`9725a313be0528214c4a02fed906ddaf7b3f712e`)
- Included tensor-parallel source PR: `Luce-Org/lucebox-ggml#39` (`a6eb14b8d678c23f111b7acfcfe6b51b2ea95c46`)
- Reconstruction: `luce-dflash@6fbe72d67069136bbd370be703e1d4f441b5e942` plus cherry-picked PRs `#35`, `#37`, upstream `llama.cpp #22298`, and the GGML delta from `Luce-Org/lucebox-ggml#39`
- Vendored paths: `LICENSE`, `common/jinja`, `common/log.h`, `common/unicode.*`, `ggml`, `gguf-py`

Open ggml feature PRs are not included unless they are explicitly listed above as a vendored source.

## Hub-local heterogeneous execution patches

PR `Luce-Org/lucebox#505` carries a temporary, reviewable patch set on top of
the snapshot above for AMD heterogeneous MoE execution:

- scheduler support for late cross-backend joins and deferred peer copies;
- ROCmFP quantized MMVQ/MMID kernels and grouped expert execution;
- fused MoE owner/combine operations and DeepSeek V4 HC/indexer kernels; and
- thread-local CUDA/HIP graph policy overrides used by fixed-topology graphs.

These changes are limited to `ggml/`. Keep their public declarations in
`ggml/include`, avoid DeepSeek-specific policy in generic kernels, and update
this provenance when the patch set is moved to `lucebox-ggml`.

## Hub-local DS4V HIP vision ops

Four inference-only ops are appended after the existing ones, so every earlier
op keeps its numeric value: `GGML_OP_MUL_MAT_BIAS_BF16`,
`GGML_OP_RMS_NORM_VISION_F32`, `GGML_OP_SOFT_MAX_VISION_F32` and
`GGML_OP_MUL_MAT_VISION_AV_F32`. Only the DS4V vision tower builds them. They
exist on the HIP backend alone; CPU, CUDA and RPC reject them in `supports_op`,
and graphs that contain them are not captured.

They are compiled only when CMake finds hipBLASLt (`GGML_HIP_DS4V_VISION`).
Without it the HIP backend builds as before and the server refuses `--mmproj`.
The fused BF16 bias matmul keeps one hipBLASLt handle and one 76 MiB workspace
per backend context that uses it, released when the context is destroyed.

Rebuild ggml-base, the backends and their consumers together; do not mix older
shared libraries with this header.
