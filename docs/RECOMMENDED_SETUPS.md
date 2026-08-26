# Recommended Setups

These profiles are starting points for supported model and hardware
combinations. Replace device indices and model paths for your system, then
validate throughput and output quality on your own workload.

Qwen 3.5 and 3.6 entries are compatibility profiles. Current Qwen performance
claims in the root README use Qwen 3.8.

Unless a row says otherwise, the settings are for `dflash_server`.

| Model | RTX 3090 (24 GB) | Strix Halo `gfx1151` | Strix Halo + R9700 `gfx1201` |
|---|---|---|---|
| **Qwen 3.8 27B IQ4_XS** | `--target-device cuda:0`<br>`--draft-device cuda:0`<br>`--draft-block-size 16`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--draft-block-size 16`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` | `HIP_VISIBLE_DEVICES=<r9700-index>`<br>`--target-device hip:0`<br>`--draft-device hip:0`<br>`--draft-block-size 16`<br>`--max-ctx 131072`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` |
| **Qwen 3.6 35B-A3B Q4_K_M** | `--target-device cuda:0`<br>`--spark`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` | `HIP_VISIBLE_DEVICES=<r9700-index>`<br>`--target-device hip:0`<br>`--kvflash auto` |
| **Laguna XS 2.1 33B Q4_K_M** | `--target-device cuda:0`<br>`--draft <path>`<br>`--prefill-drafter <path>`<br>`--max-ctx 262144`<br>`--kvflash 8192`<br>`--chunk 1024` | `--target-device hip:0`<br>`--kvflash auto` | `HIP_VISIBLE_DEVICES=<r9700-index>`<br>`--target-device hip:0`<br>`--kvflash auto` |
| **Gemma 4 26B-A4B / 31B** | `--target-device cuda:0`<br>`--draft-device cuda:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--kvflash auto` | `HIP_VISIBLE_DEVICES=<r9700-index>`<br>`--target-device hip:0`<br>`--draft-device hip:0`<br>`--kvflash auto` |
| **DeepSeek V4 Flash 0731 adaptive ROCmFPX** | Not listed | `DFLASH_DS4_SPEC=1`<br>`DFLASH_DS4_SPEC_Q=4`<br>`DFLASH_DS4_FUSED_VERIFY=1`<br>`DFLASH_DS4_DRAFT=<path>`<br>`DFLASH_DS4_DRAFT_GPU=0`<br>`--target-device hip:0`<br>`--ds4-fused-decode`<br>`--ds4-expert-top-k 6`<br>`--ds4-prefill exact` | Not listed |
| **DeepSeek V4 Flash ROCmFPX dual-GPU burn-in** | Not applicable | Not applicable | Build one HIP binary with `-DDFLASH27B_HIP_ARCHITECTURES='gfx1151;gfx1201'` and `-DGGML_HIP_GRAPHS=ON`.<br>Set `HIP_VISIBLE_DEVICES=<r9700-index>,<strix-index>`.<br>Use `DFLASH_DS4_MOE_TP=1`, `DFLASH_DS4_MOE_TP_INPROC=1`, `DFLASH_DS4_MOE_TP_GPU=1`, `DFLASH_EXPERT_BUDGET_MB=11700`, `DFLASH_DS4_SPEC=1`, `DFLASH_DS4_SPEC_Q=4`, `DFLASH_DS4_FUSED_VERIFY=1`, `DFLASH_DS4_DRAFT=<path>`, `DFLASH_DS4_DRAFT_GPU=0`, and `LUCE_MMVQ_MAX_NCOLS=4`.<br>Run with `--target-device hip:0 --peer-access --ds4-expert-top-k 4 --ds4-prefill sparse`. |
| **Qwen 3.5 0.8B Megakernel** | `uv run --directory optimizations/megakernel python final_bench.py --backend bf16` | Not supported | Not supported |

The DeepSeek dual-GPU profile is opt-in burn-in. Its top-4 routing and sparse
prefill settings are approximation policies and must be quality-qualified for
the deployment workload. For the exact top-6 RX 7900 XT + Strix Halo profile,
see [PR #604](https://github.com/Luce-Org/lucebox/pull/604).

## Multi-device placement

- Homogeneous NVIDIA tensor parallelism: use `--target-devices` with
  `--target-split-mode tensor`; add `--peer-access` when the topology supports
  it.
- Target/drafter split: set `--target-device` and `--draft-device`, or use a
  separate draft process for mixed CUDA/HIP execution.
- Mixed CUDA/HIP target layers and expert owners require backend-specific
  builds and are opt-in.

See the [server guide](../server/README.md),
[mixed-backend guide](../server/docs/MIXED_BACKEND.md), and
[environment reference](../server/docs/ENVIRONMENT.md) for the full commands
and tuning controls.
