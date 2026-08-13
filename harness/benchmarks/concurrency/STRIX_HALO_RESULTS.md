# Qwen3.6 concurrent feature results — Strix Halo

- Date: 2026-08-13
- Implementation: `568fbac03b498d53d6efc0b2ab5893044543a321`
- Stack base: PR #595 head `a90ffe45c1d4ad58f5f73c4107571d3cf6c51bfd`

These are bounded engineering measurements for the draft PR, not the
five-repeat publication matrix described in `FEATURE_MATRIX.md`. The paired
AR/DDTree screen has three fresh-process repeats. The long-context activation
rows have one fresh-process repeat per concurrency level because they are much
more expensive; treat their throughput as screening data.

## System and artifacts

- AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`), 128 GiB unified memory.
- ROCm runtime 7.2.4.
- Release HIP build for `gfx1151` with
  `DFLASH27B_HIP_SM80_EQUIV=ON`.
- Server SHA-256:
  `c77b4d2c7d1505fcc751600a6603cd65e51514b685bc66c4f7d33cd64a87c8a6`.
- Target SHA-256:
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`.
- Decode draft SHA-256:
  `e2500e90165a0f8e7b52c9882c29ed1fa391c60b300ff11b817bf10e31fa092e`.
- PFlash/KV scorer drafter SHA-256:
  `f9c9f1d3c1e21755b82d4e165f88dbbbd4355646d632fb5d6cef7c66ed4ee04e`.

Every case started a fresh server, discarded an 8-token same-concurrency
warmup, then requested exactly 64 output tokens per request with temperature
zero, seed one, and EOS ignored. Prompts were deterministic, disjoint across
concurrency levels, and identical between paired variants. The runner rotated
variant order across repeats.

`Output-window` counts all completion tokens from the earliest first output to
the last completion. `Goodput` counts completion tokens over the whole level,
including TTFT. Every reported row passed exact token accounting and the
request-ID-correlated feature proof.

The retained screening artifacts contain maximum TTFT but not median TTFT.
Their max-only columns below are an explicit screening exception, not a
protocol-complete publication result; a publication rerun must report both.

## Paired AR and adaptive DDTree

The DDTree configuration adds the local decode draft, budget 22, and target and
draft placement on `hip:0`. Values are medians over three fresh-process
repeats.

| C | Variant | N | Goodput tok/s | Output-window tok/s | vs AR goodput | Accepted/step | Steps/suspensions | Max TTFT s | Output hashes stable |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :---: |
| 1 | AR | 3 | 9.41 | 12.57 | — | — | 0/0 | 1.707 | yes |
| 1 | DDTree | 3 | 9.00 | 11.85 | -4.4% | 1.00 | 1/1 | 1.715 | yes |
| 4 | AR | 3 | 20.46 | 36.21 | — | — | 0/0 | 5.508 | no |
| 4 | DDTree | 3 | 19.43 | 33.61 | n/a | 3.08 | 4/4 | 5.533 | no |
| 8 | AR | 3 | 27.44 | 65.93 | — | — | 0/0 | 11.008 | no |
| 8 | DDTree | 3 | 25.63 | 56.32 | n/a | 2.79 | 8/8 | 11.076 | no |
| 16 | AR | 3 | 31.82 | 58.79 | — | — | 0/0 | 22.487 | no |
| 16 | DDTree | 3 | 29.36 | 54.63 | n/a | 2.00 | 16/16 | 22.514 | no |

The supplied draft had weak acceptance on this cohort. The adaptive policy
sampled one real packed-tree step, then suspended the whole cohort because its
aggregate emitted yield was below six tokens per request. At C4 and above,
the raw timings are retained only to diagnose this fallback behavior; unstable
outputs do not support a performance comparison with AR.

At C4 and above, greedy text hashes varied across fresh repeats in both the AR
control and DDTree. C1 was byte-stable. These measurements therefore establish
exact token accounting and feature execution, but do not claim bitwise text
reproducibility for concurrent batches.

## Full screenshot configuration

These rows enable the complete requested product configuration:

```text
--target-device hip:0
--draft-device hip:0
--ddtree
--ddtree-budget 22
--draft-residency persistent
--prefill-compression auto
--prefill-drafter /opt/models/Qwen3-0.6B-BF16.gguf
--kvflash auto
```

The controlled runner sets the auto PFlash threshold to 32K tokens, the keep
ratio to 0.05, and the KVFlash resident cap to 8,192 tokens. Startup telemetry
confirmed 512 physical blocks of 16 tokens, 16 configured slots, and a 65,536
logical-token bound per slot.

| C | N | Goodput tok/s | Output-window tok/s | Request decode tok/s | Raw prompt range | Effective prompt range | Max TTFT s | DDTree steps/susp. | KV page in/out | PFlash requests |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | :--- | :--- | ---: |
| 1 | 1 | 2.61 | 12.39 | 12.19 | 41,504 | 2,021 | 19.324 | 1/1 | 0/0 | 1 |
| 4 | 1 | 3.00 | 31.61 | 7.95 | 38,142–44,866 | 1,870–2,235 | 77.495 | 4/4 | 1/18 | 4 |
| 8 | 1 | 3.13 | 52.60 | 6.56 | 38,141–44,870 | 1,869–2,237 | 153.782 | 8/8 | 245/792 | 8 |
| 16 | 1 | 3.19 | 18.73 | 2.34 | 38,140–44,872 | 1,867–2,238 | 307.360 | 16/16 | 293/1,880 | 16 |

All four rows proved DDTree, PFlash, and KVFlash active. PFlash retained about
4.9% of raw prompt tokens. Output-window throughput scaled through C8, then
dropped at C16 while roughly 32K effective prompt tokens shared the 8K resident
pool; the concurrent page traffic rose accordingly.

## Feature ablations

| Workload | C | Variant | N | Goodput tok/s | Output-window tok/s | Request decode tok/s | Effective/raw | Max TTFT s | Activation evidence |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| compression | 4 | PFlash | 1 | 3.15 | 35.54 | 8.89 | 0.049 | 74.261 | 4/4 prompts compressed, 166,016 -> 8,179 tokens |
| kv-pressure | 4 | KVFlash | 1 | 1.24 | 8.56 | 5.32 | 1.000 | 197.859 | 129 resident blocks max, 0 page-ins / 3,714 page-outs |

The PFlash-only row uses the same C4 prompts as the full row; adding DDTree and
KVFlash reduced output-window throughput from 35.54 to 31.61 tok/s in this
single screening repeat. The KVFlash-only row deliberately disables PFlash and
uses 13,474–20,203-token histories against the 8K pool. It is an activation and
pressure test, not a recommended latency configuration.

## Reproduction

The exact per-case command, controlled environment, startup-observed pool,
binary/shared-library/model hashes, raw request report, server log, and
`feature-proof.json` are retained by the runner. The principal invocations were:

```bash
WORKLOADS=short CLIENTS=1,4,8,16 VARIANTS=ar,ddtree MAX_TOKENS=64 REPEATS=3 SLOTS=16 harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh

WORKLOADS=compression CLIENTS=1,4,8,16 VARIANTS=full MAX_TOKENS=64 REPEATS=1 SLOTS=16 harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh

WORKLOADS=compression CLIENTS=4 VARIANTS=pflash MAX_TOKENS=64 REPEATS=1 SLOTS=16 harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh

WORKLOADS=kv-pressure CLIENTS=4 VARIANTS=kvflash MAX_TOKENS=64 REPEATS=1 SLOTS=16 harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh
```

Set `MODEL`, `DRAFT_MODEL`, `PREFILL_DRAFTER`, and `LUCE_SERVER_BIN` as shown
in `FEATURE_MATRIX.md`. For publication-quality claims, rerun the documented
five-repeat 256-token matrix.
