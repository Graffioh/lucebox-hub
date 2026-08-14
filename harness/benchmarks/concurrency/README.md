# Qwen3.6 concurrency benchmark

This protocol measures the serving behavior targeted by packed continuous
prefill and concurrent decode. It is intentionally small: one streaming client,
one fresh-process runner, one deterministic prompt generator, and one summary
script.

## Canonical and blog workloads

The synthetic ragged profiles below isolate serving mechanics. To measure how
draft acceptance changes with real workload structure, use the canonical-suite
runner:

```bash
MODEL=/path/to/Qwen3.6-27B-Q4_K_M.gguf \
REPEATS=3 \
harness/benchmarks/concurrency/run_qwen36_canonical_concurrency.sh
```

It runs paged concurrent AR by default. Set `VARIANTS=blog-ddtree` and provide
`DRAFT_MODEL` to opt into the Strix Halo decode recipe from the AMD post:
the Q8_0 Qwen3.6 drafter,
`DFLASH27B_DRAFT_SWA=2048`, `--ddtree-budget 22`, `--fast-rollback`, and 128
forced output tokens. Adaptive fallback is disabled so every eligible decode
step measures the speculative path. It intentionally keeps paged attention on,
including at C=1, because the goal is to compare the concurrent implementation
with the standalone blog result rather than silently route C=1 elsewhere.
The serving-only differences are stated rather than hidden: every level uses
the same 16-slot concurrent graph/pool and Q4_0 paged target KV, matching the
established concurrency configuration. The standalone `test_dflash` blog bench
has neither a paged multi-sequence pool nor idle serving slots.
The DDTree variant requires a concurrent-serving build that emits
`[concurrency-metrics]` records; the harness does not substitute AR telemetry.
The speculative row also fails closed unless every measured response has a
matching `[concurrency-metrics]` record with positive DDTree work. Its JSON
report stores DDTree steps, accepted children, target forwards, mean accepted
length, and acceptance rate (the blog's 16 draft candidates per step).
An independently selectable `adaptive-ddtree` variant uses the same blog setup
with the runtime's adaptive fallback enabled; this measures whether one
configuration can retain high-acceptance workloads and fall back on low-yield
ones.

The workloads are:

- `he-raw`: the exact ten raw prompts imported from `server/scripts/bench_he.py`;
  an identity chat template preserves byte-for-byte blog prompt input.
- `he`, `gsm`, `math`, and `agent`: the checked-in canonical JSONL suites under
  `harness/benchmarks/prompts`, preserving all system/user message roles.

Each concurrency level processes the whole suite in fixed-width waves. The
ten-case suites use C=1/2/5/10; the six-case agent suite uses C=1/2/3/6. The
runner rejects a level that would create a smaller tail wave. This makes every
C comparison use the same prompt set without duplicating a prompt inside a run.
The C=1 case still launches exactly one request; the other 15 slots stay idle.
Use `SUITES=he-raw,gsm`, `VARIANTS=blog-ddtree`, or `CLIENTS=1,5` for a smaller
screen.
For a C=3 crossover measurement on a ten-case suite, set
`SUITES=he-raw CLIENTS=3 CASE_LIMIT=9`. This selects the first nine cases and runs three full
C=3 waves; the report records the limit. It never duplicates the tenth prompt
or mislabels a one-request tail as C=3.
Before starting the next wave, the client waits for a matching scheduler
retirement marker for every response. The wait is included in output goodput
wall time. This prevents nominal C=1 waves from accumulating as hidden C>1 work
when the HTTP stream closes slightly before slot retirement.

The raw HumanEval row is the direct blog-parity check. The chat-form HumanEval,
GSM8K, Math500, and agent rows answer the separate product question: whether
the drafter continues to pay off after the server's normal chat template and
on less code-like generations. Do not pool acceptance or throughput across
these suites.

### Strix Halo screening guidance

One 2026-08-14 screening repeat on gfx1151 used the exact raw HumanEval prompts,
128 forced output tokens, the Q8_0/SWA=2048 blog drafter, budget 22, and the
fixed 16-slot Q4_0 paged serving configuration. C=3 uses the first nine prompts;
C=4 uses the first eight, so both are composed entirely of full waves.

| C | Cases | AR goodput tok/s | DDTree goodput tok/s | DDTree vs AR | DDTree AL | Acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 11.59 | 15.17 | +30.9% | 5.40 | 33.8% |
| 2 | 10 | 16.66 | 22.32 | +34.0% | 5.60 | 35.0% |
| 3 | 9 | 18.88 | 25.47 | +34.9% | 6.11 | 38.2% |
| 4 | 8 | 31.43 | 28.44 | -9.5% | 5.83 | 36.4% |
| 5 | 10 | 37.50 | 25.05 | -33.2% | 5.59 | 34.9% |
| 10 | 10 | 56.13 | 24.48 | -56.4% | 5.59 | 34.9% |

The reasoning suites were then run for five clean repeats with 96 forced output
tokens. C=3 uses nine cases and C=4 uses eight, so every repeat contains only
full-width waves. `DDTree vs AR` is the median of the five same-repeat ratios,
not a ratio of independently rounded medians.

| Suite | C | AR goodput tok/s | Fixed DDTree tok/s | DDTree vs AR | DDTree AL | Acceptance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 3 | 18.88 | 24.46 | +29.4% | 5.92 | 37.0% |
| Math500 | 3 | 19.03 | 28.71 | +50.9% | 7.00 | 43.8% |
| GSM8K | 4 | 31.73 | 27.84 | -12.3% | 5.57 | 34.8% |
| Math500 | 4 | 31.81 | 32.92 | +3.1% | 7.10 | 44.4% |

At C=4, five additional adaptive-DDTree repeats measured the same eight cases.

| Suite | AR goodput tok/s | Adaptive DDTree tok/s | Adaptive vs AR | Speculative AL | Speculative acceptance |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 31.73 | 30.36 | -4.4% | 4.83 | 30.2% |
| Math500 | 31.81 | 31.53 | -1.0% | 7.56 | 47.2% |

The adaptive acceptance fields describe only the DDTree probes; they do not
measure the share of later tokens emitted by AR after fallback. Adaptive mode
substantially reduces fixed DDTree's GSM8K loss, but it did not beat AR on
either C=4 suite.

All measured requests completed with exact 96-token accounting and no errors.
Ordered output hashes were not stable across the five repeats for either AR or
DDTree. Because this also affects AR, it is not a DDTree-only signal, but it is
still a concurrent reproducibility warning; these numbers support performance
routing experiments, not a deterministic-output or correctness claim. The
canonical summary reports this check explicitly as `Stable output`.

For the measured setup, fixed blog-DDTree is the clear choice at C=1–3. At C=4
the optimum becomes workload-dependent: use fixed DDTree for a known
high-acceptance Math-like workload, and AR for GSM8K or unknown/mixed traffic.
Adaptive DDTree is a lower-risk single speculative configuration, but AR still
had the highest or statistically close aggregate goodput in this C=4 screen.
The available C=5/C=10 HumanEval screen also favors AR; reasoning-suite evidence
has not yet been collected above C=4. These are routing observations for this
hardware, model pair, prompt mix, and output length—not universal defaults.

Run a quick screening repeat:

```bash
MODEL=/path/to/Qwen3.6-27B-Q4_K_M.gguf \
LUCE_SERVER_BIN=server/build-hip/dflash_server \
LLAMA_SERVER_BIN=/path/to/llama-server \
harness/benchmarks/concurrency/run_qwen36_concurrency.sh
```

Run a decode-heavy comparison with the same harness:

```bash
MODEL=/path/to/Qwen3.6-27B-Q4_K_M.gguf \
LUCE_SERVER_BIN=server/build-hip/dflash_server \
LLAMA_SERVER_BIN=/path/to/llama-server \
WORKLOADS=short MAX_TOKENS=256 VARIANTS=luce-k8,llama REPEATS=3 \
harness/benchmarks/concurrency/run_qwen36_concurrency.sh
```

The short ragged prompts keep admission realistic while 256 forced output
tokens make generation dominate the measured window. Use `REPEATS=5` for
publication. Every measured case starts a fresh server and first runs a
discarded warmup at the same concurrency. The variants are:

- `luce-k8`: packed prefill with up to eight concurrent prefills.
- `luce-k1`: the same binary/configuration with packing width limited to one.
- `llama`: llama.cpp continuous batching with fixed `-b 2048 -ub 512`.

The 29 generated prompts are disjoint cohorts for C1/C4/C8/C16. C4 and above
contain four substantial length strata while holding the mean target length
constant. The default short, medium, and long profiles target approximately
400, 1,000, and 3,000 input tokens per request. The client refuses to wrap or
reuse a prompt; reports retain the exact server-observed token counts.

The headline metric is aggregate output goodput: exact server-reported
completion tokens divided by level wall time. It includes queueing, prefill,
and decode and must not be called decode throughput.

`Output-window tok/s` divides exact completion tokens by the interval from the
earliest observed first output to the final request completion. It removes the
initial all-prefill interval and is decode-facing, but it can still contain
staggered prefill while later requests await their first token.
`Request decode tok/s` is the median per-request estimate
`(completion_tokens - 1) / (end - first_output)`; it assumes the first
observed streaming event accounts for one token. Neither metric is pure kernel
decode throughput.

`Prompt tok/s to first` is the sum of server-reported prompt tokens divided by
the latest first-token arrival; it is a useful prefill-facing metric but still
includes admission, queueing, and transport. Report TTFT median/max alongside
all throughput metrics.

The K8-vs-K1 comparison is the causal packing ablation. The K8-vs-llama
comparison is the product comparison. Five paired repeats, the exact command
and hashes recorded in each case, zero failures, and a fixed declared output
length are required before using results in a post. The standard prefill-facing
protocol uses 64 output tokens; the decode-heavy protocol above uses 256.
Variant gains are computed as the median of same-repeat ratios, not as a ratio
of independently aggregated medians. The summarizer rejects mismatched repeat
sets. It also marks whether each variant produced the same ordered output
hashes across at least two repeats; a one-repeat screen reports stability as
`n/a`, and an unstable result is a correctness warning, not a performance win.
