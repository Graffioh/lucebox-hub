# Qwen3.6 concurrency benchmark

This protocol measures the serving behavior targeted by packed continuous
prefill and concurrent decode. It is intentionally small: one streaming client,
one fresh-process runner, one deterministic prompt generator, and one summary
script.

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
constant. The default short, medium, and long profiles target mean lengths of
400, 1,000, and 3,000 words per request. Those generator targets are not token
counts; reports retain the exact server-observed token counts for the selected
model and tokenizer. The client refuses to wrap or reuse a prompt.

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
