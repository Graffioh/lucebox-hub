# Qwen3.6 concurrency benchmark

This protocol measures the serving behavior targeted by packed continuous
prefill. It is intentionally small: one streaming client, one fresh-process
runner, one deterministic prompt generator, and one summary script.

Run a quick screening repeat:

```bash
MODEL=/path/to/Qwen3.6-27B-Q4_K_M.gguf \
LUCE_SERVER_BIN=server/build-hip/dflash_server \
LLAMA_SERVER_BIN=/path/to/llama-server \
harness/benchmarks/run_qwen36_concurrency.sh
```

Use `REPEATS=5` for publication. Every measured case starts a fresh server and
first runs a discarded warmup at the same concurrency. The variants are:

- `luce-k8`: packed prefill with up to eight concurrent prefills.
- `luce-k1`: the same binary/configuration with packing width limited to one.
- `llama`: llama.cpp continuous batching with fixed `-b 2048 -ub 512`.

The 29 generated prompts are disjoint cohorts for C1/C4/C8/C16. C4 and above
contain four substantial length strata while holding the mean target length
constant. The client refuses to wrap or reuse a prompt.

The headline metric is aggregate output goodput: exact server-reported
completion tokens divided by level wall time. It includes queueing, prefill,
and decode and must not be called decode throughput. `Prompt tok/s to first`
is the sum of server-reported prompt tokens divided by the latest first-token
arrival; it is a useful prefill-facing metric but still includes admission,
queueing, and transport. Report TTFT median/max alongside both metrics.

The K8-vs-K1 comparison is the causal packing ablation. The K8-vs-llama
comparison is the product comparison. Five paired repeats, the exact command
and hashes recorded in each case, zero failures, and fixed 64-token outputs are
required before using results in a post.
The summary also marks whether each variant produced the same ordered output
hashes across repeats; an unstable result is a correctness warning, not a
performance win.
