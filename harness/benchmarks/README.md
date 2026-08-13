# Generation Benchmarks

These checks are separate from the client harness launchers. They compare Lucebox
generation against a llama.cpp baseline on the same target GGUF, using small
deterministic prompts.

Use this when you want to know whether a server change affects output quality or
decode speed. Use `harness/clients/` when you want to know whether Codex,
OpenCode, Open WebUI, Pi, and the other clients still work.

## Bench suites (HumanEval, GSM8K, Math500, Agent)

Run standard LLM and agentic benchmarks against a running Lucebox server:

```bash
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080
```

This sends benchmark prompts through the OpenAI-compatible `/v1/chat/completions`
endpoint and reports tok/s, TTFT, and correctness scores.

### Suites

| Suite   | Description                                        | Scoring          |
|---------|----------------------------------------------------|------------------|
| `he`    | HumanEval code-completion prompts (10)             | tok/s only       |
| `gsm`   | GSM8K arithmetic reasoning prompts (10)            | tok/s only       |
| `math`  | Math500 with `\boxed{}` correctness check (10)     | tok/s + accuracy |
| `agent` | Agentic workloads at 2K/8K/24K context (6)         | TTFT + tok/s     |

### Usage

```bash
# All suites (default)
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080

# Only Math500 correctness
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite math

# HumanEval + agent
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite he,agent

# Limit to 3 prompts per suite
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --n-sample 3

# Save JSON results
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --json-out /tmp/bench.json
```

### Options

- `--url` (required): Server base URL
- `--suite`: Comma-separated list or `all` (default: `all`)
- `--model`: Model name (default: `luce-dflash`)
- `--n-sample`: Max prompts per suite (default: all in file)
- `--prompts-dir`: Override prompt files directory
- `--json-out`: Write JSON results to this path

### Prompt files

Static JSONL files in `harness/benchmarks/prompts/`:

- `bench_he.jsonl` — HumanEval code-completion
- `bench_gsm.jsonl` — GSM8K arithmetic reasoning
- `bench_math.jsonl` — Math500 with `gold_answer` field
- `bench_agent.jsonl` — Agentic prompts with `bucket` field (2k/8k/24k)

### Correctness

Math500 responses are scored by extracting `\boxed{}` answers and comparing
against gold with normalized math equivalence. Accuracy is reported in the
output but does not gate the exit code.

---

## Lucebox vs llama.cpp

Run from the repo root on the GPU host:

```bash
harness/benchmarks/run_lucebox_vs_llamacpp.sh
```

The runner starts llama.cpp first, runs the prompt set, stops it, then starts
Lucebox and runs the same prompt set. It is sequential on purpose so a 24 GB
card does not need to hold two copies of the target model.

Common overrides:

```bash
MAX_CTX=65536 MAX_TOKENS=512 harness/benchmarks/run_lucebox_vs_llamacpp.sh
LLAMA_SERVER_BIN=/path/to/llama-server harness/benchmarks/run_lucebox_vs_llamacpp.sh
PROMPTS=/tmp/my_prompts.jsonl harness/benchmarks/run_lucebox_vs_llamacpp.sh
```

Each run writes:

- `llamacpp.json`: raw llama.cpp endpoint results
- `lucebox.json`: raw Lucebox endpoint results
- `compare.json`: machine-readable comparison
- `report.md`: speed and expected-output summary

Prompt files are JSONL. Each line needs `id` and either `prompt` or `messages`.
Optional `expect_contains` and `expect_regex` fields define lightweight accuracy
checks.

---

## OFlash online-distillation benchmark

Do not evaluate an adapter on prompts that were present in its capture stream.
`oflash_benchmark.py` creates deterministic suite-stratified folds from the
existing HumanEval, GSM8K and Math500 prompt files. With the default three
folds, each adapter trains on two-thirds of every suite and is evaluated on the
unseen third; every prompt is held out exactly once across the three adapters.

```bash
python3 harness/benchmarks/oflash_benchmark.py prepare \
  --out-dir /tmp/oflash-bench/folds --folds 3 --seed oflash-v1
```

For each fold, use fresh server processes and a unique OFlash store. Disable
prefix caches in every arm. The three relevant arms are:

1. **Frozen base:** Q4 drafter, OFlash capture enabled, no trainer, generation
   zero. Run `fold-N/heldout` with `--oflash-phase heldout-base`.
2. **Adaptation:** a different fresh store, trainer enabled. Run only
   `fold-N/adapt`, repeating it as needed until a generation is promoted. Never
   run held-out prompts in this process.
3. **Frozen adapted:** stop the adaptation server, restart the same store
   without a trainer, and run `fold-N/heldout` with
   `--oflash-phase heldout-adapted`.

The phase guard reads `/props`: adaptation requires a live trainer, while both
held-out phases reject a live trainer. It also rejects an adapter-generation
change, missing capture data, or newly dropped capture records. For example:

```bash
# Frozen generation-zero server, using a clean baseline store:
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:18080 --suite he,gsm,math \
  --prompts-dir /tmp/oflash-bench/folds/fold-0/heldout \
  --max-tokens 512 \
  --oflash-phase heldout-base \
  --json-out /tmp/oflash-bench/fold-0-base.json

# Trainer-enabled server, using only adaptation prompts. Repeat this command
# for additional epochs; retain every epoch report.
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:18080 --suite he,gsm,math \
  --prompts-dir /tmp/oflash-bench/folds/fold-0/adapt \
  --max-tokens 512 \
  --oflash-phase adapt \
  --json-out /tmp/oflash-bench/fold-0-adapt-epoch-1.json

# Restart the adaptation store without --oflash-trainer-bin, confirm the
# promoted generation warm-started, then evaluate the untouched prompts:
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:18080 --suite he,gsm,math \
  --prompts-dir /tmp/oflash-bench/folds/fold-0/heldout \
  --max-tokens 512 \
  --oflash-phase heldout-adapted \
  --json-out /tmp/oflash-bench/fold-0-adapted.json

python3 harness/benchmarks/oflash_benchmark.py compare \
  --baseline /tmp/oflash-bench/fold-0-base.json \
  --candidate /tmp/oflash-bench/fold-0-adapted.json \
  --json-out /tmp/oflash-bench/fold-0-compare.json \
  --md-out /tmp/oflash-bench/fold-0-compare.md

# Repeat each frozen arm once. This gates deterministic output and acceptance
# while reporting (but not failing on) ordinary timing variation:
python3 harness/benchmarks/oflash_benchmark.py repeat \
  --first /tmp/oflash-bench/fold-0-adapted.json \
  --second /tmp/oflash-bench/fold-0-adapted-repeat.json \
  --json-out /tmp/oflash-bench/fold-0-adapted-repeat-check.json
```

After all three folds, pool the disjoint held-out reports (repeat the paired
arguments for folds 1 and 2):

```bash
python3 harness/benchmarks/oflash_benchmark.py pool \
  --baseline /tmp/oflash-bench/fold-0-base.json \
  --candidate /tmp/oflash-bench/fold-0-adapted.json \
  --baseline /tmp/oflash-bench/fold-1-base.json \
  --candidate /tmp/oflash-bench/fold-1-adapted.json \
  --baseline /tmp/oflash-bench/fold-2-base.json \
  --candidate /tmp/oflash-bench/fold-2-adapted.json \
  --json-out /tmp/oflash-bench/pooled.json \
  --md-out /tmp/oflash-bench/pooled.md
```

Pooling rejects duplicate held-out case IDs, which catches accidental fold
reuse. Keep the per-fold reports: a pooled improvement that is negative in one
domain/fold is not evidence of uniformly useful adaptation.

The benchmark preserves the server's per-request `usage.accept_rate` alongside
server-side decode timing, TTFT, output, and correctness. Comparison is paired
by suite and case ID and reports deterministic bootstrap intervals for held-out
acceptance delta and speedup; the pooled report uses a fold-aware hierarchical
bootstrap so prompts trained by the same adapter are not treated as independent
folds. A base/adapted output mismatch fails the command.
An optional target-only autoregressive report over the identical held-out cases
can be supplied with `--target-reference`; any adapted/reference mismatch then
fails as well.

OFlash phases use non-streaming chat responses because that response's
`usage` object currently carries `accept_rate`; ordinary `bench` runs remain
streaming for client-observed TTFT. Frozen OFlash reports therefore record
server prefill/decode timings but leave client TTFT unset.

The 30 short HumanEval/GSM8K/Math500 prompts are suitable for a bounded first
signal, not a publication-scale conclusion. Use all three folds and report each
fold separately before pooling. The six agent prompts can be added with
`--suites all`, but their 2K/8K/24K buckets are too sparse for a meaningful
fold-level confidence interval. The repository's 164-task HumanEval+ evaluator
under `server/scripts/quality_humaneval_plus.py` is the next larger code-quality
gate after this bounded cross-validation passes; exclude any tasks exposed in
that fold's adaptation stream or use a separate untouched corpus.
