# Qwen3.6 concurrent feature matrix

`run_qwen36_feature_matrix.sh` extends the PR #596 protocol with feature
ablations for the complete Strix Halo configuration:

- `ar`: concurrent paged autoregressive control.
- `ddtree`: adds the decode draft, DDTree, and the recorded budget.
- `pflash`: adds auto prefill compression, its drafter, and persistent draft
  residency.
- `kvflash`: adds bounded KV residency in auto mode and explicitly supplies
  the hashed prefill drafter for relevance-scored page selection; prefill
  compression remains off in this ablation.
- `full`: enables DDTree, PFlash, and KVFlash together with both devices on
  `hip:0`.
- `llama`: optional external comparison; its binary is required only when this
  variant is explicitly requested.

Run the default bounded C4 screening repeat (seven applicable fresh-server cases):

```bash
MODEL=/opt/models/Qwen3.6-27B-Q4_K_M.gguf \
DRAFT_MODEL=/opt/models/draft/dflash-draft-3.6-q4_k_m.gguf \
PREFILL_DRAFTER=/opt/models/Qwen3-0.6B-BF16.gguf \
REPEATS=1 \
harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh
```

On a 128 GiB Strix Halo host, budget roughly 45–90 minutes for this smoke run;
the long-context AR controls dominate and actual time depends on the build.
Every row remains independently selectable through `VARIANTS`. For example:

```bash
WORKLOADS=short CLIENTS=4 VARIANTS=ar,ddtree MAX_TOKENS=256 REPEATS=1 \
harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh
```

Use five fresh-process, paired repeats for published measurements:

```bash
WORKLOADS=short,compression CLIENTS=1,4,8,16 \
VARIANTS=ar,ddtree,pflash,kvflash,full MAX_TOKENS=256 REPEATS=5 \
harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh
```

That full matrix can take roughly 15–30 hours on Strix Halo; keep the generated
case directory so interrupted or suspect rows can be diagnosed rather than
quoted.

## Activation workloads

Auto features cannot be validated with the original 400–4,000 word prompts.
The extension adds two deterministic, disjoint 29-prompt cohorts:

- `compression`: 34K–40K words, chosen after observing 38,130–44,856
  tokens with the development Qwen GGUF tokenizer.
- `kv-pressure`: 12K–18K words, which produced 13,463–20,190 tokens with
  that tokenizer against the runner's explicitly recorded 8K pool cap.

The observed counts are a fixture sanity check, not a claim derived from word
count and not publication evidence. Each row records the model hash; the proof
cross-checks logged raw PFlash input against `usage.prompt_tokens`, requires
it to meet the recorded auto threshold, and uses server-reported effective
tokens plus actual page traffic for KVFlash.

The bounded default uses `short,compression` at C4. It runs PFlash and the
full configuration only on `compression`; KVFlash can also be selected on
`kv-pressure`. Inapplicable pairs are printed as skips and never appear as
successful rows. AR controls use the same prompts, so feature deltas remain
paired.

## Fail-closed feature proof

The server must write one JSON object per completed request with this prefix:

```text
[concurrency-metrics] {"request_id":"...", ...}
```

Required fields are `effective_prompt_tokens`, `ddtree_steps`,
`ddtree_accepted_tokens`, `target_forwards`, `kvflash_page_ins`,
`kvflash_page_outs`, `kvflash_resident_blocks`, `kvflash_reselects`,
`pflash_applied`, `pflash_input_tokens`, and `pflash_output_tokens`.

The proof tool correlates log objects with measured SSE request IDs and also
checks the log's effective token count against
`usage.timings.effective_prompt_tokens`. A requested feature fails the case
unless:

- DDTree has positive step and target-forward counts. Acceptance may be zero.
- PFlash reports `pflash_applied=true`, a smaller output prompt, and (in auto
  mode) an input token count at or above the recorded activation threshold.
- KVFlash always records an explicit hashed scorer drafter, reports its
  startup-observed physical pool and enabled metadata, and has a positive
  resident-block count. `kvflash`-only and `kv-pressure` rows must also show page-in or
  page-out traffic. For a `full` row, traffic is required only when a
  server-reported `effective_prompt_tokens` value exceeds that observed pool
  token limit; zero traffic is valid when PFlash compression fits in the pool.

After health succeeds, the runner fail-closed parses the backend's
`[parallel-kvflash] physical resident pool ...` and `[paged-attention] ...`
startup markers into `runtime_observed`. This distinguishes the actual resident
pool from both the requested auto cap and `--kv-pool-tokens`, which concurrent
KVFlash intentionally does not use to expand VRAM.

Each case retains the exact shell-escaped command, controlled launch
environment, literal client process arguments and client-script hash,
binary/shared-library/target/draft/PFlash-and-KV-scorer hashes, the ordered
`literal_screenshot_flags` array, all feature values, raw request report,
server log, and `feature-proof.json`. The summary refuses to
include a Lucebox row whose proof is missing or invalid.



