# GQBatching: quality-oriented ghost batching for one request

Status: implemented behind `--gqbatching`; GPU quality and latency measurements
are still pending.

GQBatching is an experimental quality-oriented decoding policy. It turns one
user generation into a temporary batch of short candidate paths, lets the
target score them, commits the best path and discards the rest.

The name means **Ghost Quality Batching**. “Ghost” means that the batch rows do
not come from independent users: they are speculative alternatives of one
request. “Quality” makes the goal explicit. This is unrelated to ghost batch
normalization.

This feature is not an exact speculative-decoding speedup. It deliberately
changes the decoding policy by choosing among target-scored paths. Latency and
throughput may improve or regress, but the design goal is better task quality
for a bounded amount of extra compute. Speed-oriented adaptive branching
belongs in DDTree, where exact speculative acceptance is preserved.

## In brief

```text
R9700: prefill + canonical target KV
                    │ captured prefix features + branch seeds
                    ▼
Strix Halo: baseline draft + one real batch-3 expansion
                    │ one combined hidden buffer
                    ▼
R9700: batched projection → one 32-row tree verify → score → commit one path
```

The default batch contains four paths of seven tokens:

```text
4 branches × 7 tokens = 28 candidate nodes
28 nodes + 1 shared root = 29 real rows
target allocation = 32 rows
```

The baseline draft is required to discover the four seeds and already provides
branch 0. Only the other three seeds need expansion, hence a real batch-3
(`K−1` for `K` total branches). Four branches also stay inside the target tile:
29 rows round to 32, while five seven-token branches need 36 real rows and
therefore the next 64-row allocation.

Thirty-two rows match the current verify tile. The target weights are read
once for the complete tree instead of once per candidate path. This raises
arithmetic intensity on the R9700, although profiling is still required to
show whether a particular model becomes compute-bound.

GQBatching runs one branching round per request. It does not keep multiple
generations alive and it does not generate several complete answers.

## Hardware split

- R9700: target prefill, target weights, canonical target KV, LM-head
  projection, tree verification, scoring, sampler and commit.
- Strix Halo: DFlash drafter and alternative-path expansion.
- Cross-device traffic: captured prefix features, seed embeddings and a bounded
  buffer of hidden rows.
- Disk and host RAM: no target KV spill is required by GQBatching.

The main prefill runs on the R9700 because its large prompt matrices benefit
most from the stronger GPU. Its target KV never moves: the committed prefix
and temporary target suffixes stay next to target attention. The Strix receives
only the captured features required by DFlash; it does not repeat the target
prefill.

The two AMD GPUs run in the same HIP process. This local path is the intended
mode for real drafter batching. The existing remote-drafter protocol remains
functional, but expands alternatives serially until it gains a batch request.

## Seed-and-Expand

The paths are built as follows:

1. Run one normal DFlash block and project its hidden rows on the R9700.
2. Take the top four tokens at the first candidate position.
3. Keep branch 0 and its existing top-1 tail as the baseline.
4. Put branches 1–3 into one Strix graph with the alternative tokens as seeds
   and the same read-only committed prefix.
5. Collect six hidden tail rows from each alternative: 18 rows in total.
6. Project all 18 rows together in one R9700 LM-head graph.
7. Build the four fully conditioned paths and verify their complete tree in
   one 32-row target graph.

```text
shared draft prefix ─┬─ baseline seed → existing 6-token tail
                     └─ one packed Strix batch
                         ├─ seed 1 → newly conditioned 6-token tail
                         ├─ seed 2 → newly conditioned 6-token tail
                         └─ seed 3 → newly conditioned 6-token tail
```

This is a real batch, not three calls grouped by the host. With the defaults,
the graph flattens `3 × 16` draft rows into one 48-row forward. Dense layers
operate on all 48 rows together. Block-diagonal attention masks let every
branch read the shared prefix and its own 16 noise rows, never another branch.
Repeated positions preserve the semantics of three independent draft blocks.

For this first implementation, the packed expansion uses the stateless DFlash
path instead of the persistent drafter-KV path. The prefix features are fused
once inside the batch, not once per branch. Supporting the same batch directly
over the persistent drafter KV is a later optimization, not required to test
whether real batching increases Strix utilization.

## Selection and commit

1. Run one ancestor-masked target verify over the complete tree.
2. Compute mean target log-probability for each path.
3. Keep branch 0 unless another path beats it by `--gqbatching-margin`.
4. Compact target KV, DeltaNet state, convolution state and captured features
   onto the selected path with the existing tree rollback.
5. Continue normal DFlash generation.

The score is a cheap first selector, not proof of better reasoning. GQBatching
changes the decoding policy, so quality must be evaluated on task results.

## Usage

Example when the R9700 is HIP device 0 and Strix Halo is HIP device 1:

```bash
./server/build/dflash_server model.gguf \
  --draft dflash-draft.gguf \
  --target-device hip:0 \
  --draft-device hip:1 \
  --gqbatching \
  --gqbatching-branches 4 \
  --gqbatching-horizon 7 \
  --gqbatching-margin 0.10
```

Check the actual HIP device order before launching. The ROCm build must contain
code objects for both GPUs.

| Flag | Default | Meaning |
|---|---:|---|
| `--gqbatching` | off | Enable one quality-oriented ghost batch per request |
| `--gqbatching-branches` | 4 | Candidate paths, from 2 to 8 |
| `--gqbatching-horizon` | 7 | Tokens in every path |
| `--gqbatching-margin` | 0.10 | Score gain required to replace branch 0 |

The configuration is reported under `/props.speculative.gqbatching`.

## Current limits

- Dense `qwen35` target on one target GPU.
- Local or second-GPU DFlash drafter required.
- Greedy generation and fast tree rollback required.
- Maximum 64 allocated verify rows.
- Thinking-budget closure, tool hints, stall recovery and the minimum-token
  floor keep their existing path; GQBatching waits for the first safe round.
- Local same-process expansion is one real packed drafter batch. Remote draft
  expansion is still serial because the IPC protocol is batch-1.
- The packed expansion currently recomputes draft prefix K/V once for the
  whole batch instead of reading the persistent drafter KV cache.
- Uncertainty-triggered activation and a proposal deadline are not implemented.
- Target KV on the Strix and remote-KV attention are not part of the design.

## What to measure

Compare normal DFlash and GQBatching with identical prompts and seeds. Record:

- task pass rate or result quality;
- end-to-end latency and time to first token;
- alternative-path win rate and score gain over branch 0;
- initial projection, Strix batch-3 expansion, 18-row projection and 32-row
  verify time;
- R9700 matrix utilization and memory bandwidth;
- Strix compute, bandwidth, power and thermal utilization.

Keep Seed-and-Expand only if independent paths improve task results or produce
useful alternative wins at an acceptable latency cost. If that succeeds, an
uncertainty trigger and a deadline can keep average latency close to normal
DFlash. Persistent-KV support for the packed Strix graph is the next low-level
optimization.

## Implementation map

- `server/src/common/gqbatching.{h,cpp}` builds and scores path trees.
- `server/src/common/dflash_draft_graph.{h,cpp}` builds the isolated packed
  Strix batch.
- `server/src/qwen35/qwen35_backend.cpp` expands paths, batches projection,
  runs the 32-row target verify and commits the winner.
- `server/test/test_gqbatching.cpp` covers path construction, row allocation,
  score margin and selection without requiring GPUs.
