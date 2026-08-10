# Ragged packed-prefill performance plan

_Status: implementation screen and follow-up plan, 2026-08-10. The small
DeltaNet concat specialization is implemented; query-tiled attention remains
design-only._

## Decision

The next concurrency-specific kernel project should be a query-tiled paged
prefill-attention specialization, beginning with four adjacent query rows per
K/V head on gfx1151. It should not begin with a 64-128-row tile.

Before that larger project, a strict 32x32 LDS transpose specialization was
added for DeltaNet's short-history-plus-transposed-token concat. In one-repeat
engineering screens against the same M-RoPE-fixed executable with only the
GGML/HIP library swapped, it improved aggregate completion goodput by 3.6% at
C=4, 4.6% at C=8, and 7.4% at C=16 on a four-stratum ragged prompt set
(roughly 213-1,221 input tokens/request). A separate shorter screen showed the
same direction. These are retention screens, not publication numbers.

Grouping equal-length DeltaNet segments is technically possible, but a
best-case end-to-end screen improved C=8 by 1.5% and C=16 by 0.5%. That case
maximized the fraction of rows groupable by the consecutive-equal-length
design. The sub-2% screen makes genuinely ragged cohorts unlikely to justify
that added lowering complexity. The experiment was removed rather than
expanded into a more complicated ragged lowering.

The attention project is larger, but it attacks work introduced specifically
by packed prompt queries: adjacent causal rows currently reload and decode the
same paged K/V data independently. It is also reusable across decoder models
and cache formats.

## What the packed-prefill replacement already packs

There are two useful meanings of "packed prefill":

1. **Graph-level packing.** Token rows from several requests share one model
   graph and one graph execution.
2. **Kernel-level packing.** A kernel recognizes sequence boundaries and
   shares traversal or state work among rows from the same request.

The packed-prefill implementation already provides the first meaning for
Qwen. The sequence engine builds
one `n_total = n_prefill + decode_bucket` token axis, uploads one packed
embedding tensor, and calls the graph once
(`qwen35_seq_engine.cpp:326-506`). Transformer-layer projections, FFNs, RoPE,
and final norm run across that packed width. The LM head runs only on gathered
committing-prefill and decode rows. Full attention is one ragged
paged-attention op per attention layer
(`qwen35_target_graph.cpp:877-906`).

Two important parts are not fully grouped inside their kernels:

- The paged-attention launch maps `grid.y` to one flattened query row. It
  shares K/V across Qwen's six GQA query heads, but not across adjacent prompt
  queries (`paged-attn.cu:173-260`, `889-935`).
- DeltaNet keeps its projections whole-batch, then lowers each prompt segment
  as an independent `S=1` convolution/recurrent/norm subgraph
  (`qwen35_target_graph.cpp:1003-1106`).

A precise description for the blog is:

> The packed-prefill replacement uses a Qwen-specific packed token graph with
> ragged sequence metadata.
> Its per-layer projections and FFNs are wide-batched, while the hybrid
> recurrent core remains segment-lowered and the paged-attention kernel
> remains query-row-oriented.

Calling the whole PR merely "round-robin interleaving" understates the current
implementation. Calling the current attention kernel "tiled prefill
FlashAttention" would overstate it.

## Negative screens already completed

These are one-repeat engineering screens, useful for rejecting weak ideas but
not publication measurements.

| Experiment | Best observed result | Disposition |
|---|---:|---|
| Native wide Q4_K WMMA prototype | no more than 0.48x the existing hipBLAS path at packed N=4096 | removed |
| Double idle/mixed token budgets | about +1.2-1.4% at C=4/C=8; C=16 exceeded graph capacity | reverted |
| Replace DeltaNet's left-fold output concat with direct slice writes | C=8 and C=16 effectively flat | reverted |
| Force paged attention to one partition | -2.9% at C=8, -1.1% at C=16 | rejected |
| Group an all-equal DeltaNet cohort into one `T x S` recurrence | +1.5% at C=8, +0.5% at C=16 | removed |
| Remove explicit DeltaNet Q/K head repeats | GPU primitive checks passed; short flat and medium +0.6% | reverted |
| Merge graph-compute and argmax synchronization | short C=8 slightly slower | reverted |

The first item showed that there is no obvious missing-native-Q4 shortcut:
vendored GGML already contains a Q4_K integer-WMMA MMQ path, and the tested
prototype lost to the selected wide hipBLAS path. A better fused wide kernel
remains possible, but it is a separate general-linear project rather than the
concurrency story.

## Why arbitrary ragged DeltaNet grouping is deferred

A small opportunistic implementation can group only maximal **consecutive**
runs with the same timestep count. A packed layout

```text
[all T rows of request A][all T rows of request B]
```

can be reshaped safely to `[channels, T, S=2]` when both `T` values match.
Non-consecutive equal lengths cannot be grouped without a gather/reorder.
Physical sequence slots also cannot be assumed adjacent, so every grouped run
needs an explicit slot-id view, state gather, and state scatter. Singleton and
unequal tails must keep the current path.

A genuinely single ragged recurrence is a larger model-specific project. It
needs per-sequence lengths, masked convolution and recurrent traversal,
dynamic state-slot descriptors, direct writes to the original token layout,
and a graph-cache key that encodes the run topology. It should be reconsidered
only with a profile showing that DeltaNet plus its segment assembly has grown
well beyond the current profile. The earlier 7.6% figure was only the coarse
sum of DeltaNet and concat kernel families, not an achievable speedup ceiling;
the direct final-concat experiment itself was flat.

## Current paged-attention limitation

Qwen supplies Q as `[D=256, n_query, Hq=24]`, paged K/V as
`[D, pool_tokens, Hkv=4]`, and one physical slot id plus inclusive causal
position per query row (`qwen35_target_graph.cpp:877-906`). The current kernel:

- quantizes each Q row once;
- gives one warp a query row and, in its widest specialization, one K/V head's
  six GQA query heads, with 3/1-head occupancy fallbacks;
- walks 32-token score tiles through 16-token physical pages;
- performs online softmax and Q4_0/Q8_0 dequantization or F16 loading;
- starts its partition count from `ceil(logical_blocks / 64)`, adds an
  occupancy floor, and combines FP16 partials when partitioning is active.

This is a good decode kernel. During prefill, however, query positions
`p, p+1, p+2, ...` revisit almost the same K/V range. Nothing guarantees that
the compressed rows loaded for query `p` remain available for query `p+1`.

The existing microbenchmark does not measure this opportunity. Its "ragged"
case varies the K/V length of several sequences but still submits one query row
per sequence (`bench_paged_attention.cpp:605-728`).

## First kernel: four-query LDS reuse

The lowest-risk material change preserves the current wave-level math:

```text
current block:  4 K/V heads x 1 query row
proposed block: 1 K/V head  x 4 adjacent query rows
```

Each of four waves retains the current six-head GQA registers, dot products,
online softmax, and value accumulator for one query row. The workgroup
cooperatively stages one 32-token compressed K/V tile for a single K/V head in
LDS, and all four waves reuse it.

For Q4_0 at D=256, one K or V row occupies 144 bytes. A 32-token K+V stage is
therefore:

```text
32 tokens * (144-byte K + 144-byte V) = 9,216 bytes
```

This fits comfortably in LDS without first redesigning the online-softmax
algorithm. Start with `Q_TILE=4`; test `Q_TILE=8` only after measuring VGPR use,
occupancy, and LDS bank behavior. A 64-128-row FlashAttention rewrite would
change the decomposition, accumulator strategy, and GQA reuse simultaneously,
which is too much for the first PR on wave32 with D=256.

### Ragged descriptor

The row-level slot ids already guarantee correctness, but they do not tell a
workgroup where a safe multi-row tile ends. Add an optional compact descriptor
tensor, conceptually:

```cpp
struct PagedQueryTile {
    int row_begin;
    int row_count;     // 1..Q_TILE
    int physical_slot;
};
```

The Qwen graph builder emits tiles wholly inside each prompt segment. Ragged
prompt tails become partial 1-3-row descriptors rather than throwing away
valid reuse; decode rows remain singleton descriptors. Non-adjacent rows are
never combined. Every wave still reads its exact `query_positions[row]` and
applies its own causal limit; the tile's largest position may size the common
page traversal but cannot replace the per-row mask.

For `row_count > 1`, one block handles one K/V head with one wave per query
row. For `row_count == 1`, the four waves retain the current four-K/V-head
topology. Select the descriptor path when any tile contains at least two rows,
while allowing singleton prompt tails and decode rows in the same op. This is
how a mixed prompt/decode traversal remains one paged-attention launch.

Invalid or missing physical pages must preserve the current `-inf` score and
zero-value semantics. Staging a missing K row as numeric zero would incorrectly
admit it into the softmax.

### Shared structure, format specializations

Keep the algorithm generic while specializing loaders at compile time:

```text
PagedQueryTile scheduler
  + shared page resolver
  + per-row causal online softmax
  + PagedKvLoader<Q4_0 | Q8_0 | F16>
```

The first dispatch should be strict: HIP, exact gfx1151, wave32, D=256,
block size 16, Q4_0 K and V, valid query-tile descriptors, and at least two
queries in at least one tile. Every other shape stays on the current kernel.
Q8_0 and F16 loaders follow only after the Q4_0 topology wins.

Keep the current direct/partitioned split and combine kernel initially. A tile
uses a common partition range; rows with no valid tokens in a partition emit
the existing sentinel partial.

## Performance model

Let a prompt chunk contain `Q` new queries after a prefix of length `P`. The
number of attended K/V rows is

```text
A = Q * (P + 1) + Q * (Q - 1) / 2
```

For Q4_0 K/V, current approximate global K/V traffic is
`288 * Hkv * A` bytes. A query tile of width `M` can reduce repeated K/V reads
toward `M`-fold, although arithmetic and LDS traffic remain unchanged.

For `Q=512`, `P=4864`, and Qwen's four K/V heads:

- `A = 2,621,696` attended rows;
- current nominal K/V traffic is about 3.02 GB per attention layer;
- `M=4` lowers the nominal traffic toward 0.755 GB;
- attention arithmetic remains about 64.4 GFLOP.

That raises nominal arithmetic intensity from roughly 21 to 85 FLOP/byte. It
does **not** promise a 4x kernel speedup: arithmetic, page lookup, barriers,
softmax, final stores, cache effects, and occupancy remain. A credible first
target is 1.6-2.3x for the attention kernel. With attention at 10.2% of the
measured prompt-heavy profile, a 2x kernel gives about 5.4% end-to-end, 3x gives
about 7.3%, and free attention would cap the gain near 11.4%.

## Staged implementation

### Stage 0: make the benchmark prefill-shaped

Extend `bench_paged_attention` before changing the kernel:

- multiple query rows per physical sequence, not one decode row;
- query lengths 256, 512, and 1024;
- prefixes 0, 2048, and about 4864 tokens;
- `K_prefill = 1, 4, 8` ragged prompt segments and
  `D_decode = 0, 4, 8, 16` singleton decode rows, including C=16-relevant
  mixed shapes;
- Q4_0/Q4_0 first, then Q8_0 and F16;
- old kernel, candidate kernel, and dense attention as a comparator/reference
  where valid;
- ten warmups, seven duration-sized samples, median time, and selected-path
  telemetry.

The benchmark must validate both GPU paths against the CPU oracle before
timing. Report query rows/s, attended K/V rows/s, median microseconds, scratch
bytes, and the actual query-tile occupancy distribution. Freeze a weighted
shape distribution from real server topology telemetry before applying the
1.6x acceptance gate.

### Stage 1: Q4_0 `Q_TILE=4`

- Add the optional descriptor input to `GGML_OP_PAGED_ATTN`.
- Build descriptors from Qwen's existing `QwenPrefillSegment` boundaries.
- Add the gfx1151 Q4_0/Q4_0 tiled specialization behind an explicit A/B mode.
- Preserve the old kernel as the default fallback until the acceptance gate
  passes.

### Stage 2: tune, then generalize

- Compare query tiles 2, 4, and 8.
- Tune 16- versus 32-token K/V stages and partition thresholds.
- Add Q8_0 and F16 loader traits without runtime branches in the hot loop.
- Generalize architecture dispatch only after per-architecture measurements.

## Correctness and acceptance gates

Add tests covering:

- ragged segment boundaries and singleton decode rows;
- positions 15/16 and 31/32 around page and score-tile boundaries;
- partial query tiles and nonmonotonic physical slot ids;
- invalid block-table entries;
- direct and multi-partition paths;
- Q lengths longer than one tile;
- old-kernel versus new-kernel output plus the independent CPU oracle.

The first performance PR should require:

- at least 1.6x on the weighted Q4_0 prefill-attention microbenchmark;
- at least 3% end-to-end at C=4, C=8, and C=16 on the ragged short workload;
- no C=1 regression greater than 2%;
- exact request success/token counts and an accepted numerical tolerance;
- one-repeat screening before five-repeat publication runs.

## Publication benchmark requirements

The current local blog fixtures are deliberately uniform: all 16 short JSONL
records have the same byte length, as do all 16 medium records. A disjoint
C=1/4/8/16 sweep consumes 29 prompts, so the benchmark wraps these fixtures;
the observed C=16 token-count spread is only about seven tokens. They are useful
for controlled saturation, but they do not demonstrate ragged serving.

The runner and aggregator also hardcode
`DFLASH_MAX_CONCURRENT_PREFILLS=2`, while the intended packed configuration
uses eight. Existing smoke metadata is internally inconsistent about K=2
versus K=8, so it cannot be used as publication provenance.

Before making the concurrency claim:

1. Record the maximum concurrent prefills plus the server commit and binary
   hashes that define the engine-owned per-step token limits. Remove the
   aggregator's K=2-only assumption.
2. Add at least 29 distinct, token-counted prompts per workload. The C=4, C=8,
   and C=16 cohorts should each contain the same four length strata, unique
   deterministic jitter, a max/min ratio of at least 1.5, and means matched
   within about 1%. Rotate request order across repeats.
3. Use the same text, model, Q4_0 K/V cache, context capacity, fixed 64 output
   tokens, greedy sampling, disabled prompt cache, and simultaneous release for
   both servers. Record both engines' observed prompt-token counts: the current
   chat paths differ by roughly two tokens per request.
4. Run C=1, C=4, C=8, and C=16 with fresh server processes and alternating
   Lucebox/llama.cpp order for at least five paired repeats. Tune llama.cpp on
   a disjoint calibration corpus, then freeze and publish its configuration.
5. Compare the packed K=8 configuration with the same build at K=1.
   Optionally include K=2 and the foundation branch to separate cohort width
   from code revision. Record a server-side
   histogram of actual prefill segment counts, tokens per segment, and mixed
   decode rows; a configured K=8 is not proof that packed steps executed.
6. Hash the executable **and** its resolved GGML/HIP shared libraries; record
   commit, submodule state, build configuration, full command, relevant
   environment, ROCm version, clocks, prompt manifest, and prompt-token counts.
7. Label aggregate completion tok/s as including queueing, prefill, and decode.
   Also publish TTFT by input-length stratum, start skew, prompt totals,
   failures, and per-request completion rate.

Aggregate completion tok/s is a good user-facing metric when every request
generates the same fixed number of tokens: it is the inverse of burst makespan.
It is not a pure decode rate and should not be labeled as one. Prompt tokens per
wall second can be shown as a secondary burst metric, but input and output
tokens should not be added together as if they had equal cost. A portable
mechanism metric is
`sum(prompt_tokens) / max(start_offset + TTFT)`; label it "cold prompt
throughput to first token" because it includes admission, queueing, first-token
compute, and SSE delivery. At C=4/8/16, report TTFT median and maximum rather
than a statistically weak p95.

There is also a publication blocker to resolve independently of this kernel.
After correcting M-RoPE position layout, repeated C=8 full 64-token waves
matched 8/8 hashes and repeated C=16 first-token waves matched 16/16. Repeated
C=16 full waves matched only 9/16, however, and C=1 versus C=16 full outputs
matched 8/16. Packed prefill and first-token output are stable in these checks,
but full C=16 autoregressive runs still diverge after the first token as
mixed-prefill/decode and decode-bucket topology evolves. This must be
localized before publication.

The concurrency advantage should be attributed to packed prefill only when all
three results agree: packed K=8 beats tuned llama.cpp end-to-end, K=8 beats the
same build at K=1, and the topology telemetry confirms multi-sequence
packed execution. Avoid "destroys llama.cpp", "llama.cpp serial prefill", "16
prompts in one packed graph", or an unscoped "X times faster". A defensible
claim names the model, quantization, GPU, ragged workload, concurrency, metric,
and repeat count.

## Ranked concurrency-specific follow-ups

1. **Specialize the per-segment non-contiguous DeltaNet history concat** — low
   backend-code cost, no graph/API change, and directly scales with ragged
   segments. Implemented and retained after a material one-repeat C=4/C=8/C=16
   screen; publication still requires the paired protocol below.
2. **Four-row tiled paged prefill attention** — medium/high implementation
   effort; estimated 4-6% end-to-end; general across decoder models and cache
   formats.
3. **One descriptor-driven ragged DeltaNet kernel** — high effort; removes
   repeated segment launches and assembly, but DeltaNet plus concat accounted
   for only about 7.6% in the coarse profile and the low-code equal-cohort
   version screened below 2%.
4. **In-place ragged recurrent-state updates and post-assembly norm/gate** —
   low/moderate graph work that can remove per-segment state copies and
   token-local launches; each needs its own numerical and C4/C8/C16 screen.
5. **Copy-free paged-attention boundaries** — validate removing unnecessary
   K/V `cont`, Q materialization, and output permute/`cont` operations before
   changing the attention math.
6. **Same-input projection/SwiGLU fusion** — potentially useful at smaller
   packed widths, but it is a general linear-stack optimization rather than the
   concurrency headline.

Skipping first-use recurrent-slot resets is a legitimate cold-admission
cleanup, but it is deliberately excluded from the headline list. The current
ascending benchmark would leave more virgin slots at higher concurrency and
therefore manufacture an order-dependent scaling gain. Screen it only after a
discarded all-slot warmup, or report cold and recycled-slot results separately.

Scheduler abstraction, allocator micro-optimization, decode-attention tuning,
launch/graph experiments, and larger token budgets do not currently have
evidence strong enough to precede the query-tiled attention work.
