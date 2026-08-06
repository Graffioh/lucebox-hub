# BTFlash — short Branch Trees for single-request quality

_Drafted 2026-08-06 against main (3f3d60b). Status: BT1 prototype implemented;
later evaluator, two-GPU, and scheduler experiments remain design work.
Initial target: Qwen3.6-27B through the `qwen35` backend on one R9700 + Strix
Halo machine, serving one user request at a time._

`COMPUTE_SATURATION.md` describes the row-budget/roofline frame. BTFlash is
the quality lever: spend a bounded amount of otherwise underused compute on
short semantic branch trees, prune them early, and continue only the most
promising state.

## tl;dr

BTFlash is not full-response best-of-K. It forks K short reasoning
continuations from one shared target cache, advances them together for a small
horizon H, evaluates them, prunes to one or two survivors, and continues from
the winner. A branch segment replaces the same H serial steps of the normal
generation; it is not appended after them:

```
                                     ┌─ branch A: H thinking tokens ─┐
  prompt → ordinary decode → fork ───┼─ branch B: H thinking tokens ─┼─ score/prune
           (shared target state)     ├─ branch C: H thinking tokens ─┤      │
                                     └─ branch D: H thinking tokens ─┘      └─ continue winner
```

If `T_forward(K)` stays close to `T_forward(1)` for small K, the branch
interval uses more arithmetic at nearly the latency of the serial interval.
The goal is deliberately modest: **increase productive compute utilization
and selected-answer quality at bounded latency**, not claim full roofline
saturation. Every performance and quality statement is a hypothesis until it
is measured on the R9700 + Strix Halo box.

The engine mechanism is a generalization of DDTree. DDTree creates short-lived
drafter-proposed token branches and uses exact target acceptance to improve
speed. BTFlash creates target-sampled semantic branches, keeps their state
across several forward passes, and uses a task-aware evaluator to improve the
eventual answer. Both should share one branch-tree runtime and one row-budget
scheduler.

## 1. Scope and non-goals

Initial scope:

- **One request, one prompt.** This is intra-request test-time compute, not
  multi-user continuous batching.
- **Target model:** Qwen3.6-27B Q4_K_M through the existing `qwen35` backend.
- **Primary GPU:** R9700 (`gfx1201`, 32 GB) owns target weights, target KV,
  DeltaNet state, target forwards, branch compaction, and the final stream.
- **Second GPU:** Strix Halo (`gfx1151`, unified memory) is used where it can
  add useful work without moving the target cache: drafter execution and
  evaluator/value-model inference.
- **Output:** one selected answer. Losing reasoning branches stay internal.

Non-goals for the first implementation:

- No K complete answers and end-of-response best-of-K in the critical path.
- No multi-tenant scheduling or cross-request batching.
- No target layer split across R9700 and Strix Halo. The GPUs have separate
  memory domains; transferring activations layer by layer would work against
  the weight/KV locality BTFlash depends on.
- No claim that higher arithmetic intensity equals measured GPU utilization.
  SQ/matrix activity, achieved FLOP/s, bandwidth, and kernel dispatch must be
  profiled.
- No long-context BTFlash while tree rows and KVFlash paging remain
  incompatible.

## 2. DDTree and BTFlash: one mechanism, different objective

| | DDTree | BTFlash |
|---|---|---|
| Objective | More accepted tokens per target step | Better final answer at bounded latency |
| Branch source | DFlash drafter | Target sampler / semantic policy |
| Lifetime | One verify round | H target steps, optionally repeated |
| Selection | Exact speculative acceptance | Vote, verifier, value model, or judge |
| Output contract | Preserve target sampling distribution | Deliberately spend test-time compute |
| Main currency | Depth | Width, then early pruning |

The common runtime owns nodes with `parent_id`, `semantic_branch_id`, absolute
position, sampler state, and recurrent-state lineage. A single row budget is
allocated between DDTree depth and BTFlash width:

```
N_total ≈ semantic branches K × speculative nodes per branch D
```

Examples under a 16–24 row cap:

| Mode | K | D | N_total | Intent |
|---|---:|---:|---:|---|
| DDTree-heavy | 1 | 16 | 16 | speed |
| light branching | 2 | 8 | 16 | speed + diversity |
| balanced | 4 | 4 | 16 | initial BTFlash target |
| quality-heavy | 6 | 4 | 24 | only if selection benefits |

These are experiment points, not defaults. The scheduler should prefer useful
rows over a fixed utilization target.

## 3. Why the branch interval can be latency-cheap

For a normal response of T target steps and a BTFlash interval of H steps:

```
T_baseline = T · t_forward(1)

T_btflash  = H · t_forward(K)
           + (T - H) · t_forward(1)
           + t_evaluate + t_compact
```

The extra latency is `H·(t_forward(K)-t_forward(1)) + evaluator + compaction`,
not the time for K full responses. Qwen3.6-27B streams roughly 15.3 GB of
target weights per forward, so several rows can amortize part of that traffic
and raise arithmetic intensity. The existing RDNA MMVQ/MMQ and flash-attention
paths make this plausible, but neither “weights paid once” nor “prefix KV read
once” is guaranteed merely by issuing one graph. BT0 measures the actual
kernel paths and traffic.

This improves end-to-end latency only indirectly. The intended result is more
quality at similar latency; DDTree remains the mechanism that converts extra
rows directly into tok/s. Combining the two lets the same target forward buy
both accepted depth and semantic width.

## 4. Branch lifecycle

### 4.1 Fork only where alternatives matter

The first version supports a fixed fork point after a short normal prefix.
The next version forks when target entropy, drafter confidence, or another
uncertainty signal crosses a threshold. Forking at the first response token is
not the default: early tokens are often formatting or boilerplate and forced
top-K first tokens can create artificial rather than semantic diversity.

At a fork, create K independent RNG substreams from the request seed and
branch id. Sample normally with a conservative diversity policy. Optional
temperature variation is small and configured; duplicate short prefixes are
collapsed or resampled. Determinism specifies branch-major draw order.

### 4.2 Short lockstep rollout

Advance K branches for H ∈ {8,16,32} target tokens. Each row attends to:

- the shared prompt and committed continuation;
- ancestors belonging to its semantic branch;
- no sibling branch suffix.

Physical cache rows append in branch-tree order while model positions follow
semantic depth. The attention mask supplies lineage visibility. All target KV
and recurrent state remain on the R9700.

If DFlash is enabled, each semantic branch receives a small DDTree micro-tree
and all micro-trees are verified in one target graph. The row-budget scheduler
can move rows between width and depth at each fork.

### 4.3 Evaluate, prune, continue

At horizon H, rank the partial trajectories and retain one or, experimentally,
two survivors. Restore/compact the winning KV, DeltaNet state, conv tail,
sampler history, and generated tokens onto the committed spine. Continue the
winner normally; a later uncertainty point may fork again.

A staged tournament is also possible:

```
K=4 for 16 tokens → keep 2 → 16 more tokens → keep 1 → finish
```

This gives the evaluator more evidence without waiting for four complete
answers. Finished, invalid, or duplicate branches must be removed from the
active width; padding a finished row does not make its matrix work free.

## 5. Qwen3.6/qwen35 state design

The full-attention part already has most primitives BTFlash needs:

- arbitrary F16 attention masks into `flash_attn_ext`;
- sibling nodes at the same absolute model position;
- contiguous tree writes followed by winner KV compaction;
- `verify_tree` and generic lineage visibility.

The recurrent half is the principal engine change. Qwen3.6 has 48 DeltaNet
layers. Existing DDTree kernels can reload a node state from an intermediate
inside one graph, but the root sentinel reloads one sequence’s `curr_state`.
It cannot yet address K persistent branch roots across forward calls.

BTFlash therefore needs:

- K persistent recurrent-state slots for every DeltaNet layer;
- K conv tails;
- an explicit root-state/branch-slot tensor in the graph and kernel ABI;
- per-frontier capture into the bank after every branch step;
- winner restore and loser-slot reuse after pruning.

The estimated state is about 151 MB f32, or roughly 40 MB Q8_0, per branch.
At K=4 this is about 0.6 GB f32 or 0.16 GB Q8_0; at K=8 it is about 1.2 GB or
0.3 GB. Exact allocation, conversion overhead, and numerical impact of a
quantized bank must be measured. Calling this “DDTree with time extended” is
conceptually useful, but the persistent state bank is a substantive change.

## 6. Two-GPU execution plan

The two GPUs are separate compute and memory pools. BTFlash should exploit
both without pretending they form one coherent target cache.

### R9700: latency-critical target owner

- Qwen3.6-27B target weights and LM head.
- Shared target KV and branch suffix KV.
- DeltaNet branch-state bank.
- Multi-row BTFlash/DDTree verification graphs.
- Winner compaction and final token streaming.

### Strix Halo: asynchronous support plane

Candidate responsibilities, enabled and measured independently:

1. **DFlash drafter.** Run the matched small drafter via the existing remote
   drafter/device boundary. Transfer compact target features/tokens rather
   than target KV. Measure whether PCIe/control latency beats colocating the
   drafter on the R9700.
2. **BTFlash evaluator/value model.** Score partial branch token streams and
   compact features while the R9700 finishes the rollout. A small verifier on
   the iGPU avoids spending an additional target-model pass on the R9700.

The support plane is failure-optional: if the Strix evaluator is late or
unavailable, use an engine-local selector or skip the fork and continue K=1.
The R9700 decode loop never blocks indefinitely on the second GPU. Transfers
are bounded, versioned messages over the existing remote-drafter style IPC;
raw target KV and DeltaNet banks never cross devices.

The scheduler must measure shared constraints that the ideal split omits:
PCIe latency/bandwidth, the whole-box power/thermal limit, and contention
between drafter and evaluator work on the Strix Halo.

### Alternative two-GPU target topologies

Moving only the target KV cache to the Strix Halo is not the default plan. If
attention still executes on the R9700, every attention layer must fetch
historical KV over PCIe. At 2K context the aggregate Qwen3.6 KV read is only
about `2048 × 34 KB ≈ 68 MB` per target forward, far below the roughly
15.3 GB target weight stream. The saved R9700 traffic is therefore small while
the design introduces 16 attention-layer transfers and synchronization
points. Remote KV becomes more relevant at 32K–128K context, but BTFlash tree
rows and KVFlash paging are not compatible yet.

A stronger variant keeps both KV and attention compute on the Strix. The
R9700 produces the current Q/K/V, sends the compact current activations, and
receives the attention output. This avoids transferring historical KV, but it
is only useful if the R9700 overlaps the wait with work for another branch;
otherwise local R9700 attention is replaced by a slower remote dependency.

The most promising target-inference experiment is a two-stage pipeline with
stage-local cache:

```
Strix Halo: contiguous early target layers + their KV/DeltaNet state
R9700:      remaining target layers + their KV/DeltaNet state + LM head

time →
Strix:  branch group A ─ branch group B ─ branch group C
R9700:                 └ branch group A ─ branch group B ─ branch group C
```

Only hidden activations cross one contiguous stage boundary. BTFlash branches
provide the otherwise-missing microbatch dimension that can fill both stages
for a single request. The first split candidate places roughly 20–30% of
measured target work on the lower-bandwidth Strix and 70–80% on the R9700;
per-layer profiling, not layer count, chooses the actual boundary.

Pipeline overlap competes with the row reuse BTFlash is trying to create. A
single `W × [K rows]` operation can amortize weights better than K independent
matrix-vector operations. Test grouped microbatches as the compromise: for
example K=8 as two groups of four, preserving four-row matmuls while the two
groups overlap across stages. Pure DDTree nodes may also have within-tree
dependencies that constrain microbatch scheduling; start with independent
semantic chains before combining pipeline parallelism with DDTree depth.

## 7. Evaluation and selection

Generating alternatives raises coverage; only a good evaluator converts that
coverage into a better emitted answer. Report these separately:

| Metric | Question answered |
|---|---|
| pass@K / oracle accuracy | Did any branch contain the right direction? |
| selected accuracy | Did the deployable evaluator choose it? |
| selection regret | How often was a correct branch generated but missed? |
| diversity | Were compute rows spent on meaningfully different paths? |

Initial evaluators:

- **Partial reasoning branches:** require a verifier/value estimate; majority
  vote is not defined until branches expose comparable candidate answers.
- **Math:** extracted-answer consensus/majority only when a short branch
  reaches a terminal answer within its horizon; otherwise use the verifier.
- **Code evaluation:** gold tests establish the oracle ceiling; syntax,
  compile checks, public tests, or a deployable verifier must be reported
  separately from hidden gold tests.
- **General responses:** normalized logprob is a cheap baseline, not a quality
  guarantee. Compare an iGPU value model or batched judge before using it as
  the product selector.

Safe early-pruning signals include duplicate continuations, invalid format,
syntax failure, and obvious repetition. Logprob gaps alone should prune only
conservatively because likelihood is not correctness.

## 8. Streaming and API

BTFlash returns one choice. Losing branches are internal and do not require an
`n>1` public API. Visible output before a fork can stream normally. Tokens in
an unresolved branch interval must be buffered; after pruning, the winning
buffer is emitted and streaming continues from its cache state.

For hidden-thinking models the branch interval naturally remains internal.
For visible reasoning, the buffer creates a local streaming pause even when
total generation latency is nearly flat, so measure both TTFT and maximum
inter-token gap.

Proposed experimental surface:

```jsonc
{
  "btflash": {
    "row_budget": 16,
    "k": 4,
    "horizon": 16,
    "survivors": 1,
    "fork_tokens": 8,
    "fork": "fixed",          // BT1 supports fixed only
    "select": "logprob"       // BT1 supports logprob only
  }
}
```

All fields remain off by default until the measurements establish useful
defaults. The BT1 implementation accepts K in {2,4}, H in {8,16}, and one
survivor. Unsupported target topologies and request combinations fall back to
ordinary autoregressive decoding; DFlash composition is deferred to BT6.

## 9. Milestones

- **BT0 — R9700 row curve, no persistent branches.** Use existing one-forward
  tree machinery at 2K context for N ∈ {1,2,4,8,16,24,32,64}. Record target
  step latency, achieved bandwidth/FLOP/s, SQ/matrix activity, MMVQ/MMQ and FA
  dispatch, logits/sampling time, and VRAM. This replaces assumptions about a
  1.2× K=8 forward with data.
- **BT1 — qwen35 persistent short branches.** Implement the DeltaNet bank,
  K ∈ {2,4}, H ∈ {8,16}, early prune to one, compaction, deterministic sampler
  state, and one final response. No DFlash composition yet.
- **BT2 — selector A/B.** Measure oracle, vote, normalized-logprob, and a
  deployable verifier on Math500/GSM8K and HumanEval. Report selection regret.
- **BT3 — two-GPU experiment A: support plane.** Keep the complete target and
  cache on the R9700; put the drafter and/or evaluator on Strix Halo. Overlap
  safe work, enforce deadlines/fallback, and establish the simplest two-GPU
  baseline including PCIe, power, and thermal effects.
- **BT4 — two-GPU experiment B: remote KV/attention.** First measure remote-KV
  storage as a falsifiable baseline, expected to lose at short context. Then
  move attention compute with its KV to Strix and transfer only current
  Q/K/V/output activations. Test context ∈ {2K,32K,128K}; proceed only if the
  R9700 can overlap useful branch work with the remote attention interval.
- **BT5 — two-GPU experiment C: pipelined target.** Split Qwen3.6 at one
  contiguous layer boundary with stage-local KV and DeltaNet state. Sweep
  K ∈ {1,2,4,8}, microbatch size ∈ {1,2,4}, and profiling-derived layer
  boundaries. Compare lost multi-row weight reuse against pipeline bubbles
  and simultaneous utilization. Start with independent BTFlash chains.
- **BT6 — DDTree composition.** Allocate a fixed N_total ∈ {16,24,32} between
  semantic width and speculative depth. Measure speed/quality/latency Pareto
  curves, first on the winning single-target topology and then on the pipeline
  only if its dependency schedule remains beneficial.
- **BT7 — dynamic tree.** Entropy-triggered fork, duplicate collapse, staged
  pruning, periodic re-fork, and prebuilt active-width graph variants.
- **BT8 — scheduler unification.** Coordinate DDTree confidence, BTFlash
  uncertainty, iGPU evaluator load, and whole-box power under one useful-row
  budget.

### Two-GPU experiment matrix

Run the topologies in this order:

| Order | Target/cache placement | Strix role | Main question |
|---:|---|---|---|
| 1 | Entire target + cache on R9700 | Drafter/evaluator | Does independent support work give a low-risk win? |
| 2 | Target compute on R9700, KV on Strix | Remote storage | Is remote KV ever competitive, and at what context? |
| 3 | Projections on R9700, attention + KV on Strix | Attention service | Can branch work hide remote-attention latency? |
| 4 | Contiguous target stages, cache local to each stage | Target pipeline stage | Can BTFlash microbatches keep both GPUs productively busy? |

For every topology record end-to-end branch-step latency, TTFT and streaming
pause, pipeline bubble fraction, PCIe bytes and wait time, achieved bandwidth
and compute activity on both GPUs, MMVQ/MMQ dispatch, peak memory, selected
quality, and quality per wall-clock. A topology advances only if it improves
latency, selected quality within the same latency budget, or usable R9700
headroom over all earlier topologies.

## 10. Go/no-go criteria

The project does not require full compute saturation. A modest, repeatable
conversion of idle capacity into quality is sufficient.

Initial gates, all subject to measurement:

- K=4 branch-forward latency no more than 1.3× K=1 at 2K context.
- End-to-end selected accuracy improves by at least 2–3 points on one suite at
  no more than 1.4× wall-clock.
- The deployable selector captures a material fraction of the oracle/pass@K
  gain; otherwise improve evaluation before increasing K.
- Peak R9700 VRAM fits Qwen3.6-27B target, draft integration, KV, and branch
  state with operational headroom.
- Strix Halo offload improves either latency, selected quality, or R9700
  headroom after PCIe and power effects; otherwise that component stays local
  or disabled.
- Profiling shows higher useful row throughput and compute activity, not just
  more padded or duplicate work.

## 11. Risks and inherited constraints

- **Weak partial-trajectory evaluation:** the central product risk. A short
  branch may look good but lead to a wrong final answer.
- **Artificial diversity:** forced distinct first tokens can waste branches;
  prefer normal sampling, entropy-triggered forks, and duplicate detection.
- **State-bank complexity:** persistent DeltaNet roots require kernel and graph
  changes beyond current DDTree.
- **Length and active-width skew:** padding finished branches still consumes
  work; pruning needs graph-width management.
- **Cache growth:** interleaved suffix rows consume K physical positions per
  semantic step and masked sibling rows remain in the attention span.
- **`fa_window == 0`:** required initially because the sampled-logit tail is
  currently unsafe in multi-row windowed batches.
- **KVFlash and target split:** tree rows remain incompatible with active
  KVFlash paging, and the existing layer-split path verifies pure chains only.
- **Second-GPU deadlines:** an evaluator that blocks the R9700 destroys the
  latency argument; every remote operation needs a timeout and K=1 fallback.
- **Streaming perception:** total wall-clock can remain close while buffered
  reasoning increases TTFT or creates a visible pause.
- **Task dependence:** literature gains do not transfer automatically to this
  model, selector, or workload. Failed hypotheses remain documented.
