# SpecLA on the qwen35 delta-net target

Implementation of *SpecLA: Efficient Speculative Decoding for Linear-Attention
Models* (arXiv:2607.16673) on the dflash spec-decode stack. The paper targets
stateful linear-attention (Gated DeltaNet) models, where speculative
verification cannot rely on KV-cache suffix truncation: every candidate token
mutates a dense recurrent state, so acceptance needs explicit state recovery.

## Where main stands today (the paper's baselines)

The pre-SpecLA runtime implements exactly the two baselines the paper argues
against, plus the snapshot/replay trade-off of its §5.1:

| Paper concept | main | file |
|---|---|---|
| Decode-kernel replay (naive 1) | fused sequential `ggml_gated_delta_net` for capture mode and tree verify | `graph_builders` |
| Prefill-kernel reuse (naive 2) | `build_delta_net_chunked` for chain verify, but **only with capture off** | `delta_net_chunked.cpp` |
| Full-state snapshotting | `capture_delta_intermediate` stores a dense per-token SSM state (F16/Q8 quantized), `rollback_to` copies one back | `qwen35_dflash_target.cpp` |
| Token replay | `restore_kv` + `verify_batch(replay_tok)` legacy path | `dflash_spec_decode.cpp` |

Consequences on main:
- Fast verify (chunked) and fast rollback (capture) are mutually exclusive;
  capture forces the sequential kernel.
- Dense per-token checkpoints are big (state = `S_v×S_v×H_v` FP32 per layer per
  token), so they are stored quantized, which makes fast rollback inexact —
  hence the `fast_rollback_threshold` / F32-checkpoint opt-in machinery in
  `chain_rollback_policy.h`.
- Every step pays a full SSM snapshot (`snapshot_kv`) before verify.

## What this branch implements

### 1. Factor capture (paper §5.1)

The chunked UT-transform verify already computes, internally, everything needed
to advance the state along any accepted prefix. With per-token projected
factors `k_t, v_t, β_t` and per-head log-decay `g_t`, the chunked algebra is:

```
g⁺_t  = Σ_{i≤t} g_i                       (cumulative gate, "g_cs")
A     = strict_lower( (k kᵀβ) ⊙ exp(g⁺_t − g⁺_i) )
T     = (I + A)⁻¹                          (unit lower-triangular, solve_tri)
ṽ     = T (β v − exp(g⁺) β k S₀)           (corrected values, "v_new")
o_t   = exp(g⁺_t) S₀ᵀ q_t + Σ_{i≤t} [strict_lower((k qᵀ)⊙decay)]_{t,i} ṽ_i
S_n   = exp(g⁺_n) S₀ + Σ_t exp(g⁺_n − g⁺_t) k_t ⊗ ṽ_t
```

Because `T` is lower-triangular, `ṽ_t` depends only on tokens `≤ t` and `S₀`.
So the state after accepting `A ≤ n` tokens is recoverable from compact
factors alone:

```
S_A = exp(g⁺_A) S₀ + Σ_{t≤A} exp(g⁺_A − g⁺_t) k_t ⊗ ṽ_t      (DeltaConstruct)
```

We capture `{k_t, ṽ_t, g⁺_t}` per layer per token (a few KiB, F32) instead of
dense per-token states (state-size × n, quantized). The existing `conv_input`
capture is kept: it already is a "factor" for the depthwise-conv state.

### 2. Verification never mutates durable state

With factor commit in place, the verify graph stops writing the speculative
end-of-window SSM state back into the cache. `S₀` stays untouched in
`cache.ssm_state` until commit, which removes both the per-step full-state
snapshot (`snapshot_kv`) and the restore on rejection for the delta-net
layers. This is the paper's "separate speculative progress from durable
state".

### 3. Fused accepted-state commit (paper §5.2, adapted)

The factor buffers are consolidated across layers (token axis outermost), and
the commit runs as ONE kernel launch (`specla_commit_cuda.cu`) that advances
every delta layer's state in place along the accepted path. This is the
paper's "fused update" adapted to this stack: ggml executes graphs as
individual kernel launches (HIP graphs are off on gfx1151), so expressing the
per-layer DeltaConstruct as graph ops costs hundreds of launches (~5 ms);
folding those same ops into the *next* verify graph (the paper's
delayed-update formulation) would not reduce the launch count. The single
fused kernel achieves the intended effect — one pass over the recurrent
state at the commit boundary, no snapshot, no replay, no dense checkpoint
traffic. The ggml-graph DeltaConstruct remains as the fallback path
(`DFLASH_SPECLA_FUSED_COMMIT=0`).

The kernel accumulates the accepted path sequentially, matching the fused
recurrence's summation order — which is what makes committed states (and
hence outputs) bit-stable against the sequential baseline.

### 4. Tree-masked parallel verification (paper §4.2)

Tree drafts are verified with the same UT-transform factorization by replacing
sequence-order masks with topology masks over DFS-ordered nodes:

```
g⁺_v  = Σ_{u ∈ ancestors(v) ∪ v} g_u        (path-cumulative gate = M_incl · g)
A     = M_strict ⊙ (k kᵀβ) ⊙ exp(g⁺_v − g⁺_u)
T     = (I + A)⁻¹                            (topo order ⇒ still lower-triangular)
```

Outputs and factors follow as in the chain case with `M`-masked attention.
No per-branch state fanout exists anywhere: the tree pass reads only `S₀`, and
the accepted root-to-leaf path commits through DeltaConstruct exactly like a
chain. This replaces the sequential per-node kernel for tree verify and the
dense per-node snapshots of `rollback_to_tree`.

### 5. Confidence-guided draft pruning (paper §6.1)

Draft-tree nodes are pruned by cumulative path log-probability with a global
margin: keep `q(v) ≥ q★ − τ_tree` (ancestor-closed by monotonicity), then
apply the node budget. Rationale: stateful verification work is the expensive
resource; low-confidence paths waste it.

### Drafting alignment (paper §6.2)

The dflash drafter already conditions on target-side features
(`target_feat` ring), i.e. it is target-aligned in the paper's sense; no
drafter change is needed.

## How to run

```
# chain verify (16-token windows)
DFLASH_SPECLA=1 build/test_dflash <target.gguf> <draft.gguf> prompt.bin 256 out.bin --fast-rollback

# tree verify (+ optional SpecLA confidence margin)
DFLASH_SPECLA=1 build/test_dflash ... --fast-rollback --ddtree --ddtree-budget=22 --ddtree-tau=6
```

- `DFLASH_SPECLA=1` — master switch; requires `--fast-rollback` (the factor
  buffers replace the dense checkpoint allocation in `migrate_prefill_cache`).
- `DFLASH_SPECLA_FUSED_COMMIT=0` — fall back from the fused commit kernel to
  the ggml-graph DeltaConstruct.
- `--ddtree-tau <T>` — §6.1 margin, off by default.
- Same switches apply to `dflash_server`; layer-split/IPC/tensor-parallel
  targets ignore SpecLA (their caches keep dense checkpoints).

## Correctness validation

- `test_delta_net_specla` (GPU, ctest `test_delta_net_specla`): the
  topology-masked builder vs the fused sequential kernel on chains, stars and
  random trees up to the exact qwen35-27B shape (S=128, H_v=48) — outputs
  match to ~1e-8; DeltaConstruct reconstruction matches the kernel's
  per-token states to ~3e-7 at every accepted endpoint.
- End-to-end on gfx1151 (Qwen3.6-27B Q4_K_M + dflash draft): tree mode and
  chain mode produce token streams bit-identical to the pre-SpecLA baseline
  (the fused commit kernel accumulates the accepted path sequentially, so
  even near-tie argmaxes resolve identically; the ggml-graph commit's batched
  reduction order can flip roughly one near-tie per hundred tokens — the same
  numerical class as the known batched-vs-stepwise verify divergence).
- `test_ddtree_tau` (CPU, in `test_server_unit`): §6.1 window semantics.

## Measured (gfx1151 Strix Halo, Qwen3.6-27B Q4_K_M + dflash draft)

All numbers from interleaved repeated runs on an idle machine
(`test_dflash`, 256 new tokens; per-step milliseconds from the harness
phase profile, which is invariant to text divergence).

**Per-step cost, chain (16-token window):**

| phase | baseline | SpecLA |
|---|---|---|
| verify_compute | 154.7 | 158.0 |
| commit (`restore_ssm`) | 1.81 | 3.80 |
| step total | 178.4 | 184.1 (+3.2%) |

**Per-step cost, tree (budget 22, N=23):** verify 194.0 → 196.6 ms,
step total 225.4 → 227.8 ms (+1.0%). Token streams are bit-identical to
baseline until a near-tie argmax flips (~1 per 100-200 tokens, the same
numerical class as the repo's known batched-vs-stepwise divergence).

The UT-factorized verify is a wash against the fused sequential kernel at
these window sizes on this machine — consistent with the paper's own
finding that the parallel path does not beat the serial kernel on short
windows (their Table 1), and with this forward being MMQ-dominated. What
the factor scheme buys on this stack:

- **Speculative-state memory: ~32×.** Dense per-token checkpoints at
  budget 22 cost 1.73 GB (F16; 3.5 GB with the F32-exact opt-in) vs
  ~55 MB of F32 factors. On a 24 GB card this is the difference between
  fitting a large tree budget or not.
- **Exact commit at any accepted length.** The F16 checkpoint restore is
  inexact, which is why `chain_rollback_policy` only fast-rollbacks at
  `accept_n >= 5` in the server loops and otherwise pays restore + a full
  replay forward (~155 ms). In the measured chain runs 42-63% of steps
  accept fewer than 5 tokens. SpecLA's F32 factor commit removes the
  threshold, the per-step snapshot, and every replay forward.
- **No state mutation during verify** — the pre-verify snapshot
  (`snapshot_kv`, a full 144 MB state copy per step in the server loops)
  is a no-op under SpecLA.

`--ddtree-tau` note: the verify graph keeps its fixed N = budget+1
allocation for gallocr reuse, so at a fixed budget the margin only changes
which candidates are submitted (pruned slots become padding). Its intended
use is enabling smaller budgets without the AL loss of naive truncation.

## Deferred

- **Chain-decomposed hybrid verification (paper §4.3)** — heavy-light
  decomposition of the tree into chains executed by the serial kernel with
  cross-chain parallelism. In ggml terms this maps to per-wave batched
  `n_seqs` invocations of the fused sequential op with boundary-state
  gathers between waves. Deferred until the parallel path's profile on
  gfx1151/3090 shows where it wins at real draft shapes.
- **State-resident serial kernel V-tiling (paper §4.1)** — the fused
  sequential ggml op already executes layer-major with state held on-chip
  across the token loop; revisiting its tiling is vendored-ggml work.
