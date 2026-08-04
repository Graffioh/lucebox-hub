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

### 3. Delayed fused update-verify (paper §5.2)

After posterior selection, the accepted factor rows are gathered (tiny D2D
copies) into a persistent pending buffer. The durable state is *not* advanced
by a standalone kernel; instead the next verify graph applies the pending
buffer first — `S_base = DeltaConstruct(S₀, pending)` — and verifies from
`S_base` in the same graph launch. A `flush_pending()` path runs the
standalone construct when the loop exits (EOS, budget) or when another
consumer needs the durable state.

Gathering into a persistent buffer (rather than referencing the capture
tensors of the previous step graph) is what makes the scheme safe under graph
rebuilds: step j+1's own captures may reuse step j's buffers.

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
