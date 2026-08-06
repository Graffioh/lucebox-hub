# QFlash — one-shot quality branching

_Status: CPU-tested MVP, pending R9700 + Strix Halo measurements._

QFlash spends one wider target forward on several short continuations of a
single user request. It is the small, testable replacement for BTFlash.

## The idea in one minute

Normal autoregressive generation processes one new row at a time. That keeps
loading a large target model to do relatively little matrix work. QFlash asks
the DFlash drafter for four short candidate continuations, verifies all of
them together on the R9700, and keeps the one that the target scores clearly
better than the normal draft continuation.

```text
                               candidate 0 (baseline, 7 tokens) ─┐
prompt + committed response ─── candidate 1 (7 tokens) ──────────┼─ one target tree verify
          shared prefix KV      candidate 2 (7 tokens) ──────────┤  → score → commit one
                               candidate 3 (7 tokens) ───────────┘

Strix Halo: DFlash proposal                 R9700: target KV + verify + commit
```

The default shape is `4 × 7`: 28 candidate nodes plus one shared root. The
target allocates 32 rows, matching the existing 32-wide verify tile. QFlash
runs once per request; normal DFlash then continues the selected path.

This is useful test-time compute, not four complete answers and not four
independent target forwards.

## What “shared KV” means here

All candidates share the same committed prefix, so the branch tree has one
logical prefix KV cache. The physical target KV stays on the R9700, next to
the target weights and attention compute. Only temporary suffix rows differ
between branches.

QFlash does **not** put the target KV on the Strix and read it over PCIe. That
would add a remote dependency to every target attention layer. It also does
not make the Strix and R9700 memories coherent. The Strix runs the small
drafter and returns candidate hidden states; the existing feature-mirror path
synchronizes the compact target features it needs after the winner is
committed.

This separation is intentional:

- R9700 owns target weights, target KV, DeltaNet state, target scoring and
  rollback.
- Strix Halo owns the DFlash drafter and its local state.
- Tokens/features cross the device boundary; the large target KV does not.

If both GPUs are visible to the same ROCm build, use separate HIP device
indices. The existing IPC drafter boundary remains available for a genuinely
mixed backend.

### Why canonical KV on Strix is MVP 2

A read-only canonical prefix on the Strix is a reasonable capacity experiment,
but it combines a second hypothesis with quality branching. The target cannot
consume that cache today without either:

1. staging the required prefix Strix → R9700 once before the branch round; or
2. a new attention path that combines a remote prefix with local branch
   suffixes.

Neither makes the 32-row target graph more compute-dense. They add transfer
and synchronization costs, so a failed combined test would not tell us whether
branching or remote KV caused the regression.

The staged roadmap is therefore:

1. QFlash with target KV local: validate branch quality and wide-verify cost.
2. A standalone read-only KV transfer benchmark over realistic context sizes.
3. If both pass, stage one prefix copy per QFlash round and measure end to end.
4. Only then consider direct remote-prefix attention with local suffixes.

The canonical Strix cache is thus a follow-up to QFlash, not discarded work.

## Algorithm

QFlash reuses the production DDTree primitives instead of adding a persistent
branch runtime:

1. Run the existing DFlash drafter once.
2. Project top-K draft tokens with `K = qflash_branches`.
3. Build K disjoint chains. Branch 0 begins with draft rank 0 and is the
   baseline; branch `b` begins with draft rank `b`. All short tails use the
   already available spine-conditioned top-1 draft rows.
4. Run one ancestor-masked `verify_tree` on the target.
5. Score every complete path by mean target log-probability.
6. Keep branch 0 unless an alternative beats it by `qflash_margin` nats per
   token.
7. Reuse `rollback_to_tree` to compact target KV, DeltaNet state, convolution
   state and captured features onto the winning path.
8. Continue with normal DFlash. QFlash does not fork again in this request.

The margin makes selection conservative. A branch that is merely different
does not replace the baseline.

## Why this is simpler than BTFlash

BTFlash proposed branches that lived across several target forwards. Qwen3.6
would then need a persistent per-branch bank for KV, 48 DeltaNet states,
convolution tails and sampler histories. It would also need scheduling,
multi-stage pruning and failure recovery for those long-lived states.

QFlash keeps every branch inside one existing tree graph. Therefore it needs:

- no persistent branch-state bank;
- no cross-forward branch lifecycle;
- no target KV replication between GPUs;
- no new attention or remote-KV kernel;
- no evaluator model for the first experiment.

It validates the important hypothesis first: whether a 32-row target step can
buy a better selected continuation at acceptable latency.

## Usage

Example when the R9700 is HIP device 0 and Strix Halo is HIP device 1:

```bash
./server/build/dflash_server model.gguf \
  --draft dflash-draft.gguf \
  --target-device hip:0 \
  --draft-device hip:1 \
  --qflash \
  --qflash-branches 4 \
  --qflash-horizon 7 \
  --qflash-margin 0.10
```

Device numbering must be checked on the actual machine. A build containing
both `gfx1201` and `gfx1151` code objects is required when both GPUs execute
inside the same ROCm process.

Flags:

| Flag | Default | Meaning |
|---|---:|---|
| `--qflash` | off | Enable one QFlash round per request |
| `--qflash-branches` | 4 | Candidate chains, from 2 to 8 |
| `--qflash-horizon` | 7 | Candidate tokens per chain |
| `--qflash-margin` | 0.10 | Required mean target log-probability gain |

The MVP is admitted only for a single-device dense `qwen35` target with a
DFlash drafter and fast rollback. The drafter may run on the second GPU.
Sampling requests and special decode regions (forced thinking close, tool
hints, stall recovery and the minimum-token floor) keep their existing path;
QFlash waits for the first safe greedy round.

## What this prototype can and cannot prove

It can measure:

- target step time for 29 actual / 32 allocated rows;
- whether the R9700 executes a more useful MMQ shape;
- total request latency with the drafter on Strix Halo;
- how often a non-baseline branch wins;
- whether selected outputs improve on task-level evaluation.

It cannot yet prove that the R9700 is fully compute-bound. Thirty-two rows
raise arithmetic intensity but remain below the ideal roofline knee. A busy
matrix kernel is also not the same as peak FLOP utilization.

It also cannot guarantee higher quality from log-probability alone. Target
likelihood is a cheap baseline selector, not a correctness verifier. Code
tests, math checkers or a small evaluator on the Strix are follow-ups only if
the one-shot mechanism passes the latency test.

## GPU validation plan

When the machine is ready, compare the same greedy prompts and seeds:

1. normal DFlash chain;
2. DDTree budget 22;
3. QFlash `4 × 7`, margin `0.10`;
4. QFlash with drafter colocated on R9700 versus drafter on Strix Halo.

Record end-to-end latency, time to first token, target verify time, selected
branch, score delta, generated tokens and task result. Then sweep only:

```text
shape:   4×7 (32 allocated rows) → 2×15 (32) → 4×8 (64)
margin:  0.00 → 0.10 → 0.25
```

Go/no-go for a larger design: keep QFlash only if it improves task results at
a latency cost small enough for the product. If branch 0 almost always wins,
improve proposal diversity before increasing the target row budget. If 32
rows are already expensive, optimize the target kernel before adding more
branch machinery.

## Interview summary

> We first considered a KV cache physically shared between two GPUs, but the
> GPUs have separate memory domains and remote target KV would put PCIe in the
> attention critical path. QFlash uses the safer meaning of shared KV: one
> prefix on the R9700, logically shared by several short branches. The Strix
> proposes candidates with the small drafter; the R9700 verifies 28 candidate
> nodes in one 32-row tree forward, conservatively selects one, compacts the
> existing KV/state with the existing rollback code, and continues normally.
> It is one optional round, so it tests compute utilization and quality without
> building a persistent multi-branch inference engine.
