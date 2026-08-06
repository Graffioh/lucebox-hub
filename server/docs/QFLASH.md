# QFlash: one-shot quality branching

Status: the base MVP is implemented behind `--qflash`; independent branch
expansion and GPU quality/latency measurements are still pending.

QFlash spends one wider target-model forward on several short continuations
of one user request. The goal is to use more of the R9700's matrix compute and
select a better continuation without generating several complete answers.

## In breve

The DFlash drafter proposes four short paths. The R9700 verifies them together,
scores them with the target model and commits one path. Generation then
continues normally.

```text
                              branch 0: 7 tokens ─┐
shared committed prefix ───── branch 1: 7 tokens ├─ one target tree verify
                              branch 2: 7 tokens ┤  → score → commit one path
                              branch 3: 7 tokens ─┘

Strix Halo: draft proposal          R9700: target model, KV, score and commit
```

"One-shot" means one branching round per request: one fork, one selection and
no persistent branches. It does not mean that the whole response uses one
forward pass.

## Why one wide forward

Running four independent batch-1 target forwards would load the target weights
four times and remain mostly memory-bound. QFlash instead places all candidate
nodes in one tree-shaped batch so the same weight stream performs more useful
matrix work.

The default shape is:

```text
4 branches × 7 tokens = 28 candidate nodes
28 nodes + 1 shared root = 29 real rows
target allocation = 32 rows
```

Thirty-two rows match the current verify tile well. This raises arithmetic
intensity, but it does not by itself prove that the R9700 is fully
compute-bound; that requires GPU profiling.

## Where the KV cache lives

All branches logically share the committed prefix. The physical target KV
cache stays on the R9700, next to the target weights and attention compute.
Temporary branch suffixes are also created there and only the winning suffix
is retained.

The Strix Halo runs the smaller DFlash drafter. Tokens and compact target
features cross the device boundary; the target KV does not. QFlash therefore
does not require coherent GPU memory or remote-KV attention over PCIe.

This asymmetric split is intentional:

- the R9700 owns the latency-sensitive target path, KV and sampler;
- the Strix uses its otherwise idle compute to create candidate continuations;
- proposals cross the boundary in one bounded transfer;
- the expensive target work stays in one wide graph.

## Current MVP limitation

The current implementation runs one DFlash block, takes the top `K` tokens at
the first branch position and reuses that block's top-1 tail for every branch.
The first token is different, but the remaining tokens were produced from the
same spine and are not conditioned on the alternative first token.

This is enough to validate tree construction, wide target verification,
scoring and rollback. It is not yet a strong quality-search proposal because
the branches are only weakly diverse.

## Recommended next MVP: Seed-and-Expand

Keep the same `4 × 7` target tree, but make the Strix produce genuine tails:

1. Run the normal DFlash block and obtain four candidate first tokens.
2. Keep branch 0 and its top-1 tail as the baseline at no extra cost.
3. Re-run the drafter for the other three seeds, always against the same
   read-only committed prefix, and take six new tail tokens from each run.
4. Return all alternative hidden rows in one combined buffer.
5. Project those rows together on the R9700, build the same 32-row tree and
   perform one target verify.
6. Score, commit and roll back exactly as in the current MVP.

```text
Strix Halo
  shared draft prefix ─┬─ baseline seed → existing tail
                       ├─ seed 1 → newly conditioned tail
                       ├─ seed 2 → newly conditioned tail
                       └─ seed 3 → newly conditioned tail
                                      │ one batched transfer
                                      ▼
R9700: batched projection → one 32-row tree verify → score → commit
```

The simplest prototype may execute the three extra drafter passes sequentially
behind one `propose_branches` call and combine their output before transfer.
The two AMD GPUs can stay in the existing process; IPC is needed only for the
already-supported remote-drafter topology. This validates whether real
diversity improves task results. Only after a positive result should the three
expansions be fused into one batch-3 drafter graph.

No target KV is copied or replicated for this flow. The Strix reuses only its
local drafter prefix state; the target prefix and all temporary target suffixes
remain on the R9700.

## How selection works

1. Obtain `K` short draft paths. In the base MVP their tails share one spine;
   Seed-and-Expand will condition every alternative tail independently.
2. Build `K` disjoint chains. Branch 0 is the drafter's normal top-1
   continuation and remains the fallback.
3. Verify the complete tree with one ancestor-masked target forward.
4. Compute the mean target log-probability of every path.
5. Keep branch 0 unless another path beats it by `--qflash-margin`.
6. Use the existing tree rollback to compact KV, DeltaNet state, convolution
   state and captured features onto the selected path.
7. Continue with normal DFlash generation.

The margin avoids replacing the baseline with a branch that is only
marginally different. Target likelihood is a cheap first selector, not a
correctness verifier. QFlash deliberately changes the generation policy, so
higher quality must be demonstrated on task-level evaluations.

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

Check the actual device order before launching. A same-process ROCm build must
contain code objects for both GPUs.

| Flag | Default | Meaning |
|---|---:|---|
| `--qflash` | off | Enable one QFlash round per request |
| `--qflash-branches` | 4 | Number of candidate chains, from 2 to 8 |
| `--qflash-horizon` | 7 | Tokens in each candidate chain |
| `--qflash-margin` | 0.10 | Required target-score gain in nats per token |

The active configuration is also reported under `/props.speculative.qflash`.

## Current limits

- Dense `qwen35` target on one target GPU.
- A local or second-GPU DFlash drafter is required.
- Fast tree rollback is required.
- QFlash currently runs only for greedy generation.
- Thinking-budget closure, tool hints, stall recovery and the minimum-token
  floor keep their existing path; QFlash waits for the first safe round.
- Branches differ at their first token. Their remaining tokens come from the
  same spine-conditioned draft block, so diversity is intentionally limited in
  this first version.
- Seed-and-Expand, uncertainty-triggered branching and a proposal deadline are
  design steps, not implemented behavior.
- Target KV on the Strix Halo and direct remote attention are not implemented.

## What to measure on the GPUs

First compare the current shared-tail MVP against Seed-and-Expand with the same
`4 × 7` shape, prompts and seeds. Record:

- end-to-end latency and time to first token;
- target tree-verify time;
- selected branch and score difference from branch 0;
- alternative-branch win rate;
- task result or test pass rate;
- Strix proposal and cross-device transfer time;
- R9700 matrix utilization and memory bandwidth;
- drafter colocated on the R9700 versus running on the Strix Halo.

The first go/no-go point is whether independent tails improve task results or
produce useful alternative wins at an acceptable latency cost. If they do,
batch the three Strix expansions. After that, trigger the one-shot only when
the draft top-1/top-2 gap is small and add a deadline that falls back to branch
0 when alternatives arrive too late. These two guards keep average latency
close to normal DFlash without complicating the first experiment.

## Implementation map

- `server/src/common/qflash.{h,cpp}` builds and scores candidate trees.
- `server/src/qwen35/qwen35_backend.cpp` runs the one-shot round and commits
  the selected path.
- `server/test/test_qflash.cpp` covers tree shape, row allocation, margin and
  branch selection without requiring a GPU.
