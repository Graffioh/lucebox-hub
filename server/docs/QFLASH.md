# QFlash: one-shot quality branching

Status: implemented behind `--qflash`; GPU quality and latency measurements
are still pending.

QFlash spends one wider target-model forward on several short continuations
of one user request. The goal is to use more of the R9700's matrix compute and
select a better continuation without generating several complete answers.

## In breve

The DFlash drafter proposes four continuations of seven tokens. The R9700
verifies the four paths together, scores them with the target model and commits
one path. Generation then continues normally.

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

## How selection works

1. Run one normal DFlash draft block.
2. Take the top `K` draft tokens at the first branch position.
3. Build `K` short, disjoint chains. Branch 0 is the drafter's normal top-1
   continuation.
4. Verify the complete tree with one ancestor-masked target forward.
5. Compute the mean target log-probability of every path.
6. Keep branch 0 unless another path beats it by `--qflash-margin`.
7. Use the existing tree rollback to compact KV, DeltaNet state, convolution
   state and captured features onto the selected path.
8. Continue with normal DFlash generation.

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
- Target KV on the Strix Halo and direct remote attention are not implemented.

## What to measure on the GPUs

Compare the same prompts and seeds with normal DFlash, DDTree and QFlash.
Record:

- end-to-end latency and time to first token;
- target tree-verify time;
- selected branch and score difference from branch 0;
- task result or test pass rate;
- R9700 matrix utilization and memory bandwidth;
- drafter colocated on the R9700 versus running on the Strix Halo.

The first go/no-go point is simple: keep the design only if branch selection
improves task results at an acceptable latency cost. If branch 0 almost always
wins, improve proposal diversity before increasing the target batch.

## Implementation map

- `server/src/common/qflash.{h,cpp}` builds and scores candidate trees.
- `server/src/qwen35/qwen35_backend.cpp` runs the one-shot round and commits
  the selected path.
- `server/test/test_qflash.cpp` covers tree shape, row allocation, margin and
  branch selection without requiring a GPU.
