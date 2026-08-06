# Uncertainty-gated packed DDTree drafting

Adaptive DDTree keeps the normal DFlash chain as the default route and spends
additional draft and target work only when the first draft continuation is
uncertain. It is opt-in and currently supported by the single-device dense
Qwen3.5/Qwen3.6 backend.

## Enable it

Pass both flags with the usual target and draft arguments:

```sh
./build/dflash_server \
  --model models/target.gguf \
  --draft models/draft/dflash-draft.gguf \
  --ddtree-adaptive \
  --ddtree-branch-margin 0.35
```

`--ddtree-adaptive` also enables DDTree and fast rollback. The margin has no
built-in default: calibrate it on a representative prompt set, then keep it
fixed while comparing throughput and acceptance. Branching is selected when
the first continuation's top-1 minus top-2 log-probability margin is strictly
less than the threshold. Equality stays on the chain route.

## Fixed shapes

Confident rounds use the existing 16-row DFlash chain: one root plus 15 draft
continuations.

Uncertain rounds retain that complete baseline chain and add three independent
five-node paths:

```text
1 root + 15 baseline nodes + (3 branches × 5 nodes) = 31 real rows
target allocation/verify tile                              = 32 rows
```

The three alternatives are expanded in one real packed drafter graph. Its
query dimension is `3 × 16 = 48` rows, with block-diagonal full-attention and
sliding-window masks. All branches share the committed target-feature prefix,
but no branch can attend to another branch's noise rows.

There is no quality-score path selector. The target verifies the 31-row tree
once and the existing exact DDTree follower accepts only tokens that match the
target path. Greedy generation remains greedy, and sampled generation keeps
using the target sampler chain.

## Remote drafter IPC

A remote drafter sends the complete 48-row expansion as one `propose_batch`
command and receives one combined hidden-state response. File, pipe, and
shared-memory payload transports implement the same command-level batching;
the adaptive path has no serial three-request fallback.

This makes a CUDA target plus HIP Strix Halo drafter useful for measuring the
cross-device boundary. In particular, an RTX 3090 target plus Strix Halo
drafter is a measurement setup only:

- a positive result is useful evidence that packed IPC is viable;
- a negative result is not conclusive because target speed and interconnect
  overhead differ from the intended deployment;
- final performance and the production threshold must be confirmed with an
  R9700 target plus Strix Halo drafter.

## Telemetry

Every adaptive round logs its route, measured margin, threshold, draft graph
count, packed width, target verify rows, and acceptance. The request summary
reports:

- confident and uncertain round counts and branch activation rate;
- average margin and configured threshold;
- draft/packed graph counts and average packed width;
- target verify rows, accepted tokens per round, and accepted tokens per target
  forward;
- baseline draft, packed expansion, projection, verify, and rollback time.

Use these counters to sweep the margin. A useful threshold raises acceptance
enough to pay for the second draft graph and wider 32-row target verification;
activation rate alone is not the objective.

## Constraints

Adaptive DDTree currently requires:

- a draft model and tree verify support;
- dense `qwen35`/`qwen36` target architecture;
- single-device target execution (no target layer split);
- fast rollback;
- a DFlash block size of at least 16.

The feature gate rejects unsupported combinations at startup instead of
silently changing decoding semantics.
