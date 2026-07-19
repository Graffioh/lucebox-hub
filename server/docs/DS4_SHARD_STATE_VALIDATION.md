# DeepSeek4 shard-state validation

This validation-only branch adds opt-in state tracing and a focused unit-test
mode for the HC shard handoff, layer-range runtime cache, and device-scoped HC
scratch changes. The PR branch remains unchanged.

## Focused unit tests

Build `test_deepseek4_unit` in both backend build directories, then run:

```bash
DFLASH_DS4_VALIDATION_TESTS_ONLY=1 \
DFLASH_DS4_VALIDATE_STATE=1 \
HIP_VISIBLE_DEVICES=0,1 \
./server/build-hip/test_deepseek4_unit 2>&1 | tee /tmp/ds4-unit-hip.log
```

```bash
DFLASH_DS4_VALIDATION_TESTS_ONLY=1 \
DFLASH_DS4_VALIDATE_STATE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
./server/build-cuda/test_deepseek4_unit 2>&1 | tee /tmp/ds4-unit-cuda.log
```

The focused mode checks:

- 43-layer range partitioning;
- HC dimensions (`4 * 4096` floats per token);
- embedding input for shard zero and HC input for later shards;
- exact HC replication, copy, alias preservation, and size-mismatch rejection;
- deterministic state digest behavior;
- distinct layer-range caches and complete ownership matching;
- device sequence `0 -> 1 -> 0` for HC scratch, when two devices are visible.

The scratch test reports `skipped` when the backend sees fewer than two GPUs.

## Runtime state trace

Add this environment variable to either the local HIP split or mixed CUDA/HIP
server command:

```bash
DFLASH_DS4_VALIDATE_STATE=1
```

For a local split, the `output-hc` record from one shard must exactly match the
next shard's `input-hc` record by `elements` and `hash`. The consumer must also
report `alias_hc=1`.

For a mixed split, the CUDA parent's `send-hc` record must exactly match the HIP
daemon's `receive-hc` record. Every state must report `nonfinite=0`.

Cache records must show one stable `runtime` per `cache_owner`. The first
forward reports `create`; later decode forwards report `reuse`. Any
`reinitialize` event is a validation failure.

Scratch records must show one stable slot per device. When two devices are
tested, their slots must differ and returning to device zero must reuse its
original slot.

Validate a captured server log automatically with:

```bash
python3 server/scripts/validate_ds4_state_trace.py /tmp/ds4-server.log
```

If parent and daemon output were captured separately, pass both files in
forward-order:

```bash
python3 server/scripts/validate_ds4_state_trace.py \
  /tmp/ds4-cuda-parent.log /tmp/ds4-hip-daemon.log
```
