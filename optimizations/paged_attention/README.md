<p align="left">
  <a href="../../README.md">← lucebox-hub</a>
</p>

<h1 align="center">Luce Paged Attention</h1>

<p align="center">
  <strong>Exact fixed-block K/V for long-context serving.</strong><br/>
  Every sequence owns a block table; decode reads reusable 16-token physical blocks straight through it.<br/>
  Ragged 8-request profile (<code>128K + 7x8K</code>) on one RTX 3090:
  <strong>82% less K/V storage and 1.35x faster attention</strong>.
</p>

---

## How it works

- **Blocks.** Full-attention K/V rows live in fixed 16-token physical blocks in
  one pool. Nothing is evicted and nothing is approximated — this is a layout
  and allocation mechanism, not a residency policy.
- **Block tables.** Each sequence owns a logical→physical map. Decode walks it,
  so a sequence never has to occupy one contiguous K/V range, and the pool only
  ever holds live, block-rounded tokens.
- **One decode op.** `GGML_OP_PAGED_ATTN` on CUDA and HIP: D=256 GQA, F16/Q4_0/Q8_0
  K/V. One warp covers all query heads of a K/V group per row loaded, and long
  contexts split into partitions merged by a stable split-softmax.
- **Resident metadata.** The block table and sequence length live in the
  persistent target cache next to the pool, so a decode step uploads 4 bytes per
  newly filled block instead of the whole table every token.
- **Prefill remains exact.** Prompt chunks attend a contiguous staging copy of
  their K/V while dual-writing rows into the sequence's physical blocks.

Not to be confused with [KVFlash](../kvflash/README.md): that one bounds how much
context stays resident and evicts cold chunks. Paging keeps every token.

## Usage

```bash
./server/build/dflash_server model.gguf --paged-attention --max-ctx 131072
```

Concurrent serving — N requests decoded together, one batched paged step
per token, per-request SSE streams:

```bash
./server/build/dflash_server model.gguf --paged-attention --max-ctx 4096 \
    --max-concurrency 8         # 8 decode slots
    # --kv-pool-tokens 16384    # optional: shared pool smaller than 8 full contexts
```

## Compatibility

| | |
|---|---|
| Architecture | `qwen35` — dense Qwen3.5 / Qwen3.6 |
| Placement | one local CUDA or HIP device |
| Attention | full only (`--fa-window 0`) |
| K/V types | F16, Q4_0, Q8_0 |
| Decode | autoregressive; up to 64 concurrent sequences with `--max-concurrency` |
| Block size | 16 tokens, fixed |

**Rejected at startup** (exit 2, with the reason): `--draft`, `--ddtree`,
`--target-devices`, `--target-shard-ipc-bin`, a non-zero `--fa-window`,
`--prefill-compression`, and `DFLASH_KVFLASH`. The rules live with every other
launch-admission rule in `check_feature_compatibility()`; which architecture and
placement they are allowed on is one row in `model_capabilities.h`.

**Disabled, not rejected:** prefix, prefill, and disk snapshots. Their format
assumes contiguous K/V rows, so `--paged-attention` zeroes the caps and says so.

**Concurrent serving (`--max-concurrency N`).** Qwen3.6 is hybrid: 48 of its 64 layers
hold recurrent Gated DeltaNet state rather than an attention cache. With
`--max-concurrency N` that state is sequence-indexed — each slot owns a contiguous
slab of every layer's state tensor (~150 MB/slot) plus one block-table column —
and decode runs one batched step per token across all live slots (the token
axis of the graph becomes the sequence axis; GDN, `ssm_conv`, and the paged
kernel all iterate sequences independently, bitwise-identically to running
them alone — see `ctest -R batched_gdn`). Admission claims a slot but allocates
no K/V. Each prefill chunk allocates only the rows it is about to write, and
decode adds one block at each 16-token boundary. A prompt that fits the whole
pool but not the blocks currently free waits in the queue; a prefilling request
that temporarily runs out of blocks keeps its completed chunks and pauses.
Decode exhaustion fails that request explicitly rather than truncating it.
At the default pool size (`N x max_ctx`) those exhaustion paths are unreachable;
`--kv-pool-tokens` can explicitly select a smaller shared pool.

**Non-pausing admission.** `SeqEngine::admit()` claims a slot and queues its
prompt without allocating K/V or running the full prefill. Each scheduler
iteration then allocates one prompt chunk and calls `SeqEngine::step()` with
that chunk and every live decode slot in the same forward pass. Prompt rows use
the exact dense attention path against staging K/V while decode rows read the
paged pool; the committing chunk copies the prompt's recurrent state into its
slot and returns the first sampled token. This keeps existing streams advancing
through an admission instead of stopping for the whole prefill. One staging set
means only one prompt can be prefilling at a time, so cold-burst admissions are
still serialized.

## Numbers

Attention ops only, Q4_0 K/V, Qwen dims (`D=256, Hq=24, Hkv=4`). Uniform batches
of 1/2/4/8 sequences, as `paged ÷ contiguous` throughput — above 1.0x is paged
winning.

| Context | RTX 3090 (CUDA) | Radeon 8060S (HIP) |
|---|---|---|
| 4K | 1.10–1.20x | 4.28–5.60x |
| 32K | 1.12–1.23x | 4.72–5.16x |
| 128K | 1.23–1.97x | 5.00–6.42x |

Ragged 8-request profile, `128K + 7x8K` on the RTX 3090. Paging stores only live,
block-rounded tokens (188,416) instead of padding every sequence to 128K
(1,048,576 slots):

| | contiguous | paged |
|---|---|---|
| attention step | 3.007 ms | **2.235 ms** (1.35x) |
| K/V, one layer | 1152 MiB | **207 MiB** |
| K/V, 16 full-attn layers | 18.0 GiB | **3.23 GiB** (−82%) |

Two caveats. The HIP ratios are inflated by a weak native HIP
`flash_attn_ext_vec` baseline — don't read them as whole-model throughput. And
small ragged batches are still behind (`4K + 7x256` measures 0.48x), because
short sequences carry dead partitions sized by the longest sequence; the stream-K
kernel in step 4 is what fixes that.

### End-to-end concurrent serving (`--max-concurrency`)

The following measurements are the blocking-admission PR1 baseline. They
predate both non-pausing admission and the benchmark's unique expansion of
reused base prompts, so they are retained for historical comparison rather
than presented as current PR2 results.

Whole-model HTTP serving, Qwen3.6-27B Q4_K_M, Q8_0 K/V, greedy, ~300-token
prompts, 256 generated tokens per request, 2 requests per stream
(`harness/benchmarks/concurrent_benchmark.py`, 2026-07-28). Aggregate is the
sum of generated tokens over the level's wall clock; per-stream is decode-only
tok/s of one stream; TTFT includes queueing and serialized cold-burst prefills.

RTX 3090 (CUDA, `--max-ctx 4096`):

| Streams | Aggregate tok/s | Per-stream tok/s | TTFT p95 | vs 1 stream |
|---:|---:|---:|---:|---:|
| 1 | 21.0 | 21.8 | 0.45 s | 1.00x |
| 4 | 60.1 | 16.4 | 1.49 s | 2.86x |
| 8 | 88.2 | 12.6 | 2.90 s | 4.20x |
| 16 | 110.3 | 8.2 | 5.78 s | 5.25x |

Radeon 8060S (HIP gfx1151, `--max-ctx 2048`, host RAM constrained by an
unrelated 100 GB tenant during the run):

| Streams | Aggregate tok/s | Per-stream tok/s | TTFT p95 | vs 1 stream |
|---:|---:|---:|---:|---:|
| 1 | 11.0 | 11.5 | 1.06 s | 1.00x |
| 4 | 26.5 | 7.2 | 3.29 s | 2.42x |
| 8 | 42.7 | 6.1 | 6.25 s | 3.89x |
| 16 | 56.4 | 4.2 | 11.51 s | 5.15x |

## Roadmap

1. **Done.** Decode-only `ggml_paged_attn`, block manager, CUDA/HIP integration,
   benchmarks.
2. **Done.** Scheduler groundwork: sequence-indexed DeltaNet/conv state,
   iteration-level scheduling, batched decode (`--max-concurrency`).
3. **Done** (via staging): chunk prefill writes arbitrary block-table rows
   through `set_rows` while attending a contiguous staging copy — exact, any
   block layout. A paged-aware prefill *read* path (no staging copy) remains
   open.
4. **Partly done.** On-demand block allocation, client-disconnect cancellation,
   and non-pausing prefill/decode fusion are in. The physical pool still
   defaults to `max-concurrency x max-ctx`; memory-derived sizing and
   recompute preemption remain follow-up work. Also open: decode batches sized
   to live slots (today the batch is fixed at `--max-concurrency` width; idle
   slots decode a dummy row), more than one in-flight prefill, and the stream-K
   decode kernel for ragged batches.
5. **Variable-length batched prefill.**
6. **Prefix caching / CoW.** Shared prefix blocks with reference counting and
   copy-on-write.

## Benchmarks and tests

```bash
cmake --build "$BUILD_DIR" --target bench_paged_attention
"$BUILD_DIR/bench_paged_attention" --context 131072 --k-type q4_0 --v-type q4_0
```

Each run compares one native padded-contiguous attention op against one paged op
at `n_seq=1,2,4,8`, then adds a ragged 8-request row. Both layouts get identical
logical Q/K/V and identical quantized K/V rows, and both must pass a
double-precision CPU oracle before anything is timed. The CSV reports median step
time, aggregate query throughput, exact K/V and metadata bytes, and each path's
oracle error. It measures attention ops — not the full Qwen graph, HTTP
scheduling, or continuous-batching throughput.

`ctest -R paged` covers the host allocator plus the nine F16/Q4_0/Q8_0 K/V
combinations, in both the partitioned and the forced-single-partition kernel
paths.

The fixed-block/block-table shape follows the
[llama.cpp paged-attention discussion][llama-discussion].

[llama-discussion]: https://github.com/ggml-org/llama.cpp/discussions/21961
