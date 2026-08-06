# OFlash: Online Drafter Adaptation

Status: draft design, not implemented.
Scope: `server/` (qwen35 DFlash path first), plus a new trainer sidecar.
Prior art: OSD (ICML 2024, arXiv:2310.07177), OmniDraft (NeurIPS 2025,
arXiv:2507.02659), OnlineSPEC (ICML 2026). The algorithm is established;
our contribution is the production engine integration and the
iGPU-trains / dGPU-serves split on Strix Halo + R9700 class machines.

See also: `COMPUTE_SATURATION.md` (why this is the highest-leverage knob
for the roofline on this box) and `BFLASH.md` (the quality-side twin:
spending the same idle compute on parallel branches).

## 1. Goal

Make decode speed climb during use. The DFlash drafter is static today:
its acceptance rate α is fixed by offline distillation and degrades on
out-of-distribution workloads (user's codebase, prose style, language).
Every verify step already computes the exact data needed to fix this —
target hidden states and target tokens at every draft position — and
throws it away. We capture it and continuously fine-tune a LoRA on the
drafter in a background process on the iGPU, while the dGPU keeps
serving. Acceptance rate, and therefore tok/s, improves within a session
and persists across sessions.

Correctness is not at risk: speculative verification guarantees the
output distribution equals the target's regardless of draft quality. A
bad adapter update can only cost speed, never correctness, and the
acceptance guard (§7) bounds the speed downside at "static drafter".

Non-goals (this iteration):

- No target-model changes of any kind.
- No adaptation of prefill/PFlash.
- No multi-tenant or batched-serving concerns; single-box, low
  concurrency is the design point.
- No ggml backprop. Training happens in a PyTorch ROCm sidecar.

## 2. Why the hardware makes this ~free

| | R9700 (target) | Strix Halo iGPU (trainer) |
|---|---|---|
| BW | 640 GB/s | 256 GB/s |
| FP16 | 191 TFLOPS | 60 TFLOPS |
| Memory | 32 GB GDDR6 | 128 GB unified |

Decode is bandwidth-bound on the dGPU; the iGPU is idle during
generation. A fwd+bwd micro-batch of 256 rows on a ~2B-param drafter is
roughly `6 * 2e9 * 256 ≈ 3 TFLOP` → well under 1 s on the iGPU even at
poor efficiency, and it runs asynchronously — training lag never blocks
decode. Unified memory means captured features cross to the trainer
without PCIe copies.

Known unknowns to measure, not assume (§9, M4): shared power budget
(~500 W sustained for the whole box) and memory-controller contention
when the drafter itself runs on the iGPU.

## 3. Architecture

```
        dGPU (R9700)                          iGPU / CPU (Strix Halo)
┌───────────────────────────────┐      ┌─────────────────────────────────┐
│ Qwen35Backend decode loop     │      │ OFlash trainer sidecar (Python) │
│                               │      │                                 │
│ verify_batch / verify_tree ───┼──┐   │  shm ring buffer (consumer)     │
│   target logits per node      │  │   │        │                        │
│   target_hidden_cat features  │  └──▶│  replay buffer + rejection      │
│   accept/reject flags         │ shm  │  weighting                      │
│                               │ ring │        │                        │
│ draft_graph (ggml)            │      │  bf16 drafter mirror + LoRA     │
│   base GGUF weights (frozen)  │      │  fwd+bwd, AdamW on LoRA only    │
│   + LoRA slot A/B  ◀──────────┼──────┼─ adapter export (safetensors)   │
│                               │ swap │        │                        │
│ acceptance guard ─────────────┼──────┼─▶ accept / rollback signal      │
└───────────────────────────────┘      └─────────────────────────────────┘
```

Components:

1. **Capture hook** in the DFlash target verify path. Emits one record
   per verify step into a lock-free shared-memory ring. Never blocks:
   if the ring is full, drop the record and bump a counter.
2. **Trainer sidecar**: separate process, PyTorch ROCm pinned to the
   iGPU (`HIP_VISIBLE_DEVICES` selects it). Holds a bf16 mirror of the
   drafter loaded from the published safetensors, trains a LoRA,
   periodically exports it.
3. **LoRA slots** in the ggml draft graph: two preallocated adapter
   buffers (A/B). The engine drafts with the active slot while the
   inactive one is being written; swap is a pointer flip at a
   draft-block boundary.
4. **Acceptance guard**: rolling acceptance-length windows before/after
   each swap; auto-rollback on regression.
5. **Adapter store**: per-profile LoRA files on disk so adaptation
   survives restarts.

### Process model

The sidecar follows the existing remote-drafter subprocess pattern
(`DFlashDraftIpcClient`, `server/src/common/dflash_draft_ipc.h`): the
server spawns it, supervises it, and communicates over a UDS control
socket. Bulk data (features) goes through the shm ring, not the socket.
If the sidecar dies, the engine keeps serving with the current adapter;
this feature must never take down inference.

## 4. Data: what we capture

The drafter is EAGLE-style (`server/src/draft/draft_graph.h`): it
conditions on `target_hidden_cat` — concatenated hidden states from 5
target layers, `[5*hidden, ctx_len]` — plus noise embeddings, proposes a
16-token block, and shares the target LM head. Training therefore needs
the same features the drafter saw, plus target-side labels.

Per verify step, one record:

| Field | Shape / type | Source |
|---|---|---|
| `feat` | `[5*hidden, n_rows]` bf16 | already assembled for the draft call |
| `draft_tokens` | `[n_rows]` i32 | proposed block (chain) or accepted path + siblings (tree) |
| `target_topk` | `[n_rows, K=32]` (id, logprob) | target logits at each draft position, top-K only |
| `accept_flags` | `[n_rows]` u8 | verify outcome |
| `bonus/corrected token` | i32 | target sample at first rejection |
| `pos`, `seq_id`, timestamp | | bookkeeping |

Sizing: hidden = 5120 → `feat` ≈ 50 KB/row bf16 ≈ 0.8 MB per 16-row
block. At ~15 verify steps/s that's ~12 MB/s into a ring of ~2 GB
(unified memory, allocated host-side), i.e. minutes of history. Full
vocab logits are deliberately not captured (150k floats/row); top-32
covers the KL term (§5) at negligible size. The trainer subsamples: it
keeps every rejection-adjacent row and a fraction of accepted rows.

Privacy note: the ring and replay buffer contain user text (token ids)
and activations. Everything stays on-box; persisted adapters are
weights only, never data. Ring memory is freed on exit; nothing raw is
written to disk by default (a `--oflash-dump` debug flag may override).

## 5. Training

- **Trainable parameters**: LoRA rank 16, α=32, on the drafter's
  attention projections (q/k/v/o) and MLP up/down. Base weights frozen.
  Adapter size ~10–30 MB bf16 — cheap to snapshot, export, and store.
  The fc/embedding that ingests `target_hidden_cat` is included; the
  shared LM head is excluded (it belongs to the target).
- **Loss**: hybrid, following OmniDraft:
  `L = CE(drafter logits, target tokens) + λ · KL(target_topk ‖ drafter)`
  computed at every captured position, with rejection positions
  up-weighted ~3× (they are exactly where α is lost). λ ≈ 0.5 to start;
  tune in M1.
- **Optimizer**: AdamW on LoRA params only, lr 1e-4, cosine-free
  (constant lr; this is a non-stationary stream, not a convergence
  run). Grad-norm clip 1.0.
- **Cadence**: accumulate ≥ N=512 fresh rows → one micro-batch step →
  after every E=8 steps, export adapter and request a swap. All numbers
  are config, tuned in M1/M2.
- **Replay buffer**: reservoir of ~50k rows mixing recent and older
  rows (recency-weighted). This is the forgetting knob: pure-recent
  training thrashes on domain switches (OSD's data-mix experiment);
  the reservoir keeps a tail of the old domain.
- **Warm start**: on session start, load the profile's persisted
  adapter (§8) into the active slot before serving.

## 6. Engine changes (by file)

1. `server/src/qwen35/qwen35_dflash_target.cpp`
   (`verify_batch` ~L233, `verify_tree` ~L396): after logits/argmax are
   computed, materialize top-K and hand the record to the capture
   module. Feature rows are already on-device inputs to the draft call;
   copy them to the ring (device→host on dGPU; on iGPU-resident
   drafters it is a no-op view).
2. New `server/src/common/oflash/`:
   - `oflash_ring.{h,cpp}` — SPSC shm ring, drop-on-full, counters.
   - `oflash_adapter.{h,cpp}` — LoRA slot management: mmap adapter
     file, validate shapes against the loaded drafter GGUF, A/B swap,
     acceptance guard state machine.
   - `oflash_supervisor.{h,cpp}` — sidecar spawn/health, UDS control
     protocol (swap-ready, rollback, stats).
3. `server/src/draft/draft_graph.cpp`: add optional LoRA to the
   selected projections: `y = W x + (α/r) · B (A x)`. Adapter tensors
   live in a dedicated ggml buffer per slot; when no adapter is loaded
   the graph is unchanged (zero overhead for existing users).
4. `server/src/server/server_main.cpp` (~L267, next to
   `--draft-device`): `--oflash` (off by default), `--oflash-device`,
   `--oflash-profile <name>`, `--oflash-lora-rank`, `--oflash-dir`.
5. `/props` + logs: expose acceptance-length rolling stats, adapter
   generation counter, swap/rollback counts (extends the existing
   speed-profile/props surface in `docs/specs/`).
6. New `optimizations/oflash/`: the trainer package (Python, uv),
   README with setup + benchmark notes, matching the repo's
   one-directory-per-optimization convention. The trainer loads the
   published safetensors drafter, consumes the ring, and writes
   adapters. It has no build-time coupling to the C++ server beyond the
   ring/adapter file formats, which get a small versioned header.

Format note: adapter files are single-file safetensors with a JSON
metadata block (drafter hash, rank, target tensors, generation, parent
adapter hash). The engine refuses adapters whose drafter hash mismatches
the loaded GGUF.

## 7. Correctness and the acceptance guard

Speculative sampling accepts drafted token x with probability
`min(1, p_target(x)/p_draft(x))` and resamples from the residual on
rejection — output distribution is exactly the target's for any draft
distribution. One engine-side invariant must hold: **the p_draft used in
the accept rule must come from the same adapter generation that
produced the draft block**. Swaps at draft-block boundaries guarantee
this; a swap mid-block would not corrupt the target distribution but
would corrupt the guard's attribution, so we forbid it anyway.

Guard state machine per swap:

```
serving(gen N) → swap(gen N+1) → probation: 32 verify steps
  AL(probation) ≥ AL(baseline) - ε  → promote: baseline ← rolling AL
  else                              → rollback to gen N, quarantine N+1,
                                      back off next swap by 2×
```

AL = mean accepted length. ε absorbs noise (start: 0.15 tokens; tune).
Persistent rollbacks (≥3 consecutive) disable training for the session
and log loudly — that's a bug signal, not a tuning signal.

Kill switches: `--oflash` off by default; SIGHUP-style runtime disable
via the control socket; sidecar crash → engine continues on the last
promoted adapter.

## 8. Persistence and profiles

Adapters are stored under `~/.lucebox/oflash/<drafter-hash>/<profile>/`
with generation history (keep last 4). Profile selection: explicit
`--oflash-profile` per launch; default profile is `"default"`. Per-user
or per-project separation is an operator choice, not automagic.
Cross-session snowball = warm-starting from the promoted adapter.
Catastrophic forgetting across profiles is a non-issue (separate
files); within a profile, the reservoir replay (§5) is the mitigation,
and its adequacy is an explicit evaluation item (M3).

## 9. Milestones

- **M0 — telemetry (engine only, no training).** Capture hook + ring +
  a file sink. Log per-step AL and α-per-position over long sessions on
  the harness benchmarks. Deliverable: acceptance-vs-decode-step curves
  for HumanEval/Math500/GSM8K and one real long agentic coding session.
  This is the baseline every later claim is measured against.
- **M1 — offline replay training.** Train LoRA in the sidecar on
  M0-captured data, then re-run benchmarks with the adapter loaded
  statically. Proves the loss/rank/λ recipe moves α on our drafters
  before any online machinery exists. Go/no-go: ≥ +0.05 α on a held-out
  slice of the same distribution.
- **M2 — online loop.** Ring → live trainer → A/B swap → guard.
  Deliverable: within-session AL climb on a domain-shifted workload
  (e.g. non-English prose, or a codebase in a language the drafter is
  weak on), and zero regressions with the guard on adversarial (rapidly
  switching) workloads.
- **M3 — persistence.** Warm start, profiles, generation GC. Measure
  session-2 speedup at t=0 vs session-1.
- **M4 — hardware interference study.** Same benchmarks with trainer
  pinned to iGPU vs CPU vs disabled; measure target tok/s delta, power,
  and thermals under sustained load. This is the publishable systems
  measurement; it also decides the default `--oflash-device`.
- **M5 (optional) — co-adaptation with DDTree.** As α climbs, optimal
  tree width/depth changes; feed the guard's α estimate into
  `build_ddtree` budgets (`server/src/common/ddtree.cpp`). Out of scope
  until M2 numbers exist.

## 10. Evaluation

Primary metric: decode tok/s as a function of decode-step index and of
wall-clock session time, static vs adaptive, same seeds. Secondary:
AL, α per tree depth, swap/rollback counts, TTFT (must be unchanged),
target tok/s interference from the trainer (M4), peak memory both
devices. Harness: extend `harness/` clients to emit per-step timing
rather than per-request aggregates. Every claim in the README-style
card follows the repo convention: measured vs vendored llama.cpp
baseline plus vs static-DFlash baseline.

## 11. Risks

- **Drafter mirror fidelity.** The sidecar trains a bf16 mirror while
  the engine serves the Q4_K_M GGUF; a LoRA fitted on bf16 may transfer
  imperfectly onto the quantized base. Mitigation: M1 explicitly
  evaluates the adapter on the quantized engine, not in PyTorch; if the
  gap is large, add quantization-aware noise to the mirror's frozen
  weights during training (QLoRA-style) — the mirror can literally
  dequantize the GGUF instead of loading safetensors.
- **Feature capture bandwidth** on dGPU-resident drafting (device→host
  copies of `feat`). Mitigation: copies are async on the existing
  streams and drop-on-full; M0 measures the overhead before anything
  else is built. Budget: <1% target tok/s.
- **Non-stationarity**: rapid domain switching could keep the adapter
  perpetually behind. The guard bounds the damage; the reservoir buffer
  reduces thrash; if a workload defeats both, the correct behavior is
  "converges to static drafter performance", which the guard enforces.
- **ROCm PyTorch on the iGPU** (gfx1151) is less battle-tested than
  dGPU targets. Fallback: trainer on CPU (Zen 5 cores are sufficient
  for LoRA-only steps at ~10× lower cadence) — this is why the trainer
  is cadence-tolerant by design.
- **Scope creep into other model families.** This design is qwen35
  DFlash first. `dflash_target.h`'s target-agnostic contract keeps the
  capture hook generic, but gemma4/laguna wiring is explicitly later
  work.

## 12. Implementation notes (v1, as built)

The v1 implementation follows this design with four deliberate deviations,
each forced by an engine fact the draft above predated:

1. **Pipes, not UDS.** The repo has no Unix domain socket anywhere; the
   remote-drafter pattern is fork/execv + stdin text commands + a binary
   status/stream pipe. The trainer sidecar follows that: parent→child
   control lines on stdin (`promote <gen>`, `rollback <gen>`, `disable`,
   `quit`), child→parent an int32 ready handshake then newline-JSON events
   (`swap_ready`, `log`) on an inherited `--stream-fd`. Supervision lives in
   `oflash_supervisor.{h,cpp}` (not `BackendIpcProcess`, whose blocking
   waitpid and fatal-spawn call sites contradict "never take down
   inference"): spawn failure degrades to capture-only, a wedged child is
   SIGKILLed after a grace period, respawn is bounded with backoff.
2. **Content-overwrite swaps, not A/B pointer flips.** CUDA/HIP-graph
   replay keys on tensor data addresses, and the (default-on) KV-cached
   drafter graph is built once and replayed. LoRA A/B tensors are therefore
   preallocated once (rank fixed by `--oflash-lora-rank`), zero-filled
   (delta = 0 is bit-exact with the base drafter), and swaps overwrite
   their contents at a draft-block boundary. The "A/B slot" of §3 survives
   host-side: the previous generation is staged in RAM so guard rollback is
   an upload, not a file reload. Every swap/rollback also runs
   `draft_kv_reset` — cached context K/V rows embed the previous wk/wv
   delta and would otherwise mix adapter generations.
3. **The capture hook lives in the decode loop** (`do_spec_decode`), not in
   `verify_batch`: accept flags and acceptance length exist only there, and
   `sg_.logits` is clobbered by the replay forward, so top-K is
   materialized between verify and replay (GPU top-K via
   `geometric_extract_draft_topk_cuda`, CPU fallback). Feature rows are
   read at the same point the loop syncs the draft feature mirror — they
   are bit-identical to what the drafter conditions on.
4. **Acceptance is exact-match in this engine** (greedy argmax match, or
   sampled-chain match under `DFLASH_SAMPLED_VERIFY`); there is no
   p_target/p_draft residual rule and no p_draft anywhere. §7's correctness
   claim holds a fortiori; swap-at-block-boundary is still enforced for the
   guard's attribution.

Identity is SHA-256 (`read_gguf_metadata`, sidecar-cached): full hex in
adapter metadata (`oflash.drafter_sha256`), first 16 hex chars as the store
directory name and (as u64) in the ring header. Formats live in
`server/src/common/oflash/oflash_format.h`, mirrored by
`optimizations/oflash/src/oflash/ring_format.py`, cross-checked by golden
tests on both sides. The trainer builds its bf16 mirror by dequantizing the
drafter GGUF (§11's mitigation — no safetensors drafter ships on this box)
and dequantizes the target's `output.weight` + `token_embd` for the loss
head and noise embeddings.

Admission is intentionally narrower than qwen35 generally: the server rejects
OFlash with PFlash compression, request-scoped/`--lazy-draft` residency, and
layer-split or tensor-parallel targets. Those modes park or replace the local
decode drafter and cannot yet preserve the pointer-stable LoRA graph. Shared
memory is fully reserved at startup so capacity failures disable capture
instead of surfacing later as SIGBUS; stale dead-process rings are reclaimed
on the next OFlash startup.
