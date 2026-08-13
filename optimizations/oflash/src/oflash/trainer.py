"""Online training loop for the OFlash trainer sidecar (OFLASH.md §5).

Consumes the engine's capture ring (ring_format.RingReader), maintains a
per-sequence shadow of the drafter's feature window plus verify-step labels,
and fine-tunes a LoRA on the low-precision mirror (mirror.DrafterMirror).
Every `export_every` optimizer steps the adapter is written via
adapter_export and a `swap_ready` event is emitted; the engine A/B-swaps it
behind the acceptance guard and answers with promote/rollback commands.

Robustness contract: bad records and transient errors are dropped; OOM,
accelerator faults and mirror-load failures shed the training model and enter
capture-only drain mode. The process keeps draining the ring because the
engine sizes its drop-on-full budget assuming a live consumer. Only corrupt
ring framing exits for a supervisor respawn.
"""

from __future__ import annotations

import gc
import math
import os
import queue
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from .adapter_export import (adapter_path, export_adapter, gc_generations,
                             load_validated_adapter)
from .mirror import DrafterMirror
from .identity import fnv1a64, require_sha256
from .ring_format import (
    REC_CONTEXT,
    REC_SEQ_END,
    REC_STEP,
    Record,
    RingReader,
    bf16_bits_to_f32,
)

# A training window shorter than this carries too little context to teach the
# drafter anything its block-attention will see at serve time; skip it.
MIN_WINDOW_ROWS = 64
# Upper bound on StepSamples per optimizer step. Samples backprop serially;
# context length, not this work/cadence bound, controls peak activation memory.
MICRO_BATCH_MAX = 16
# Main-loop sleep when the ring is drained and nothing trained.
POLL_SLEEP_S = 0.05
# Max ring records ingested per loop iteration, so stdin commands stay
# responsive even under a large backlog.
INGEST_CHUNK = 256
# Loss EMA smoothing (per optimizer step).
EMA_DECAY = 0.98


def _is_oom(exc: BaseException) -> bool:
    """Recognize both host and torch allocator exhaustion."""
    torch_oom = getattr(torch, "OutOfMemoryError", None)
    return (isinstance(exc, MemoryError)
            or (torch_oom is not None and isinstance(exc, torch_oom))
            or "out of memory" in str(exc).lower())


def _is_fatal_accelerator_error(exc: BaseException) -> bool:
    """Errors after which retrying work in the same HIP context is unsafe."""
    message = str(exc).lower()
    return any(marker in message for marker in (
        "device-side assert",
        "illegal memory access",
        "unspecified launch failure",
        "hip error",
        "device lost",
    ))


def _max_adapter_generation(out_dir: str) -> int:
    """Highest immutable adapter generation already present on disk."""
    highest = 0
    try:
        entries = os.scandir(out_dir)
    except OSError:
        return highest
    with entries:
        for entry in entries:
            name = entry.name
            prefix, suffix = "adapter-gen", ".safetensors"
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            try:
                highest = max(highest,
                              int(name[len(prefix):-len(suffix)]))
            except ValueError:
                continue
    return highest


class ControlLike(Protocol):
    """The slice of cli.Control the trainer needs (kept import-cycle-free)."""

    commands: queue.Queue[tuple[str, int]]

    def send_event(self, event: str, **fields) -> None: ...

    def log(self, message: str) -> None: ...


@dataclass
class TrainerConfig:
    """Knobs the CLI passes through; defaults live in cli.parse_args."""

    drafter_gguf: str
    target_gguf: str
    out_dir: str
    profile: str
    rank: int
    alpha: float
    drafter_sha256: str
    target_sha256: str
    start_generation: int
    requested_device: str
    dtype: str
    resolved_rope_theta: float | None
    resolved_swa_window: int | None
    resolved_swa_pattern: tuple[bool, ...] | None
    resolved_mask_token_id: int | None
    drafter_semantics: str
    lr: float
    kl_lambda: float
    reject_weight: float
    batch_rows: int
    export_every: int
    train_ctx: int
    reservoir_rows: int
    keep_generations: int
    seed: int


@dataclass
class StepSample:
    """Labels of one verify step, tied to feature rows held by its SeqStore."""

    pos: int                       # first feature-row position of the step
    draft_tok: np.ndarray          # i32 [n_labels]; [0] is the seed token
    target_tok: np.ndarray         # i32 [n_labels]
    accept_flags: np.ndarray       # u8 [n_labels]; 0 = rejected
    accept_len: int
    bonus_tok: int
    topk_ids: np.ndarray | None    # i32 [n_labels, K] or None
    topk_lp: np.ndarray | None     # f32 [n_labels, K] or None
    block_size: int = 0            # full proposal width from ring header

    @property
    def n_labels(self) -> int:
        return int(self.draft_tok.shape[0])

    @property
    def has_reject(self) -> bool:
        return bool((self.accept_flags == 0).any()
                    or self.bonus_tok >= 0
                    or (self.block_size > 0
                        and self.n_labels < self.block_size))

    def valid_train_rows(self, q_len: int) -> int:
        """Rows whose target logits have valid autoregressive conditioning.

        Verification after the first mismatch is conditioned on an
        uncommitted draft suffix and cannot supervise the mirror. accept_len
        includes the seed, so it also counts the accepted transitions plus
        the first target correction that remain safe to train on.
        """
        return max(0, min(self.accept_len, self.n_labels, q_len - 1))


@dataclass
class SeqStore:
    """Shadow state of one engine sequence: feature rows + step samples.

    Rows stay as raw bf16 bit patterns (u16) and are converted to f32 lazily
    at batch-build time — most rows are never trained on.
    """

    seq_id: int
    first_pos: int                 # lowest feature-row position ever seen
    max_pos: int = 0               # one past the highest row position seen
    feat: dict[int, np.ndarray] = field(default_factory=dict)
    steps: list[StepSample] = field(default_factory=list)
    evicted: bool = False


class Trainer:
    """The sidecar main loop: ring in, LoRA gradients, adapters out."""

    def __init__(self, cfg: TrainerConfig, ring: RingReader,
                 control: ControlLike):
        self.cfg = cfg
        self._ring = ring
        self._control = control

        self._mirror: DrafterMirror | None = None
        self._optimizer: torch.optim.AdamW | None = None
        self._requested_device = getattr(cfg, "requested_device", "cpu")
        self._device = ("cpu" if self._requested_device == "cpu" else
                        "cuda")
        dtype_name = getattr(cfg, "dtype", "auto")
        if dtype_name == "auto":
            dtype_name = "fp32" if self._device == "cpu" else "fp16"
        self._dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype_name]

        # Live sequences and the replay reservoir of retired ones (oldest
        # first; evicted whole when the global row budget is exceeded).
        self._seqs: dict[int, SeqStore] = {}
        self._reservoir: deque[SeqStore] = deque()
        self._reservoir_row_count = 0
        # Keep replay selection reproducible without mutating Python's global
        # RNG state (the mirror and callers may have their own randomness).
        self._rng = random.Random(cfg.seed)

        # Fresh samples since the last optimizer step, with their stores.
        self._fresh: list[tuple[SeqStore, StepSample]] = []
        self._fresh_rows = 0

        self._enabled = True
        self._micro_max = MICRO_BATCH_MAX
        self._train_ctx = cfg.train_ctx
        self._generation = max(
            cfg.start_generation, _max_adapter_generation(cfg.out_dir))
        self._promoted_path: str | None = None
        self._promoted_generation: int | None = None
        self._drafter_semantics = ""

        self._opt_steps = 0
        self._rows_seen = 0
        self._labels_seen = 0
        self._loss_ema: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def run(self) -> int:
        """Main loop; returns the process exit code."""
        if self.cfg.train_ctx < MIN_WINDOW_ROWS:
            self._control.log(
                f"--train-ctx {self.cfg.train_ctx} is below the minimum "
                f"window {MIN_WINDOW_ROWS}; no sample could train — exiting")
            return 1
        if not self._validate_ring_identity():
            return 1
        if self._device == "cuda" and not torch.cuda.is_available():
            self._disable(
                f"requested HIP device {self._requested_device!r} is not "
                "available; refusing unsafe CPU fallback",
                release_model=True)
        if not self._enabled:
            return self._drain_only_loop()
        try:
            self._preflight_accelerator()
        except Exception as e:
            self._disable(
                f"accelerator preflight failed ({e!r}); refusing a large "
                "mirror allocation",
                release_model=True)
        # Entering the never-returning drain loop from inside the except block
        # would keep `e` and its traceback alive.  In particular, a failed
        # mirror constructor's traceback can own several GiB of partially
        # allocated device buffers and dequantized host arrays.
        if not self._enabled:
            self._release_training_memory()
            return self._drain_only_loop()
        try:
            self._load_mirror()
        except Exception as e:
            reason = ("mirror load exhausted memory" if _is_oom(e)
                      else "mirror load failed")
            self._disable(
                f"{reason} ({e!r}); capture-only mode avoids a supervisor "
                "allocation retry loop",
                release_model=True)
        if not self._enabled:
            self._release_training_memory()
            return self._drain_only_loop()
        if not self._validate_ring_dims():
            self._disable("incompatible capture/mirror dimensions",
                          release_model=True)
            return self._drain_only_loop()

        while True:
            rc = self._drain_commands()
            if rc is not None:
                return rc
            try:
                progressed = self._consume_ring()
            except RuntimeError as e:
                # Corrupt framing is unrecoverable: tail cannot skip a record
                # whose size it cannot trust. Exit; the supervisor respawns.
                self._control.log(f"ring read failed: {e!r}; exiting")
                return 1
            trained = False
            if self._enabled and self._fresh_rows >= self.cfg.batch_rows:
                self._train_step()
                trained = True
            if not progressed and not trained:
                time.sleep(POLL_SLEEP_S)

    def _drain_only_loop(self) -> int:
        """Stay attached after a safety shutdown so the ring cannot fill."""
        while True:
            rc = self._drain_commands()
            if rc is not None:
                return rc
            try:
                progressed = self._consume_ring()
            except RuntimeError as e:
                self._control.log(f"ring read failed: {e!r}; exiting")
                return 1
            if not progressed:
                time.sleep(POLL_SLEEP_S)

    def _validate_ring_identity(self) -> bool:
        """Refuse a stream from different model files or draft semantics."""
        info = self._ring.info
        checks = (
            ("drafter", self.cfg.drafter_sha256, info.drafter_hash),
            ("target", self.cfg.target_sha256, info.target_hash),
        )
        for label, sha, got in checks:
            if not sha:
                self._control.log(
                    f"no --{label}-sha256 given; refusing unlabeled ring")
                return False
            try:
                canonical = require_sha256(sha, f"{label} SHA-256")
                want = int(canonical[:16], 16)
            except ValueError as exc:
                self._control.log(f"malformed {label} identity ({exc}); exiting")
                return False
            if got != want:
                self._control.log(
                    f"ring {label}_hash {got:#018x} != {label} GGUF "
                    f"{want:#018x} ({canonical[:16]}…); refusing to train — "
                    "exiting")
                return False
        if self.cfg.drafter_semantics:
            return self._validate_ring_semantics(self.cfg.drafter_semantics)
        return True

    def _validate_ring_semantics(self, semantics: str) -> bool:
        want = fnv1a64(semantics)
        got = self._ring.info.drafter_semantics_hash
        if got != want:
            self._control.log(
                f"ring drafter_semantics_hash {got:#018x} != resolved "
                f"drafter semantics {want:#018x}; refusing to train")
            return False
        return True

    def _load_mirror(self) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        if self._device == "cuda":
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            self._control.log(
                f"trainer GPU: requested ordinal={self._requested_device} "
                f"visible={props.name!r} free={free / 2**30:.1f}GiB "
                f"total={total / 2**30:.1f}GiB dtype={self._dtype}")
        t0 = time.monotonic()
        self._mirror = DrafterMirror.from_gguf(
            self.cfg.drafter_gguf, self.cfg.target_gguf, self.cfg.rank,
            self.cfg.alpha, self._device, self._dtype,
            resolved_rope_theta=self.cfg.resolved_rope_theta,
            resolved_swa_window=self.cfg.resolved_swa_window,
            resolved_swa_pattern=self.cfg.resolved_swa_pattern,
            resolved_mask_token_id=self.cfg.resolved_mask_token_id)
        dt = time.monotonic() - t0
        m = self._mirror
        from .gguf_reader import drafter_semantics
        actual_semantics = drafter_semantics(m.meta)
        if (self.cfg.drafter_semantics
                and self.cfg.drafter_semantics != actual_semantics):
            raise ValueError(
                "resolved drafter semantics disagree with engine: "
                f"mirror={actual_semantics!r} "
                f"engine={self.cfg.drafter_semantics!r}")
        if not self._validate_ring_semantics(actual_semantics):
            raise ValueError("capture ring has different drafter semantics")
        self._drafter_semantics = (self.cfg.drafter_semantics
                                   or actual_semantics)
        self._control.log(
            f"mirror loaded in {dt:.1f}s: device={self._device} "
            f"dtype={self._dtype} layers={m.dims.n_layer} "
            f"hidden={m.dims.hidden} fc_in={m.dims.fc_in} vocab={m.vocab} "
            f"block={m.block_size} rank={self.cfg.rank} "
            f"rope={m.meta.rope_theta:g} swa={m.meta.swa_window} "
            f"pattern={m.meta.sliding_window_pattern} "
            f"mask={m.meta.mask_token_id}")
        self._warm_start()
        self._make_optimizer()

    def _preflight_accelerator(self) -> None:
        """Exercise mixed-precision forward/backward before the multi-GiB load.

        The probe stays below one MiB (apart from lazy runtime workspaces) and
        rejects a generic CUDA/CPU torch wheel on a requested HIP device.
        """
        if self._device != "cuda":
            return
        if not getattr(torch.version, "hip", None):
            raise RuntimeError(
                "requested a HIP device but torch is not a ROCm build")
        x = w = lora_a = lora_b = optimizer = None
        try:
            x = torch.randn((16, 64), device="cuda", dtype=self._dtype)
            w = torch.randn((64, 64), device="cuda", dtype=self._dtype)
            lora_a = torch.nn.Parameter(torch.randn(
                (4, 64), device="cuda", dtype=torch.float32) * 0.01)
            lora_b = torch.nn.Parameter(torch.zeros(
                (64, 4), device="cuda", dtype=torch.float32))
            optimizer = torch.optim.AdamW(
                (lora_a, lora_b), lr=1e-4, weight_decay=0.0)
            y = (torch.nn.functional.linear(x, w).float()
                 + (x.float() @ lora_a.t()) @ lora_b.t())
            loss = y.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite preflight loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (lora_a, lora_b), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if not all(torch.isfinite(p).all() for p in (lora_a, lora_b)):
                raise FloatingPointError("nonfinite preflight parameter")
            self._control.log(
                f"accelerator preflight passed: ROCm {torch.version.hip} "
                f"dtype={self._dtype}")
        finally:
            del optimizer, lora_b, lora_a, w, x
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    def _warm_start(self) -> None:
        """Resume from the generation the engine is serving, if it exists."""
        assert self._mirror is not None
        gen = self.cfg.start_generation
        if gen <= 0:
            return
        path = os.path.abspath(adapter_path(self.cfg.out_dir, gen))
        if not os.path.exists(path):
            self._control.log(f"warm start: gen {gen} adapter missing at "
                              f"{path}; starting from zero LoRA")
            return
        try:
            state = load_validated_adapter(
                path, self.cfg.drafter_sha256, self.cfg.target_sha256,
                self._drafter_semantics, self.cfg.rank, self.cfg.alpha,
                generation=gen, profile=self.cfg.profile)
            self._mirror.load_lora_state_numpy(state)
            # The engine serves this generation: protect it from GC and make
            # it the rollback target until the first promote arrives.
            self._promoted_path = path
            self._promoted_generation = gen
            self._control.log(f"warm start: loaded {path}")
        except Exception as e:
            self._mirror.reset_lora()
            self._control.log(f"warm start failed ({e!r}); zero LoRA")

    def _validate_ring_dims(self) -> bool:
        assert self._mirror is not None
        info = self._ring.info
        if info.row_elems != self._mirror.dims.fc_in:
            self._control.log(
                f"ring feature width {info.row_elems} != mirror fc_in "
                f"{self._mirror.dims.fc_in}; incompatible capture")
            return False
        if info.block_size != self._mirror.block_size:
            self._control.log(
                f"ring block_size {info.block_size} != mirror "
                f"{self._mirror.block_size}; incompatible labels")
            return False
        if info.vocab != self._mirror.vocab:
            self._control.log(
                f"ring vocab {info.vocab} != mirror target vocab "
                f"{self._mirror.vocab}; incompatible labels")
            return False
        return True

    def _make_optimizer(self) -> None:
        assert self._mirror is not None
        # Constant lr on purpose: this is a non-stationary stream, not a
        # convergence run (OFLASH.md §5).
        self._optimizer = torch.optim.AdamW(
            self._mirror.lora_parameters(), lr=self.cfg.lr, weight_decay=0.0)

    # ── engine commands ──────────────────────────────────────────────

    def _drain_commands(self) -> int | None:
        """Apply queued engine commands; a return code means exit now."""
        while True:
            try:
                cmd, gen = self._control.commands.get_nowait()
            except queue.Empty:
                return None
            if cmd == "quit":
                return 0
            if cmd == "promote":
                path = os.path.abspath(adapter_path(self.cfg.out_dir, gen))
                self._promoted_path = path
                self._promoted_generation = gen
                self._control.log(f"promote: gen {gen} is now the baseline")
            elif cmd == "rollback":
                self._rollback(gen)
                # This shares the ordered event stream with swap_ready. The
                # engine drops stale announcements until this matching ack.
                self._control.send_event("rollback_ack", generation=gen)
            elif cmd == "disable":
                self._disable()
            else:
                self._control.log(f"unknown command {cmd!r}; ignored")

    def _rollback(self, gen: int) -> None:
        """Quarantined generation: retreat to the last promoted weights."""
        if self._mirror is None:
            self._control.log(
                f"rollback gen {gen} ignored: trainer is capture-only")
            return
        try:
            if self._promoted_path and os.path.exists(self._promoted_path):
                state = load_validated_adapter(
                    self._promoted_path, self.cfg.drafter_sha256,
                    self.cfg.target_sha256, self._drafter_semantics,
                    self.cfg.rank, self.cfg.alpha,
                    generation=self._promoted_generation,
                    profile=self.cfg.profile)
                self._mirror.load_lora_state_numpy(state)
                self._control.log(f"rollback (quarantining gen {gen}): "
                                  f"reloaded {self._promoted_path}")
            else:
                self._mirror.reset_lora()
                self._control.log(f"rollback (quarantining gen {gen}): no "
                                  "promoted adapter; LoRA reset to zero")
        except Exception as e:
            self._mirror.reset_lora()
            self._control.log(f"rollback reload failed ({e!r}); LoRA reset")
        # Optimizer moments were fitted to the rejected trajectory.
        self._make_optimizer()

    def _release_training_memory(self) -> None:
        """Collect dead model objects, then return cached device allocations."""
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    def _disable(self, reason: str = "engine command",
                 release_model: bool = True) -> None:
        """Stop training/exporting, shed memory, but keep draining the ring."""
        self._enabled = False
        self._seqs.clear()
        self._reservoir.clear()
        self._reservoir_row_count = 0
        self._fresh.clear()
        self._fresh_rows = 0
        if release_model:
            self._optimizer = None
            self._mirror = None
            self._release_training_memory()
        self._control.log(
            f"training disabled ({reason}); ring consumption continues")
        self._control.send_event("training_disabled", reason=reason)

    # ── ring ingestion ───────────────────────────────────────────────

    def _consume_ring(self) -> bool:
        """Ingest up to INGEST_CHUNK records; True if any were consumed."""
        consumed = False
        for _ in range(INGEST_CHUNK):
            rec = self._ring.read_next()
            if rec is None:
                break
            consumed = True
            if not self._enabled:
                continue
            try:
                self._ingest(rec)
            except Exception as e:
                # The tail already advanced past this record; dropping it is
                # safe and beats taking the sidecar down.
                self._control.log(f"record ingest failed: {e!r}; dropped")
        return consumed

    def _ingest(self, rec: Record) -> None:
        # Records are globally ordered and request ids increase strictly.
        # If SEQ_END was dropped under ring pressure, the first record for a
        # newer request proves that older live stores are complete.
        for old_id in [sid for sid in self._seqs if sid < rec.seq_id]:
            self._retire(old_id)

        if rec.type == REC_SEQ_END:
            self._retire(rec.seq_id)
            return
        if rec.type not in (REC_CONTEXT, REC_STEP):
            return
        store = self._seqs.get(rec.seq_id)
        if store is None:
            store = SeqStore(seq_id=rec.seq_id, first_pos=rec.pos,
                             max_pos=rec.pos)
            self._seqs[rec.seq_id] = store
        store.first_pos = min(store.first_pos, rec.pos)
        store.max_pos = max(store.max_pos, rec.pos + rec.n_rows)
        if rec.feat_u16 is not None and rec.n_rows > 0:
            for i in range(rec.n_rows):
                store.feat[rec.pos + i] = rec.feat_u16[i]
            self._rows_seen += rec.n_rows
        if rec.type == REC_STEP and rec.n_labels > 0 \
                and rec.draft_tok is not None:
            sample = StepSample(
                pos=rec.pos,
                draft_tok=rec.draft_tok,
                target_tok=rec.target_tok,
                accept_flags=rec.accept_flags,
                accept_len=rec.accept_len,
                bonus_tok=rec.bonus_tok,
                topk_ids=rec.topk_ids,
                topk_lp=rec.topk_lp,
                block_size=self._ring.info.block_size,
            )
            store.steps.append(sample)
            self._fresh.append((store, sample))
            self._fresh_rows += rec.n_labels
            self._labels_seen += rec.n_labels
        if len(store.feat) > self.cfg.reservoir_rows:
            self._prune_runaway(store)

    def _retire(self, seq_id: int) -> None:
        """SEQ_END: move the sequence into the replay reservoir."""
        store = self._seqs.pop(seq_id, None)
        if store is None:
            return
        # Windows can only shrink from here on; drop steps that will never
        # be trainable so eviction accounting stays honest.
        store.steps = [s for s in store.steps
                       if self._window_start(store, s) is not None]
        if not store.steps or not store.feat:
            store.evicted = True
            return
        self._reservoir.append(store)
        self._reservoir_row_count += len(store.feat)
        # Recency-weighted keep, implemented as whole-seq FIFO eviction:
        # the newest retired seq is never evicted.
        while (self._reservoir_row_count > self.cfg.reservoir_rows
               and len(self._reservoir) > 1):
            old = self._reservoir.popleft()
            old.evicted = True
            self._reservoir_row_count -= len(old.feat)

    def _prune_runaway(self, store: SeqStore) -> None:
        """One live sequence outgrew the whole reservoir budget: drop its
        oldest half so a marathon request cannot exhaust host memory."""
        positions = sorted(store.feat)
        cut = positions[len(positions) // 2]
        for p in positions:
            if p >= cut:
                break
            del store.feat[p]
        store.first_pos = max(store.first_pos, cut)
        store.steps = [s for s in store.steps
                       if s.pos - cut >= MIN_WINDOW_ROWS]
        self._control.log(
            f"seq {store.seq_id} exceeded the row budget; pruned rows below "
            f"pos {cut} ({len(store.feat)} kept)")

    def _window_start(self, store: SeqStore,
                      sample: StepSample) -> int | None:
        """First row of the sample's training context, or None if the window
        is incomplete or shorter than MIN_WINDOW_ROWS."""
        start = max(sample.pos - self._train_ctx, store.first_pos)
        if sample.pos - start < MIN_WINDOW_ROWS:
            return None
        feat = store.feat
        for p in range(start, sample.pos):
            if p not in feat:
                return None
        return start

    # ── training ─────────────────────────────────────────────────────

    def _build_batch(self) -> list[tuple[SeqStore, StepSample, int]]:
        """Every fresh rejection-adjacent sample, topped up with
        recency-biased draws from the reservoir, capped at _micro_max."""
        cap = self._micro_max
        batch: list[tuple[SeqStore, StepSample, int]] = []
        remaining: list[tuple[SeqStore, StepSample]] = []
        for store, sample in self._fresh:
            if store.evicted or not sample.has_reject:
                continue
            start = self._window_start(store, sample)
            if start is not None:
                if len(batch) < cap:
                    batch.append((store, sample, start))
                else:
                    remaining.append((store, sample))
        # Keep valid fresh samples that did not fit this micro-batch. Invalid
        # or permanently incomplete samples are discarded once, not retried
        # forever. The main loop drains remaining batches on later turns.
        self._fresh = remaining
        self._fresh_rows = sum(s.n_labels for _, s in remaining)
        n_res = len(self._reservoir)
        attempts = 0
        while len(batch) < cap and n_res > 0 and attempts < 4 * cap:
            attempts += 1
            # Density ∝ index: newest retired sequences are drawn most.
            idx = min(int(math.sqrt(self._rng.random()) * n_res), n_res - 1)
            store = self._reservoir[idx]
            sample = self._rng.choice(store.steps)
            start = self._window_start(store, sample)
            if start is not None:
                batch.append((store, sample, start))
        return batch

    def _train_step(self) -> None:
        """One optimizer step over a micro-batch; never raises."""
        assert self._optimizer is not None
        batch = self._build_batch()
        if not batch:
            return
        try:
            loss = self._optimize(batch)
        except (RuntimeError, MemoryError) as e:
            optimizer = self._optimizer
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if _is_oom(e):
                old_ctx = self._train_ctx
                if old_ctx > MIN_WINDOW_ROWS:
                    self._train_ctx = max(MIN_WINDOW_ROWS, old_ctx // 2)
                    if self._device == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass
                    self._control.log(
                        f"OOM in train step; context reduced "
                        f"{old_ctx}->{self._train_ctx} rows")
                else:
                    self._disable(
                        f"repeated OOM at minimum {MIN_WINDOW_ROWS}-row "
                        "training context",
                        release_model=True)
            elif _is_fatal_accelerator_error(e):
                self._disable(
                    f"accelerator context is unsafe after {e!r}",
                    release_model=True)
            else:
                self._control.log(f"train step failed: {e!r}; skipped")
            return
        except Exception as e:
            optimizer = self._optimizer
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            self._control.log(f"train step failed: {e!r}; skipped")
            return
        self._loss_ema = (loss if self._loss_ema is None else
                          EMA_DECAY * self._loss_ema
                          + (1.0 - EMA_DECAY) * loss)
        self._micro_max = min(MICRO_BATCH_MAX, self._micro_max + 1)
        if self._opt_steps % self.cfg.export_every == 0:
            self._export()

    def _optimize(self,
                  batch: list[tuple[SeqStore, StepSample, int]]) -> float:
        """Grad-accumulate over the batch, clip, step. Returns mean loss."""
        assert self._mirror is not None and self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        inv = 1.0 / len(batch)
        total = 0.0
        for store, sample, win_start in batch:
            loss = self._sample_loss(store, sample, win_start)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            (loss * inv).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(
            self._mirror.lora_parameters(), 1.0, error_if_nonfinite=True)
        self._optimizer.step()
        params_finite = all(
            torch.isfinite(p).all()
            for p in self._mirror.lora_parameters())
        state_finite = all(
            torch.isfinite(value).all()
            for state in self._optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value))
        if not params_finite or not state_finite:
            # The candidate is poisoned. Restore the promoted adapter (or
            # zero LoRA) and discard all optimizer moments before any export.
            self._rollback(self._generation)
            raise FloatingPointError("nonfinite optimizer state; rolled back")
        self._opt_steps += 1
        return total * inv

    def _sample_loss(self, store: SeqStore, sample: StepSample,
                     win_start: int) -> torch.Tensor:
        """Weighted CE + λ·KL(target top-K ‖ drafter) for one step sample.

        Label alignment (oflash_format.h §STEP): target_tok[i] is the
        target's choice AFTER consuming draft_tok[0..i], i.e. the ground
        truth for draft position i+1. Drafter row j therefore trains
        against target_tok[j-1] (and topk[j-1]); row 0 reproduces the
        known seed and carries no signal. accept_flags[j] describes
        draft_tok[j] — the row the drafter produced — so the rejection
        upweight indexes rows, not labels.
        """
        assert self._mirror is not None
        m = self._mirror
        rows = np.stack([store.feat[p]
                         for p in range(win_start, sample.pos)])
        feat_np = bf16_bits_to_f32(rows)
        if not np.isfinite(feat_np).all():
            raise ValueError("capture contains nonfinite feature values")

        q_len = m.block_size
        n_train = sample.valid_train_rows(q_len)
        if n_train < 1:
            # Preserve the old zero-loss behavior without touching a device.
            return next(iter(m.lora_parameters())).sum() * 0.0
        seed = int(sample.draft_tok[0])
        target_ids = sample.target_tok[:n_train]
        if seed < 0 or seed >= m.vocab:
            raise ValueError(f"draft seed token {seed} outside vocabulary")
        if ((target_ids < 0) | (target_ids >= m.vocab)).any():
            raise ValueError("target token outside vocabulary")
        if sample.topk_ids is not None:
            topk_ids = sample.topk_ids[:n_train]
            if ((topk_ids < 0) | (topk_ids >= m.vocab)).any():
                raise ValueError("top-K token outside vocabulary")
        if (sample.topk_lp is not None
                and not np.isfinite(sample.topk_lp[:n_train]).all()):
            raise ValueError("top-K log-probability is nonfinite")

        feat = torch.from_numpy(feat_np).to(
            self._device, dtype=m.dtype)

        noise = torch.full((q_len,), m.mask_token_id, dtype=torch.long)
        noise[0] = seed
        logits = m.forward(feat, noise.to(self._device))  # [q_len, vocab] f32
        if not torch.isfinite(logits).all():
            raise FloatingPointError("mirror produced nonfinite logits")

        # Only the accepted span plus its first correction is valid. Target
        # logits later in the verify batch saw an uncommitted draft suffix.
        logp = torch.log_softmax(
            logits[1:n_train + 1].float(), dim=-1)
        tgt = torch.from_numpy(
            sample.target_tok[:n_train].astype(np.int64)).to(self._device)
        ce = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)

        reject_rows = np.zeros(n_train, dtype=bool)
        n_flagged = min(n_train, max(0, sample.accept_flags.size - 1))
        reject_rows[:n_flagged] = \
            sample.accept_flags[1:1 + n_flagged] == 0
        if sample.has_reject:
            reject_rows[n_train - 1] = True
        rej = torch.from_numpy(reject_rows).to(self._device)
        w = torch.ones(n_train, dtype=torch.float32, device=self._device)
        w = w.masked_fill(rej, float(self.cfg.reject_weight))
        loss_vec = w * ce

        if (sample.topk_ids is not None and sample.topk_lp is not None
                and self.cfg.kl_lambda > 0.0):
            lp = torch.from_numpy(sample.topk_lp[:n_train]).to(self._device)
            ids = torch.from_numpy(
                sample.topk_ids[:n_train].astype(np.int64)).to(self._device)
            log_p = torch.log_softmax(lp, dim=-1)  # renormalized over top-K
            p = log_p.exp()
            q_lp = logp.gather(1, ids)
            kl = (p * (log_p - q_lp)).sum(dim=-1)
            loss_vec = loss_vec + self.cfg.kl_lambda * kl

        return loss_vec.mean()

    # ── export ───────────────────────────────────────────────────────

    def _export(self) -> None:
        """Write the next adapter generation and announce it. Never raises."""
        assert self._mirror is not None
        self._generation += 1
        gen = self._generation
        path = os.path.abspath(adapter_path(self.cfg.out_dir, gen))
        try:
            export_adapter(path, self._mirror.lora_state_numpy(),
                           self._mirror.dims, self.cfg.rank, self.cfg.alpha,
                           self.cfg.drafter_sha256,
                           self.cfg.target_sha256,
                           self._drafter_semantics,
                           gen, self.cfg.profile)
            gc_generations(self.cfg.out_dir, self.cfg.keep_generations,
                           protect=self._promoted_path)
        except Exception as e:
            self._control.log(f"adapter export gen {gen} failed: {e!r}")
            return
        self._control.send_event("swap_ready", path=path, generation=gen)
        ema = f"{self._loss_ema:.4f}" if self._loss_ema is not None else "n/a"
        self._control.log(
            f"gen {gen} exported: rows={self._rows_seen} "
            f"labels={self._labels_seen} opt_steps={self._opt_steps} "
            f"loss_ema={ema} backlog={self._ring.backlog()}B "
            f"dropped={self._ring.dropped_records} "
            f"reservoir={self._reservoir_row_count}rows/"
            f"{len(self._reservoir)}seqs")
