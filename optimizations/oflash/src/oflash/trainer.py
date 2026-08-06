"""Online training loop for the OFlash trainer sidecar (OFLASH.md §5).

Consumes the engine's capture ring (ring_format.RingReader), maintains a
per-sequence shadow of the drafter's feature window plus verify-step labels,
and fine-tunes a LoRA on the bf16 drafter mirror (mirror.DrafterMirror).
Every `export_every` optimizer steps the adapter is written via
adapter_export and a `swap_ready` event is emitted; the engine A/B-swaps it
behind the acceptance guard and answers with promote/rollback commands.

Robustness contract: a failed train step (bad record, transient torch error,
OOM) is logged and skipped — this process keeps draining the ring no matter
what, because the engine sizes its drop-on-full budget assuming a live
consumer. Only ring corruption or a broken mirror load exits (the supervisor
respawns with backoff).
"""

from __future__ import annotations

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
from safetensors.numpy import load_file

from .adapter_export import adapter_path, export_adapter, gc_generations
from .mirror import DrafterMirror
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
# Upper bound on StepSamples per optimizer step (halved on OOM, floor 1).
MICRO_BATCH_MAX = 16
# Main-loop sleep when the ring is drained and nothing trained.
POLL_SLEEP_S = 0.05
# Max ring records ingested per loop iteration, so stdin commands stay
# responsive even under a large backlog.
INGEST_CHUNK = 256
# Loss EMA smoothing (per optimizer step).
EMA_DECAY = 0.98


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
    start_generation: int
    lr: float
    kl_lambda: float
    reject_weight: float
    batch_rows: int
    export_every: int
    train_ctx: int
    reservoir_rows: int
    keep_generations: int


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

    @property
    def n_labels(self) -> int:
        return int(self.draft_tok.shape[0])

    @property
    def has_reject(self) -> bool:
        return bool((self.accept_flags == 0).any())


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
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = (torch.bfloat16 if self._device == "cuda"
                       else torch.float32)

        # Live sequences and the replay reservoir of retired ones (oldest
        # first; evicted whole when the global row budget is exceeded).
        self._seqs: dict[int, SeqStore] = {}
        self._reservoir: deque[SeqStore] = deque()
        self._reservoir_row_count = 0

        # Fresh samples since the last optimizer step, with their stores.
        self._fresh: list[tuple[SeqStore, StepSample]] = []
        self._fresh_rows = 0

        self._enabled = True
        self._micro_max = MICRO_BATCH_MAX
        self._generation = cfg.start_generation
        self._promoted_path: str | None = None

        self._opt_steps = 0
        self._rows_seen = 0
        self._labels_seen = 0
        self._loss_ema: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def run(self) -> int:
        """Main loop; returns the process exit code."""
        if not self._validate_ring_identity():
            return 1
        try:
            self._load_mirror()
        except Exception as e:
            self._control.log(f"mirror load failed: {e!r}; exiting")
            return 1
        if not self._validate_ring_dims():
            return 1

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

    def _validate_ring_identity(self) -> bool:
        """Refuse to train against a ring produced by a different drafter."""
        sha = self.cfg.drafter_sha256
        if not sha:
            self._control.log("no --drafter-sha256 given; skipping ring "
                              "identity check")
            return True
        try:
            want = int(sha[:16], 16)
        except ValueError:
            self._control.log(f"malformed --drafter-sha256 {sha!r}; exiting")
            return False
        got = self._ring.info.drafter_hash
        if got != want:
            self._control.log(
                f"ring drafter_hash {got:#018x} != drafter GGUF "
                f"{want:#018x} ({sha[:16]}…); refusing to train — exiting")
            return False
        return True

    def _load_mirror(self) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        t0 = time.monotonic()
        self._mirror = DrafterMirror.from_gguf(
            self.cfg.drafter_gguf, self.cfg.target_gguf, self.cfg.rank,
            self.cfg.alpha, self._device, self._dtype)
        dt = time.monotonic() - t0
        m = self._mirror
        self._control.log(
            f"mirror loaded in {dt:.1f}s: device={self._device} "
            f"dtype={self._dtype} layers={m.dims.n_layer} "
            f"hidden={m.dims.hidden} fc_in={m.dims.fc_in} vocab={m.vocab} "
            f"block={m.block_size} rank={self.cfg.rank}")
        self._warm_start()
        self._make_optimizer()

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
            self._mirror.load_lora_state_numpy(load_file(path))
            # The engine serves this generation: protect it from GC and make
            # it the rollback target until the first promote arrives.
            self._promoted_path = path
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
                f"{self._mirror.dims.fc_in}; incompatible capture — exiting")
            return False
        if info.block_size != self._mirror.block_size:
            self._control.log(
                f"warning: ring block_size {info.block_size} != mirror "
                f"{self._mirror.block_size}; scoring the overlap only")
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
                self._control.log(f"promote: gen {gen} is now the baseline")
            elif cmd == "rollback":
                self._rollback(gen)
            elif cmd == "disable":
                self._disable()
            else:
                self._control.log(f"unknown command {cmd!r}; ignored")

    def _rollback(self, gen: int) -> None:
        """Quarantined generation: retreat to the last promoted weights."""
        assert self._mirror is not None
        try:
            if self._promoted_path and os.path.exists(self._promoted_path):
                self._mirror.load_lora_state_numpy(
                    load_file(self._promoted_path))
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

    def _disable(self) -> None:
        """Stop training/exporting but keep draining the ring."""
        self._enabled = False
        self._seqs.clear()
        self._reservoir.clear()
        self._reservoir_row_count = 0
        self._fresh.clear()
        self._fresh_rows = 0
        self._control.log("training disabled; ring consumption continues")

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
        start = max(sample.pos - self.cfg.train_ctx, store.first_pos)
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
        for store, sample in self._fresh:
            if len(batch) >= cap:
                break
            if store.evicted or not sample.has_reject:
                continue
            start = self._window_start(store, sample)
            if start is not None:
                batch.append((store, sample, start))
        n_res = len(self._reservoir)
        attempts = 0
        while len(batch) < cap and n_res > 0 and attempts < 4 * cap:
            attempts += 1
            # Density ∝ index: newest retired sequences are drawn most.
            idx = min(int(math.sqrt(random.random()) * n_res), n_res - 1)
            store = self._reservoir[idx]
            sample = random.choice(store.steps)
            start = self._window_start(store, sample)
            if start is not None:
                batch.append((store, sample, start))
        return batch

    def _train_step(self) -> None:
        """One optimizer step over a micro-batch; never raises."""
        assert self._optimizer is not None
        batch = self._build_batch()
        self._fresh.clear()
        self._fresh_rows = 0
        if not batch:
            return
        try:
            loss = self._optimize(batch)
        except RuntimeError as e:
            self._optimizer.zero_grad(set_to_none=True)
            if "out of memory" in str(e).lower():
                self._micro_max = max(1, self._micro_max // 2)
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                self._control.log(
                    f"OOM in train step; micro-batch cap now "
                    f"{self._micro_max}")
            else:
                self._control.log(f"train step failed: {e!r}; skipped")
            return
        except Exception as e:
            self._optimizer.zero_grad(set_to_none=True)
            self._control.log(f"train step failed: {e!r}; skipped")
            return
        self._loss_ema = (loss if self._loss_ema is None else
                          EMA_DECAY * self._loss_ema
                          + (1.0 - EMA_DECAY) * loss)
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
            (loss * inv).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self._mirror.lora_parameters(), 1.0)
        self._optimizer.step()
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
        feat = torch.from_numpy(bf16_bits_to_f32(rows)).to(self._device)

        q_len = m.block_size
        noise = torch.full((q_len,), m.mask_token_id, dtype=torch.long)
        noise[0] = int(sample.draft_tok[0])
        logits = m.forward(feat, noise.to(self._device))  # [q_len, vocab] f32

        n = min(sample.n_labels, q_len)
        if n < 2:
            return logits.sum() * 0.0  # no trainable rows; keep the graph
        logp = torch.log_softmax(logits[1:n].float(), dim=-1)  # rows 1..n-1
        tgt = torch.from_numpy(
            sample.target_tok[:n - 1].astype(np.int64)).to(self._device)
        ce = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)  # [n-1]

        rej = torch.from_numpy(
            sample.accept_flags[1:n] == 0).to(self._device)
        w = torch.ones(n - 1, dtype=torch.float32, device=self._device)
        w = w.masked_fill(rej, float(self.cfg.reject_weight))
        loss_vec = w * ce

        if (sample.topk_ids is not None and sample.topk_lp is not None
                and self.cfg.kl_lambda > 0.0):
            lp = torch.from_numpy(sample.topk_lp[:n - 1]).to(self._device)
            ids = torch.from_numpy(
                sample.topk_ids[:n - 1].astype(np.int64)).to(self._device)
            log_p = torch.log_softmax(lp, dim=-1)  # renormalized over top-K
            p = log_p.exp()
            q_lp = logp.gather(1, ids)             # [n-1, K]
            kl = (p * (log_p - q_lp)).sum(dim=-1)  # [n-1]
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
                           self.cfg.drafter_sha256, gen, self.cfg.profile)
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
