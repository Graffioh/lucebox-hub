"""oflash-trainer entry point — the sidecar the engine spawns.

Process contract (server/src/common/oflash/oflash_supervisor.cpp):

  argv:   oflash-trainer <drafter.gguf>
              --ring-name=/lucebox-oflash-<pid>  --out-dir=<profile dir>
              --profile=<name> --rank=N --alpha=F --device=<cpu|ordinal>
              --drafter-sha256=<hex> --target-sha256=<hex>
              --start-generation=N --stream-fd=N
  ready:  one int32 0 on --stream-fd, sent AFTER attaching the ring but
          BEFORE any heavy import (torch) or model load, so server startup
          never blocks on us. Accelerator/model-load failures after ready
          enter drain-only capture mode instead of retrying a large allocation.
  events: newline JSON on --stream-fd afterwards:
              {"event":"swap_ready","path":...,"generation":N}
              {"event":"rollback_ack","generation":N}
              {"event":"training_disabled","reason":...}
              {"event":"log","message":...}
  stdin:  newline commands from the engine:
              promote <gen> | rollback <gen> | disable | quit

Device selection happens here (HIP_VISIBLE_DEVICES) before importing torch —
the engine deliberately does not manipulate the child environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import struct
import sys
import threading

from .identity import (require_sha256, resolved_drafter_semantics,
                       sha256_file)


def _finite_positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _swa_pattern(raw: str) -> tuple[bool, ...]:
    values = raw.split(",")
    if not values or any(value not in ("0", "1") for value in values):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of 0/1 values")
    return tuple(value == "1" for value in values)


def _sha256(raw: str) -> str:
    try:
        return require_sha256(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("drafter", help="drafter GGUF path")
    ap.add_argument("--ring-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=_finite_positive_float, default=32.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--drafter-sha256", type=_sha256)
    ap.add_argument("--target-sha256", type=_sha256)
    ap.add_argument("--start-generation", type=int, default=0)
    ap.add_argument("--stream-fd", type=int, default=-1)
    ap.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"),
                    default="auto",
                    help="mirror compute dtype; auto uses fp16 on GPU and "
                         "fp32 on CPU")
    ap.add_argument("--target", default="",
                    help="target GGUF for lm_head/token_embd; defaults to "
                         "the value stored in <out-dir>/trainer.json or "
                         "$OFLASH_TARGET_GGUF")
    # Resolved engine metadata. Integrated launches always pass these after
    # applying legacy GGUF overrides; direct trainer invocations may omit
    # them and use the file metadata unchanged.
    ap.add_argument("--resolved-rope-theta", type=_finite_positive_float)
    ap.add_argument("--resolved-swa-window", type=_nonnegative_int)
    ap.add_argument("--resolved-swa-pattern", type=_swa_pattern)
    ap.add_argument("--resolved-mask-token-id", type=_nonnegative_int)
    ap.add_argument("--drafter-semantics", default="",
                    help="engine-computed resolved drafter identity; direct "
                         "invocations derive it from the resolved metadata")
    # Training knobs (OFLASH.md §5 defaults).
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-lambda", type=float, default=0.5)
    ap.add_argument("--reject-weight", type=float, default=3.0)
    ap.add_argument("--batch-rows", type=int, default=128)
    ap.add_argument("--export-every", type=int, default=8)
    ap.add_argument("--train-ctx", type=int, default=128)
    ap.add_argument("--reservoir-rows", type=int, default=10_000)
    ap.add_argument("--keep-generations", type=int, default=4)
    args = ap.parse_args(argv)
    if ((args.resolved_swa_window is None)
            != (args.resolved_swa_pattern is None)):
        ap.error("--resolved-swa-window and --resolved-swa-pattern must be "
                 "provided together")
    resolved = (
        args.resolved_rope_theta,
        args.resolved_swa_window,
        args.resolved_swa_pattern,
        args.resolved_mask_token_id,
    )
    if not args.drafter_semantics and all(value is not None
                                           for value in resolved):
        try:
            args.drafter_semantics = resolved_drafter_semantics(*resolved)
        except ValueError as exc:
            ap.error(str(exc))
    return args


class Control:
    """Engine-facing I/O: ready handshake, events out, commands in."""

    def __init__(self, stream_fd: int):
        self._fd = stream_fd
        self._lock = threading.Lock()
        self.commands: "queue.Queue[tuple[str, int]]" = queue.Queue()
        self._stdin_thread = threading.Thread(target=self._read_stdin,
                                              daemon=True)
        self._stdin_thread.start()

    def send_ready(self) -> None:
        if self._fd >= 0:
            os.write(self._fd, struct.pack("<i", 0))

    def send_event(self, event: str, **fields) -> None:
        payload = {"event": event, **fields}
        line = (json.dumps(payload) + "\n").encode()
        with self._lock:
            if self._fd >= 0:
                try:
                    os.write(self._fd, line)
                except OSError:
                    pass  # engine gone; the stdin EOF path exits us
            else:
                sys.stderr.write(line.decode())

    def log(self, message: str) -> None:
        self.send_event("log", message=message)

    def _read_stdin(self) -> None:
        for raw in sys.stdin:
            parts = raw.strip().split()
            if not parts:
                continue
            cmd = parts[0]
            gen = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            self.commands.put((cmd, gen))
            if cmd == "quit":
                return
        # stdin EOF = engine closed our pipe = shut down.
        self.commands.put(("quit", 0))


def resolve_device(device: str) -> None:
    """Scope GPU visibility BEFORE torch is imported."""
    if device == "cpu":
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device:
        os.environ["HIP_VISIBLE_DEVICES"] = device
        os.environ["CUDA_VISIBLE_DEVICES"] = device


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    resolve_device(args.device)

    # Ring first: cheap, and the ready handshake gates only on it.
    ring_path = args.ring_name
    if not ring_path.startswith("/dev/shm/"):
        ring_path = "/dev/shm/" + ring_path.lstrip("/")
    from .ring_format import RingReader  # numpy only — safe pre-ready
    try:
        ring = RingReader(ring_path)
    except (OSError, RuntimeError) as e:
        sys.stderr.write(f"[oflash-trainer] ring attach failed: {e}\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        return 1

    target = args.target or os.environ.get("OFLASH_TARGET_GGUF", "")
    try:
        if not target:
            cfg_path = os.path.join(args.out_dir, "trainer.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    target = json.load(f).get("target_gguf", "")
    except (OSError, ValueError, TypeError) as e:
        sys.stderr.write(f"[oflash-trainer] target config failed: {e}\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        ring.close()
        return 1
    if not isinstance(target, str) or not target or not os.path.exists(target):
        sys.stderr.write(
            "[oflash-trainer] no target GGUF configured (--target / "
            "$OFLASH_TARGET_GGUF / trainer.json target_gguf)\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        ring.close()
        return 1
    if not os.path.exists(args.drafter):
        sys.stderr.write(
            f"[oflash-trainer] drafter GGUF not found: {args.drafter}\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        ring.close()
        return 1
    if args.train_ctx < 64:  # keep in sync with trainer.MIN_WINDOW_ROWS
        sys.stderr.write(
            f"[oflash-trainer] --train-ctx {args.train_ctx} must be >= 64\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        ring.close()
        return 1
    if args.batch_rows <= 0 or args.export_every <= 0 \
            or args.reservoir_rows <= 0 or args.keep_generations <= 0:
        sys.stderr.write(
            "[oflash-trainer] batch/export/reservoir/generation limits must "
            "be positive\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        ring.close()
        return 1
    try:
        supplied = [
            ("drafter", args.drafter, args.drafter_sha256),
            ("target", target, args.target_sha256),
        ]
        if args.stream_fd >= 0 and any(
                digest is None for _, _, digest in supplied):
            raise ValueError(
                "engine launch omitted required model SHA-256 identity")
        # The integrated child trusts the engine, which just fingerprinted the
        # already-loaded models. Direct mode verifies even supplied hashes so
        # a mistyped path cannot mislabel training against another target.
        for label, path, supplied_digest in supplied:
            if args.stream_fd >= 0:
                continue
            sys.stderr.write(
                f"[oflash-trainer] verifying {label} GGUF for adapter "
                f"identity: {path}\n")
            actual_digest = sha256_file(path)
            if supplied_digest is not None and supplied_digest != actual_digest:
                raise ValueError(
                    f"{label} SHA-256 does not match {path}")
            if label == "drafter":
                args.drafter_sha256 = actual_digest
            else:
                args.target_sha256 = actual_digest
    except (OSError, ValueError) as e:
        sys.stderr.write(f"[oflash-trainer] model identity failed: {e}\n")
        if args.stream_fd >= 0:
            os.write(args.stream_fd, struct.pack("<i", -1))
        ring.close()
        return 1
    control = Control(args.stream_fd)
    control.send_ready()

    # Heavy imports only after ready.
    try:
        from .trainer import Trainer, TrainerConfig
    except ImportError as e:
        control.log(f"train extras missing ({e}); install "
                    "'oflash[train]' — exiting")
        return 1

    cfg = TrainerConfig(
        drafter_gguf=args.drafter,
        target_gguf=target,
        out_dir=args.out_dir,
        profile=args.profile,
        rank=args.rank,
        alpha=args.alpha,
        drafter_sha256=args.drafter_sha256,
        target_sha256=args.target_sha256,
        start_generation=args.start_generation,
        requested_device=args.device,
        dtype=args.dtype,
        resolved_rope_theta=args.resolved_rope_theta,
        resolved_swa_window=args.resolved_swa_window,
        resolved_swa_pattern=args.resolved_swa_pattern,
        resolved_mask_token_id=args.resolved_mask_token_id,
        drafter_semantics=args.drafter_semantics,
        lr=args.lr,
        kl_lambda=args.kl_lambda,
        reject_weight=args.reject_weight,
        batch_rows=args.batch_rows,
        export_every=args.export_every,
        train_ctx=args.train_ctx,
        reservoir_rows=args.reservoir_rows,
        keep_generations=args.keep_generations,
    )
    trainer = Trainer(cfg, ring, control)
    return trainer.run()


if __name__ == "__main__":
    raise SystemExit(main())
