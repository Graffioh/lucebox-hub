"""oflash-trainer entry point — the sidecar the engine spawns.

Process contract (server/src/common/oflash/oflash_supervisor.cpp):

  argv:   oflash-trainer <drafter.gguf>
              --ring-name=/lucebox-oflash-<pid>  --out-dir=<profile dir>
              --profile=<name> --rank=N --alpha=F --device=<cpu|ordinal>
              --drafter-sha256=<hex> --start-generation=N --stream-fd=N
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
import os
import queue
import struct
import sys
import threading


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("drafter", help="drafter GGUF path")
    ap.add_argument("--ring-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--drafter-sha256", default="")
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
    # Training knobs (OFLASH.md §5 defaults).
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-lambda", type=float, default=0.5)
    ap.add_argument("--reject-weight", type=float, default=3.0)
    ap.add_argument("--batch-rows", type=int, default=128)
    ap.add_argument("--export-every", type=int, default=8)
    ap.add_argument("--train-ctx", type=int, default=128)
    ap.add_argument("--reservoir-rows", type=int, default=10_000)
    ap.add_argument("--keep-generations", type=int, default=4)
    return ap.parse_args(argv)


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
        start_generation=args.start_generation,
        requested_device=args.device,
        dtype=args.dtype,
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
