"""Luce ODistill — online DFlash drafter adaptation (trainer sidecar).

Tooling (this package) runs outside the inference daemon: it consumes the
capture ring the engine exposes in shared memory, fine-tunes a LoRA on a
low-precision mirror of the drafter, and exports adapter files the engine
hot-swaps.
The engine half lives in ../../server/ (src/common/odistill/); the only
contracts between the two are the ring layout and the adapter safetensors
format, both defined in server/src/common/odistill/odistill_format.h and
mirrored here in ring_format.py / adapter_export.py.
"""

__version__ = "0.1.0"
