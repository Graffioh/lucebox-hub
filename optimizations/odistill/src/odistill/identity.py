"""Lightweight model-identity helpers shared before torch is imported."""

from __future__ import annotations

import hashlib
import re
import struct


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def require_sha256(value: str, label: str = "SHA-256") -> str:
    """Return a canonical digest or reject partial/uppercase identities."""
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly 64 lowercase hex characters")
    return value


def sha256_file(path: str) -> str:
    """Hash a model file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64(value: str) -> int:
    """Stable 64-bit tag used by the engine for ring/profile identities."""
    digest = 14695981039346656037
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return digest


def resolved_drafter_semantics(
        rope_theta: float,
        swa_window: int,
        swa_pattern: tuple[bool, ...],
        mask_token_id: int,
        ) -> str:
    """Build the engine's canonical post-override drafter identity."""
    try:
        rope_bits = struct.unpack("<I", struct.pack("<f", rope_theta))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("RoPE theta is not representable as f32") from exc
    pattern = "".join("1" if enabled else "0" for enabled in swa_pattern)
    return (f"v1;rope={rope_bits:08x};swa={swa_window};"
            f"pattern={pattern};mask={mask_token_id}")
