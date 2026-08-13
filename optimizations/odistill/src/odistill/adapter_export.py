"""Adapter safetensors export — the file format the engine hot-swaps.

Contract: server/src/common/odistill/odistill_format.h (names, shapes, metadata
keys) and odistill_adapter.cpp (refusal rules). Shapes are torch-conventional
row-major: lora_a is [rank, in_features], lora_b is [out_features, rank] —
exactly `lora_A.weight` / `lora_B.weight`. dtype F16 (F32/BF16 also accepted
by the engine). The engine refuses files whose draft/target hashes or rank
mismatch, or whose resolved model semantics or tensor set differs from its
expectation in ANY way, so exports are all-or-nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .identity import require_sha256

# LoRA-targeted projections per ODISTILL.md §5: attention q/k/v/o + MLP
# up/down + the feature-fusion fc. NOT ffn_gate, NOT the (target-owned)
# LM head. Keep in lockstep with odistill_lora_expected_tensors().
LAYER_TARGETS = ("attn_q", "attn_k", "attn_v", "attn_output",
                 "ffn_up", "ffn_down")


@dataclass(frozen=True)
class DrafterDims:
    n_layer: int
    hidden: int
    q_dim: int
    kv_dim: int
    intermediate: int
    fc_in: int  # n_target_layers * hidden


def expected_tensor_shapes(dims: DrafterDims, rank: int) -> dict[str, tuple[int, int]]:
    """Safetensors-shape ([rows, cols] row-major) for every adapter tensor."""
    inout = {
        "attn_q": (dims.hidden, dims.q_dim),
        "attn_k": (dims.hidden, dims.kv_dim),
        "attn_v": (dims.hidden, dims.kv_dim),
        "attn_output": (dims.q_dim, dims.hidden),
        "ffn_up": (dims.hidden, dims.intermediate),
        "ffn_down": (dims.intermediate, dims.hidden),
    }
    shapes: dict[str, tuple[int, int]] = {
        "dflash.fc.lora_a": (rank, dims.fc_in),
        "dflash.fc.lora_b": (dims.hidden, rank),
    }
    for i in range(dims.n_layer):
        for t in LAYER_TARGETS:
            in_dim, out_dim = inout[t]
            shapes[f"blk.{i}.{t}.lora_a"] = (rank, in_dim)
            shapes[f"blk.{i}.{t}.lora_b"] = (out_dim, rank)
    return shapes


def export_adapter(path: str,
                   tensors: dict[str, np.ndarray],
                   dims: DrafterDims,
                   rank: int,
                   alpha: float,
                   drafter_sha256: str,
                   target_sha256: str,
                   drafter_semantics: str,
                   generation: int,
                   profile: str) -> None:
    """Validate against the engine's expectation and write atomically.

    Generations are immutable. Refusing an existing destination prevents a
    restarted trainer from silently replacing an adapter the engine may
    still have resident or queued for validation.
    """
    if os.path.exists(path):
        raise FileExistsError(f"adapter generation already exists: {path}")
    require_sha256(drafter_sha256, "drafter SHA-256")
    require_sha256(target_sha256, "target SHA-256")
    if not drafter_semantics:
        raise ValueError("drafter semantics must not be empty")
    expected = expected_tensor_shapes(dims, rank)
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(f"adapter tensor set mismatch: missing={missing[:3]} "
                         f"extra={extra[:3]}")
    out: dict[str, np.ndarray] = {}
    for name, want in expected.items():
        arr = np.asarray(tensors[name])
        if tuple(arr.shape) != want:
            raise ValueError(
                f"{name}: shape {tuple(arr.shape)} != expected {want}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: adapter contains NaN or infinity")
        with np.errstate(over="ignore", invalid="ignore"):
            packed = np.ascontiguousarray(arr.astype(np.float16))
        if not np.isfinite(packed).all():
            raise ValueError(f"{name}: adapter overflows float16 export")
        out[name] = packed
    metadata = {
        "odistill.format": "3",
        "odistill.drafter_sha256": drafter_sha256,
        "odistill.target_sha256": target_sha256,
        "odistill.drafter_semantics": drafter_semantics,
        "odistill.rank": str(rank),
        # The engine's alpha is f32. Nine significant decimal digits round-trip
        # every f32 value and stay inside the shared validation tolerance.
        "odistill.alpha": format(float(np.float32(alpha)), ".9g"),
        "odistill.generation": str(generation),
        "odistill.profile": profile,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        save_file(out, tmp, metadata=metadata)
        # A same-filesystem hard link is atomic and, unlike os.replace,
        # refuses to clobber a generation that appeared concurrently.
        os.link(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def _validate_adapter_metadata(metadata: dict[str, str],
                               drafter_sha256: str,
                               target_sha256: str,
                               drafter_semantics: str,
                               rank: int,
                               alpha: float,
                               generation: int | None = None,
                               profile: str | None = None) -> None:
    if metadata.get("odistill.format") != "3":
        raise ValueError("adapter format version mismatch")
    require_sha256(drafter_sha256, "drafter SHA-256")
    require_sha256(target_sha256, "target SHA-256")
    if metadata.get("odistill.drafter_sha256") != drafter_sha256:
        raise ValueError("adapter drafter hash mismatch")
    if metadata.get("odistill.target_sha256") != target_sha256:
        raise ValueError("adapter target hash mismatch")
    if metadata.get("odistill.drafter_semantics") != drafter_semantics:
        raise ValueError("adapter drafter semantics mismatch")
    if metadata.get("odistill.rank") != str(rank):
        raise ValueError("adapter rank mismatch")
    try:
        file_alpha = float(metadata.get("odistill.alpha", ""))
    except ValueError as exc:
        raise ValueError("adapter alpha is invalid") from exc
    if (not np.isfinite(file_alpha)
            or abs(file_alpha - alpha) > 1e-6 * max(1.0, abs(alpha))):
        raise ValueError("adapter alpha mismatch")
    if generation is not None:
        try:
            file_generation = int(metadata.get("odistill.generation", ""))
        except ValueError as exc:
            raise ValueError("adapter generation is invalid") from exc
        if file_generation != generation:
            raise ValueError("adapter generation mismatch")
    if profile is not None and metadata.get("odistill.profile") != profile:
        raise ValueError("adapter profile mismatch")


def validate_adapter_metadata(path: str,
                              drafter_sha256: str,
                              target_sha256: str,
                              drafter_semantics: str,
                              rank: int,
                              alpha: float,
                              generation: int | None = None,
                              profile: str | None = None) -> None:
    """Apply the engine's identity checks before a Python warm start."""
    with safe_open(path, framework="numpy") as handle:
        _validate_adapter_metadata(
            handle.metadata() or {}, drafter_sha256, target_sha256,
            drafter_semantics, rank, alpha, generation, profile)


def load_validated_adapter(path: str,
                           drafter_sha256: str,
                           target_sha256: str,
                           drafter_semantics: str,
                           rank: int,
                           alpha: float,
                           generation: int | None = None,
                           profile: str | None = None) -> dict[str, np.ndarray]:
    """Validate metadata and copy tensors through the same open handle."""
    with safe_open(path, framework="numpy") as handle:
        _validate_adapter_metadata(
            handle.metadata() or {}, drafter_sha256, target_sha256,
            drafter_semantics, rank, alpha, generation, profile)
        return {name: handle.get_tensor(name) for name in handle.keys()}


def gc_generations(profile_dir: str, keep: int = 4,
                   protect: str | None = None) -> None:
    """Keep the newest `keep` adapter files (by generation in the name),
    never deleting `protect` (the promoted one)."""
    files = []
    for name in os.listdir(profile_dir):
        if name.startswith("adapter-gen") and name.endswith(".safetensors"):
            try:
                gen = int(name[len("adapter-gen"):-len(".safetensors")])
            except ValueError:
                continue
            files.append((gen, os.path.join(profile_dir, name)))
    files.sort(reverse=True)
    for _, fpath in files[keep:]:
        if protect and os.path.abspath(fpath) == os.path.abspath(protect):
            continue
        try:
            os.remove(fpath)
        except OSError:
            pass


def adapter_path(profile_dir: str, generation: int) -> str:
    return os.path.join(profile_dir, f"adapter-gen{generation}.safetensors")
