"""Adapter safetensors export — the file format the engine hot-swaps.

Contract: server/src/common/oflash/oflash_format.h (names, shapes, metadata
keys) and oflash_adapter.cpp (refusal rules). Shapes are torch-conventional
row-major: lora_a is [rank, in_features], lora_b is [out_features, rank] —
exactly `lora_A.weight` / `lora_B.weight`. dtype F16 (F32/BF16 also accepted
by the engine). The engine refuses files whose drafter sha or rank mismatch,
or whose tensor set differs from its expectation in ANY way, so exports are
all-or-nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from safetensors.numpy import save_file

# LoRA-targeted projections per OFLASH.md §5: attention q/k/v/o + MLP
# up/down + the feature-fusion fc. NOT ffn_gate, NOT the (target-owned)
# LM head. Keep in lockstep with oflash_lora_expected_tensors().
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
                   generation: int,
                   profile: str) -> None:
    """Validate against the engine's expectation and write atomically.

    Generations are immutable. Refusing an existing destination prevents a
    restarted trainer from silently replacing an adapter the engine may
    still have resident or queued for validation.
    """
    if os.path.exists(path):
        raise FileExistsError(f"adapter generation already exists: {path}")
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
        "oflash.format": "1",
        "oflash.drafter_sha256": drafter_sha256,
        "oflash.rank": str(rank),
        "oflash.alpha": f"{alpha:g}",
        "oflash.generation": str(generation),
        "oflash.profile": profile,
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
