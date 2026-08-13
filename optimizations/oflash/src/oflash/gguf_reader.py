"""GGUF loading + dequantization for the OFlash trainer sidecar.

The trainer builds its low-precision drafter mirror by dequantizing the same GGUFs the
engine serves (OFLASH.md §11/§12) — no safetensors drafter ships on this box.
Two loaders live here:

  load_drafter(path)      -> (dense weight dict, DrafterMeta) for the DFlash
                             drafter GGUF (arch "qwen35-dflash-draft"). Key
                             handling mirrors server/src/draft/
                             draft_gguf_loader.cpp: metadata fills hparams,
                             the weights are ground truth for derived counts.
  load_target_head(path)  -> dequantized lm_head + token embeddings of the
                             target GGUF (the drafter shares the target's
                             head and embeddings; draft_graph.cpp §4).

Requires the `gguf` package (>= 0.10; the oflash[train] extra). Imports are
lazy so ring-only deployments never pay for it; DrafterMeta itself is
importable with numpy alone (mirror.py's synthetic-weights tests rely on
that).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct
from typing import Any

import numpy as np

# Fallback hparams mirroring server/include/dflash27b.h. Converter-produced
# GGUFs carry every key; these only fill gaps in hand-rolled test files.
DEFAULT_HIDDEN = 5120
DEFAULT_N_LAYER = 5
DEFAULT_N_HEAD = 32
DEFAULT_N_HEAD_KV = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_N_FF = 17408
DEFAULT_BLOCK_SIZE = 16
DEFAULT_N_TARGET_LAYERS = 5
DEFAULT_MASK_TOKEN_ID = 248070

DRAFTER_ARCHS = ("qwen35-dflash-draft", "dflash-draft")


@dataclass(frozen=True)
class DrafterMeta:
    """Drafter hyperparameters as the engine's GGUF loader resolves them."""

    n_layer: int
    n_head: int
    n_head_kv: int
    head_dim: int
    hidden: int            # n_embd
    n_ff: int
    n_target_layers: int   # capture slices fused by dflash.fc
    block_size: int        # draft q_len
    mask_token_id: int
    rope_theta: float
    swa_window: int
    # Per-layer SWA flags ('<arch>.attention.sliding_window_pattern');
    # None when the GGUF carries no pattern (all layers full, non-causal).
    sliding_window_pattern: tuple[bool, ...] | None
    context_kv_layer_norm: bool
    has_aux_hidden_norms: bool
    rms_eps: float = 1e-6  # DFLASH27B_RMS_EPS; not a GGUF key, engine constant

    @property
    def q_dim(self) -> int:
        return self.n_head * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_head_kv * self.head_dim

    @property
    def fc_in(self) -> int:
        return self.n_target_layers * self.hidden

    def layer_is_swa(self, il: int) -> bool:
        if self.sliding_window_pattern is None:
            return False
        return self.sliding_window_pattern[il]


def apply_resolved_drafter_meta(
        meta: DrafterMeta,
        *,
        rope_theta: float | None = None,
        swa_window: int | None = None,
        sliding_window_pattern: tuple[bool, ...] | None = None,
        mask_token_id: int | None = None,
        ) -> DrafterMeta:
    """Apply the engine's post-load draft metadata to the trainer mirror.

    Older Qwen3.6 GGUFs need a runtime SWA override.  The engine passes the
    fully resolved values after applying that override so online training
    cannot silently mirror the stale on-disk attention layout.
    """
    resolved_rope = meta.rope_theta if rope_theta is None else float(rope_theta)
    resolved_window = meta.swa_window if swa_window is None else int(swa_window)
    resolved_pattern = (meta.sliding_window_pattern
                        if sliding_window_pattern is None
                        else tuple(bool(v) for v in sliding_window_pattern))
    resolved_mask = (meta.mask_token_id if mask_token_id is None
                     else int(mask_token_id))
    if not np.isfinite(resolved_rope) or resolved_rope <= 0.0:
        raise ValueError(f"resolved rope theta must be finite and positive, got "
                         f"{resolved_rope!r}")
    if resolved_window < 0:
        raise ValueError(f"resolved SWA window must be non-negative, got "
                         f"{resolved_window}")
    if resolved_pattern is not None and len(resolved_pattern) != meta.n_layer:
        raise ValueError(
            f"resolved SWA pattern has {len(resolved_pattern)} layers, expected "
            f"{meta.n_layer}")
    if resolved_pattern is not None and any(resolved_pattern) \
            and resolved_window <= 0:
        raise ValueError("resolved SWA pattern enables layers with window=0")
    if resolved_mask < 0:
        raise ValueError(f"resolved mask token must be non-negative, got "
                         f"{resolved_mask}")
    return replace(meta, rope_theta=resolved_rope,
                   swa_window=resolved_window,
                   sliding_window_pattern=resolved_pattern,
                   mask_token_id=resolved_mask)


def drafter_semantics(meta: DrafterMeta) -> str:
    """Canonical identity for runtime-resolved draft behavior.

    The engine writes this into adapter metadata and profile namespaces. RoPE
    is represented by its exact IEEE-754 f32 bits, avoiding language-specific
    decimal formatting.
    """
    rope = struct.unpack("<I", struct.pack("<f", np.float32(meta.rope_theta)))[0]
    pattern = meta.sliding_window_pattern
    if pattern is None:
        pattern = (False,) * meta.n_layer
    if len(pattern) != meta.n_layer:
        raise ValueError(
            f"SWA pattern has {len(pattern)} layers, expected {meta.n_layer}")
    bits = "".join("1" if enabled else "0" for enabled in pattern)
    return (f"v1;rope={rope:08x};swa={meta.swa_window};"
            f"pattern={bits};mask={meta.mask_token_id}")


# ── GGUF access helpers ──────────────────────────────────────────────

def _scalar(reader: Any, key: str, default: Any) -> Any:
    field = reader.get_field(key)
    if field is None:
        return default
    return field.contents()


def _dequantize(t: Any, out_dtype: Any = np.float32,
                chunk_rows: int = 4096) -> np.ndarray:
    """Dequantize a ReaderTensor to a dense array of `out_dtype`.

    2-D tensors are processed in row chunks so the peak transient is one
    chunk of f32, not the whole tensor twice (the target lm_head is
    vocab x hidden ≈ 5 GB in f32).
    """
    from gguf.quants import dequantize

    data = t.data
    if data.ndim != 2:
        return np.array(dequantize(data, t.tensor_type), dtype=out_dtype)
    n_rows = int(data.shape[0])
    n_cols = int(t.shape[0])  # ggml ne[0] = logical fastest-varying dim
    out = np.empty((n_rows, n_cols), dtype=out_dtype)
    for i in range(0, n_rows, chunk_rows):
        j = min(i + chunk_rows, n_rows)
        out[i:j] = dequantize(data[i:j], t.tensor_type)
    return out


def _check_shape(arr: np.ndarray, want: tuple[int, ...], name: str) -> None:
    if tuple(arr.shape) != want:
        raise ValueError(
            f"drafter GGUF tensor {name}: shape {tuple(arr.shape)} != expected {want}")


# ── Drafter ──────────────────────────────────────────────────────────

def load_drafter(path: str,
                 dtype: Any = np.float32,
                 ) -> tuple[dict[str, np.ndarray], DrafterMeta]:
    """Load and dequantize the DFlash drafter GGUF.

    Returns (weights, meta). `weights` maps canonical GGUF tensor names
    ("dflash.fc.weight", "blk.<i>.attn_q.weight", ...) to float32 arrays in
    torch/row-major orientation [out_features, in_features] (numpy sees ggml
    ne reversed, which is exactly that). Alternate spellings the engine
    accepts (dflash_fc.weight, blk.<i>.post_attention_norm.weight) are
    canonicalized. Pass dtype=np.float16 for a low-peak-memory GPU load; the
    mirror converts to its requested torch dtype while copying each tensor.
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)

    arch = _scalar(reader, "general.architecture", None)
    if arch is None:
        raise ValueError(f"{path}: missing general.architecture")
    if arch not in DRAFTER_ARCHS and not arch.endswith("-dflash-draft"):
        raise ValueError(f"{path}: unexpected draft arch {arch!r} "
                         f"(expected one of {DRAFTER_ARCHS})")

    def u32(suffix: str, default: int) -> int:
        return int(_scalar(reader, f"{arch}.{suffix}", default))

    def f32(suffix: str, default: float) -> float:
        return float(_scalar(reader, f"{arch}.{suffix}", default))

    hidden = u32("embedding_length", DEFAULT_HIDDEN)
    n_layer = u32("block_count", DEFAULT_N_LAYER)
    n_ff = u32("feed_forward_length", DEFAULT_N_FF)
    n_head = u32("attention.head_count", DEFAULT_N_HEAD)
    n_head_kv = u32("attention.head_count_kv", DEFAULT_N_HEAD_KV)
    head_dim = u32("attention.key_length", DEFAULT_HEAD_DIM)
    block_size = u32("dflash.block_size", DEFAULT_BLOCK_SIZE)
    mask_token_id = u32("dflash.mask_token_id", DEFAULT_MASK_TOKEN_ID)
    rope_theta = f32("rope.freq_base", 0.0)
    swa_window = u32("attention.sliding_window", 0)

    if min(hidden, n_layer, n_ff, n_head, n_head_kv, head_dim, block_size) <= 0:
        raise ValueError(f"{path}: non-positive hparam in metadata")
    if n_head % n_head_kv != 0 or n_head_kv > n_head:
        raise ValueError(f"{path}: n_head={n_head} not divisible by "
                         f"n_head_kv={n_head_kv}")

    # n_target_layers: scalar KV, then target_layer_ids length, then
    # n_target_features/hidden — and finally the fc weight overrides all
    # (draft_gguf_loader.cpp: "the weights are ground truth").
    n_tgt = u32("dflash.n_target_layers", 0)
    layer_ids = _scalar(reader, f"{arch}.dflash.target_layer_ids", None)
    if n_tgt == 0 and isinstance(layer_ids, list):
        n_tgt = len(layer_ids)
    if n_tgt == 0:
        n_feat = u32("dflash.n_target_features", 0)
        if n_feat > 0 and n_feat % hidden == 0:
            n_tgt = n_feat // hidden

    pattern_raw = _scalar(reader, f"{arch}.attention.sliding_window_pattern", None)
    pattern: tuple[bool, ...] | None = None
    if isinstance(pattern_raw, list):
        flags = [bool(v) for v in pattern_raw[:n_layer]]
        flags += [False] * (n_layer - len(flags))
        pattern = tuple(flags)

    tmap = {t.name: t for t in reader.tensors}

    def pick(name: str, alt: str | None = None) -> Any:
        t = tmap.get(name)
        if t is None and alt is not None:
            t = tmap.get(alt)
        if t is None:
            raise ValueError(f"{path}: missing tensor {name}"
                             + (f" (or {alt})" if alt else ""))
        return t

    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    weights: dict[str, np.ndarray] = {}

    fc = _dequantize(pick("dflash.fc.weight", "dflash_fc.weight"), dtype)
    if fc.ndim != 2 or fc.shape[0] != hidden or fc.shape[1] % hidden != 0:
        raise ValueError(f"{path}: dflash.fc.weight shape {tuple(fc.shape)} "
                         f"inconsistent with hidden={hidden}")
    derived_tgt = int(fc.shape[1]) // hidden
    if n_tgt != derived_tgt:
        n_tgt = derived_tgt  # weights win over metadata
    weights["dflash.fc.weight"] = fc

    hidden_norm = _dequantize(
        pick("dflash.hidden_norm.weight", "dflash_hidden_norm.weight"), dtype)
    _check_shape(hidden_norm, (hidden,), "dflash.hidden_norm.weight")
    weights["dflash.hidden_norm.weight"] = hidden_norm

    out_norm = _dequantize(pick("output_norm.weight"), dtype)
    _check_shape(out_norm, (hidden,), "output_norm.weight")
    weights["output_norm.weight"] = out_norm

    n_aux = 0
    while f"dflash.aux_hidden_norm.{n_aux}.weight" in tmap:
        name = f"dflash.aux_hidden_norm.{n_aux}.weight"
        aux = _dequantize(tmap[name], dtype)
        _check_shape(aux, (hidden,), name)
        weights[name] = aux
        n_aux += 1
    if n_aux and n_aux != n_tgt:
        raise ValueError(f"{path}: {n_aux} aux hidden norms != "
                         f"fc-derived capture count {n_tgt}")

    n_gate_layers = 0
    for il in range(n_layer):
        blk = f"blk.{il}"
        shapes = {
            "attn_norm.weight": (hidden,),
            "attn_q.weight": (q_dim, hidden),
            "attn_k.weight": (kv_dim, hidden),
            "attn_v.weight": (kv_dim, hidden),
            "attn_output.weight": (hidden, q_dim),
            "attn_q_norm.weight": (head_dim,),
            "attn_k_norm.weight": (head_dim,),
            "ffn_gate.weight": (n_ff, hidden),
            "ffn_up.weight": (n_ff, hidden),
            "ffn_down.weight": (hidden, n_ff),
        }
        for suffix, want in shapes.items():
            arr = _dequantize(pick(f"{blk}.{suffix}"), dtype)
            _check_shape(arr, want, f"{blk}.{suffix}")
            weights[f"{blk}.{suffix}"] = arr
        ffn_norm = _dequantize(pick(f"{blk}.ffn_norm.weight",
                                    f"{blk}.post_attention_norm.weight"), dtype)
        _check_shape(ffn_norm, (hidden,), f"{blk}.ffn_norm.weight")
        weights[f"{blk}.ffn_norm.weight"] = ffn_norm
        gate_t = tmap.get(f"{blk}.attn_gate.weight")
        if gate_t is not None:
            gate = _dequantize(gate_t, dtype)
            if tuple(gate.shape) not in ((n_head, hidden), (q_dim, hidden)):
                raise ValueError(
                    f"{path}: {blk}.attn_gate.weight shape {tuple(gate.shape)} "
                    f"!= [{n_head},{hidden}] or [{q_dim},{hidden}]")
            weights[f"{blk}.attn_gate.weight"] = gate
            n_gate_layers += 1
    if n_gate_layers not in (0, n_layer):
        raise ValueError(f"{path}: incomplete attention gate tensors: "
                         f"{n_gate_layers}/{n_layer} layers")

    context_kv_layer_norm = (bool(u32("dflash.context_kv_layer_norm", 0))
                             or n_gate_layers > 0 or n_aux > 0)

    meta = DrafterMeta(
        n_layer=n_layer,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        hidden=hidden,
        n_ff=n_ff,
        n_target_layers=n_tgt,
        block_size=block_size,
        mask_token_id=mask_token_id,
        rope_theta=rope_theta,
        swa_window=swa_window,
        sliding_window_pattern=pattern,
        context_kv_layer_norm=context_kv_layer_norm,
        has_aux_hidden_norms=n_aux > 0,
    )
    return weights, meta


# ── Target head ──────────────────────────────────────────────────────

def load_target_head(path: str,
                     dtype: Any = np.float32,
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Dequantize the target GGUF's lm_head and token embeddings.

    Returns (lm_head [vocab, hidden], token_embd [vocab, hidden], vocab).
    'output.weight' falls back to 'token_embd.weight' when absent (tied
    head); in that case both returns are the SAME array — don't mutate.
    Pass dtype=np.float16 to halve resident size (~2.5 GB instead of ~5 GB
    per tensor at vocab 248320 x hidden 5120).
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)
    tmap = {t.name: t for t in reader.tensors}

    te_t = tmap.get("token_embd.weight")
    if te_t is None:
        raise ValueError(f"{path}: missing token_embd.weight")
    token_embd = _dequantize(te_t, dtype)
    if token_embd.ndim != 2:
        raise ValueError(f"{path}: token_embd.weight is not 2-D")

    out_t = tmap.get("output.weight")
    lm_head = token_embd if out_t is None else _dequantize(out_t, dtype)
    if lm_head.shape != token_embd.shape:
        raise ValueError(
            f"{path}: output.weight {tuple(lm_head.shape)} != "
            f"token_embd.weight {tuple(token_embd.shape)}")

    return lm_head, token_embd, int(token_embd.shape[0])
