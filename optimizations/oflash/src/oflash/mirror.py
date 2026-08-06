"""Torch mirror of the DFlash drafter + trainable LoRA.

Numerically replicates the legacy one-shot ggml graph in
server/src/draft/draft_graph.cpp build_draft_graph():

  - feature fusion: optional per-capture-slice rms_norm * aux_hidden_norm,
    fc @ target_hidden_cat, rms_norm (eps 1e-6) * hidden_norm
  - per layer: pre-norm; Q from noise rows with per-head q_norm; K/V from
    BOTH the fused context features (optionally pre-normed with the layer's
    attn_norm — context_kv_layer_norm) AND the noise activations through the
    SAME wk/wv; per-head k_norm over the concatenated K; NEOX RoPE at
    absolute positions (context 0..ctx-1, noise ctx..ctx+q_len-1); attention
    non-causal on full layers, SWA layers window the context rows to
    swa_window and mask the noise rows causally; optional softplus attention
    gate; wo + residual; SwiGLU FFN
  - final rms_norm * output_norm, then the TARGET's lm_head

The noise rows attend jointly and non-causally over [context, noise] on full
layers — this mirrors the graph, not an autoregressive transformer.

LoRA (scale alpha/rank) wraps fc, attn_q/k/v, attn_output, ffn_up, ffn_down
— exactly the engine's oflash_lora_mm call sites; ffn_gate and the
target-owned lm_head are excluded. The k/v pairs apply to both context and
noise activations because they wrap the shared wk/wv modules. Base weights
are frozen buffers in the compute dtype (bf16 by default); LoRA params stay
float32 for optimizer stability and are used in f32 inside forward.

Adapter tensor names/shapes match adapter_export.expected_tensor_shapes():
lora_a [rank, in], lora_b [out, rank], delta = scale * B @ (A @ x), B=0 at
init so a fresh adapter is bit-identical to the base drafter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .adapter_export import DrafterDims, expected_tensor_shapes
from .gguf_reader import DEFAULT_MASK_TOKEN_ID, DrafterMeta

RMS_EPS = 1e-6  # DFLASH27B_RMS_EPS


def _coerce_meta(meta: DrafterMeta | Mapping[str, Any]) -> DrafterMeta:
    """Accept a DrafterMeta or a plain hparam dict (tests pass dicts keyed
    like DraftWeights: n_embd for hidden, optional swa/gate/aux fields)."""
    if isinstance(meta, DrafterMeta):
        return meta
    d = dict(meta)
    hidden = int(d.get("hidden", d.get("n_embd", 0)))
    pattern = d.get("sliding_window_pattern")
    return DrafterMeta(
        n_layer=int(d["n_layer"]),
        n_head=int(d["n_head"]),
        n_head_kv=int(d["n_head_kv"]),
        head_dim=int(d["head_dim"]),
        hidden=hidden,
        n_ff=int(d["n_ff"]),
        n_target_layers=int(d["n_target_layers"]),
        block_size=int(d["block_size"]),
        mask_token_id=int(d.get("mask_token_id", DEFAULT_MASK_TOKEN_ID)),
        rope_theta=float(d["rope_theta"]),
        swa_window=int(d.get("swa_window", 0)),
        sliding_window_pattern=(tuple(bool(x) for x in pattern)
                                if pattern is not None else None),
        context_kv_layer_norm=bool(d.get("context_kv_layer_norm", False)),
        has_aux_hidden_norms=bool(d.get("has_aux_hidden_norms", False)),
        rms_eps=float(d.get("rms_eps", RMS_EPS)),
    )


def _rms_mul(x: torch.Tensor, weight: torch.Tensor,
             eps: float = RMS_EPS) -> torch.Tensor:
    """ggml_rms_norm (f32 internals) followed by ggml_mul with `weight`.

    Normalizes over the last dim; `weight` broadcasts from the right, which
    covers [n, hidden] * [hidden] and per-head [n, heads, hd] * [hd].
    """
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


def _rope_neox(x: torch.Tensor, pos: torch.Tensor,
               theta: float) -> torch.Tensor:
    """GGML_ROPE_TYPE_NEOX on x [n, heads, head_dim] at positions pos [n].

    Pair (i, i + hd/2) rotates by pos * theta^(-2i/hd) — identical to
    ggml_rope_ext with n_dims=head_dim, freq_scale=1, ext_factor=0.
    """
    hd = x.shape[-1]
    half = hd // 2
    inv = theta ** (-torch.arange(half, device=x.device,
                                  dtype=torch.float32) * (2.0 / hd))
    ang = pos.to(torch.float32)[:, None] * inv[None, :]     # [n, half]
    cos = ang.cos()[:, None, :]                             # [n, 1, half]
    sin = ang.sin()[:, None, :]
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class _LoraPair(nn.Module):
    """One lora_a/lora_b pair; delta(x) = B @ (A @ x) computed in f32."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank
        self.a = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32))
        self.b = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.float32))
        self.reset()

    def reset(self) -> None:
        """Standard LoRA init: A ~ N(0, 1/r), B = 0 → delta starts at 0."""
        with torch.no_grad():
            self.a.normal_(mean=0.0, std=1.0 / self.rank)
            self.b.zero_()

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.a.t()) @ self.b.t()


def _lora_linear(x: torch.Tensor, w: torch.Tensor,
                 pair: _LoraPair, scale: float) -> torch.Tensor:
    """y = W x + scale * B (A x)  (oflash_lora_mm)."""
    y = F.linear(x, w)
    return y + (pair.delta(x) * scale).to(y.dtype)


def _get_weight(weights: dict[str, np.ndarray], name: str,
                alt: str | None = None) -> np.ndarray:
    if name in weights:
        return weights[name]
    if alt is not None and alt in weights:
        return weights[alt]
    raise KeyError(f"drafter weights: missing {name}")


class _DraftLayer(nn.Module):
    """Frozen weights + LoRA pairs of one drafter decoder layer."""

    def __init__(self, il: int, weights: dict[str, np.ndarray],
                 meta: DrafterMeta, rank: int, dtype: torch.dtype):
        super().__init__()
        blk = f"blk.{il}"
        hidden, q_dim, kv_dim = meta.hidden, meta.q_dim, meta.kv_dim

        def buf(name: str, arr: np.ndarray, want: tuple[int, ...]) -> None:
            if tuple(arr.shape) != want:
                raise ValueError(f"{blk}.{name}: shape {tuple(arr.shape)} "
                                 f"!= expected {want}")
            self.register_buffer(
                name, torch.tensor(np.ascontiguousarray(arr), dtype=dtype),
                persistent=False)

        buf("attn_norm", _get_weight(weights, f"{blk}.attn_norm.weight"),
            (hidden,))
        buf("ffn_norm", _get_weight(weights, f"{blk}.ffn_norm.weight",
                                    f"{blk}.post_attention_norm.weight"),
            (hidden,))
        buf("wq", _get_weight(weights, f"{blk}.attn_q.weight"),
            (q_dim, hidden))
        buf("wk", _get_weight(weights, f"{blk}.attn_k.weight"),
            (kv_dim, hidden))
        buf("wv", _get_weight(weights, f"{blk}.attn_v.weight"),
            (kv_dim, hidden))
        buf("wo", _get_weight(weights, f"{blk}.attn_output.weight"),
            (hidden, q_dim))
        buf("q_norm", _get_weight(weights, f"{blk}.attn_q_norm.weight"),
            (meta.head_dim,))
        buf("k_norm", _get_weight(weights, f"{blk}.attn_k_norm.weight"),
            (meta.head_dim,))
        buf("w_gate", _get_weight(weights, f"{blk}.ffn_gate.weight"),
            (meta.n_ff, hidden))
        buf("w_up", _get_weight(weights, f"{blk}.ffn_up.weight"),
            (meta.n_ff, hidden))
        buf("w_down", _get_weight(weights, f"{blk}.ffn_down.weight"),
            (hidden, meta.n_ff))

        gate = weights.get(f"{blk}.attn_gate.weight")
        self.attn_gate_per_head = False
        if gate is None:
            self.register_buffer("attn_gate", None, persistent=False)
        else:
            if tuple(gate.shape) == (meta.n_head, hidden):
                self.attn_gate_per_head = True
            elif tuple(gate.shape) != (q_dim, hidden):
                raise ValueError(f"{blk}.attn_gate.weight: bad shape "
                                 f"{tuple(gate.shape)}")
            self.register_buffer(
                "attn_gate",
                torch.tensor(np.ascontiguousarray(gate), dtype=dtype),
                persistent=False)

        self.is_swa = meta.layer_is_swa(il)

        self.lora_q = _LoraPair(hidden, q_dim, rank)
        self.lora_k = _LoraPair(hidden, kv_dim, rank)
        self.lora_v = _LoraPair(hidden, kv_dim, rank)
        self.lora_o = _LoraPair(q_dim, hidden, rank)
        self.lora_up = _LoraPair(hidden, meta.n_ff, rank)
        self.lora_down = _LoraPair(meta.n_ff, hidden, rank)


class DrafterMirror(nn.Module):
    """bf16 drafter mirror; only the LoRA pairs are trainable parameters."""

    def __init__(self, weights: dict[str, np.ndarray],
                 meta: DrafterMeta | Mapping[str, Any],
                 lm_head: np.ndarray, token_embd: np.ndarray,
                 rank: int, alpha: float,
                 device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        meta = _coerce_meta(meta)
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if meta.rope_theta <= 0.0:
            raise ValueError("drafter meta has no rope_theta (rope.freq_base "
                             "missing from GGUF); RoPE would be wrong")
        self.meta = meta
        self.rank = rank
        self.alpha = float(alpha)
        self.lora_scale = float(alpha) / rank

        hidden, fc_in = meta.hidden, meta.fc_in

        def buf(name: str, arr: np.ndarray, want: tuple[int, ...]) -> None:
            if tuple(arr.shape) != want:
                raise ValueError(f"{name}: shape {tuple(arr.shape)} != "
                                 f"expected {want}")
            self.register_buffer(
                name, torch.tensor(np.ascontiguousarray(arr), dtype=dtype),
                persistent=False)

        buf("fc", _get_weight(weights, "dflash.fc.weight",
                              "dflash_fc.weight"), (hidden, fc_in))
        buf("hidden_norm", _get_weight(weights, "dflash.hidden_norm.weight",
                                       "dflash_hidden_norm.weight"),
            (hidden,))
        buf("out_norm", _get_weight(weights, "output_norm.weight"), (hidden,))

        if meta.has_aux_hidden_norms:
            aux = [_get_weight(weights, f"dflash.aux_hidden_norm.{i}.weight")
                   for i in range(meta.n_target_layers)]
            buf("aux_hidden_norms", np.stack(aux),
                (meta.n_target_layers, hidden))
        else:
            self.register_buffer("aux_hidden_norms", None, persistent=False)

        lm_head = np.asarray(lm_head)
        token_embd = np.asarray(token_embd)
        if lm_head.ndim != 2 or lm_head.shape[1] != hidden:
            raise ValueError(f"lm_head shape {tuple(lm_head.shape)} != "
                             f"[vocab, {hidden}]")
        if token_embd.shape != lm_head.shape:
            raise ValueError(f"token_embd shape {tuple(token_embd.shape)} != "
                             f"lm_head shape {tuple(lm_head.shape)}")
        buf("lm_head", lm_head, tuple(lm_head.shape))
        buf("token_embd", token_embd, tuple(token_embd.shape))
        self._vocab = int(lm_head.shape[0])

        self.lora_fc = _LoraPair(fc_in, hidden, rank)
        self.layers = nn.ModuleList(
            _DraftLayer(il, weights, meta, rank, dtype)
            for il in range(meta.n_layer))

        self.to(device)

    # ── constructors ─────────────────────────────────────────────────
    @classmethod
    def from_arrays(cls, weights: dict[str, np.ndarray],
                    meta: DrafterMeta | Mapping[str, Any],
                    lm_head: np.ndarray, token_embd: np.ndarray,
                    rank: int, alpha: float,
                    device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.bfloat16) -> DrafterMirror:
        """Build from in-memory arrays (tests, synthetic weights)."""
        return cls(weights, meta, lm_head, token_embd, rank=rank,
                   alpha=alpha, device=device, dtype=dtype)

    @classmethod
    def from_gguf(cls, drafter_path: str, target_path: str,
                  rank: int, alpha: float,
                  device: str | torch.device = "cpu",
                  dtype: torch.dtype = torch.bfloat16) -> DrafterMirror:
        """Dequantize the drafter + target-head GGUFs and build the mirror."""
        from .gguf_reader import load_drafter, load_target_head
        weights, meta = load_drafter(drafter_path)
        lm_head, token_embd, _vocab = load_target_head(target_path)
        return cls(weights, meta, lm_head, token_embd, rank=rank,
                   alpha=alpha, device=device, dtype=dtype)

    # ── properties ───────────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return self.fc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.fc.dtype

    @property
    def dims(self) -> DrafterDims:
        m = self.meta
        return DrafterDims(n_layer=m.n_layer, hidden=m.hidden, q_dim=m.q_dim,
                           kv_dim=m.kv_dim, intermediate=m.n_ff,
                           fc_in=m.fc_in)

    @property
    def mask_token_id(self) -> int:
        return self.meta.mask_token_id

    @property
    def block_size(self) -> int:
        return self.meta.block_size

    @property
    def vocab(self) -> int:
        return self._vocab

    # ── forward ──────────────────────────────────────────────────────
    def forward(self, feat: torch.Tensor,
                noise_ids: torch.Tensor) -> torch.Tensor:
        """One drafter forward: feat [ctx, fc_in] float, noise_ids [q_len]
        int → logits [q_len, vocab] float32."""
        m = self.meta
        feat = torch.as_tensor(feat).to(self.device, self.dtype)
        ids = torch.as_tensor(noise_ids).to(self.device).long()
        if feat.ndim != 2 or feat.shape[1] != m.fc_in:
            raise ValueError(f"feat shape {tuple(feat.shape)} != "
                             f"[ctx, {m.fc_in}]")
        if ids.ndim != 1 or ids.shape[0] == 0 or feat.shape[0] == 0:
            raise ValueError("noise_ids must be non-empty 1-D and ctx >= 1")
        ctx, q_len = feat.shape[0], ids.shape[0]

        # Feature fusion (draft_fuse_features).
        x = feat
        if self.aux_hidden_norms is not None:
            x = _rms_mul(x.view(ctx, m.n_target_layers, m.hidden),
                         self.aux_hidden_norms,
                         eps=m.rms_eps).reshape(ctx, m.fc_in)
        tf = _lora_linear(x, self.fc, self.lora_fc, self.lora_scale)
        tf = _rms_mul(tf, self.hidden_norm, eps=m.rms_eps)

        h = self.token_embd[ids]                       # [q_len, hidden]
        pos_noise = torch.arange(ctx, ctx + q_len, device=self.device)

        for layer in self.layers:
            h = self._layer_forward(layer, h, tf, ctx, pos_noise)

        out = _rms_mul(h, self.out_norm, eps=m.rms_eps)
        return F.linear(out, self.lm_head).float()

    def _layer_forward(self, L: _DraftLayer, h: torch.Tensor,
                       tf: torch.Tensor, ctx: int,
                       pos_noise: torch.Tensor) -> torch.Tensor:
        m = self.meta
        scale = self.lora_scale
        eps = m.rms_eps
        q_len = h.shape[0]

        # Attention pre-norm; Q from noise rows only.
        hn = _rms_mul(h, L.attn_norm, eps=eps)
        q = _lora_linear(hn, L.wq, L.lora_q, scale)
        q = q.view(q_len, m.n_head, m.head_dim)
        q = _rms_mul(q, L.q_norm, eps=eps)
        q = _rope_neox(q, pos_noise, m.rope_theta)

        # SWA layers window the context rows to the last swa_window.
        off = 0
        if L.is_swa and m.swa_window > 0 and ctx > m.swa_window:
            off = ctx - m.swa_window
        tf_l = tf[off:]
        n_ctx = tf_l.shape[0]
        pos_ctx = torch.arange(off, ctx, device=self.device)

        tf_kv = (_rms_mul(tf_l, L.attn_norm, eps=eps)
                 if m.context_kv_layer_norm else tf_l)
        # Same wk/wv (and the same LoRA pair) over ctx features AND noise.
        k = torch.cat([_lora_linear(tf_kv, L.wk, L.lora_k, scale),
                       _lora_linear(hn, L.wk, L.lora_k, scale)], dim=0)
        v = torch.cat([_lora_linear(tf_kv, L.wv, L.lora_v, scale),
                       _lora_linear(hn, L.wv, L.lora_v, scale)], dim=0)
        total = n_ctx + q_len
        k = k.view(total, m.n_head_kv, m.head_dim)
        # per-head k_norm on the CONCATENATED K
        k = _rms_mul(k, L.k_norm, eps=eps)
        k = _rope_neox(k, torch.cat([pos_ctx, pos_noise]), m.rope_theta)
        v = v.view(total, m.n_head_kv, m.head_dim)

        rep = m.n_head // m.n_head_kv
        if rep > 1:  # GQA: q head hh reads kv head hh // rep
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float())
        scores = scores * (1.0 / math.sqrt(m.head_dim))
        if L.is_swa:
            # Windowed ctx rows all visible; noise rows causal. Full layers
            # run unmasked (non-causal over [context, noise]).
            kj = torch.arange(total, device=scores.device)[None, :]
            qi = torch.arange(q_len, device=scores.device)[:, None]
            allowed = (kj < n_ctx) | (kj - n_ctx <= qi)
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        probs = scores.softmax(dim=-1)
        attn = torch.einsum("hqk,khd->qhd", probs, v.float())

        if L.attn_gate is not None:
            gate = F.softplus(F.linear(hn, L.attn_gate).float())
            if L.attn_gate_per_head:
                gate = gate.view(q_len, m.n_head, 1)
            else:
                gate = gate.view(q_len, m.n_head, m.head_dim)
            attn = attn * gate
        attn = attn.reshape(q_len, m.q_dim).to(h.dtype)
        h = h + _lora_linear(attn, L.wo, L.lora_o, scale)

        # SwiGLU FFN; LoRA on up/down only (ffn_gate stays base).
        hf = _rms_mul(h, L.ffn_norm, eps=eps)
        g = F.silu(F.linear(hf, L.w_gate))
        u = _lora_linear(hf, L.w_up, L.lora_up, scale)
        return h + _lora_linear(g * u, L.w_down, L.lora_down, scale)

    # ── LoRA state ───────────────────────────────────────────────────
    def _lora_pairs(self) -> dict[str, _LoraPair]:
        """Adapter tensor prefix → pair, in expected_tensor_shapes order."""
        pairs: dict[str, _LoraPair] = {"dflash.fc": self.lora_fc}
        for i, layer in enumerate(self.layers):
            pairs[f"blk.{i}.attn_q"] = layer.lora_q
            pairs[f"blk.{i}.attn_k"] = layer.lora_k
            pairs[f"blk.{i}.attn_v"] = layer.lora_v
            pairs[f"blk.{i}.attn_output"] = layer.lora_o
            pairs[f"blk.{i}.ffn_up"] = layer.lora_up
            pairs[f"blk.{i}.ffn_down"] = layer.lora_down
        return pairs

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for pair in self._lora_pairs().values()
                for p in (pair.a, pair.b)]

    def lora_state_numpy(self) -> dict[str, np.ndarray]:
        """f32 arrays under exactly the adapter_export tensor names."""
        out: dict[str, np.ndarray] = {}
        for name, pair in self._lora_pairs().items():
            out[f"{name}.lora_a"] = pair.a.detach().cpu().numpy().copy()
            out[f"{name}.lora_b"] = pair.b.detach().cpu().numpy().copy()
        return out

    def load_lora_state_numpy(self, state: dict[str, np.ndarray]) -> None:
        pairs = self._lora_pairs()
        expected = {f"{n}.{s}" for n in pairs for s in ("lora_a", "lora_b")}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(f"lora state mismatch: missing={missing[:3]} "
                             f"extra={extra[:3]}")
        with torch.no_grad():
            for name, pair in pairs.items():
                for suffix, param in (("lora_a", pair.a), ("lora_b", pair.b)):
                    arr = np.asarray(state[f"{name}.{suffix}"],
                                     dtype=np.float32)
                    if tuple(arr.shape) != tuple(param.shape):
                        raise ValueError(
                            f"{name}.{suffix}: shape {tuple(arr.shape)} != "
                            f"{tuple(param.shape)}")
                    param.copy_(torch.from_numpy(arr))

    # Spec-test spelling (tests/test_mirror.py) — same operation.
    load_lora_state = load_lora_state_numpy

    def reset_lora(self) -> None:
        """Zero every B, re-draw every A — delta back to exactly 0."""
        for pair in self._lora_pairs().values():
            pair.reset()


# ── self-test: tiny synthetic mirror, no GGUF files needed ───────────

def _self_test() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hidden, n_head, n_head_kv, head_dim = 8, 2, 1, 4
    n_layer, n_ff, n_tgt, block, vocab = 2, 16, 2, 4, 32
    q_dim, kv_dim, fc_in = n_head * head_dim, n_head_kv * head_dim, n_tgt * hidden
    meta = DrafterMeta(
        n_layer=n_layer, n_head=n_head, n_head_kv=n_head_kv,
        head_dim=head_dim, hidden=hidden, n_ff=n_ff, n_target_layers=n_tgt,
        block_size=block, mask_token_id=7, rope_theta=1e4, swa_window=2,
        sliding_window_pattern=(True, False), context_kv_layer_norm=True,
        has_aux_hidden_norms=True)

    def r(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    w = {
        "dflash.fc.weight": r(hidden, fc_in),
        "dflash.hidden_norm.weight": 1.0 + r(hidden),
        "output_norm.weight": 1.0 + r(hidden),
    }
    for i in range(n_tgt):
        w[f"dflash.aux_hidden_norm.{i}.weight"] = 1.0 + r(hidden)
    for i in range(n_layer):
        w[f"blk.{i}.attn_norm.weight"] = 1.0 + r(hidden)
        w[f"blk.{i}.ffn_norm.weight"] = 1.0 + r(hidden)
        w[f"blk.{i}.attn_q.weight"] = r(q_dim, hidden)
        w[f"blk.{i}.attn_k.weight"] = r(kv_dim, hidden)
        w[f"blk.{i}.attn_v.weight"] = r(kv_dim, hidden)
        w[f"blk.{i}.attn_output.weight"] = r(hidden, q_dim)
        w[f"blk.{i}.attn_gate.weight"] = r(n_head, hidden)
        w[f"blk.{i}.attn_q_norm.weight"] = 1.0 + r(head_dim)
        w[f"blk.{i}.attn_k_norm.weight"] = 1.0 + r(head_dim)
        w[f"blk.{i}.ffn_gate.weight"] = r(n_ff, hidden)
        w[f"blk.{i}.ffn_up.weight"] = r(n_ff, hidden)
        w[f"blk.{i}.ffn_down.weight"] = r(hidden, n_ff)
    lm_head, token_embd = r(vocab, hidden), r(vocab, hidden)

    m = DrafterMirror.from_arrays(w, meta, lm_head, token_embd,
                                  rank=4, alpha=8.0)
    feat = torch.from_numpy(r(6, fc_in))
    ids = torch.tensor([1, meta.mask_token_id, meta.mask_token_id,
                        meta.mask_token_id])

    # (a) forward shape/dtype.
    logits = m(feat, ids)
    assert logits.shape == (4, vocab) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    # (b) B=0 → identical to no-LoRA (a different rank changes nothing).
    m2 = DrafterMirror.from_arrays(w, meta, lm_head, token_embd,
                                   rank=2, alpha=4.0)
    assert torch.equal(logits, m2(feat, ids))

    # State names/shapes match the adapter file contract exactly.
    state = m.lora_state_numpy()
    expected = expected_tensor_shapes(m.dims, 4)
    assert set(state) == set(expected)
    assert all(tuple(state[k].shape) == expected[k] for k in expected)

    # Nonzero B changes logits; reset_lora restores the base function.
    for key in state:
        if key.endswith("lora_b"):
            state[key] = (rng.standard_normal(state[key].shape)
                          * 0.05).astype(np.float32)
    m.load_lora_state_numpy(state)
    assert not torch.equal(logits, m(feat, ids))
    m.reset_lora()
    assert torch.equal(logits, m(feat, ids))

    # Gradients reach the LoRA pairs (B first: dL/dA is 0 while B=0).
    assert [tuple(p.shape) for p in m.lora_parameters()] == \
        [s for k in m._lora_pairs()
         for s in (expected[f"{k}.lora_a"], expected[f"{k}.lora_b"])]
    m(feat, ids).sum().backward()
    grads_b = [pair.b.grad for pair in m._lora_pairs().values()]
    assert all(g is not None for g in grads_b)
    assert any(g.abs().sum() > 0 for g in grads_b)

    print("mirror self-test OK")


if __name__ == "__main__":
    _self_test()
