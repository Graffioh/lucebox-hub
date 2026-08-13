"""Executable spec for odistill.mirror.DrafterMirror (torch drafter mirror).

Both `torch` and `odistill.mirror` are importorskip'd, so this file skips
cleanly until the module lands, then becomes binding. The pinned API:

  DrafterMirror.from_arrays(weights, meta, lm_head, token_embd, rank, alpha)
    weights     dict[str, np.ndarray] keyed by the drafter GGUF tensor names
                (dequantized), torch Linear layout [out_features, in_features]:
                dflash.fc.weight, dflash.hidden_norm.weight,
                output_norm.weight, and per layer blk.<i>.{attn_norm,
                ffn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm,
                attn_k_norm, ffn_gate, ffn_up, ffn_down}.weight
    meta        dict of DraftWeights-style hparams: n_layer, n_head,
                n_head_kv, head_dim, n_embd, n_ff, n_target_layers,
                block_size, rope_theta, rms_eps, mask_token_id
    lm_head     np.ndarray [vocab, hidden] (target-owned, frozen)
    token_embd  np.ndarray [vocab, hidden] (noise-row embeddings)

  m.forward(feat, q_tokens) -> logits [q_len, vocab]
    feat [ctx_len, n_target_layers*hidden] float32, q_tokens [q_len] int64.
    Mirrors build_draft_graph: q_tokens are embedded via token_embd as the
    noise rows; positions follow [ctx_len, ctx_len + q_len).

  m.lora_state_numpy() -> dict[str, np.ndarray] in adapter_export naming,
    shapes == expected_tensor_shapes(dims, rank); lora_b zero-initialized
    (fresh LoRA is bit-exact with the base drafter).
  m.load_lora_state(dict[str, np.ndarray]) -> None
    inverse of lora_state_numpy (warm start / rollback path).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mirror = pytest.importorskip("odistill.mirror",
                             reason="odistill.mirror not implemented yet")

from odistill.adapter_export import DrafterDims, expected_tensor_shapes  # noqa: E402

N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8            # q_dim = 32, kv_dim = 16
HIDDEN = 16
N_FF = 24
N_TARGET_LAYERS = 3     # fc_in = 48
VOCAB = 64
BLOCK = 4
RANK = 4
ALPHA = 8.0

DIMS = DrafterDims(n_layer=N_LAYER, hidden=HIDDEN, q_dim=N_HEAD * HEAD_DIM,
                   kv_dim=N_HEAD_KV * HEAD_DIM, intermediate=N_FF,
                   fc_in=N_TARGET_LAYERS * HIDDEN)

META = {
    "n_layer": N_LAYER,
    "n_head": N_HEAD,
    "n_head_kv": N_HEAD_KV,
    "head_dim": HEAD_DIM,
    "n_embd": HIDDEN,
    "n_ff": N_FF,
    "n_target_layers": N_TARGET_LAYERS,
    "block_size": BLOCK,
    "rope_theta": 10000.0,
    "rms_eps": 1e-6,
    "mask_token_id": 0,
}


def synthetic_weights(seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    q_dim = N_HEAD * HEAD_DIM
    kv_dim = N_HEAD_KV * HEAD_DIM
    weights: dict[str, np.ndarray] = {
        "dflash.fc.weight": w(HIDDEN, N_TARGET_LAYERS * HIDDEN),
        "dflash.hidden_norm.weight": np.ones(HIDDEN, dtype=np.float32),
        "output_norm.weight": np.ones(HIDDEN, dtype=np.float32),
    }
    for i in range(N_LAYER):
        weights.update({
            f"blk.{i}.attn_norm.weight": np.ones(HIDDEN, dtype=np.float32),
            f"blk.{i}.ffn_norm.weight": np.ones(HIDDEN, dtype=np.float32),
            f"blk.{i}.attn_q.weight": w(q_dim, HIDDEN),
            f"blk.{i}.attn_k.weight": w(kv_dim, HIDDEN),
            f"blk.{i}.attn_v.weight": w(kv_dim, HIDDEN),
            f"blk.{i}.attn_output.weight": w(HIDDEN, q_dim),
            f"blk.{i}.attn_q_norm.weight": np.ones(HEAD_DIM, dtype=np.float32),
            f"blk.{i}.attn_k_norm.weight": np.ones(HEAD_DIM, dtype=np.float32),
            f"blk.{i}.ffn_gate.weight": w(N_FF, HIDDEN),
            f"blk.{i}.ffn_up.weight": w(N_FF, HIDDEN),
            f"blk.{i}.ffn_down.weight": w(HIDDEN, N_FF),
        })
    lm_head = w(VOCAB, HIDDEN)
    token_embd = w(VOCAB, HIDDEN)
    return weights, lm_head, token_embd


@pytest.fixture
def tiny_mirror():
    weights, lm_head, token_embd = synthetic_weights()
    return mirror.DrafterMirror.from_arrays(weights, META, lm_head, token_embd,
                                            rank=RANK, alpha=ALPHA)


def forward_inputs(ctx_len: int = 6, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(ctx_len, N_TARGET_LAYERS * HIDDEN, generator=g)
    q_tokens = torch.randint(0, VOCAB, (BLOCK,), generator=g)
    return feat, q_tokens


def test_forward_shape_and_determinism(tiny_mirror):
    feat, q_tokens = forward_inputs()
    out1 = tiny_mirror.forward(feat, q_tokens)
    assert tuple(out1.shape) == (BLOCK, VOCAB)
    assert torch.isfinite(out1.float()).all()
    out2 = tiny_mirror.forward(feat, q_tokens)
    torch.testing.assert_close(out1, out2)


def test_fresh_lora_is_identity(tiny_mirror):
    """B=0 at init means the LoRA delta is exactly zero: a fresh mirror must
    match the no-LoRA drafter, and randomizing A alone must change nothing."""
    feat, q_tokens = forward_inputs()
    base = tiny_mirror.forward(feat, q_tokens)

    state = tiny_mirror.lora_state_numpy()
    for name, arr in state.items():
        if name.endswith(".lora_b"):
            assert not arr.any(), f"{name} not zero-initialized"

    rng = np.random.default_rng(3)
    for name in state:
        if name.endswith(".lora_a"):
            state[name] = rng.standard_normal(state[name].shape).astype(state[name].dtype)
    tiny_mirror.load_lora_state(state)
    still_base = tiny_mirror.forward(feat, q_tokens)
    torch.testing.assert_close(still_base, base, rtol=1e-3, atol=1e-3)

    # Engage B: the delta must actually reach the logits.
    for name in state:
        if name.endswith(".lora_b"):
            state[name] = (rng.standard_normal(state[name].shape) * 0.5).astype(
                state[name].dtype)
    tiny_mirror.load_lora_state(state)
    perturbed = tiny_mirror.forward(feat, q_tokens)
    assert not torch.allclose(perturbed, base, rtol=1e-3, atol=1e-3)


def test_lora_state_matches_adapter_export_shapes(tiny_mirror):
    state = tiny_mirror.lora_state_numpy()
    expected = expected_tensor_shapes(DIMS, RANK)
    assert set(state) == set(expected)
    for name, want in expected.items():
        assert tuple(state[name].shape) == want, name


def test_tied_target_head_stays_tied_in_mirror():
    weights, lm_head, _token_embd = synthetic_weights()
    m = mirror.DrafterMirror.from_arrays(
        weights, META, lm_head, lm_head, rank=RANK, alpha=ALPHA)
    assert m.lm_head.data_ptr() == m.token_embd.data_ptr()


def test_lora_state_rejects_nonfinite_values(tiny_mirror):
    state = tiny_mirror.lora_state_numpy()
    state["dflash.fc.lora_a"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        tiny_mirror.load_lora_state_numpy(state)


def test_untied_target_weights_use_distinct_storage():
    weights, lm_head, token_embd = synthetic_weights()
    m = mirror.DrafterMirror.from_arrays(
        weights, META, lm_head, token_embd, rank=RANK, alpha=ALPHA)
    assert m.lm_head.data_ptr() != m.token_embd.data_ptr()
