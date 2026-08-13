"""Resolved drafter metadata shared by the engine and Torch mirror."""

import pytest

from oflash.gguf_reader import (DrafterMeta, apply_resolved_drafter_meta,
                                drafter_semantics)


@pytest.fixture
def legacy_qwen36_meta() -> DrafterMeta:
    return DrafterMeta(
        n_layer=5, n_head=32, n_head_kv=8, head_dim=128,
        hidden=5120, n_ff=17408, n_target_layers=5, block_size=16,
        mask_token_id=248070, rope_theta=1_000_000.0, swa_window=0,
        sliding_window_pattern=None, context_kv_layer_norm=False,
        has_aux_hidden_norms=False)


def test_engine_resolved_swa_metadata_overrides_legacy_gguf(
        legacy_qwen36_meta):
    resolved = apply_resolved_drafter_meta(
        legacy_qwen36_meta, rope_theta=1_000_000.0, swa_window=2048,
        sliding_window_pattern=(True, True, True, True, False),
        mask_token_id=248071)
    assert resolved.rope_theta == 1_000_000.0
    assert resolved.swa_window == 2048
    assert resolved.sliding_window_pattern == (
        True, True, True, True, False)
    assert resolved.mask_token_id == 248071
    assert drafter_semantics(resolved) == (
        "v1;rope=49742400;swa=2048;pattern=11110;mask=248071")
    assert legacy_qwen36_meta.swa_window == 0
    assert legacy_qwen36_meta.sliding_window_pattern is None


def test_engine_resolved_swa_metadata_rejects_inconsistent_layout(
        legacy_qwen36_meta):
    with pytest.raises(ValueError, match="4 layers, expected 5"):
        apply_resolved_drafter_meta(
            legacy_qwen36_meta, swa_window=2048,
            sliding_window_pattern=(True, True, True, False))
    with pytest.raises(ValueError, match="window=0"):
        apply_resolved_drafter_meta(
            legacy_qwen36_meta, swa_window=0,
            sliding_window_pattern=(True, True, True, True, False))
    with pytest.raises(ValueError, match="mask token"):
        apply_resolved_drafter_meta(legacy_qwen36_meta, mask_token_id=-1)


@pytest.mark.parametrize("rope", [0.0, -1.0, float("nan"), float("inf")])
def test_engine_resolved_rope_rejects_invalid_values(
        legacy_qwen36_meta, rope):
    with pytest.raises(ValueError, match="rope theta"):
        apply_resolved_drafter_meta(legacy_qwen36_meta, rope_theta=rope)
