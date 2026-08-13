"""Contract tests for odistill.adapter_export against odistill_format.h §Adapter.

The tensor set/shape expectations are pinned on the C++ side by
server/test/test_odistill_unit.cpp (adapter_expected_tensor_specs, using the
same tiny drafter: n_layer=2, hidden=16, q_dim=32, kv_dim=16, n_ff=24,
fc_in=3*16=48, rank=4 -> 26 tensors). Keep both in lockstep.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file

from odistill.adapter_export import (
    LAYER_TARGETS,
    DrafterDims,
    adapter_path,
    expected_tensor_shapes,
    export_adapter,
    gc_generations,
    load_validated_adapter,
    validate_adapter_metadata,
)

DIMS = DrafterDims(n_layer=2, hidden=16, q_dim=32, kv_dim=16, intermediate=24, fc_in=48)
RANK = 4
SHA = "a" * 64
TARGET_SHA = "b" * 64
SEMANTICS = "v1;rope=49742400;swa=2048;pattern=10;mask=0"


def make_tensors(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in expected_tensor_shapes(DIMS, RANK).items()}


def test_expected_shapes_match_cpp_spec():
    shapes = expected_tensor_shapes(DIMS, RANK)
    # fc pair + 6 pairs per layer x 2 layers = 13 pairs = 26 tensors.
    assert len(shapes) == 26
    want_names = {"dflash.fc.lora_a", "dflash.fc.lora_b"}
    for i in range(2):
        for t in LAYER_TARGETS:
            want_names |= {f"blk.{i}.{t}.lora_a", f"blk.{i}.{t}.lora_b"}
    assert set(shapes) == want_names
    # ffn_gate and the target-owned LM head are NOT adapted.
    assert not any("ffn_gate" in n or "output" == n.split(".")[0] for n in shapes)

    # Dims pinned by the C++ test (safetensors [rows, cols] row-major).
    assert shapes["dflash.fc.lora_a"] == (4, 48)     # [rank, 3*hidden]
    assert shapes["dflash.fc.lora_b"] == (16, 4)     # [hidden, rank]
    assert shapes["blk.1.attn_output.lora_a"] == (4, 32)   # in = q_dim
    assert shapes["blk.1.attn_output.lora_b"] == (16, 4)   # out = hidden
    assert shapes["blk.1.ffn_down.lora_a"] == (4, 24)      # in = intermediate
    assert shapes["blk.1.ffn_down.lora_b"] == (16, 4)      # out = hidden
    assert shapes["blk.0.attn_q.lora_a"] == (4, 16)
    assert shapes["blk.0.attn_q.lora_b"] == (32, 4)        # out = q_dim
    assert shapes["blk.0.attn_k.lora_b"] == (16, 4)        # out = kv_dim
    assert shapes["blk.0.attn_v.lora_b"] == (16, 4)
    assert shapes["blk.0.ffn_up.lora_a"] == (4, 16)
    assert shapes["blk.0.ffn_up.lora_b"] == (24, 4)        # out = intermediate


def test_export_writes_valid_safetensors_and_reloads(tmp_path):
    tensors = make_tensors()
    path = str(tmp_path / "adapter-gen7.safetensors")
    export_adapter(path, tensors, DIMS, RANK, alpha=32.0, drafter_sha256=SHA,
                   target_sha256=TARGET_SHA, drafter_semantics=SEMANTICS,
                   generation=7,
                   profile="default")
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")  # atomic: tmp file replaced

    loaded = load_file(path)
    expected = expected_tensor_shapes(DIMS, RANK)
    assert set(loaded) == set(expected)
    for name, want in expected.items():
        assert loaded[name].shape == want
        assert loaded[name].dtype == np.float16
        np.testing.assert_array_equal(loaded[name],
                                      tensors[name].astype(np.float16))

    # Metadata keys exactly per odistill_format.h §Adapter (all string values).
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata()
    assert meta == {
        "odistill.format": "3",
        "odistill.drafter_sha256": SHA,
        "odistill.target_sha256": TARGET_SHA,
        "odistill.drafter_semantics": SEMANTICS,
        "odistill.rank": "4",
        "odistill.alpha": "32",
        "odistill.generation": "7",
        "odistill.profile": "default",
    }
    validate_adapter_metadata(
        path, SHA, TARGET_SHA, SEMANTICS, RANK, 32.0)
    assert set(load_validated_adapter(
        path, SHA, TARGET_SHA, SEMANTICS, RANK, 32.0)) == set(expected)
    with pytest.raises(ValueError, match="target hash"):
        validate_adapter_metadata(
            path, SHA, "c" * 64, SEMANTICS, RANK, 32.0)
    with pytest.raises(ValueError, match="semantics"):
        validate_adapter_metadata(
            path, SHA, TARGET_SHA, SEMANTICS + "-other", RANK, 32.0)
    with pytest.raises(ValueError, match="rank"):
        validate_adapter_metadata(
            path, SHA, TARGET_SHA, SEMANTICS, RANK + 1, 32.0)


def test_export_requires_resolved_semantics(tmp_path):
    with pytest.raises(ValueError, match="semantics"):
        export_adapter(str(tmp_path / "a.safetensors"), make_tensors(),
                       DIMS, RANK, alpha=32.0, drafter_sha256=SHA,
                       target_sha256=TARGET_SHA, drafter_semantics="",
                       generation=1,
                       profile="default")


@pytest.mark.parametrize("bad_hash", ["", "a" * 16, "A" * 64])
def test_export_requires_complete_canonical_model_hashes(tmp_path, bad_hash):
    with pytest.raises(ValueError, match="64 lowercase hex"):
        export_adapter(str(tmp_path / "a.safetensors"), make_tensors(),
                       DIMS, RANK, alpha=32.0,
                       drafter_sha256=bad_hash,
                       target_sha256=TARGET_SHA,
                       drafter_semantics=SEMANTICS, generation=1,
                       profile="default")


def test_nonround_alpha_and_store_identity_round_trip(tmp_path):
    path = str(tmp_path / "adapter-gen9.safetensors")
    alpha = 1.23456776
    export_adapter(path, make_tensors(), DIMS, RANK, alpha=alpha,
                   drafter_sha256=SHA, target_sha256=TARGET_SHA,
                   drafter_semantics=SEMANTICS, generation=9,
                   profile="coding")
    validate_adapter_metadata(
        path, SHA, TARGET_SHA, SEMANTICS, RANK, alpha,
        generation=9, profile="coding")
    with pytest.raises(ValueError, match="generation"):
        validate_adapter_metadata(
            path, SHA, TARGET_SHA, SEMANTICS, RANK, alpha,
            generation=8, profile="coding")
    with pytest.raises(ValueError, match="profile"):
        validate_adapter_metadata(
            path, SHA, TARGET_SHA, SEMANTICS, RANK, alpha,
            generation=9, profile="other")


def test_export_rejects_shape_mismatch(tmp_path):
    tensors = make_tensors()
    tensors["blk.0.attn_q.lora_a"] = np.zeros((5, 16), dtype=np.float32)  # rank+1
    with pytest.raises(ValueError, match="blk.0.attn_q.lora_a"):
        export_adapter(str(tmp_path / "a.safetensors"), tensors, DIMS, RANK,
                       alpha=32.0, drafter_sha256=SHA,
                       target_sha256=TARGET_SHA,
                       drafter_semantics=SEMANTICS, generation=1,
                       profile="default")
    assert not os.listdir(tmp_path)  # nothing written on failure


def test_export_rejects_missing_and_extra_tensors(tmp_path):
    tensors = make_tensors()
    del tensors["blk.1.ffn_down.lora_b"]
    with pytest.raises(ValueError, match="mismatch"):
        export_adapter(str(tmp_path / "a.safetensors"), tensors, DIMS, RANK,
                       alpha=32.0, drafter_sha256=SHA,
                       target_sha256=TARGET_SHA,
                       drafter_semantics=SEMANTICS, generation=1,
                       profile="default")

    tensors = make_tensors()
    tensors["blk.0.ffn_gate.lora_a"] = np.zeros((4, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        export_adapter(str(tmp_path / "a.safetensors"), tensors, DIMS, RANK,
                       alpha=32.0, drafter_sha256=SHA,
                       target_sha256=TARGET_SHA,
                       drafter_semantics=SEMANTICS, generation=1,
                       profile="default")
    assert not os.listdir(tmp_path)


def test_export_never_overwrites_an_existing_generation(tmp_path):
    path = str(tmp_path / "adapter-gen1.safetensors")
    export_adapter(path, make_tensors(seed=1), DIMS, RANK, alpha=32.0,
                   drafter_sha256=SHA, target_sha256=TARGET_SHA,
                   drafter_semantics=SEMANTICS,
                   generation=1, profile="default")
    original = (tmp_path / "adapter-gen1.safetensors").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        export_adapter(path, make_tensors(seed=2), DIMS, RANK, alpha=32.0,
                       drafter_sha256=SHA, target_sha256=TARGET_SHA,
                       drafter_semantics=SEMANTICS,
                       generation=1, profile="default")

    assert (tmp_path / "adapter-gen1.safetensors").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp.*"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, 70_000.0])
def test_export_rejects_nonfinite_or_float16_overflow(tmp_path, bad):
    tensors = make_tensors()
    tensors["dflash.fc.lora_a"][0, 0] = bad
    path = str(tmp_path / "adapter-gen1.safetensors")

    with pytest.raises(ValueError, match="NaN|infinity|overflows"):
        export_adapter(path, tensors, DIMS, RANK, alpha=32.0,
                       drafter_sha256=SHA, target_sha256=TARGET_SHA,
                       drafter_semantics=SEMANTICS,
                       generation=1, profile="default")

    assert not os.listdir(tmp_path)


def test_adapter_path_naming(tmp_path):
    assert adapter_path(str(tmp_path), 12) == str(tmp_path / "adapter-gen12.safetensors")


def test_gc_keeps_newest_and_protects_promoted(tmp_path):
    d = str(tmp_path)
    for gen in range(1, 7):
        open(adapter_path(d, gen), "wb").close()
    # Non-adapter and malformed names must survive untouched.
    open(os.path.join(d, "trainer.json"), "wb").close()
    open(os.path.join(d, "adapter-genX.safetensors"), "wb").close()

    promoted = adapter_path(d, 1)
    gc_generations(d, keep=3, protect=promoted)

    left = sorted(os.listdir(d))
    assert left == sorted([
        "adapter-gen1.safetensors",   # protected promoted, oldest
        "adapter-gen4.safetensors",
        "adapter-gen5.safetensors",
        "adapter-gen6.safetensors",
        "adapter-genX.safetensors",
        "trainer.json",
    ])


def test_gc_without_protect_keeps_newest_n(tmp_path):
    d = str(tmp_path)
    for gen in (3, 10, 2, 8):
        open(adapter_path(d, gen), "wb").close()
    gc_generations(d, keep=2)
    assert sorted(os.listdir(d)) == [
        "adapter-gen10.safetensors",
        "adapter-gen8.safetensors",
    ]
