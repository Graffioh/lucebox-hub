"""Contract tests for oflash.adapter_export against oflash_format.h §Adapter.

The tensor set/shape expectations are pinned on the C++ side by
server/test/test_oflash_unit.cpp (adapter_expected_tensor_specs, using the
same tiny drafter: n_layer=2, hidden=16, q_dim=32, kv_dim=16, n_ff=24,
fc_in=3*16=48, rank=4 -> 26 tensors). Keep both in lockstep.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file

from oflash.adapter_export import (
    LAYER_TARGETS,
    DrafterDims,
    adapter_path,
    expected_tensor_shapes,
    export_adapter,
    gc_generations,
)

DIMS = DrafterDims(n_layer=2, hidden=16, q_dim=32, kv_dim=16, intermediate=24, fc_in=48)
RANK = 4
SHA = "a" * 64


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
                   generation=7, profile="default")
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

    # Metadata keys exactly per oflash_format.h §Adapter (all string values).
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata()
    assert meta == {
        "oflash.format": "1",
        "oflash.drafter_sha256": SHA,
        "oflash.rank": "4",
        "oflash.alpha": "32",
        "oflash.generation": "7",
        "oflash.profile": "default",
    }


def test_export_rejects_shape_mismatch(tmp_path):
    tensors = make_tensors()
    tensors["blk.0.attn_q.lora_a"] = np.zeros((5, 16), dtype=np.float32)  # rank+1
    with pytest.raises(ValueError, match="blk.0.attn_q.lora_a"):
        export_adapter(str(tmp_path / "a.safetensors"), tensors, DIMS, RANK,
                       alpha=32.0, drafter_sha256=SHA, generation=1,
                       profile="default")
    assert not os.listdir(tmp_path)  # nothing written on failure


def test_export_rejects_missing_and_extra_tensors(tmp_path):
    tensors = make_tensors()
    del tensors["blk.1.ffn_down.lora_b"]
    with pytest.raises(ValueError, match="mismatch"):
        export_adapter(str(tmp_path / "a.safetensors"), tensors, DIMS, RANK,
                       alpha=32.0, drafter_sha256=SHA, generation=1,
                       profile="default")

    tensors = make_tensors()
    tensors["blk.0.ffn_gate.lora_a"] = np.zeros((4, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        export_adapter(str(tmp_path / "a.safetensors"), tensors, DIMS, RANK,
                       alpha=32.0, drafter_sha256=SHA, generation=1,
                       profile="default")
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
