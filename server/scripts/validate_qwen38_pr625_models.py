#!/usr/bin/env python3
"""Fail closed unless target and drafter match the PR #625 Qwen3.8 recipe."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


DRAFT_ARCH = "qwen35-dflash-draft"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def field(reader: GGUFReader, name: str):
    value = reader.fields.get(name)
    return None if value is None else value.contents()


def validate_target(path: Path, errors: list[str]) -> None:
    reader = GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)

    require(field(reader, "general.architecture") == "qwen35",
            "target architecture must be qwen35", errors)
    require(counts == Counter({"IQ4_XS": 440, "F32": 360, "Q6_K": 65, "Q5_K": 1}),
            f"target tensor-type counts differ from PR #625: {dict(counts)}", errors)
    require(tensors.get("output.weight") is not None and
            tensors["output.weight"].tensor_type.name == "Q5_K",
            "target output.weight must be Q5_K", errors)

    q6_names = {
        tensor.name for tensor in reader.tensors if tensor.tensor_type.name == "Q6_K"
    }
    invalid_q6 = sorted(
        name for name in q6_names
        if not (name.endswith("ssm_out.weight") or name.endswith("attn_v.weight"))
    )
    require(not invalid_q6,
            f"target has unexpected Q6_K tensors: {invalid_q6}", errors)
    require(all(
        tensor.tensor_type.name == "Q6_K"
        for name, tensor in tensors.items()
        if name.endswith("ssm_out.weight") or name.endswith("attn_v.weight")
    ), "every target ssm_out/attn_v tensor must be Q6_K", errors)


def validate_draft(path: Path, scheme: str, errors: list[str]) -> None:
    reader = GGUFReader(path)
    prefix = DRAFT_ARCH + "."
    counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)

    require(field(reader, "general.architecture") == DRAFT_ARCH,
            f"drafter architecture must be {DRAFT_ARCH}", errors)
    expected_counts = {
        "f16": Counter({"F16": 39, "F32": 23}),
        "q8_0": Counter({"Q8_0": 39, "F32": 23}),
        "q4-mix": Counter({"Q4_0": 35, "F32": 23, "Q8_0": 4}),
    }[scheme]
    require(counts == expected_counts,
            f"drafter tensor-type counts differ from {scheme}: {dict(counts)}", errors)
    require(field(reader, prefix + "rope.freq_base") == 10_000_000.0,
            "drafter rope.freq_base must be 10000000", errors)
    require(not any("rope.scaling" in name for name in reader.fields),
            "drafter must not contain YaRN/rope.scaling metadata", errors)

    expected = {
        "dflash.n_target_layers": 5,
        "dflash.block_size": 7,
        "dflash.mask_token_id": 248077,
        "dflash.target_layer_ids": [4, 16, 28, 40, 52],
        "dflash.dspark.enabled": 1,
        "dflash.dspark.markov_rank": 256,
        "dflash.dspark.vocab_size": 248320,
        "dflash.dspark.confidence_dim": 5376,
        "dflash.dspark.confidence.enabled": 1,
    }
    for key, wanted in expected.items():
        actual = field(reader, prefix + key)
        require(actual == wanted,
                f"drafter {key} must be {wanted!r}, got {actual!r}", errors)

    if scheme == "q4-mix":
        invalid_q8 = sorted(
            tensor.name for tensor in reader.tensors
            if tensor.tensor_type.name == "Q8_0" and not tensor.name.startswith("dflash.")
        )
        require(not invalid_q8,
                f"q4-mix has non-head Q8_0 tensors: {invalid_q8}", errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--draft-scheme", choices=("f16", "q8_0", "q4-mix"),
                        default="q8_0")
    args = parser.parse_args()

    errors: list[str] = []
    for label, path in (("target", args.target), ("draft", args.draft)):
        if not path.is_file():
            errors.append(f"{label} is not a readable file: {path}")
    if not errors:
        validate_target(args.target, errors)
        validate_draft(args.draft, args.draft_scheme, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"PR #625 target OK: {args.target}")
    print(f"PR #625 no-YaRN {args.draft_scheme} drafter OK: {args.draft}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
