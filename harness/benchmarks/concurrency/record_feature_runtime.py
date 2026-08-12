#!/usr/bin/env python3
"""Add startup-observed pool dimensions to one feature-case metadata file."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any


KVFLASH_POOL_RE = re.compile(
    r"\[parallel-kvflash\] physical resident pool (?P<tokens>\d+) tokens; "
    r"logical per-slot cap (?P<max_ctx>\d+) across (?P<slots>\d+) slots"
)
PAGED_POOL_RE = re.compile(
    r"\[paged-attention\] (?P<blocks>\d+) physical blocks x "
    r"(?P<block_size>\d+) tokens \((?P<tokens>\d+) pool tokens, "
    r"per-sequence max_ctx (?P<max_ctx>\d+)\)"
)


def _one_consistent(matches: list[dict[str, int]], label: str) -> dict[str, int] | None:
    if not matches:
        return None
    first = matches[0]
    if any(row != first for row in matches[1:]):
        raise ValueError(f"conflicting {label} startup markers: {matches}")
    return first


def observe_startup(log_text: str) -> dict[str, Any]:
    kvflash = _one_consistent(
        [
            {key: int(value) for key, value in match.groupdict().items()}
            for match in KVFLASH_POOL_RE.finditer(log_text)
        ],
        "KVFlash pool",
    )
    paged = _one_consistent(
        [
            {key: int(value) for key, value in match.groupdict().items()}
            for match in PAGED_POOL_RE.finditer(log_text)
        ],
        "paged pool",
    )
    if paged and paged["blocks"] * paged["block_size"] != paged["tokens"]:
        raise ValueError("paged-attention startup marker has inconsistent dimensions")
    if kvflash and paged:
        if kvflash["tokens"] != paged["tokens"]:
            raise ValueError("KVFlash and paged-attention startup pool sizes disagree")
        if kvflash["max_ctx"] != paged["max_ctx"]:
            raise ValueError("KVFlash and paged-attention logical max_ctx values disagree")

    return {
        "kvflash_active": kvflash is not None,
        "physical_kv_pool_tokens": (
            kvflash["tokens"] if kvflash else paged["tokens"] if paged else None
        ),
        "physical_kv_pool_blocks": paged["blocks"] if paged else None,
        "kv_block_size_tokens": paged["block_size"] if paged else None,
        "logical_per_slot_max_ctx": (
            kvflash["max_ctx"] if kvflash else paged["max_ctx"] if paged else None
        ),
        "configured_slots": kvflash["slots"] if kvflash else None,
        "proof_sources": {
            "kvflash_pool_startup_marker": kvflash is not None,
            "paged_pool_startup_marker": paged is not None,
        },
    }


def update_metadata(metadata: dict[str, Any], log_text: str) -> dict[str, Any]:
    observed = observe_startup(log_text)
    feature_config = metadata.get("feature_config") or {}
    kvflash_mode = feature_config.get("kvflash")
    kvflash_requested = isinstance(kvflash_mode, str) and kvflash_mode not in (
        "", "off", "0",
    )
    if kvflash_requested and not observed["kvflash_active"]:
        raise ValueError(
            "KVFlash metadata is enabled but its physical-pool startup marker is missing"
        )
    if observed["physical_kv_pool_tokens"] is None:
        raise ValueError("paged physical-pool startup marker is missing")

    result = dict(metadata)
    result["schema_version"] = max(3, int(result.get("schema_version", 0)))
    result["runtime_observed"] = observed
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    updated = update_metadata(
        metadata,
        args.server_log.read_text(encoding="utf-8", errors="replace"),
    )
    # Replace atomically so a killed run never leaves half-written metadata.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.metadata.parent,
        prefix=f".{args.metadata.name}.", delete=False,
    ) as handle:
        json.dump(updated, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
