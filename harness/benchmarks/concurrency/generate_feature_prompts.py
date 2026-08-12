#!/usr/bin/env python3
"""Generate deterministic long-context cohorts that force feature activation."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_ragged_prompts import (
    PROFILES as BASE_PROFILES,
    build_profile_records,
    write_records,
)


PROFILES = {
    **BASE_PROFILES,
    # These word counts were chosen after observing 38,130--44,856 tokens
    # with the development Qwen GGUF tokenizer. Word count is never treated as
    # activation proof: runtime wire/log telemetry is checked against the
    # recorded PFlash threshold. The stable word bank keeps hashes distinct.
    "compression": (34000, 36000, 38000, 40000),
    # The development Qwen GGUF tokenizer produced 13,463--20,190 tokens here.
    # Runtime effective-token and paging telemetry, not this estimate, proves
    # KVFlash pressure.
    "kv-pressure": (12000, 14000, 16000, 18000),
}


def build_records(profile: str) -> list[dict[str, object]]:
    activation_target = (
        "pflash-auto" if profile == "compression"
        else "kvflash-pressure" if profile == "kv-pressure"
        else "none"
    )
    return build_profile_records(
        profile, PROFILES,
        lambda _profile: {"activation_target": activation_target},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(args.profile)
    try:
        write_records(args.out, records)
    except FileExistsError as exc:
        parser.error(str(exc))
    print(f"wrote {len(records)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
