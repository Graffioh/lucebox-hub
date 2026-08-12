#!/usr/bin/env python3
"""Generate a small deterministic ragged-prompt manifest for concurrency runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping


PROFILES = {
    "short": (250, 350, 450, 550),
    "medium": (650, 850, 1150, 1350),
    "long": (2000, 2600, 3400, 4000),
}

WORD_BANK = (
    "systems engineers compare latency throughput scheduling memory kernels queues "
    "batches requests tokens caches pages attention arithmetic bandwidth occupancy "
    "profiling measurement fairness reproducibility workloads concurrency admission "
    "prefill decoding evidence tradeoffs implementation validation production service"
).split()


def prompt_text(profile: str, cohort: str, index: int, target_words: int) -> str:
    prefix = (
        f"Ragged benchmark {profile} cohort {cohort} request {index}. "
        "Write a structured engineering analysis of the following observations, "
        "including assumptions, likely bottlenecks, and a concise conclusion."
    ).split()
    words = list(prefix)
    cursor = (index * 7 + target_words) % len(WORD_BANK)
    while len(words) < target_words:
        words.append(WORD_BANK[cursor % len(WORD_BANK)])
        cursor += 1
    return " ".join(words[:target_words])


ExtraFields = Callable[[str], Mapping[str, object]]


def build_profile_records(
    profile: str,
    profiles: Mapping[str, tuple[int, ...]],
    extra_fields: ExtraFields | None = None,
) -> list[dict[str, object]]:
    """Build the standard disjoint C1/C4/C8/C16 cohort layout."""
    strata = profiles[profile]
    if not strata:
        raise ValueError(f"profile {profile!r} has no length strata")
    layout = [
        ("c1", [sum(strata) // len(strata)]),
        ("c4", list(strata)),
        ("c8", list(strata) * 2),
        ("c16", list(strata) * 4),
    ]
    records: list[dict[str, object]] = []
    for cohort, targets in layout:
        for target in targets:
            index = len(records)
            record: dict[str, object] = {
                "id": f"{profile}-{index:02d}",
                "cohort": cohort,
                "stratum": strata.index(target) if target in strata else "mean",
                "target_words": target,
                "prompt": prompt_text(profile, cohort, index, target),
            }
            if extra_fields is not None:
                additions = dict(extra_fields(profile))
                overlap = record.keys() & additions.keys()
                if overlap:
                    raise ValueError(
                        "extra profile fields may not replace standard fields: "
                        f"{sorted(overlap)}"
                    )
                record.update(additions)
            records.append(record)
    return records


def build_records(profile: str) -> list[dict[str, object]]:
    return build_profile_records(profile, PROFILES)


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
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
