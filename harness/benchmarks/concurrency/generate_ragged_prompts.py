#!/usr/bin/env python3
"""Generate a small deterministic ragged-prompt manifest for concurrency runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROFILES = {
    "short": (300, 500, 750, 1050),
    "medium": (1700, 2800, 4200, 5700),
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


def build_records(profile: str) -> list[dict[str, object]]:
    strata = PROFILES[profile]
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
            records.append({
                "id": f"{profile}-{index:02d}",
                "cohort": cohort,
                "stratum": strata.index(target) if target in strata else "mean",
                "target_words": target,
                "prompt": prompt_text(profile, cohort, index, target),
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(args.profile)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"wrote {len(records)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
