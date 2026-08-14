#!/usr/bin/env python3
"""Summarize canonical concurrent benchmark reports into one comparison table."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def fmt(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def output_signature(report: dict[str, Any]) -> tuple[tuple[str, str, str], ...] | None:
    level = report["levels"][0]
    rows = []
    for wave in level.get("wave_results", []):
        for request in wave.get("requests_detail", []):
            try:
                rows.append((
                    request["case_id"], request["content_sha256"],
                    request["reasoning_content_sha256"],
                ))
            except KeyError:
                return None
    if len(rows) != level["requests"]:
        return None
    return tuple(rows)


def summarize(root: Path) -> str:
    GroupKey = tuple[str, int, str, int | None, int]
    FamilyKey = tuple[str, int, int | None, int]
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    repeat_ids: dict[GroupKey, set[int]] = defaultdict(set)
    variant_repeats: dict[FamilyKey, dict[str, set[int]]] = defaultdict(dict)
    for path in root.glob("*/c*/r*/*/bench.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        level = report["levels"][0]
        metadata = report["server_metadata"]
        suite = report.get("suite")
        variant = metadata.get("variant")
        clients = level.get("clients")
        requests = level.get("requests")
        repeat = metadata.get("repeat")
        case_limit = report.get("case_limit")
        if not isinstance(suite, str) or not isinstance(variant, str):
            raise ValueError(f"invalid suite or variant metadata: {path}")
        if isinstance(clients, bool) or not isinstance(clients, int) or clients < 1:
            raise ValueError(f"invalid client count: {path}")
        if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
            raise ValueError(f"invalid request count: {path}")
        if case_limit is not None and (
            isinstance(case_limit, bool) or not isinstance(case_limit, int) or case_limit < 1
        ):
            raise ValueError(f"invalid case_limit: {path}")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"missing or invalid repeat id: {path}")
        if level["failures"] or level["fixed_token_workload_valid"] is not True:
            raise ValueError(f"invalid measured report: {path}")
        if variant.endswith("ddtree"):
            proof = report.get("ddtree_proof")
            if not isinstance(proof, dict):
                raise ValueError(f"missing positive DDTree proof: {path}")
            steps = proof.get("ddtree_steps")
            if (
                isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
                or proof.get("requests_proven") != requests
            ):
                raise ValueError(f"missing positive DDTree proof: {path}")
        key = (suite, clients, variant, case_limit, requests)
        if repeat in repeat_ids[key]:
            raise ValueError(f"duplicate repeat id {repeat} for {key}")
        repeat_ids[key].add(repeat)
        groups[key].append(report)
        family = (suite, clients, case_limit, requests)
        variant_repeats.setdefault(family, {}).setdefault(variant, set()).add(repeat)
    for family, variants in variant_repeats.items():
        if len({frozenset(repeats) for repeats in variants.values()}) > 1:
            raise ValueError(f"mismatched repeat sets for {family}")
    if not groups:
        raise ValueError(f"no canonical reports under {root}")
    lines = [
        "# Canonical Qwen3.6 concurrency benchmark", "",
        "| Suite | C | Cases | Variant | Repeats | Goodput tok/s | Output-window tok/s | "
        "Prompt tok/s to first | Request decode tok/s | TTFT median s | TTFT max s | "
        "DDTree AL | Acceptance | Stable output |",
        "| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    sort_key = lambda item: (
        item[0][0], item[0][1], item[0][2], item[0][3] is not None,
        item[0][3] or 0, item[0][4],
    )
    for (suite, clients, variant, _case_limit, _requests), reports in sorted(
        groups.items(), key=sort_key
    ):
        levels = [report["levels"][0] for report in reports]
        proofs = [report.get("ddtree_proof") for report in reports]
        med = lambda key: statistics.median(level[key] for level in levels)
        al = (
            statistics.median(proof["mean_accepted_length"] for proof in proofs)
            if all(proof is not None for proof in proofs) else None
        )
        acceptance = (
            statistics.median(proof["acceptance_rate"] for proof in proofs)
            if all(proof is not None for proof in proofs) else None
        )
        acceptance_text = f"{100 * acceptance:.1f}%" if acceptance is not None else "n/a"
        signatures = [output_signature(report) for report in reports]
        complete = len(reports) >= 2 and all(signature is not None for signature in signatures)
        stable = "YES" if complete and len(set(signatures)) == 1 else "NO" if complete else "n/a"
        lines.append(
            f"| {suite} | {clients} | {levels[0]['requests']} | {variant} | {len(reports)} | "
            f"{fmt(med('aggregate_tok_s'))} | {fmt(med('output_window_tok_s'))} | "
            f"{fmt(med('prompt_tokens_per_s_to_first_token'))} | "
            f"{fmt(med('request_decode_tok_s_median'))} | "
            f"{fmt(med('ttft_median_s'), 3)} | {fmt(med('ttft_max_s'), 3)} | "
            f"{fmt(al)} | {acceptance_text} | {stable} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = summarize(args.root)
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
