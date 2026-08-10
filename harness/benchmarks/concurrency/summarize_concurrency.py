#!/usr/bin/env python3
"""Summarize paired Lucebox/llama.cpp concurrency benchmark reports."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_reports(root: Path) -> list[dict]:
    reports = []
    for path in sorted(root.rglob("bench.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        meta = report.get("server_metadata") or {}
        if len(report.get("levels", [])) != 1:
            raise ValueError(f"{path}: expected exactly one client level")
        level = report["levels"][0]
        if level.get("failures") or not level.get("token_count_complete"):
            raise ValueError(f"{path}: failed or incomplete token accounting")
        if report.get("ignore_eos") and level.get("fixed_token_workload_valid") is not True:
            raise ValueError(f"{path}: fixed-token validation failed")
        reports.append({"path": path, "report": report, "level": level, "meta": meta})
    if not reports:
        raise ValueError(f"{root}: no bench.json files found")
    return reports


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize(reports: list[dict]) -> str:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for item in reports:
        meta, level = item["meta"], item["level"]
        key = (str(meta["workload"]), int(level["clients"]), str(meta["variant"]))
        grouped[key].append(item)

    lines = [
        "# Concurrency benchmark summary", "",
        "Aggregate output goodput includes queueing, prefill, and decode. "
        "Prompt tok/s to first token includes admission and TTFT.", "",
        "| Workload | C | Variant | Repeats | Output goodput tok/s | "
        "Prompt tok/s to first | TTFT max s | Stable output | vs llama | K8 vs K1 |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    for workload, clients, variant in sorted(grouped):
        items = grouped[(workload, clients, variant)]
        hashes = {item["level"]["selected_prompt_set_sha256"] for item in items}
        if len(hashes) != 1:
            raise ValueError(f"{workload} C={clients} {variant}: prompt sets differ")
        goodput = median([item["level"]["aggregate_tok_s"] for item in items])
        prompt_rate_values = [
            item["level"]["prompt_tokens_per_s_to_first_token"] for item in items
            if item["level"].get("prompt_tokens_per_s_to_first_token") is not None
        ]
        prompt_rate = median(prompt_rate_values) if prompt_rate_values else None
        ttft = median([item["level"]["ttft_max_s"] for item in items])
        output_hashes = {
            item["level"].get("selected_output_set_sha256") for item in items
        }
        stable = "yes" if len(output_hashes) == 1 else "NO"

        def delta(other: str) -> str:
            peers = grouped.get((workload, clients, other), [])
            if not peers:
                return "n/a"
            peer_hashes = {p["level"]["selected_prompt_set_sha256"] for p in peers}
            if peer_hashes != hashes:
                raise ValueError(f"{workload} C={clients}: {variant}/{other} prompts differ")
            base = median([p["level"]["aggregate_tok_s"] for p in peers])
            return f"{(goodput / base - 1.0) * 100:+.1f}%"

        vs_llama = delta("llama") if variant == "luce-k8" else "—"
        vs_k1 = delta("luce-k1") if variant == "luce-k8" else "—"
        lines.append(
            f"| {workload} | {clients} | {variant} | {len(items)} | {goodput:.2f} | "
            f"{prompt_rate:.2f} | {ttft:.3f} | {stable} | {vs_llama} | {vs_k1} |"
            if prompt_rate is not None else
            f"| {workload} | {clients} | {variant} | {len(items)} | {goodput:.2f} | "
            f"n/a | {ttft:.3f} | {stable} | {vs_llama} | {vs_k1} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = summarize(load_reports(args.root))
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
