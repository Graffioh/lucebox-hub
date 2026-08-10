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
        if (
            level.get("failures")
            or not level.get("token_count_complete")
            or not level.get("prompt_token_count_complete")
        ):
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
    for key, items in grouped.items():
        repeats = [int(item["meta"]["repeat"]) for item in items]
        if len(repeats) != len(set(repeats)):
            raise ValueError(f"{key}: duplicate repeat")

    lines = [
        "# Concurrency benchmark summary", "",
        "Aggregate output goodput includes queueing, prefill, and decode. "
        "Output-window goodput starts at the first observed output and is decode-facing, "
        "but it can include staggered prefill. Prompt tok/s to first token includes "
        "admission and TTFT.", "",
        "| Workload | C | Variant | Repeats | Output goodput tok/s | "
        "Output-window tok/s | Request decode tok/s | Prompt tok/s to first | "
        "TTFT max s | Stable output | vs llama | Decode vs llama | K8 vs K1 |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | "
        ":---: | ---: | ---: | ---: |",
    ]
    for workload, clients, variant in sorted(grouped):
        items = grouped[(workload, clients, variant)]
        hashes = {item["level"]["selected_prompt_set_sha256"] for item in items}
        if len(hashes) != 1:
            raise ValueError(f"{workload} C={clients} {variant}: prompt sets differ")
        goodput = median([item["level"]["aggregate_tok_s"] for item in items])
        output_window_values = [
            item["level"].get("output_window_tok_s") for item in items
            if item["level"].get("output_window_tok_s") is not None
        ]
        output_window = median(output_window_values) if output_window_values else None
        request_decode_values = [
            item["level"].get("request_decode_tok_s_median") for item in items
            if item["level"].get("request_decode_tok_s_median") is not None
        ]
        request_decode = median(request_decode_values) if request_decode_values else None
        prompt_rate_values = [
            item["level"]["prompt_tokens_per_s_to_first_token"] for item in items
            if item["level"].get("prompt_tokens_per_s_to_first_token") is not None
        ]
        prompt_rate = median(prompt_rate_values) if prompt_rate_values else None
        ttft = median([item["level"]["ttft_max_s"] for item in items])
        output_hashes = {
            item["level"].get("selected_output_set_sha256") for item in items
        }
        stable = (
            "n/a" if len(items) < 2
            else "yes" if len(output_hashes) == 1
            else "NO"
        )

        def delta(other: str, metric: str) -> str:
            peers = grouped.get((workload, clients, other), [])
            if not peers:
                return "n/a"
            peer_hashes = {p["level"]["selected_prompt_set_sha256"] for p in peers}
            if peer_hashes != hashes:
                raise ValueError(f"{workload} C={clients}: {variant}/{other} prompts differ")
            by_repeat = {int(item["meta"]["repeat"]): item for item in items}
            peers_by_repeat = {int(item["meta"]["repeat"]): item for item in peers}
            if by_repeat.keys() != peers_by_repeat.keys():
                raise ValueError(
                    f"{workload} C={clients}: {variant}/{other} repeat sets differ"
                )
            ratios = []
            for repeat in sorted(by_repeat):
                value = by_repeat[repeat]["level"].get(metric)
                base = peers_by_repeat[repeat]["level"].get(metric)
                if value is None or base is None:
                    return "n/a"
                if base <= 0:
                    raise ValueError(
                        f"{workload} C={clients} repeat={repeat}: "
                        f"non-positive {other} {metric}"
                    )
                ratios.append(value / base - 1.0)
            return f"{median(ratios) * 100:+.1f}%"

        vs_llama = (
            delta("llama", "aggregate_tok_s")
            if variant == "luce-k8" else "—"
        )
        decode_vs_llama = (
            delta("llama", "output_window_tok_s")
            if variant == "luce-k8" else "—"
        )
        vs_k1 = (
            delta("luce-k1", "aggregate_tok_s")
            if variant == "luce-k8" else "—"
        )
        output_window_text = f"{output_window:.2f}" if output_window is not None else "n/a"
        request_decode_text = f"{request_decode:.2f}" if request_decode is not None else "n/a"
        prompt_rate_text = f"{prompt_rate:.2f}" if prompt_rate is not None else "n/a"
        lines.append(
            f"| {workload} | {clients} | {variant} | {len(items)} | {goodput:.2f} | "
            f"{output_window_text} | {request_decode_text} | {prompt_rate_text} | "
            f"{ttft:.3f} | {stable} | {vs_llama} | {decode_vs_llama} | {vs_k1} |"
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
