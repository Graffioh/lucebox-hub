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


def complete_median(values: list[float | None]) -> float | None:
    """Return a median only when every repeat measured the metric."""
    if not values or any(value is None for value in values):
        return None
    return median([value for value in values if value is not None])


def output_stability(items: list[dict]) -> str:
    output_digests = [
        item["level"].get("selected_output_set_sha256") for item in items
    ]
    complete = all(isinstance(value, str) and bool(value) for value in output_digests)
    hashes = {value for value in output_digests if isinstance(value, str)}
    return (
        "n/a" if len(items) < 2 or not complete
        else "yes" if len(hashes) == 1
        else "NO"
    )


def run_signature(item: dict) -> tuple[object, ...]:
    report, meta = item["report"], item["meta"]
    max_tokens = report.get("max_tokens")
    ignore_eos = report.get("ignore_eos")
    temperature = report.get("temperature")
    seed = report.get("seed")
    model_sha256 = meta.get("model_sha256")
    if (
        type(max_tokens) is not int or max_tokens <= 0
        or not isinstance(ignore_eos, bool)
        or type(temperature) not in (int, float)
        or type(seed) is not int
        or not isinstance(model_sha256, str) or not model_sha256
    ):
        raise ValueError("incomplete run metadata")
    return max_tokens, ignore_eos, temperature, seed, model_sha256


def report_key(item: dict) -> tuple[str, int, str]:
    meta, level = item["meta"], item["level"]
    workload = meta.get("workload")
    variant = meta.get("variant")
    clients = level.get("clients")
    if not isinstance(workload, str) or not workload:
        raise ValueError("incomplete report metadata: missing workload")
    if not isinstance(variant, str) or not variant:
        raise ValueError("incomplete report metadata: missing variant")
    if type(clients) is not int or clients <= 0:
        raise ValueError("incomplete report metadata: invalid clients")
    return workload, clients, variant


def summarize(reports: list[dict]) -> str:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for item in reports:
        grouped[report_key(item)].append(item)
    for key, items in grouped.items():
        repeats = [item["meta"].get("repeat") for item in items]
        if any(type(repeat) is not int or repeat <= 0 for repeat in repeats):
            raise ValueError(f"{key}: invalid or missing repeat")
        if len(repeats) != len(set(repeats)):
            raise ValueError(f"{key}: duplicate repeat")
        if len({run_signature(item) for item in items}) != 1:
            raise ValueError(f"{key}: incompatible run metadata")

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
        prompt_digests = [
            item["level"].get("selected_prompt_set_sha256") for item in items
        ]
        if not all(isinstance(value, str) and value for value in prompt_digests):
            raise ValueError(
                f"{workload} C={clients} {variant}: missing selected prompt set hash"
            )
        hashes = set(prompt_digests)
        if len(hashes) != 1:
            raise ValueError(f"{workload} C={clients} {variant}: prompt sets differ")
        goodput_values = [item["level"].get("aggregate_tok_s") for item in items]
        if any(type(value) not in (int, float) for value in goodput_values):
            raise ValueError(
                f"{workload} C={clients} {variant}: missing aggregate token rate"
            )
        goodput = median(goodput_values)
        output_window = complete_median([
            item["level"].get("output_window_tok_s") for item in items
        ])
        request_decode = complete_median([
            item["level"].get("request_decode_tok_s_median") for item in items
        ])
        prompt_rate = complete_median([
            item["level"].get("prompt_tokens_per_s_to_first_token") for item in items
        ])
        ttft = complete_median([
            item["level"].get("ttft_max_s") for item in items
        ])
        stable = output_stability(items)

        def delta(other: str, metric: str) -> str:
            peers = grouped.get((workload, clients, other), [])
            if not peers:
                return "n/a"
            peer_hashes = {p["level"]["selected_prompt_set_sha256"] for p in peers}
            if peer_hashes != hashes:
                raise ValueError(f"{workload} C={clients}: {variant}/{other} prompts differ")
            if {run_signature(item) for item in items} != {
                run_signature(peer) for peer in peers
            }:
                raise ValueError(
                    f"{workload} C={clients}: {variant}/{other} run metadata differs"
                )
            by_repeat = {int(item["meta"]["repeat"]): item for item in items}
            peers_by_repeat = {int(item["meta"]["repeat"]): item for item in peers}
            if by_repeat.keys() != peers_by_repeat.keys():
                raise ValueError(
                    f"{workload} C={clients}: {variant}/{other} repeat sets differ"
                )
            if stable == "NO" or output_stability(peers) == "NO":
                return "n/a"
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
        ttft_text = f"{ttft:.3f}" if ttft is not None else "n/a"
        lines.append(
            f"| {workload} | {clients} | {variant} | {len(items)} | {goodput:.2f} | "
            f"{output_window_text} | {request_decode_text} | {prompt_rate_text} | "
            f"{ttft_text} | {stable} | {vs_llama} | {decode_vs_llama} | {vs_k1} |"
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
