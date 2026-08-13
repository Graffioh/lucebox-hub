#!/usr/bin/env python3
"""Summarize Qwen3.6 concurrent feature ablations and activation proof."""

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
        variant = str(meta.get("variant", ""))
        if (
            level.get("failures")
            or not level.get("token_count_complete")
            or not level.get("prompt_token_count_complete")
            or (variant != "llama"
                and not level.get("effective_prompt_token_count_complete"))
        ):
            raise ValueError(f"{path}: failed or incomplete token accounting")
        if report.get("ignore_eos") and level.get("fixed_token_workload_valid") is not True:
            raise ValueError(f"{path}: fixed-token validation failed")
        proof_path = path.with_name("feature-proof.json")
        proof = None
        if variant != "llama":
            if not proof_path.is_file():
                raise ValueError(f"{path}: missing feature-proof.json")
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            if proof.get("valid") is not True:
                raise ValueError(f"{proof_path}: activation proof failed")
            expected_by_variant = {
                "ar": [], "ddtree": ["ddtree"], "pflash": ["pflash"],
                "kvflash": ["kvflash"], "full": ["ddtree", "kvflash", "pflash"],
            }
            if variant not in expected_by_variant:
                raise ValueError(f"{path}: unknown Lucebox variant {variant!r}")
            if proof.get("expected_features") != expected_by_variant[variant]:
                raise ValueError(
                    f"{proof_path}: expected_features does not match variant {variant}"
                )
        reports.append({
            "path": path, "report": report, "level": level,
            "meta": meta, "proof": proof,
        })
    if not reports:
        raise ValueError(f"{root}: no bench.json files found")
    return reports


def median(values: list[float]) -> float:
    return statistics.median(values)


def fmt(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def complete_median(values: list[float | None]) -> float | None:
    """Return a median only when every repeat measured the metric."""
    if not values or any(value is None for value in values):
        return None
    return median([value for value in values if value is not None])


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
        if len({run_signature(item) for item in items}) != 1:
            raise ValueError(f"{key}: incompatible run metadata")

    lines = [
        "# Qwen3.6 concurrent feature matrix", "",
        "Every Lucebox row is included only after request-correlated server telemetry "
        "proves its requested features executed. Throughput is the median across fresh-process repeats.",
        "",
        "| Workload | C | Variant | N | Output goodput | Output-window | vs AR | "
        "Effective/wire | DDTree accepted/step | DDTree steps/susp. | "
        "Target forwards | KV in/out | PFlash requests | TTFT max s | "
        "Stable output |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | "
        ":--- | ---: | :--- | ---: | ---: | :---: |",
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
        prompt_hashes = set(prompt_digests)
        if len(prompt_hashes) != 1:
            raise ValueError(f"{workload} C={clients} {variant}: prompt sets differ")
        goodput = median([item["level"]["aggregate_tok_s"] for item in items])
        window = complete_median([
            item["level"].get("output_window_tok_s") for item in items
        ])
        ratio = complete_median([
            item["level"].get("effective_to_wire_prompt_ratio") for item in items
        ])
        ttft = complete_median([
            item["level"].get("ttft_max_s") for item in items
        ])
        stable = output_stability(items)

        peers = grouped.get((workload, clients, "ar"), [])
        vs_ar = "—" if variant == "ar" else "n/a"
        if variant != "ar" and peers:
            peer_hashes = {p["level"]["selected_prompt_set_sha256"] for p in peers}
            if peer_hashes != prompt_hashes:
                raise ValueError(f"{workload} C={clients}: {variant}/ar prompts differ")
            if {run_signature(item) for item in items} != {
                run_signature(peer) for peer in peers
            }:
                raise ValueError(
                    f"{workload} C={clients}: {variant}/ar run metadata differs"
                )
            by_repeat = {int(item["meta"]["repeat"]): item for item in items}
            peers_by_repeat = {int(item["meta"]["repeat"]): item for item in peers}
            if by_repeat.keys() != peers_by_repeat.keys():
                raise ValueError(
                    f"{workload} C={clients}: {variant}/ar repeat sets differ"
                )
            if stable != "NO" and output_stability(peers) != "NO":
                ratios = []
                for repeat in sorted(by_repeat):
                    value = by_repeat[repeat]["level"].get("aggregate_tok_s")
                    base = peers_by_repeat[repeat]["level"].get("aggregate_tok_s")
                    if value is None or base is None or base <= 0:
                        raise ValueError(
                            f"{workload} C={clients} repeat={repeat}: invalid AR goodput"
                        )
                    ratios.append(value / base - 1.0)
                vs_ar = f"{median(ratios) * 100:+.1f}%"

        proofs = [item["proof"] for item in items if item["proof"] is not None]
        aggregates = [p["aggregate"] for p in proofs]
        steps = sum(a["ddtree_steps"] for a in aggregates)
        accepted = sum(a["ddtree_accepted_tokens"] for a in aggregates)
        accepted_per_step = accepted / steps if steps else None
        median_steps = median(
            [a["ddtree_steps"] for a in aggregates]
        ) if aggregates else None
        median_suspensions = median(
            [a["ddtree_suspensions"] for a in aggregates]
        ) if aggregates else None
        ddtree_activity = (
            f"{median_steps:.0f}/{median_suspensions:.0f}"
            if median_steps is not None and median_suspensions is not None
            else "n/a"
        )
        target_forwards = median([a["target_forwards"] for a in aggregates]) if aggregates else None
        page_ins = median([a["kvflash_page_ins"] for a in aggregates]) if aggregates else None
        page_outs = median([a["kvflash_page_outs"] for a in aggregates]) if aggregates else None
        pflash_requests = median([a["pflash_applied_requests"] for a in aggregates]) if aggregates else None
        kv_text = (
            f"{page_ins:.0f}/{page_outs:.0f}"
            if page_ins is not None and page_outs is not None else "n/a"
        )
        lines.append(
            f"| {workload} | {clients} | {variant} | {len(items)} | {goodput:.2f} | "
            f"{fmt(window)} | {vs_ar} | {fmt(ratio, 3)} | {fmt(accepted_per_step)} | "
            f"{ddtree_activity} | {fmt(target_forwards, 0)} | {kv_text} | "
            f"{fmt(pflash_requests, 0)} | {fmt(ttft, 3)} | {stable} |"
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
