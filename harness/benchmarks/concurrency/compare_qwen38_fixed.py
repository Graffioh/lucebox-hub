#!/usr/bin/env python3
"""Validate and compare replicated Qwen3.8 fixed-work benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

VARIANTS = ("ar", "speculation", "adaptive")


def load_profile(
    server_log: Path,
    max_tokens: int,
    requests: int,
    require_profile_context: bool,
) -> dict[str, Any]:
    metrics = []
    epochs = []
    profile_context_methods = set()
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[concurrency-metrics] "):
            row = json.loads(line.split(" ", 1)[1])
            if row.get("output_tokens") == max_tokens:
                metrics.append(row)
        elif line.startswith("[spec-epoch] "):
            epochs.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("[spec-profile] ") and "context_method=" in line:
            method = line.split("context_method=", 1)[1].split()[0]
            profile_context_methods.add(method)
    if len(metrics) != requests:
        raise ValueError(
            f"expected {requests} measured telemetry rows in {server_log}, "
            f"found {len(metrics)}"
        )

    request_ids = {row.get("engine_request_id") for row in metrics}
    if None in request_ids or len(request_ids) != requests:
        raise ValueError(f"measured telemetry request IDs are incomplete in {server_log}")
    if require_profile_context and len(profile_context_methods) != 1:
        raise ValueError(
            f"adaptive profile context method is missing or ambiguous in {server_log}"
        )
    routes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    profiled_costs = []
    predicted_costs = []
    for epoch in epochs:
        measured_decisions = [
            decision for decision in epoch.get("requests", [])
            if decision.get("request_id") in request_ids
        ]
        if not measured_decisions:
            continue
        routes.update(decision.get("route", "missing") for decision in measured_decisions)
        reasons.update(decision.get("reason", "missing") for decision in measured_decisions)
        if isinstance(epoch.get("profiled_cost_us"), (int, float)):
            profiled_costs.append(float(epoch["profiled_cost_us"]))
        if isinstance(epoch.get("predicted_cost_us"), (int, float)):
            predicted_costs.append(float(epoch["predicted_cost_us"]))

    accepted = sum(int(row.get("spec_accepted_tokens", 0)) for row in metrics)
    steps = sum(int(row.get("spec_steps", 0)) for row in metrics)
    output_tokens = sum(int(row["output_tokens"]) for row in metrics)
    return {
        "requests": requests,
        "output_tokens": output_tokens,
        "spec_accepted_tokens": accepted,
        "spec_steps": steps,
        "accepted_tokens_per_spec_step": accepted / steps if steps else 0.0,
        "spec_accepted_output_share": accepted / output_tokens,
        "spec_service_ar_steps": sum(
            int(row.get("spec_service_ar_steps", 0)) for row in metrics
        ),
        "target_forwards": sum(int(row.get("target_forwards", 0)) for row in metrics),
        "selector_routes": dict(sorted(routes.items())),
        "selector_reasons": dict(sorted(reasons.items())),
        "profiled_cost_us_mean": (
            statistics.mean(profiled_costs) if profiled_costs else None
        ),
        "predicted_cost_us_mean": (
            statistics.mean(predicted_costs) if predicted_costs else None
        ),
        "profile_context_method": (
            next(iter(profile_context_methods)) if profile_context_methods else None
        ),
    }


def load_case(
    replicate: Path, variant: str,
) -> tuple[float, tuple[Any, ...], dict[str, Any]]:
    case = replicate / variant
    bench_path = case / "bench.json"
    runtime_path = case / "runtime-metadata.json"
    gpu_path = case / "gpu-identity.txt"
    server_log = case / "server.log"
    for path in (bench_path, runtime_path, gpu_path, server_log):
        if not path.is_file():
            raise ValueError(f"missing benchmark proof artifact: {path}")

    report = json.loads(bench_path.read_text(encoding="utf-8"))
    json.loads(runtime_path.read_text(encoding="utf-8"))
    levels = report.get("levels")
    if not isinstance(levels, list) or len(levels) != 1:
        raise ValueError(f"{bench_path} must contain exactly one concurrency level")
    level = levels[0]
    if (
        level.get("failures") != 0
        or level.get("requests_ok") != level.get("requests")
        or level.get("fixed_token_workload_valid") is not True
    ):
        raise ValueError(f"failed or incomplete requests in {bench_path}")
    rate = level.get("aggregate_tok_s")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0:
        raise ValueError(f"invalid aggregate_tok_s in {bench_path}")

    metadata = report.get("server_metadata") or {}
    fingerprint = (
        report.get("max_tokens"),
        report.get("prompt_file_sha256"),
        metadata.get("model_sha256"),
        metadata.get("server_binary_sha256"),
        level.get("clients"),
        level.get("completion_tokens_total"),
        level.get("selected_prompt_set_sha256"),
        gpu_path.read_text(encoding="utf-8"),
    )
    if any(value in (None, "") for value in fingerprint):
        raise ValueError(f"incomplete workload fingerprint in {bench_path}")
    profile = load_profile(
        server_log,
        int(report["max_tokens"]),
        int(level["requests"]),
        require_profile_context=variant == "adaptive",
    )
    return float(rate), fingerprint, profile


def compare(
    replicates: list[Path],
    *,
    max_relative_range: float,
    min_adaptive_ratio: float,
) -> dict[str, Any]:
    rates = {variant: [] for variant in VARIANTS}
    profiles = {variant: [] for variant in VARIANTS}
    expected_fingerprint: tuple[Any, ...] | None = None
    for replicate in replicates:
        for variant in VARIANTS:
            rate, fingerprint, profile = load_case(replicate, variant)
            if expected_fingerprint is None:
                expected_fingerprint = fingerprint
            elif fingerprint != expected_fingerprint:
                raise ValueError(
                    f"workload fingerprint mismatch in {replicate / variant}"
                )
            rates[variant].append(rate)
            profiles[variant].append(profile)
    if len(replicates) < 2:
        raise ValueError("at least two benchmark replicates are required")

    summaries: dict[str, Any] = {}
    for variant, values in rates.items():
        mean = statistics.mean(values)
        summaries[variant] = {
            "values": values,
            "mean": mean,
            "relative_range": (max(values) - min(values)) / mean,
            "sample_cv": statistics.stdev(values) / mean,
        }
    adaptive_ratios = [
        adaptive / ar
        for adaptive, ar in zip(rates["adaptive"], rates["ar"], strict=True)
    ]
    speculation_ratios = [
        speculation / ar
        for speculation, ar in zip(
            rates["speculation"], rates["ar"], strict=True
        )
    ]
    stability = {
        variant: row["relative_range"] <= max_relative_range
        for variant, row in summaries.items()
    }
    adaptive_ratio_mean = statistics.mean(adaptive_ratios)
    if not stability["adaptive"]:
        performance_observation = "adaptive_variable"
    elif adaptive_ratio_mean >= min_adaptive_ratio:
        performance_observation = "adaptive_above_ar"
    else:
        performance_observation = "adaptive_below_ar"

    return {
        "schema_version": 1,
        "replicates": [str(path.resolve()) for path in replicates],
        "thresholds": {
            "max_relative_range": max_relative_range,
            "min_adaptive_vs_ar": min_adaptive_ratio,
        },
        "variants": summaries,
        "profiles": profiles,
        "paired": {
            "adaptive_vs_ar": adaptive_ratios,
            "adaptive_vs_ar_mean": adaptive_ratio_mean,
            "speculation_vs_ar": speculation_ratios,
            "speculation_vs_ar_mean": statistics.mean(speculation_ratios),
        },
        "measurement_status": "valid",
        "stability": stability,
        "performance_observation": performance_observation,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "| Variant | Replicate goodput tok/s | Mean | Relative range |",
        "| :-- | :-- | --: | --: |",
    ]
    for variant in VARIANTS:
        row = result["variants"][variant]
        values = ", ".join(f"{value:.2f}" for value in row["values"])
        lines.append(
            f"| {variant} | {values} | {row['mean']:.2f} | "
            f"{row['relative_range']:.2%} |"
        )
    lines.extend((
        "",
        f"Adaptive / AR mean: {result['paired']['adaptive_vs_ar_mean']:.3f}",
        f"Forced speculation / AR mean: "
        f"{result['paired']['speculation_vs_ar_mean']:.3f}",
        f"Measurement status: {result['measurement_status']}",
        f"Performance observation: {result['performance_observation']}",
    ))
    lines.extend((
        "",
        "| Replicate | Adaptive accepted tokens | Spec steps | Accepted / step | "
        "Target forwards | Selector spec / AR decisions |",
        "| --: | --: | --: | --: | --: | --: |",
    ))
    for index, profile in enumerate(result["profiles"]["adaptive"], start=1):
        routes = profile["selector_routes"]
        lines.append(
            f"| {index} | {profile['spec_accepted_tokens']} | "
            f"{profile['spec_steps']} | "
            f"{profile['accepted_tokens_per_spec_step']:.2f} | "
            f"{profile['target_forwards']} | "
            f"{routes.get('speculation', 0)} / {routes.get('ar', 0)} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replicates", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-relative-range", type=float, default=0.05)
    parser.add_argument("--min-adaptive-ratio", type=float, default=1.0)
    parser.add_argument("--require-stable", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.max_relative_range < 1:
        parser.error("--max-relative-range must be in [0, 1)")
    if args.min_adaptive_ratio <= 0:
        parser.error("--min-adaptive-ratio must be positive")

    result = compare(
        args.replicates,
        max_relative_range=args.max_relative_range,
        min_adaptive_ratio=args.min_adaptive_ratio,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(render(result))
    return int(args.require_stable and not all(result["stability"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
