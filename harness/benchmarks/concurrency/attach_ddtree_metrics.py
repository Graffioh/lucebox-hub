#!/usr/bin/env python3
"""Attach and validate concurrent DDTree server telemetry for a measured report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MARKER = re.compile(r"\[concurrency-metrics\]\s+(\{.*\})")
COUNTERS = ("ddtree_steps", "ddtree_accepted_tokens", "target_forwards")


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MARKER.search(line)
        if not match:
            continue
        value = json.loads(match.group(1))
        response_id = value.get("response_id") or value.get("request_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("concurrency metric is missing response_id")
        if response_id in found:
            raise ValueError(f"duplicate concurrency metric for {response_id}")
        for key in COUNTERS:
            if (
                isinstance(value.get(key), bool)
                or not isinstance(value.get(key), int)
                or value[key] < 0
            ):
                raise ValueError(f"{response_id}: invalid {key}")
        found[response_id] = value
    return found


def attach(report: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> None:
    requests = [
        request
        for level in report.get("levels", [])
        for wave in level.get("wave_results", [])
        for request in wave.get("requests_detail", [])
        if request.get("error") is None
    ]
    totals = {key: 0 for key in COUNTERS}
    for request in requests:
        response_id = request.get("response_id")
        if not isinstance(response_id, str) or response_id not in metrics:
            raise ValueError(f"missing concurrency metric for response {response_id!r}")
        value = metrics[response_id]
        if value["ddtree_steps"] <= 0:
            raise ValueError(f"{response_id}: ddtree_steps must be positive")
        request["ddtree_metrics"] = {key: value[key] for key in COUNTERS}
        for key in COUNTERS:
            totals[key] += value[key]
    steps = totals["ddtree_steps"]
    if not requests or steps <= 0:
        raise ValueError("DDTree proof requires at least one successful request and step")
    emitted = totals["ddtree_accepted_tokens"] + steps
    report["ddtree_proof"] = {
        **totals,
        "speculative_emitted_tokens": emitted,
        "mean_accepted_length": emitted / steps,
        "acceptance_rate": emitted / (16 * steps),
        "acceptance_denominator_tokens_per_step": 16,
        "requests_proven": len(requests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("server_log", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        attach(report, load_metrics(args.server_log))
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        proof = report["ddtree_proof"]
        print(
            f"DDTree AL={proof['mean_accepted_length']:.2f} "
            f"acceptance={100 * proof['acceptance_rate']:.1f}% "
            f"steps={proof['ddtree_steps']}"
        )
        return 0
    except Exception as exc:
        print(f"[ddtree-proof] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
