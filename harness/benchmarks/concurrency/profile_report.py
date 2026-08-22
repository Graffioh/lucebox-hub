#!/usr/bin/env python3
"""Build reports from a Lucebox concurrency profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from profile_payload import (
    build_folded,
    build_summary,
    format_folded_weight,
    step_phase_buckets,
)
from profile_view import build_html, load_device_specs

FOLDED_DIMENSIONS = frozenset(("path", "cohort", "phase"))
DEFAULT_FOLDED_STACK = ("path", "cohort", "phase")


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error.msg}") from error
            if not isinstance(record, dict) or "type" not in record:
                raise ValueError(f"{path}:{line_number}: record needs a type")
            records.append(record)
    if not records or records[0].get("schema") != "lucebox.concurrency.v1":
        raise ValueError(f"{path}: unsupported or missing profile schema")
    return records


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100.0 * value:.1f}%"


def build_markdown(summary: dict[str, Any]) -> str:
    run = summary["run"]
    capture = summary["capture"]
    latency = summary["latency_ms"]
    speculation = summary["speculation"]
    padding = summary["padding"]

    def format_ms(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f} ms"

    def format_ratio(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    def format_pair(values: dict[str, float | None]) -> str:
        return f"{format_ms(values['p50'])} / {format_ms(values['p95'])}"

    lines = [
        "# Lucebox concurrency profile",
        "",
        "## Run summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Git SHA | `{run.get('git_sha', 'unknown')}` |",
        f"| Model | `{run.get('model_name', 'unknown')}` |",
        f"| Configured concurrency | {run.get('max_concurrency', 'unknown')} |",
        f"| Captured rounds | {capture['rounds']} |",
        f"| Requests | {capture['requests']} |",
        f"| Failed requests | {capture['failed_requests']} |",
        f"| Queue delay p50 / p95 | {format_pair(latency['queue'])} |",
        f"| TTFT p50 / p95 | {format_pair(latency['ttft'])} |",
        f"| End-to-end p50 / p95 | {format_pair(latency['end_to_end'])} |",
        f"| Inter-burst token interval p50 / p95 | {format_pair(latency['inter_burst_per_token'])} |",
        f"| Target padding | {padding['target_padding_rows']} / "
        f"{padding['target_rows']} "
        f"({format_ratio(padding['target_padding_ratio'])}) |",
        f"| Draft padding | {padding['draft_padding_rows']} / "
        f"{padding['draft_rows']} "
        f"({format_ratio(padding['draft_padding_ratio'])}) |",
        "",
        "The inter-burst interval divides each gap by the number of tokens "
        "made ready in the later burst. It is a scheduler-level estimate, "
        "not a per-token GPU timestamp.",
        "",
        "## Speculation funnel",
        "",
        "| Stage | Count | Conversion from previous |",
        "| --- | ---: | ---: |",
    ]
    funnel = [
        ("Eligible lanes", speculation["spec_eligible_lanes"]),
        ("Reserved lanes", speculation["spec_reserved_lanes"]),
        ("Attempted lanes", speculation["spec_attempted_lanes"]),
        ("Proposed draft tokens",
         speculation["spec_proposed_draft_tokens"]),
        ("Verified draft tokens",
         speculation["spec_verified_draft_tokens"]),
        ("Accepted draft tokens",
         speculation["spec_accepted_draft_tokens"]),
        ("Durable draft tokens",
         speculation["spec_durable_draft_tokens"]),
        ("Scheduler-consumed draft tokens",
         speculation["spec_scheduler_consumed_tokens"]),
    ]
    previous = 0
    for index, (name, value) in enumerate(funnel):
        conversion = (
            "n/a" if previous == 0 or index == 3
            else percent(ratio(value, previous))
        )
        lines.append(f"| {name} | {value} | {conversion} |")
        previous = value

    lines.extend([
        "",
        "### Suppression reasons",
        "",
        "| Decision | Decode lanes |",
        "| --- | ---: |",
    ])
    for decision, count in speculation["decisions"].items():
        lines.append(f"| `{decision}` | {count} |")

    lines.extend([
        "",
        "## Concurrency cohorts",
        "",
        "| Live slots | Rounds | Mean round | Target padding | "
        "Draft acceptance | Paths |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for live_slots, cohort in sorted(
            summary["cohorts"].items(), key=lambda item: int(item[0])):
        path_text = ", ".join(
            f"{name}={count}" for name, count in cohort["paths"].items()
        )
        lines.append(
            f"| {live_slots} | {cohort['rounds']} | "
            f"{cohort['mean_round_ms']:.2f} ms | "
            f"{format_ratio(cohort['target_padding_ratio'])} | "
            f"{format_ratio(cohort['draft_acceptance_ratio'])} | "
            f"{path_text} |"
        )

    phase_total = sum(summary["phase_ns"].values())
    lines.extend([
        "",
        "## Phase time",
        "",
        "| Phase | Total | Share of measured phase time |",
        "| --- | ---: | ---: |",
    ])
    for phase, nanoseconds in summary["phase_ns"].items():
        lines.append(
            f"| `{phase}` | {nanoseconds / 1_000_000:.2f} ms | "
            f"{percent(ratio(nanoseconds, phase_total))} |"
        )

    accepted = speculation["spec_accepted_draft_tokens"]
    durable = speculation["spec_durable_draft_tokens"]
    warnings: list[str] = []
    if capture["failed_requests"]:
        warnings.append(
            f"{capture['failed_requests']}/{capture['requests']} captured "
            "requests failed. Inspect the first incomplete funnel or phase "
            "boundary."
        )
    if accepted != durable:
        warnings.append(
            "Accepted and durable draft token counts differ. Inspect state "
            "promotion or commit before tuning proposal quality."
        )
    if not capture["complete"]:
        warnings.append("Capture is incomplete.")
    dropped = (
        capture["dropped_steps"]
        + capture["dropped_requests"]
        + capture["dropped_token_bursts"]
    )
    if dropped:
        warnings.append(f"Capture dropped {dropped} records.")
    paths = ", ".join(
        f"{name}={count}" for name, count in capture["paths"].items()
    ) or "none"
    lines.extend([
        "",
        "## Capture integrity",
        "",
        f"- Complete: {'yes' if capture['complete'] else 'no'}",
        f"- Paths: {paths}",
        f"- Dropped steps: {capture['dropped_steps']}",
        f"- Dropped requests: {capture['dropped_requests']}",
        f"- Dropped token bursts: {capture['dropped_token_bursts']}",
    ])
    if warnings:
        lines.extend(["", "### Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)


def build_perfetto(records: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for record in records:
        record_type = record["type"]
        if record_type == "step":
            started_ns = int(record.get("started_ns", 0))
            round_id = int(record.get("round_id", 0))
            for span in record.get("phases", []):
                events.append({
                    "name": str(span.get("phase", "unknown")),
                    "cat": "lucebox.round",
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "ts": (started_ns + int(span.get("start_offset_ns", 0))) / 1000,
                    "dur": int(span.get("duration_ns", 0)) / 1000,
                    "args": {
                        "round_id": round_id,
                        "path": record.get("path", "unknown"),
                        "live_slots": int(record.get("live_slots", 0)),
                    },
                })
        elif record_type == "request":
            request_id = int(record.get("request_id", 0))
            spans = [
                ("queue", int(record.get("queued_ns", 0)), int(record.get("admitted_ns", 0))),
                ("prefill", int(record.get("admitted_ns", 0)), int(record.get("prefill_completed_ns", 0))),
                ("decode", int(record.get("prefill_completed_ns", 0)), int(record.get("completed_ns", 0))),
            ]
            for name, start, end in spans:
                if start and end >= start:
                    events.append({
                        "name": name,
                        "cat": "lucebox.request",
                        "ph": "X",
                        "pid": 1,
                        "tid": 1000 + request_id,
                        "ts": start / 1000,
                        "dur": (end - start) / 1000,
                        "args": {"request_id": request_id},
                    })
        elif record_type == "token_burst":
            events.append({
                "name": "tokens_ready",
                "cat": "lucebox.request",
                "ph": "i",
                "s": "t",
                "pid": 1,
                "tid": 1000 + int(record.get("request_id", 0)),
                "ts": int(record.get("ready_ns", 0)) / 1000,
                "args": {
                    "round_id": int(record.get("round_id", 0)),
                    "token_count": int(record.get("token_count", 0)),
                },
            })
    events.sort(key=lambda event: (event.get("ts", 0), event.get("tid", 0)))
    return {"displayTimeUnit": "ms", "traceEvents": events}


def parse_folded_stack(value: str) -> tuple[str, ...]:
    dimensions = tuple(value.split(","))
    if len(dimensions) != 3 or set(dimensions) != FOLDED_DIMENSIONS:
        raise argparse.ArgumentTypeError(
            "stack must be a permutation of path,cohort,phase"
        )
    return dimensions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--perfetto", type=Path)
    parser.add_argument("--json-summary", type=Path)
    parser.add_argument("--folded", type=Path)
    parser.add_argument("--folded-per-token", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--device",
        help="device_specs.json key used for analytic boundness coloring",
    )
    parser.add_argument(
        "--baseline-device",
        help="device key for the baseline capture; defaults to --device",
    )
    parser.add_argument(
        "--stack", type=parse_folded_stack, default=DEFAULT_FOLDED_STACK,
        metavar="path,cohort,phase",
    )
    args = parser.parse_args()

    if args.baseline and not args.html:
        parser.error("--baseline requires --html")
    if args.baseline_device and not args.baseline:
        parser.error("--baseline-device requires --baseline")

    records = load_records(args.profile)
    summary = build_summary(records)
    markdown = build_markdown(summary)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.perfetto:
        args.perfetto.write_text(
            json.dumps(build_perfetto(records), indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json_summary:
        args.json_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.folded:
        args.folded.write_text(
            build_folded(records, stack=args.stack), encoding="utf-8"
        )
    if args.folded_per_token:
        args.folded_per_token.write_text(
            build_folded(records, stack=args.stack, per_token=True),
            encoding="utf-8",
        )
    if args.html:
        device_specs = load_device_specs()
        if args.device:
            device_specs["selected_device"] = args.device
        if args.baseline_device:
            device_specs["baseline_device"] = args.baseline_device
        baseline_records = load_records(args.baseline) if args.baseline else None
        args.html.write_text(
            build_html(records, baseline_records, device_specs),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
