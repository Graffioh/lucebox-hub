#!/usr/bin/env python3
"""Summarize a Lucebox concurrency profile and emit a Perfetto trace."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def duration_ms(end: int, start: int) -> float:
    return (end - start) / 1_000_000 if end and start and end >= start else math.nan


def fmt_ms(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f} ms"


def sum_field(records: Iterable[dict[str, Any]], key: str) -> int:
    return sum(int(record.get(key, 0)) for record in records)


def build_markdown(records: list[dict[str, Any]]) -> str:
    steps = [record for record in records if record["type"] == "step"]
    requests = [record for record in records if record["type"] == "request"]
    bursts = [record for record in records if record["type"] == "token_burst"]
    footer = next(
        (record for record in reversed(records) if record["type"] == "footer"),
        {},
    )
    failed_requests = sum(
        not bool(request.get("ok")) for request in requests
    )

    phases: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    cohorts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    paths: Counter[str] = Counter()
    for step in steps:
        cohorts[int(step.get("live_slots", 0))].append(step)
        paths[str(step.get("path", "unknown"))] += 1
        for span in step.get("phases", []):
            phases[str(span.get("phase", "unknown"))] += int(
                span.get("duration_ns", 0)
            )
        for lane in step.get("lanes", []):
            if lane.get("kind") == "decode":
                decisions[str(lane.get("spec", "none"))] += 1

    queue_ms = [
        duration_ms(int(request.get("admitted_ns", 0)), int(request.get("queued_ns", 0)))
        for request in requests
    ]
    ttft_ms = [
        duration_ms(int(request.get("first_token_ns", 0)), int(request.get("queued_ns", 0)))
        for request in requests
    ]
    e2e_ms = [
        duration_ms(int(request.get("completed_ns", 0)), int(request.get("queued_ns", 0)))
        for request in requests
    ]
    queue_ms = [value for value in queue_ms if not math.isnan(value)]
    ttft_ms = [value for value in ttft_ms if not math.isnan(value)]
    e2e_ms = [value for value in e2e_ms if not math.isnan(value)]

    burst_times: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for burst in bursts:
        burst_times[int(burst["request_id"])].append(
            (int(burst["ready_ns"]), int(burst.get("token_count", 0)))
        )
    inter_token_ms: list[float] = []
    for request_bursts in burst_times.values():
        request_bursts.sort()
        for previous, current in zip(request_bursts, request_bursts[1:]):
            token_count = max(1, current[1])
            inter_token_ms.append((current[0] - previous[0]) / 1_000_000 / token_count)

    eligible = sum_field(steps, "spec_eligible_lanes")
    reserved = sum_field(steps, "spec_reserved_lanes")
    attempted = sum_field(steps, "spec_attempted_lanes")
    proposed = sum_field(steps, "spec_proposed_draft_tokens")
    verified = sum_field(steps, "spec_verified_draft_tokens")
    accepted = sum_field(steps, "spec_accepted_draft_tokens")
    durable = sum_field(steps, "spec_durable_draft_tokens")
    consumed = sum_field(steps, "spec_scheduler_consumed_tokens")
    target_rows = sum_field(steps, "target_rows")
    target_padding = sum_field(steps, "target_padding_rows")
    draft_rows = sum_field(steps, "draft_rows")
    draft_padding = sum_field(steps, "draft_padding_rows")
    phase_total = sum(phases.values())

    lines = [
        "# Lucebox concurrency profile",
        "",
        "## Run summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Captured rounds | {len(steps)} |",
        f"| Requests | {len(requests)} |",
        f"| Failed requests | {failed_requests} |",
        f"| Queue delay p50 / p95 | {fmt_ms(percentile(queue_ms, 0.50))} / {fmt_ms(percentile(queue_ms, 0.95))} |",
        f"| TTFT p50 / p95 | {fmt_ms(percentile(ttft_ms, 0.50))} / {fmt_ms(percentile(ttft_ms, 0.95))} |",
        f"| End-to-end p50 / p95 | {fmt_ms(percentile(e2e_ms, 0.50))} / {fmt_ms(percentile(e2e_ms, 0.95))} |",
        f"| Inter-burst token interval p50 / p95 | {fmt_ms(percentile(inter_token_ms, 0.50))} / {fmt_ms(percentile(inter_token_ms, 0.95))} |",
        f"| Target padding | {target_padding} / {target_rows} ({percent(ratio(target_padding, target_rows))}) |",
        f"| Draft padding | {draft_padding} / {draft_rows} ({percent(ratio(draft_padding, draft_rows))}) |",
        "",
        "The inter-burst interval divides each gap by the number of tokens made ready in the later burst. It is a scheduler-level estimate, not a per-token GPU timestamp.",
        "",
        "## Speculation funnel",
        "",
        "| Stage | Count | Conversion from previous |",
        "| --- | ---: | ---: |",
    ]
    funnel = [
        ("Eligible lanes", eligible),
        ("Reserved lanes", reserved),
        ("Attempted lanes", attempted),
        ("Proposed draft tokens", proposed),
        ("Verified draft tokens", verified),
        ("Accepted draft tokens", accepted),
        ("Durable draft tokens", durable),
        ("Scheduler-consumed draft tokens", consumed),
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
    for decision, count in sorted(decisions.items()):
        lines.append(f"| `{decision}` | {count} |")

    lines.extend([
        "",
        "## Concurrency cohorts",
        "",
        "| Live slots | Rounds | Mean round | Target padding | Draft acceptance | Paths |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for live_slots, cohort in sorted(cohorts.items()):
        mean_ms = statistics.fmean(int(step.get("duration_ns", 0)) for step in cohort) / 1_000_000
        cohort_target = sum_field(cohort, "target_rows")
        cohort_target_padding = sum_field(cohort, "target_padding_rows")
        cohort_proposed = sum_field(cohort, "spec_proposed_draft_tokens")
        cohort_accepted = sum_field(cohort, "spec_accepted_draft_tokens")
        cohort_paths = Counter(str(step.get("path", "unknown")) for step in cohort)
        path_text = ", ".join(f"{name}={count}" for name, count in sorted(cohort_paths.items()))
        lines.append(
            f"| {live_slots} | {len(cohort)} | {mean_ms:.2f} ms | "
            f"{percent(ratio(cohort_target_padding, cohort_target))} | "
            f"{percent(ratio(cohort_accepted, cohort_proposed))} | {path_text} |"
        )

    lines.extend([
        "",
        "## Phase time",
        "",
        "| Phase | Total | Share of measured phase time |",
        "| --- | ---: | ---: |",
    ])
    for phase, nanoseconds in phases.most_common():
        lines.append(
            f"| `{phase}` | {nanoseconds / 1_000_000:.2f} ms | "
            f"{percent(ratio(nanoseconds, phase_total))} |"
        )

    signals: list[str] = []
    if failed_requests:
        signals.append(
            f"{failed_requests}/{len(requests)} captured requests failed. "
            "Inspect the first incomplete funnel or phase boundary."
        )
    if accepted != durable:
        signals.append(
            "Accepted and durable draft token counts differ. Inspect state "
            "promotion or commit before tuning proposal quality."
        )
    if target_rows and ratio(target_padding, target_rows) > 0.20:
        signals.append("Target graph padding exceeds 20%. Inspect cohort bucket shapes.")
    if proposed and ratio(accepted, proposed) < 0.35:
        signals.append("Draft acceptance is below 35%. Inspect proposal quality before increasing speculative width.")
    if eligible and ratio(attempted, eligible) < 0.75:
        signals.append("Fewer than 75% of eligible lanes reach an attempt. Inspect suppression reasons and prompt mixing.")
    if requests and percentile(queue_ms, 0.95) > percentile(ttft_ms, 0.95) * 0.40:
        signals.append("Queueing accounts for a large part of p95 TTFT. Inspect admission and KV pressure.")
    if not signals:
        signals.append("No default threshold fired. Use the cohort and phase tables to choose the next experiment.")

    lines.extend(["", "## Signals", ""])
    lines.extend(f"- {signal}" for signal in signals)
    lines.extend([
        "",
        "## Capture integrity",
        "",
        f"- Paths: {', '.join(f'{name}={count}' for name, count in sorted(paths.items())) or 'none'}",
        f"- Dropped steps: {int(footer.get('dropped_steps', 0))}",
        f"- Dropped requests: {int(footer.get('dropped_requests', 0))}",
        f"- Dropped token bursts: {int(footer.get('dropped_token_bursts', 0))}",
        "",
    ])
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--perfetto", type=Path)
    args = parser.parse_args()

    records = load_records(args.profile)
    markdown = build_markdown(records)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.perfetto:
        args.perfetto.write_text(
            json.dumps(build_perfetto(records), indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
