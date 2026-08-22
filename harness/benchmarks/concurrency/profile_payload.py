from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TypeAlias

ReportPayload: TypeAlias = dict[str, Any]

PHASE_ORDER = (
    "scheduler_plan",
    "input_staging",
    "draft_prepare",
    "draft_compute",
    "proposal_select",
    "target_graph_build",
    "metadata_upload",
    "target_compute",
    "readback_sync",
    "acceptance",
    "state_promotion",
    "sampling_commit",
    "output_processing",
    "client_flush",
)

FUNNEL_KEYS = (
    "spec_eligible_lanes",
    "spec_reserved_lanes",
    "spec_attempted_lanes",
    "spec_proposed_draft_tokens",
    "spec_verified_draft_tokens",
    "spec_accepted_draft_tokens",
    "spec_durable_draft_tokens",
    "spec_scheduler_consumed_tokens",
)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


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


def optional_number(value: float) -> float | None:
    return None if math.isnan(value) else value


def percentile_pair(values: Iterable[float]) -> dict[str, float | None]:
    finite = [value for value in values if not math.isnan(value)]
    return {
        "p50": optional_number(percentile(finite, 0.50)),
        "p95": optional_number(percentile(finite, 0.95)),
    }


def duration_ms(end: int, start: int) -> float:
    return (end - start) / 1_000_000 if end and start and end >= start else math.nan


def sum_field(records: Iterable[dict[str, Any]], key: str) -> int:
    return sum(int(record.get(key, 0)) for record in records)


def folded_atom(value: Any) -> str:
    atom = str(value)
    if not atom or ";" in atom or any(character.isspace() for character in atom):
        raise ValueError(f"invalid folded-stack frame: {atom!r}")
    return atom


def step_phase_buckets(step: dict[str, Any]) -> Counter[str]:
    duration_ns = max(0, int(step.get("duration_ns", 0)))
    events: dict[int, Counter[str]] = defaultdict(Counter)
    for span in step.get("phases", []):
        phase = folded_atom(span.get("phase", "unknown"))
        raw_start = int(span.get("start_offset_ns", 0))
        raw_end = raw_start + max(0, int(span.get("duration_ns", 0)))
        start = min(duration_ns, max(0, raw_start))
        end = min(duration_ns, max(0, raw_end))
        if end <= start:
            continue
        events[start][phase] += 1
        events[end][phase] -= 1

    buckets: Counter[str] = Counter()
    active: Counter[str] = Counter()
    previous = 0
    for offset in sorted({0, duration_ns, *events}):
        if offset > previous:
            phases = sorted(
                phase for phase, count in active.items() if count > 0
            )
            if not phases:
                label = "unattributed"
            elif len(phases) == 1:
                label = phases[0]
            else:
                label = f"overlap({'+'.join(phases)})"
            buckets[label] += offset - previous
        active.update(events[offset])
        previous = offset
    return buckets


def phase_sort_key(phase: str) -> tuple[int, str]:
    if phase in PHASE_ORDER:
        return PHASE_ORDER.index(phase), phase
    if phase == "unattributed":
        return len(PHASE_ORDER) + 1, phase
    if phase.startswith("overlap("):
        return len(PHASE_ORDER) + 2, phase
    return len(PHASE_ORDER), phase


def inter_round_idle(steps: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        steps,
        key=lambda step: (
            int(step.get("started_ns", 0)),
            int(step.get("round_id", 0)),
        ),
    )
    gaps: list[int] = []
    if ordered:
        previous_end = int(ordered[0].get("started_ns", 0)) + max(
            0, int(ordered[0].get("duration_ns", 0))
        )
        for step in ordered[1:]:
            started_ns = int(step.get("started_ns", 0))
            gaps.append(max(0, started_ns - previous_end))
            previous_end = max(
                previous_end,
                started_ns + max(0, int(step.get("duration_ns", 0))),
            )
    return {
        "phase": "idle;inter_round",
        "total_ns": sum(gaps),
        "gaps": len(gaps),
        "p50_ns": optional_number(percentile(gaps, 0.50)),
        "p95_ns": optional_number(percentile(gaps, 0.95)),
        "note": "Positive host-clock gaps between retained rounds. No cohort or token denominator applies.",
    }


def _decode_tokens(step: dict[str, Any]) -> int:
    return sum(
        max(0, int(lane.get("scheduler_consumed_tokens", 0)))
        for lane in step.get("lanes", [])
        if lane.get("kind") == "decode"
    )


def _resolve_model_spec(
    metadata: dict[str, Any],
    catalog: dict[str, Any],
    *,
    draft: bool = False,
) -> tuple[str | None, dict[str, Any] | None]:
    models = catalog.get("models", {})
    if not isinstance(models, dict):
        return None, None
    candidates: list[str] = []
    if draft:
        draft_path = str(metadata.get("draft_path", ""))
        if draft_path:
            candidates.append(Path(draft_path).name)
    else:
        model_name = str(metadata.get("model_name", ""))
        model_path = str(metadata.get("model_path", ""))
        if model_name:
            candidates.append(model_name)
        if model_path:
            candidates.append(Path(model_path).name)
    for candidate in candidates:
        spec = models.get(candidate)
        if isinstance(spec, dict):
            return candidate, spec
    return None, None


def _classification(
    phase: str,
    steps: list[dict[str, Any]],
    metadata: dict[str, Any],
    catalog: dict[str, Any],
    device_key: str | None,
) -> dict[str, Any]:
    devices = catalog.get("devices", {})
    device = devices.get(device_key) if isinstance(devices, dict) and device_key else None
    if not isinstance(device, dict):
        return {"class": "neutral", "reason": "no device spec"}
    if phase == "unattributed" or phase.startswith("idle;"):
        return {"class": "idle", "reason": "idle or unattributed host time"}
    if phase not in {"target_compute", "draft_compute"}:
        return {"class": "overhead", "reason": "non-compute phase"}

    draft = phase == "draft_compute"
    model_key, model = _resolve_model_spec(metadata, catalog, draft=draft)
    if not isinstance(model, dict):
        return {
            "class": "overhead" if draft else "neutral",
            "reason": "no draft model spec" if draft else "no target model spec",
        }

    rounds = len(steps)
    rows_key = "draft_rows" if draft else "target_rows"
    padding_key = "draft_padding_rows" if draft else "target_padding_rows"
    rows = sum_field(steps, rows_key) / rounds if rounds else 0.0
    mean_max_kv_len = sum_field(steps, "max_kv_len") / rounds if rounds else 0.0
    live_slots = int(steps[0].get("live_slots", 0)) if steps else 0
    weight_bytes = float(model["weight_bytes"])
    active_params = float(model["active_params"])
    kv_bytes = float(model["kv_bytes_per_token_per_seq"])
    bytes_moved = weight_bytes + kv_bytes * mean_max_kv_len * live_slots
    flops = 2.0 * active_params * rows
    arithmetic_intensity = flops / bytes_moved if bytes_moved else 0.0
    machine_balance = (
        float(device["fp16_tflops"]) * 1_000.0 / float(device["mem_bw_gbps"])
    )
    headroom = arithmetic_intensity / machine_balance if machine_balance else None
    boundness = "bandwidth" if arithmetic_intensity < machine_balance else "compute"
    total_rows = sum_field(steps, rows_key)
    padding_fraction = (
        sum_field(steps, padding_key) / total_rows if total_rows else 0.0
    )
    padding_note = None
    if padding_fraction:
        if boundness == "bandwidth":
            padding_note = "padding ≈ free when bandwidth-bound"
        else:
            padding_note = f"padding is {padding_fraction:.1%} of executed rows"
    return {
        "class": boundness,
        "reason": "analytic roofline",
        "device": device_key,
        "device_name": device.get("name", device_key),
        "model": model_key,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "machine_balance_flops_per_byte": machine_balance,
        "headroom": headroom,
        "estimated_flops": flops,
        "estimated_bytes": bytes_moved,
        "mean_rows": rows,
        "mean_max_kv_len": mean_max_kv_len,
        "padding_fraction": padding_fraction,
        "padding_note": padding_note,
    }


def _group_payload(
    steps: list[dict[str, Any]],
    path: str,
    cohort: int,
    metadata: dict[str, Any],
    catalog: dict[str, Any],
    device_key: str | None,
    classification_resolver: Any = None,
) -> dict[str, Any]:
    per_round = [step_phase_buckets(step) for step in steps]
    phase_names = sorted(
        {phase for buckets in per_round for phase in buckets},
        key=phase_sort_key,
    )
    round_count = len(steps)
    token_count = sum(_decode_tokens(step) for step in steps)
    prefill_tokens = sum_field(steps, "executed_prefill_tokens")
    serviced_tokens = token_count + prefill_tokens
    total_ns = sum(max(0, int(step.get("duration_ns", 0))) for step in steps)
    phases = []
    for phase in phase_names:
        values = [int(buckets.get(phase, 0)) for buckets in per_round]
        phase_total = sum(values)
        classification = (
            classification_resolver(phase) if classification_resolver else None
        )
        if classification is None:
            classification = _classification(
                phase, steps, metadata, catalog, device_key
            )
        phases.append({
            "phase": phase,
            "total_ns": phase_total,
            "ns_per_round": phase_total / round_count if round_count else None,
            "ns_per_token": phase_total / token_count if token_count else None,
            "ns_per_serviced_token": (
                phase_total / serviced_tokens if serviced_tokens else None
            ),
            "wall_share": phase_total / total_ns if total_ns else None,
            "p50_ns": optional_number(percentile(values, 0.50)),
            "p95_ns": optional_number(percentile(values, 0.95)),
            "classification": classification,
        })
    return {
        "path": path,
        "cohort": cohort,
        "label": f"C={cohort}",
        "rounds": round_count,
        "durable_tokens": token_count,
        "executed_prefill_tokens": prefill_tokens,
        "serviced_tokens": serviced_tokens,
        "total_ns": total_ns,
        "mean_target_rows": sum_field(steps, "target_rows") / round_count,
        "mean_draft_rows": sum_field(steps, "draft_rows") / round_count,
        "mean_max_kv_len": sum_field(steps, "max_kv_len") / round_count,
        "phases": phases,
    }


def _inherited_classification_resolver(
    paths_payload: dict[str, list[dict[str, Any]]],
) -> Any:
    """Resolve merged-group boundness from per-path rooflines.

    A merged "all" group blends packed and speculative row shapes that never
    executed together, so its roofline must come from the per-path groups
    rather than a blended mean.
    """
    by_cohort_phase: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for path, groups in paths_payload.items():
        for group in groups:
            for phase in group["phases"]:
                by_cohort_phase.setdefault(
                    (group["cohort"], phase["phase"]), {}
                )[path] = phase["classification"]

    def resolver_for(cohort: int) -> Any:
        def resolve(phase_name: str) -> dict[str, Any] | None:
            if phase_name not in {"target_compute", "draft_compute"}:
                return None
            by_path = by_cohort_phase.get((cohort, phase_name))
            if not by_path:
                return None
            if len(by_path) == 1:
                path, classification = next(iter(by_path.items()))
                return {**classification, "inherited_from": path}
            per_path = {
                path: {
                    "class": classification["class"],
                    "arithmetic_intensity_flops_per_byte": classification.get(
                        "arithmetic_intensity_flops_per_byte"
                    ),
                    "headroom": classification.get("headroom"),
                }
                for path, classification in sorted(by_path.items())
            }
            classes = {c["class"] for c in per_path.values()}
            if len(classes) == 1:
                return {
                    "class": classes.pop(),
                    "reason": "inherited: consistent across paths",
                    "per_path": per_path,
                }
            return {
                "class": "mixed",
                "reason": "paths disagree: " + ", ".join(
                    f"{path}={c['class']}"
                    for path, c in sorted(per_path.items())
                ),
                "per_path": per_path,
            }
        return resolve

    return resolver_for


def build_phase_groups(
    records: list[dict[str, Any]],
    *,
    device_specs: dict[str, Any] | None = None,
    device_key: str | None = None,
) -> dict[str, Any]:
    metadata = next(
        (record for record in records if record.get("type") == "metadata"), {}
    )
    steps = [record for record in records if record.get("type") == "step"]
    catalog = device_specs or {}
    all_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    path_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        path = folded_atom(step.get("path", "unknown"))
        cohort = max(0, int(step.get("live_slots", 0)))
        all_groups[cohort].append(step)
        path_groups[path, cohort].append(step)
    paths_payload = {
        path: [
            _group_payload(group, path, cohort, metadata, catalog, device_key)
            for (group_path, cohort), group in sorted(path_groups.items())
            if group_path == path
        ]
        for path in sorted({path for path, _ in path_groups})
    }
    resolver_for = _inherited_classification_resolver(paths_payload)
    return {
        "all": [
            _group_payload(
                group, "all", cohort, metadata, catalog, device_key,
                classification_resolver=resolver_for(cohort),
            )
            for cohort, group in sorted(all_groups.items())
        ],
        "paths": paths_payload,
        "inter_round_idle": inter_round_idle(steps),
    }


def format_folded_weight(duration_ns: int, tokens: int | None) -> str:
    if tokens is None or duration_ns % tokens == 0:
        return str(duration_ns if tokens is None else duration_ns // tokens)
    return f"{duration_ns / tokens:.6f}".rstrip("0").rstrip(".")


def build_folded(
    records: list[dict[str, Any]],
    *,
    stack: tuple[str, ...] = ("path", "cohort", "phase"),
    per_token: bool = False,
) -> str:
    if len(stack) != 3 or set(stack) != {"path", "cohort", "phase"}:
        raise ValueError("stack must be a permutation of path,cohort,phase")
    phase_groups = build_phase_groups(records)
    lines: list[str] = []
    for path_groups in phase_groups["paths"].values():
        for group in path_groups:
            if per_token and not group["durable_tokens"]:
                continue
            for phase in group["phases"]:
                frames = {
                    "path": group["path"],
                    "cohort": group["label"],
                    "phase": phase["phase"],
                }
                folded_stack = ";".join(frames[dimension] for dimension in stack)
                lines.append(
                    f"{folded_stack} "
                    f"{format_folded_weight(phase['total_ns'], group['durable_tokens'] if per_token else None)}"
                )
    idle_ns = phase_groups["inter_round_idle"]["total_ns"]
    if not per_token and idle_ns:
        lines.append(f"idle;inter_round {idle_ns}")
    return "" if not lines else "\n".join(sorted(lines)) + "\n"


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = next(
        (record for record in records if record["type"] == "metadata"), {}
    )
    footer = next(
        (record for record in reversed(records) if record["type"] == "footer"), {}
    )
    steps = [record for record in records if record["type"] == "step"]
    requests = [record for record in records if record["type"] == "request"]
    bursts = [record for record in records if record["type"] == "token_burst"]

    phases: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    cohorts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    proposed_by_position: list[int] = []
    accepted_by_position: list[int] = []
    for step in steps:
        live_slots = int(step.get("live_slots", 0))
        cohorts[live_slots].append(step)
        paths[str(step.get("path", "unknown"))] += 1
        for span in step.get("phases", []):
            phases[str(span.get("phase", "unknown"))] += int(
                span.get("duration_ns", 0)
            )
        for lane in step.get("lanes", []):
            if lane.get("kind") == "decode":
                decisions[str(lane.get("spec", "none"))] += 1
        for source, target in (
            (step.get("proposed_by_position", []), proposed_by_position),
            (step.get("accepted_by_position", []), accepted_by_position),
        ):
            if len(target) < len(source):
                target.extend([0] * (len(source) - len(target)))
            for index, value in enumerate(source):
                target[index] += int(value)

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
    burst_times: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for burst in bursts:
        burst_times[int(burst["request_id"])].append(
            (int(burst["ready_ns"]), int(burst.get("token_count", 0)))
        )
    inter_burst_ms: list[float] = []
    for request_bursts in burst_times.values():
        request_bursts.sort()
        for previous, current in zip(request_bursts, request_bursts[1:]):
            inter_burst_ms.append(
                (current[0] - previous[0]) / 1_000_000 / max(1, current[1])
            )

    funnel = {key: sum_field(steps, key) for key in FUNNEL_KEYS}
    acceptance_by_position = [
        optional_number(ratio(accepted, proposed))
        for proposed, accepted in zip(proposed_by_position, accepted_by_position)
    ]
    cohort_summary: dict[str, Any] = {}
    for live_slots, cohort in sorted(cohorts.items()):
        target_rows = sum_field(cohort, "target_rows")
        target_padding = sum_field(cohort, "target_padding_rows")
        draft_rows = sum_field(cohort, "draft_rows")
        draft_padding = sum_field(cohort, "draft_padding_rows")
        proposed = sum_field(cohort, "spec_proposed_draft_tokens")
        accepted = sum_field(cohort, "spec_accepted_draft_tokens")
        cohort_summary[str(live_slots)] = {
            "rounds": len(cohort),
            "mean_round_ms": statistics.fmean(
                int(step.get("duration_ns", 0)) for step in cohort
            ) / 1_000_000,
            "target_padding_ratio": optional_number(ratio(target_padding, target_rows)),
            "draft_padding_ratio": optional_number(ratio(draft_padding, draft_rows)),
            "draft_acceptance_ratio": optional_number(ratio(accepted, proposed)),
            "paths": dict(sorted(Counter(
                str(step.get("path", "unknown")) for step in cohort
            ).items())),
        }

    target_rows = sum_field(steps, "target_rows")
    target_padding = sum_field(steps, "target_padding_rows")
    draft_rows = sum_field(steps, "draft_rows")
    draft_padding = sum_field(steps, "draft_padding_rows")
    return {
        "schema": "lucebox.concurrency.summary.v1",
        "run": {
            key: value for key, value in metadata.items()
            if key not in {"type", "schema"}
        },
        "capture": {
            "complete": bool(footer.get("complete", False)),
            "rounds": len(steps),
            "requests": len(requests),
            "failed_requests": sum(
                int(request.get("completed_ns", 0)) != 0 and request.get("ok") is False
                for request in requests
            ),
            "dropped_steps": int(footer.get("dropped_steps", 0)),
            "dropped_requests": int(footer.get("dropped_requests", 0)),
            "dropped_token_bursts": int(footer.get("dropped_token_bursts", 0)),
            "paths": dict(sorted(paths.items())),
        },
        "latency_ms": {
            "queue": percentile_pair(queue_ms),
            "ttft": percentile_pair(ttft_ms),
            "end_to_end": percentile_pair(e2e_ms),
            "inter_burst_per_token": percentile_pair(inter_burst_ms),
        },
        "speculation": {
            **funnel,
            "tree_widths": sorted({
                int(step.get("spec_tree_width", 0)) for step in steps
                if int(step.get("spec_tree_width", 0)) > 0
            }),
            "decisions": dict(sorted(decisions.items())),
            "proposed_by_position": proposed_by_position,
            "accepted_by_position": accepted_by_position,
            "acceptance_by_position": acceptance_by_position,
        },
        "padding": {
            "target_rows": target_rows,
            "target_padding_rows": target_padding,
            "target_padding_ratio": optional_number(ratio(target_padding, target_rows)),
            "draft_rows": draft_rows,
            "draft_padding_rows": draft_padding,
            "draft_padding_ratio": optional_number(ratio(draft_padding, draft_rows)),
        },
        "cohorts": cohort_summary,
        "phase_ns": dict(phases.most_common()),
    }


def _capture_bounds(records: list[dict[str, Any]]) -> dict[str, int | None]:
    timestamps: list[int] = []
    request_keys = (
        "queued_ns",
        "admitted_ns",
        "prefill_completed_ns",
        "first_token_ns",
        "completed_ns",
    )
    for record in records:
        record_type = record.get("type")
        if record_type == "step":
            started_ns = int(record.get("started_ns", 0))
            if started_ns > 0:
                timestamps.append(started_ns)
                timestamps.append(
                    started_ns + max(0, int(record.get("duration_ns", 0)))
                )
        elif record_type == "request":
            timestamps.extend(
                timestamp
                for key in request_keys
                if (timestamp := int(record.get(key, 0))) > 0
            )
        elif record_type == "token_burst":
            ready_ns = int(record.get("ready_ns", 0))
            if ready_ns > 0:
                timestamps.append(ready_ns)
    earliest_ns = min(timestamps) if timestamps else None
    latest_ns = max(timestamps) if timestamps else None

    return {
        "earliest_steady_ns": earliest_ns,
        "latest_steady_ns": latest_ns,
        "duration_ns": (
            latest_ns - earliest_ns if earliest_ns is not None else None
        ),
    }


def _request_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    requests = sorted(
        (record for record in records if record.get("type") == "request"),
        key=lambda request: (
            int(request.get("queued_ns", 0)),
            int(request.get("request_id", 0)),
        ),
    )
    embedded = requests[:1000]
    boundaries = [
        int(request.get(key, 0))
        for request in embedded
        for key in (
            "queued_ns", "admitted_ns", "prefill_completed_ns",
            "first_token_ns", "completed_ns",
        )
        if int(request.get(key, 0)) > 0
    ]
    origin = min(boundaries) if boundaries else 0
    end = max(boundaries) if boundaries else origin

    def span(request: dict[str, Any], start_key: str, end_key: str) -> int | None:
        start = int(request.get(start_key, 0))
        finish = int(request.get(end_key, 0))
        return finish - start if start and finish >= start else None

    return {
        "total": len(requests),
        "embedded": len(embedded),
        "display_limit": 200,
        "origin_ns": origin,
        "end_ns": end,
        "rows": [
            {
                "request_id": int(request.get("request_id", 0)),
                "ok": request.get("ok"),
                "open_ended": request.get("ok") is None or not int(request.get("completed_ns", 0)),
                "prompt_tokens": int(request.get("prompt_tokens", 0)),
                "output_tokens": int(request.get("output_tokens", 0)),
                "start_offset_ns": max(0, int(request.get("queued_ns", 0)) - origin),
                "queue_ns": span(request, "queued_ns", "admitted_ns"),
                "prefill_ns": span(request, "admitted_ns", "prefill_completed_ns"),
                "first_decode_ns": span(request, "prefill_completed_ns", "first_token_ns"),
                "decode_ns": span(request, "first_token_ns", "completed_ns"),
            }
            for request in embedded
        ],
    }


def _run_notices(
    records: list[dict[str, Any]],
    catalog: dict[str, Any],
    device_key: str | None,
) -> list[dict[str, str]]:
    metadata = next(
        (record for record in records if record.get("type") == "metadata"), {}
    )
    notices: list[dict[str, str]] = []
    devices = catalog.get("devices", {})
    if not device_key:
        notices.append({
            "kind": "device",
            "level": "warn",
            "message": "No device selected. Boundness coloring is neutral. Pass --device with a checked-in device key.",
        })
    elif not isinstance(devices, dict) or device_key not in devices:
        notices.append({
            "kind": "device",
            "level": "warn",
            "message": f"No device spec for {device_key}. Boundness coloring is neutral.",
        })
    else:
        target_key, target_spec = _resolve_model_spec(metadata, catalog)
        if target_spec is None:
            notices.append({
                "kind": "model",
                "level": "warn",
                "message": "No target model spec matched model_name or the model_path basename. Target compute is neutral.",
            })
        draft_compute = any(
            span.get("phase") == "draft_compute"
            for record in records if record.get("type") == "step"
            for span in record.get("phases", [])
        )
        draft_key, draft_spec = _resolve_model_spec(metadata, catalog, draft=True)
        if draft_compute and draft_spec is None:
            notices.append({
                "kind": "model",
                "level": "warn",
                "message": "No draft model spec matched draft_path. Draft compute remains overhead.",
            })
        del target_key, draft_key
    return notices


def _device_payload(catalog: dict[str, Any], device_key: str | None) -> dict[str, Any]:
    devices = catalog.get("devices", {})
    device = devices.get(device_key) if isinstance(devices, dict) and device_key else None
    if not isinstance(device, dict):
        return {"key": device_key, "known": False, "name": device_key or "not selected"}
    return {
        "key": device_key,
        "known": True,
        "name": device.get("name", device_key),
        "mem_bw_gbps": device.get("mem_bw_gbps"),
        "fp16_tflops": device.get("fp16_tflops"),
        "note": device.get("note", ""),
    }


def build_run_payload(
    records: list[dict[str, Any]],
    *,
    device_specs: dict[str, Any] | None = None,
    device_key: str | None = None,
) -> ReportPayload:
    catalog = device_specs or {}
    summary = build_summary(records)
    cohorts = sorted(int(cohort) for cohort in summary["cohorts"])
    return {
        "run": summary["run"],
        "capture": summary["capture"],
        "capture_bounds": _capture_bounds(records),
        "latency_ms": summary["latency_ms"],
        "phase_groups": build_phase_groups(
            records, device_specs=catalog, device_key=device_key
        ),
        "requests": _request_payload(records),
        "speculation": summary["speculation"],
        "padding": summary["padding"],
        "device": _device_payload(catalog, device_key),
        "notices": _run_notices(records, catalog, device_key),
        "mixed_run_cohorts": len(cohorts) > 1,
        "cohorts": cohorts,
    }


def _group_index(run: ReportPayload) -> dict[tuple[str, int, str], dict[str, Any]]:
    # Per-path groups only: an "all" row would repeat every per-path row's
    # time and double-count the table.
    return {
        (group["path"], group["cohort"], phase["phase"]): phase
        for path_groups in run["phase_groups"]["paths"].values()
        for group in path_groups
        for phase in group["phases"]
    }


def _diff_payload(current: ReportPayload, baseline: ReportPayload) -> dict[str, Any]:
    warnings: list[str] = []
    for key, label in (("model_name", "model"), ("arch", "model architecture")):
        current_value = current["run"].get(key)
        baseline_value = baseline["run"].get(key)
        if current_value != baseline_value:
            warnings.append(
                f"Baseline {label} mismatch: {baseline_value or 'unknown'} vs {current_value or 'unknown'}."
            )
    current_device = current["device"].get("key")
    baseline_device = baseline["device"].get("key")
    if current_device != baseline_device:
        warnings.append(
            f"Baseline device mismatch: {baseline_device or 'not selected'} "
            f"vs {current_device or 'not selected'}."
        )
    current_index = _group_index(current)
    baseline_index = _group_index(baseline)
    rows = []
    for path, cohort, phase_name in sorted(
        current_index.keys() | baseline_index.keys(),
        key=lambda key: (key[0], key[1], phase_sort_key(key[2])),
    ):
        current_phase = current_index.get((path, cohort, phase_name), {})
        baseline_phase = baseline_index.get((path, cohort, phase_name), {})
        current_value = current_phase.get("ns_per_token")
        baseline_value = baseline_phase.get("ns_per_token")
        delta = (
            current_value - baseline_value
            if current_value is not None and baseline_value is not None else None
        )
        delta_percent = (
            delta / baseline_value
            if delta is not None and baseline_value else None
        )
        rows.append({
            "path": path,
            "cohort": cohort,
            "phase": phase_name,
            "baseline_ns_per_token": baseline_value,
            "current_ns_per_token": current_value,
            "delta_ns_per_token": delta,
            "delta_percent": delta_percent,
        })
    rows.sort(key=lambda row: (
        row["delta_ns_per_token"] is None,
        -abs(row["delta_ns_per_token"] or 0.0),
    ))
    return {"warnings": warnings, "rows": rows}


def build_report_payload(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]] | None = None,
    *,
    device_specs: dict[str, Any] | None = None,
    device_key: str | None = None,
    baseline_device_key: str | None = None,
) -> ReportPayload:
    current = build_run_payload(
        records, device_specs=device_specs, device_key=device_key
    )
    baseline = (
        build_run_payload(
            baseline_records,
            device_specs=device_specs,
            device_key=baseline_device_key or device_key,
        )
        if baseline_records is not None else None
    )
    return {
        "schema": "lucebox.concurrency.report.v1",
        "phase_order": list(PHASE_ORDER),
        "current": current,
        "baseline": baseline,
        "diff": _diff_payload(current, baseline) if baseline is not None else None,
    }
