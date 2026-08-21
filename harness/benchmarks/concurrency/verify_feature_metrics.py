#!/usr/bin/env python3
"""Fail closed unless server telemetry proves requested features executed."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PREFIX = "[concurrency-metrics] "
COUNTERS = (
    "ddtree_steps", "ddtree_suspensions", "ddtree_accepted_tokens",
    "spec_steps", "spec_accepted_tokens",
    "target_forwards", "kvflash_page_ins", "kvflash_page_outs",
    "kvflash_reselects",
)
DECODE_MODES = ("ar", "speculation", "adaptive")
SPECULATOR_STARTUP_PREFIX = "[parallel-chain] speculator="
SPEC_PROFILE_PREFIX = "[spec-profile] context="
REQUIRED_KEYS = (
    "request_id", "effective_prompt_tokens", *COUNTERS,
    "kvflash_resident_blocks", "pflash_applied", "pflash_input_tokens",
    "pflash_output_tokens",
)


def parse_markers(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        raw = line[marker + len(PREFIX):]
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid concurrency metric JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: concurrency metric must be an object")
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"{path}:{line_no}: missing telemetry keys {missing}")
        rows.append(row)
    return rows


def measured_requests(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for level in report.get("levels") or []:
        for row in level.get("requests_detail") or []:
            if "error" not in row:
                raise ValueError("bench report request lacks an explicit error status")
            if row["error"] is not None:
                continue
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("bench report lacks a request ID for a successful request")
            if request_id in requests:
                raise ValueError(f"duplicate measured request ID {request_id}")
            requests[request_id] = row
    if not requests:
        raise ValueError("bench report has no successful measured requests")
    return requests


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("telemetry request_id must be a non-empty string")
        if request_id in aggregate:
            raise ValueError(f"duplicate telemetry request ID {request_id}")
        dst = {
            "request_id": request_id,
            "effective_prompt_tokens": row["effective_prompt_tokens"],
            "kvflash_resident_blocks": row["kvflash_resident_blocks"],
            "pflash_applied": False,
            "pflash_input_tokens": row["pflash_input_tokens"],
            "pflash_output_tokens": row["pflash_output_tokens"],
            **{key: 0 for key in COUNTERS},
        }
        aggregate[request_id] = dst
        for key in COUNTERS:
            value = row[key]
            if key == "ddtree_suspensions":
                if type(value) is not int or value not in (0, 1):
                    raise ValueError(
                        f"{request_id}: ddtree_suspensions must be 0 or 1 "
                        "per request"
                    )
            elif type(value) is not int or value < 0:
                raise ValueError(f"{request_id}: {key} must be a non-negative integer")
            dst[key] += value
        for key in (
            "effective_prompt_tokens", "kvflash_resident_blocks",
            "pflash_input_tokens", "pflash_output_tokens",
        ):
            value = row[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"{request_id}: {key} must be a non-negative integer")
            dst[key] = value
        if not isinstance(row["pflash_applied"], bool):
            raise ValueError(f"{request_id}: pflash_applied must be boolean")
        dst["pflash_applied"] = dst["pflash_applied"] or row["pflash_applied"]
    for request_id, row in aggregate.items():
        if row["ddtree_suspensions"] not in (0, 1):
            raise ValueError(
                f"{request_id}: ddtree_suspensions must be 0 or 1 per request"
            )
        if row["spec_accepted_tokens"] > 0 and row["spec_steps"] == 0:
            raise ValueError(
                f"{request_id}: spec_accepted_tokens requires positive spec_steps"
            )
    return aggregate


def verify(
    report: dict[str, Any], markers: list[dict[str, Any]], expected: set[str],
    server_log_text: str | None = None,
) -> dict[str, Any]:
    measured = measured_requests(report)
    all_rows = aggregate_rows(markers)
    rows = {request_id: all_rows[request_id] for request_id in measured if request_id in all_rows}
    errors: list[str] = []
    missing = sorted(set(measured) - set(rows))
    if missing:
        errors.append(f"missing concurrency telemetry for {len(missing)} measured request(s): {missing}")

    metadata = report.get("server_metadata") or {}
    feature_config = metadata.get("feature_config") or {}
    decode_mode = feature_config.get("decode_mode")
    if decode_mode is not None and decode_mode not in DECODE_MODES:
        errors.append(f"invalid recorded decode_mode {decode_mode!r}")

    for request_id, measured_row in measured.items():
        metric = rows.get(request_id)
        if metric is None:
            continue
        wire_effective = measured_row.get("effective_prompt_tokens")
        if type(wire_effective) is not int or wire_effective < 0:
            errors.append(f"{request_id}: missing usage.timings.effective_prompt_tokens")
        elif metric["effective_prompt_tokens"] != wire_effective:
            errors.append(
                f"{request_id}: log effective_prompt_tokens={metric['effective_prompt_tokens']} "
                f"does not match wire value {wire_effective}"
            )
        if "pflash" in expected:
            wire_input = measured_row.get("prompt_tokens")
            if type(wire_input) is not int or wire_input < 0:
                errors.append(f"{request_id}: missing usage.prompt_tokens")
            elif metric["pflash_input_tokens"] != wire_input:
                errors.append(
                    f"{request_id}: log pflash_input_tokens="
                    f"{metric['pflash_input_tokens']} does not match wire value "
                    f"{wire_input}"
                )
        if "ddtree" in expected:
            if metric["ddtree_steps"] <= 0:
                errors.append(f"{request_id}: DDTree requested but ddtree_steps is zero")
            if metric["target_forwards"] <= 0:
                errors.append(f"{request_id}: DDTree requested but target_forwards is zero")
        if "chain" in expected:
            if metric["target_forwards"] <= 0:
                errors.append(
                    f"{request_id}: chain decode requested but target_forwards is zero"
                )
            if metric["ddtree_steps"] != 0 or metric["ddtree_accepted_tokens"] != 0:
                errors.append(
                    f"{request_id}: chain run must keep DDTree counters at zero"
                )
            if metric["ddtree_suspensions"] != 0:
                errors.append(
                    f"{request_id}: chain run must keep ddtree_suspensions at zero"
                )
            if decode_mode == "speculation" and metric["spec_steps"] <= 0:
                errors.append(
                    f"{request_id}: forced chain speculation requested but "
                    "spec_steps is zero"
                )
        if "pflash" in expected:
            if metric["pflash_applied"] is not True:
                errors.append(f"{request_id}: PFlash requested but pflash_applied is false")
            if not (0 < metric["pflash_output_tokens"] < metric["pflash_input_tokens"]):
                errors.append(
                    f"{request_id}: PFlash did not reduce prompt tokens "
                    f"({metric['pflash_input_tokens']} -> {metric['pflash_output_tokens']})"
                )

    if "chain" in expected:
        if decode_mode not in ("speculation", "adaptive"):
            errors.append(
                "chain requested but metadata decode_mode is not speculation or adaptive"
            )
        log_text = server_log_text or ""
        startup_mode = (
            isinstance(decode_mode, str)
            and re.search(
                rf"^.*{re.escape(SPECULATOR_STARTUP_PREFIX)}\S+.*$",
                log_text,
                flags=re.MULTILINE,
            )
        )
        if not startup_mode:
            errors.append(
                "chain requested but matching speculator adapter startup proof is missing"
            )
        if decode_mode == "adaptive" and SPEC_PROFILE_PREFIX not in log_text:
            errors.append(
                "adaptive chain requested but startup cost-profile proof is missing"
            )
    elif decode_mode == "ar":
        active_spec = sorted(
            request_id for request_id, row in rows.items()
            if row["spec_steps"] != 0 or row["spec_accepted_tokens"] != 0
        )
        if active_spec:
            errors.append(
                f"AR decode_mode emitted chain speculation for request(s): {active_spec}"
            )

    totals = {key: sum(row[key] for row in rows.values()) for key in COUNTERS}
    resident = [row["kvflash_resident_blocks"] for row in rows.values()]
    variant = str(metadata.get("variant") or "")
    workload = str(metadata.get("workload") or "")
    pflash_mode = feature_config.get("prefill_compression")
    pflash_threshold = feature_config.get("prefill_threshold")
    if "pflash" in expected:
        if not isinstance(pflash_mode, str) or pflash_mode in ("", "off", "0"):
            errors.append("PFlash requested but metadata does not prove it was enabled")
        if pflash_mode == "auto":
            if type(pflash_threshold) is not int or pflash_threshold <= 0:
                errors.append(
                    "PFlash auto row lacks a positive recorded token threshold"
                )
            else:
                below_threshold = sorted(
                    request_id for request_id, row in rows.items()
                    if row["pflash_input_tokens"] < pflash_threshold
                )
                if below_threshold:
                    errors.append(
                        "PFlash auto input did not reach its recorded token threshold "
                        f"for request(s): {below_threshold}"
                    )
    kvflash_mode = feature_config.get("kvflash")
    requested_pool_tokens = feature_config.get("kvflash_max_pool_tokens")
    runtime_observed = metadata.get("runtime_observed") or {}
    pool_tokens = runtime_observed.get("physical_kv_pool_tokens")
    kvflash_page_traffic_required = False
    kvflash_page_traffic_reason = "not-requested"
    if "kvflash" in expected:
        if not isinstance(kvflash_mode, str) or kvflash_mode in ("", "off", "0"):
            errors.append("KVFlash requested but metadata does not prove it was enabled")
        if (type(requested_pool_tokens) is not int
                or requested_pool_tokens <= 0):
            errors.append(
                "KVFlash requested but its recorded pool-token cap is not a positive integer")
        scorer_drafter = feature_config.get("kvflash_scorer_drafter")
        scorer_sha256 = feature_config.get("kvflash_scorer_drafter_sha256")
        if not isinstance(scorer_drafter, str) or not scorer_drafter:
            errors.append(
                "KVFlash requested but no explicit scorer drafter was recorded"
            )
        if (
            not isinstance(scorer_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", scorer_sha256) is None
        ):
            errors.append(
                "KVFlash requested but the scorer drafter hash is not a valid SHA-256 digest"
            )
        if runtime_observed.get("kvflash_active") is not True:
            errors.append(
                "KVFlash requested but its physical-pool startup marker was not recorded"
            )
        if type(pool_tokens) is not int or pool_tokens <= 0:
            errors.append(
                "KVFlash requested but no positive startup-observed physical pool was recorded"
            )
        if not resident or max(resident) <= 0:
            errors.append("KVFlash requested but resident block count never became positive")

        effective_demand_by_level = []
        for level in report.get("levels") or []:
            demand = 0
            for request in level.get("requests_detail") or []:
                request_id = request.get("request_id")
                if request.get("error") is None and request_id in rows:
                    demand += rows[request_id]["effective_prompt_tokens"]
            effective_demand_by_level.append(demand)
        if variant not in ("kvflash", "full"):
            kvflash_page_traffic_required = True
            kvflash_page_traffic_reason = "unknown-variant"
        elif variant == "kvflash":
            kvflash_page_traffic_required = True
            kvflash_page_traffic_reason = "kvflash-only-ablation"
        elif workload == "kv-pressure":
            kvflash_page_traffic_required = True
            kvflash_page_traffic_reason = "kv-pressure-workload"
        elif variant == "full":
            if type(pool_tokens) is not int or pool_tokens <= 0:
                errors.append(
                    "full KVFlash row lacks a positive recorded pool-token limit"
                )
                kvflash_page_traffic_reason = "missing-pool-limit"
            else:
                kvflash_page_traffic_required = any(
                    demand > pool_tokens
                    for demand in effective_demand_by_level
                )
                kvflash_page_traffic_reason = (
                    "effective-prompt-exceeds-pool"
                    if kvflash_page_traffic_required
                    else "compressed-prompt-fits-pool"
                )

        if (
            kvflash_page_traffic_required
            and totals["kvflash_page_ins"] + totals["kvflash_page_outs"] <= 0
        ):
            errors.append(
                "KVFlash paging was required but no page-in/page-out was observed"
            )

    return {
        "schema_version": 5,
        "decode_mode": decode_mode,
        "expected_features": sorted(expected),
        "valid": not errors,
        "errors": errors,
        "measured_request_count": len(measured),
        "matched_metric_count": len(rows),
        "ignored_marker_count": len(markers) - len(rows),
        "kvflash_page_traffic_required": kvflash_page_traffic_required,
        "kvflash_page_traffic_reason": kvflash_page_traffic_reason,
        "kvflash_pool_tokens": pool_tokens if type(pool_tokens) is int else None,
        "kvflash_requested_max_pool_tokens": (
            requested_pool_tokens if type(requested_pool_tokens) is int else None
        ),
        "pflash_threshold_tokens": (
            pflash_threshold if type(pflash_threshold) is int else None
        ),
        "aggregate": {
            **totals,
            "kvflash_resident_blocks_max": max(resident) if resident else None,
            "pflash_applied_requests": sum(row["pflash_applied"] is True for row in rows.values()),
            "pflash_input_tokens": sum(row["pflash_input_tokens"] for row in rows.values()),
            "pflash_output_tokens": sum(row["pflash_output_tokens"] for row in rows.values()),
        },
        "requests": [rows[key] for key in sorted(rows)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument(
        "--expect", action="append",
        choices=("ddtree", "chain", "pflash", "kvflash"), default=[],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.bench.read_text(encoding="utf-8"))
        log_text = args.server_log.read_text(encoding="utf-8", errors="replace")
        result = verify(
            report, parse_markers(args.server_log), set(args.expect), log_text,
        )
    except Exception as exc:
        print(f"[proof] error: {exc}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["valid"]:
        for error in result["errors"]:
            print(f"[proof] {error}", file=sys.stderr)
        return 1
    print(
        f"[proof] valid features={','.join(result['expected_features']) or 'ar'} "
        f"requests={result['matched_metric_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
