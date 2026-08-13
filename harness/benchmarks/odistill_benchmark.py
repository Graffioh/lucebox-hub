#!/usr/bin/env python3
"""Prepare and compare leakage-resistant ODistill benchmark folds.

This script deliberately does not start a model server. It creates deterministic
suite-stratified adaptation/held-out folds from the repository's existing
benchmark prompts, then compares frozen baseline and adapted reports produced by
``harness/client_test_runner.py bench``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable


SUITE_FILES = {
    "he": "bench_he.jsonl",
    "gsm": "bench_gsm.jsonl",
    "math": "bench_math.jsonl",
    "agent": "bench_agent.jsonl",
}
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_no}: missing string id")
            rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate case id")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stable_order(rows: list[dict[str, Any]], suite: str, seed: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        material = f"{seed}\0{suite}\0{row['id']}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    return sorted(rows, key=key)


def prepare_folds(
    prompts_dir: Path,
    out_dir: Path,
    suites: list[str],
    folds: int,
    seed: str,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    source: dict[str, list[dict[str, Any]]] = {}
    for suite in suites:
        rows = _read_jsonl(prompts_dir / SUITE_FILES[suite])
        if len(rows) < folds:
            raise ValueError(f"suite {suite} has {len(rows)} cases, fewer than {folds} folds")
        source[suite] = _stable_order(rows, suite, seed)

    manifest: dict[str, Any] = {
        "schema": 1,
        "seed": seed,
        "folds": folds,
        "suites": suites,
        "source_dir": str(prompts_dir),
        "fold": [],
    }
    heldout_occurrences: dict[str, int] = {}
    for fold in range(folds):
        fold_entry: dict[str, Any] = {"index": fold, "suites": {}}
        for suite, ordered in source.items():
            heldout = [row for i, row in enumerate(ordered) if i % folds == fold]
            adapt = [row for i, row in enumerate(ordered) if i % folds != fold]
            adapt_ids = {row["id"] for row in adapt}
            heldout_ids = {row["id"] for row in heldout}
            if adapt_ids & heldout_ids:
                raise AssertionError(f"internal split overlap in {suite} fold {fold}")
            for case_id in heldout_ids:
                key = f"{suite}:{case_id}"
                heldout_occurrences[key] = heldout_occurrences.get(key, 0) + 1

            filename = SUITE_FILES[suite]
            _write_jsonl(out_dir / f"fold-{fold}" / "adapt" / filename, adapt)
            _write_jsonl(out_dir / f"fold-{fold}" / "heldout" / filename, heldout)
            fold_entry["suites"][suite] = {
                "adapt_ids": sorted(adapt_ids),
                "heldout_ids": sorted(heldout_ids),
            }
        manifest["fold"].append(fold_entry)

    expected = sum(len(rows) for rows in source.values())
    if len(heldout_occurrences) != expected or set(heldout_occurrences.values()) != {1}:
        raise AssertionError("each case must occur in exactly one held-out fold")
    manifest["total_cases"] = expected
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("command") != "bench" or not isinstance(report.get("suites"), dict):
        raise ValueError(f"{path}: not a client_test_runner bench report")
    return report


def _flatten(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flat: dict[str, dict[str, Any]] = {}
    for suite, suite_report in report["suites"].items():
        for result in suite_report.get("results", []):
            key = f"{suite}:{result['id']}"
            if key in flat:
                raise ValueError(f"duplicate result {key}")
            flat[key] = {"suite": suite, **result}
    return flat


def _require_frozen_phase(report: dict[str, Any], expected: str, label: str) -> None:
    actual = report.get("odistill_phase")
    if actual != expected:
        raise ValueError(f"{label} report must use --odistill-phase {expected}, got {actual!r}")
    before = report.get("odistill_before") or {}
    after = report.get("odistill_after") or {}
    if before.get("trainer_alive") or after.get("trainer_alive"):
        raise ValueError(f"{label} report was collected with a live trainer")
    if before.get("adapter_generation") != after.get("adapter_generation"):
        raise ValueError(f"{label} adapter generation changed during held-out evaluation")


def _number(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _server_decode_tps(row: dict[str, Any]) -> float | None:
    timings = row.get("server_timings")
    if isinstance(timings, dict):
        value = timings.get("decode_tokens_per_sec")
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return _number(row, "output_tok_s")


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float | None],
    samples: int,
    seed: int,
) -> list[float] | None:
    if not rows or samples <= 0:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        estimate = metric(sample)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return [lo, hi]


def _hierarchical_bootstrap_ci(
    folds: list[list[dict[str, Any]]],
    metric: Callable[[list[dict[str, Any]]], float | None],
    samples: int,
    seed: int,
) -> list[float] | None:
    if not folds or samples <= 0:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample: list[dict[str, Any]] = []
        for _fold in folds:
            source = folds[rng.randrange(len(folds))]
            sample.extend(source[rng.randrange(len(source))] for _row in source)
        estimate = metric(sample)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return [lo, hi]


def _mean_delta(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        row[f"candidate_{field}"] - row[f"baseline_{field}"]
        for row in rows
        if row.get(f"candidate_{field}") is not None
        and row.get(f"baseline_{field}") is not None
    ]
    return _mean(values)


def compare_reports(
    baseline_path: Path,
    candidate_path: Path,
    reference_path: Path | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_report = _load_report(baseline_path)
    candidate_report = _load_report(candidate_path)
    _require_frozen_phase(baseline_report, "heldout-base", "baseline")
    _require_frozen_phase(candidate_report, "heldout-adapted", "candidate")
    if baseline_report.get("model") != candidate_report.get("model"):
        raise ValueError("baseline and candidate model request names differ")
    if baseline_report["suites"].keys() != candidate_report["suites"].keys():
        raise ValueError("baseline and candidate suite selections differ")
    for suite in baseline_report["suites"]:
        base_fingerprint = baseline_report["suites"][suite].get("prompt_fingerprint")
        candidate_fingerprint = candidate_report["suites"][suite].get("prompt_fingerprint")
        if not base_fingerprint or base_fingerprint != candidate_fingerprint:
            raise ValueError(f"baseline and candidate prompt fingerprints differ for {suite}")
    baseline = _flatten(baseline_report)
    candidate = _flatten(candidate_report)
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            f"reports do not contain identical cases; missing candidate={missing_candidate}, "
            f"missing baseline={missing_baseline}"
        )

    reference = None
    if reference_path is not None:
        reference_report = _load_report(reference_path)
        if reference_report["suites"].keys() != baseline_report["suites"].keys():
            raise ValueError("target-only reference suite selection differs")
        for suite in baseline_report["suites"]:
            if (reference_report["suites"][suite].get("prompt_fingerprint") !=
                    baseline_report["suites"][suite].get("prompt_fingerprint")):
                raise ValueError(f"target-only prompt fingerprint differs for {suite}")
        reference = _flatten(reference_report)
        if reference.keys() != baseline.keys():
            raise ValueError("target-only reference does not contain the identical held-out cases")

    rows: list[dict[str, Any]] = []
    for key in sorted(baseline):
        base = baseline[key]
        cand = candidate[key]
        base_tps = _server_decode_tps(base)
        cand_tps = _server_decode_tps(cand)
        base_accept = _number(base, "accept_rate")
        cand_accept = _number(cand, "accept_rate")
        rows.append({
            "key": key,
            "suite": base["suite"],
            "id": base["id"],
            "baseline_accept_rate": base_accept,
            "candidate_accept_rate": cand_accept,
            "baseline_decode_tok_s": base_tps,
            "candidate_decode_tok_s": cand_tps,
            "speedup": cand_tps / base_tps if base_tps and cand_tps is not None else None,
            "baseline_ttft_s": _number(base, "ttft_s"),
            "candidate_ttft_s": _number(cand, "ttft_s"),
            "baseline_correct": base.get("correct"),
            "candidate_correct": cand.get("correct"),
            "exact_output_match": base.get("text") == cand.get("text"),
            "target_reference_match": (
                cand.get("text") == reference[key].get("text")
                if reference is not None else None
            ),
        })

    summary = {
        "cases": len(rows),
        "exact_output_matches": sum(row["exact_output_match"] for row in rows),
        "target_reference_matches": (
            sum(row["target_reference_match"] is True for row in rows)
            if reference is not None else None
        ),
        "baseline_mean_accept_rate": _mean([
            row["baseline_accept_rate"] for row in rows
            if row["baseline_accept_rate"] is not None
        ]),
        "candidate_mean_accept_rate": _mean([
            row["candidate_accept_rate"] for row in rows
            if row["candidate_accept_rate"] is not None
        ]),
        "mean_accept_rate_delta": _mean_delta(rows, "accept_rate"),
        "baseline_mean_decode_tok_s": _mean([
            row["baseline_decode_tok_s"] for row in rows
            if row["baseline_decode_tok_s"] is not None
        ]),
        "candidate_mean_decode_tok_s": _mean([
            row["candidate_decode_tok_s"] for row in rows
            if row["candidate_decode_tok_s"] is not None
        ]),
        "mean_speedup": _mean([
            row["speedup"] for row in rows if row["speedup"] is not None
        ]),
        "mean_ttft_delta_s": _mean_delta(rows, "ttft_s"),
        "baseline_correct": sum(row["baseline_correct"] is True for row in rows),
        "candidate_correct": sum(row["candidate_correct"] is True for row in rows),
    }
    summary["accept_rate_delta_ci95"] = _bootstrap_ci(
        rows, lambda sample: _mean_delta(sample, "accept_rate"),
        bootstrap_samples, bootstrap_seed,
    )
    summary["speedup_ci95"] = _bootstrap_ci(
        rows,
        lambda sample: _mean([
            row["speedup"] for row in sample if row["speedup"] is not None
        ]),
        bootstrap_samples,
        bootstrap_seed + 1,
    )
    return {
        "schema": 1,
        "baseline_report": str(baseline_path),
        "candidate_report": str(candidate_path),
        "target_reference_report": str(reference_path) if reference_path else None,
        "summary": summary,
        "cases": rows,
    }


def compare_repeats(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Compare duplicate frozen runs without treating timing jitter as drift."""
    first_report = _load_report(first_path)
    second_report = _load_report(second_path)
    phase = first_report.get("odistill_phase")
    if phase not in ("heldout-base", "heldout-adapted"):
        raise ValueError(f"repeat reports must use a frozen held-out phase, got {phase!r}")
    if second_report.get("odistill_phase") != phase:
        raise ValueError("repeat reports use different ODistill phases")
    _require_frozen_phase(first_report, phase, "first repeat")
    _require_frozen_phase(second_report, phase, "second repeat")
    first_gen = (first_report.get("odistill_before") or {}).get("adapter_generation")
    second_gen = (second_report.get("odistill_before") or {}).get("adapter_generation")
    if first_gen != second_gen:
        raise ValueError("repeat reports use different adapter generations")
    if first_report.get("model") != second_report.get("model"):
        raise ValueError("repeat report model request names differ")
    if first_report["suites"].keys() != second_report["suites"].keys():
        raise ValueError("repeat report suite selections differ")
    for suite in first_report["suites"]:
        if (first_report["suites"][suite].get("prompt_fingerprint") !=
                second_report["suites"][suite].get("prompt_fingerprint")):
            raise ValueError(f"repeat prompt fingerprints differ for {suite}")

    first = _flatten(first_report)
    second = _flatten(second_report)
    if first.keys() != second.keys():
        raise ValueError("repeat reports do not contain identical cases")
    rows = []
    for key in sorted(first):
        left, right = first[key], second[key]
        left_accept = _number(left, "accept_rate")
        right_accept = _number(right, "accept_rate")
        left_tps = _server_decode_tps(left)
        right_tps = _server_decode_tps(right)
        rows.append({
            "key": key,
            "exact_output_match": left.get("text") == right.get("text"),
            "exact_acceptance_match": (
                left_accept is not None and left_accept == right_accept),
            "acceptance_delta": (
                right_accept - left_accept
                if left_accept is not None and right_accept is not None else None
            ),
            "decode_speed_ratio": (
                right_tps / left_tps
                if left_tps and right_tps is not None else None
            ),
        })
    deltas = [abs(row["acceptance_delta"]) for row in rows
              if row["acceptance_delta"] is not None]
    ratios = [row["decode_speed_ratio"] for row in rows
              if row["decode_speed_ratio"] is not None]
    return {
        "schema": 1,
        "first_report": str(first_path),
        "second_report": str(second_path),
        "summary": {
            "phase": phase,
            "adapter_generation": first_gen,
            "cases": len(rows),
            "exact_output_matches": sum(row["exact_output_match"] for row in rows),
            "exact_acceptance_matches": sum(
                row["exact_acceptance_match"] for row in rows),
            "max_abs_acceptance_delta": max(deltas, default=None),
            "mean_decode_speed_ratio": _mean(ratios),
        },
        "cases": rows,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# ODistill frozen held-out comparison",
        "",
        "| Metric | Baseline | Adapted | Change |",
        "| --- | ---: | ---: | ---: |",
        f"| Mean acceptance rate | {_fmt(summary['baseline_mean_accept_rate'])} | "
        f"{_fmt(summary['candidate_mean_accept_rate'])} | {_fmt(summary['mean_accept_rate_delta'])} |",
        f"| Mean decode tok/s | {_fmt(summary['baseline_mean_decode_tok_s'], 2)} | "
        f"{_fmt(summary['candidate_mean_decode_tok_s'], 2)} | "
        f"{_fmt(summary['mean_speedup'], 3)}x |",
        f"| Correct | {summary['baseline_correct']}/{summary['cases']} | "
        f"{summary['candidate_correct']}/{summary['cases']} | — |",
        f"| Exact output matches | — | {summary['exact_output_matches']}/{summary['cases']} | — |",
        "",
        f"Acceptance delta 95% paired-bootstrap CI: `{summary['accept_rate_delta_ci95']}`",
        "",
        f"Mean per-case speedup 95% paired-bootstrap CI: `{summary['speedup_ci95']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_suites(value: str) -> list[str]:
    suites = list(SUITE_FILES) if value == "all" else [part.strip() for part in value.split(",")]
    unknown = [suite for suite in suites if suite not in SUITE_FILES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown suites: {','.join(unknown)}")
    return suites


def cmd_prepare(args: argparse.Namespace) -> int:
    manifest = prepare_folds(
        args.prompts_dir, args.out_dir, _parse_suites(args.suites), args.folds, args.seed
    )
    print(
        f"[odistill-bench] wrote {args.folds} folds with {manifest['total_cases']} cases "
        f"to {args.out_dir}"
    )
    print(f"[odistill-bench] manifest sha256={manifest['manifest_sha256']}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    report = compare_reports(
        args.baseline, args.candidate, args.target_reference,
        args.bootstrap_samples, args.bootstrap_seed
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out:
        write_markdown(report, args.md_out)
    summary = report["summary"]
    print(
        f"[odistill-bench] held-out n={summary['cases']} "
        f"accept_delta={_fmt(summary['mean_accept_rate_delta'])} "
        f"speedup={_fmt(summary['mean_speedup'], 3)}x "
        f"exact={summary['exact_output_matches']}/{summary['cases']}"
    )
    parity_ok = summary["exact_output_matches"] == summary["cases"]
    if summary["target_reference_matches"] is not None:
        parity_ok = parity_ok and summary["target_reference_matches"] == summary["cases"]
    # Target-equivalent output is a hard correctness gate for this comparison.
    return 0 if parity_ok else 1


def cmd_repeat(args: argparse.Namespace) -> int:
    report = compare_repeats(args.first, args.second)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"[odistill-bench] repeat n={summary['cases']} "
        f"outputs={summary['exact_output_matches']}/{summary['cases']} "
        f"acceptance={summary['exact_acceptance_matches']}/{summary['cases']} "
        f"timing_ratio={_fmt(summary['mean_decode_speed_ratio'], 3)}x"
    )
    deterministic = (
        summary["exact_output_matches"] == summary["cases"]
        and summary["exact_acceptance_matches"] == summary["cases"]
    )
    return 0 if deterministic else 1


def cmd_pool(args: argparse.Namespace) -> int:
    if len(args.baseline) != len(args.candidate):
        raise ValueError("--baseline and --candidate must have the same number of reports")
    if len(args.baseline) < 2:
        raise ValueError("pool requires at least two disjoint held-out fold pairs")
    if args.target_reference and len(args.target_reference) != len(args.baseline):
        raise ValueError("--target-reference count must match the fold-pair count")

    fold_reports = []
    fold_rows: list[list[dict[str, Any]]] = []
    all_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (baseline, candidate) in enumerate(zip(args.baseline, args.candidate)):
        target_reference = args.target_reference[index] if args.target_reference else None
        fold = compare_reports(
            baseline, candidate, target_reference,
            bootstrap_samples=0, bootstrap_seed=args.bootstrap_seed + index,
        )
        for row in fold["cases"]:
            if row["key"] in seen:
                raise ValueError(f"held-out case appears in multiple folds: {row['key']}")
            seen.add(row["key"])
        all_rows.extend(fold["cases"])
        fold_rows.append(fold["cases"])
        fold_reports.append({
            "index": index,
            "baseline_report": str(baseline),
            "candidate_report": str(candidate),
            "summary": fold["summary"],
        })

    summary = {
        "cases": len(all_rows),
        "folds": len(fold_reports),
        "exact_output_matches": sum(row["exact_output_match"] for row in all_rows),
        "target_reference_matches": (
            sum(row["target_reference_match"] is True for row in all_rows)
            if args.target_reference else None
        ),
        "baseline_mean_accept_rate": _mean([
            row["baseline_accept_rate"] for row in all_rows
            if row["baseline_accept_rate"] is not None
        ]),
        "candidate_mean_accept_rate": _mean([
            row["candidate_accept_rate"] for row in all_rows
            if row["candidate_accept_rate"] is not None
        ]),
        "mean_accept_rate_delta": _mean_delta(all_rows, "accept_rate"),
        "baseline_mean_decode_tok_s": _mean([
            row["baseline_decode_tok_s"] for row in all_rows
            if row["baseline_decode_tok_s"] is not None
        ]),
        "candidate_mean_decode_tok_s": _mean([
            row["candidate_decode_tok_s"] for row in all_rows
            if row["candidate_decode_tok_s"] is not None
        ]),
        "mean_speedup": _mean([
            row["speedup"] for row in all_rows if row["speedup"] is not None
        ]),
        "mean_ttft_delta_s": _mean_delta(all_rows, "ttft_s"),
        "baseline_correct": sum(row["baseline_correct"] is True for row in all_rows),
        "candidate_correct": sum(row["candidate_correct"] is True for row in all_rows),
    }
    summary["accept_rate_delta_ci95"] = _hierarchical_bootstrap_ci(
        fold_rows, lambda sample: _mean_delta(sample, "accept_rate"),
        args.bootstrap_samples, args.bootstrap_seed,
    )
    summary["speedup_ci95"] = _hierarchical_bootstrap_ci(
        fold_rows,
        lambda sample: _mean([
            row["speedup"] for row in sample if row["speedup"] is not None
        ]),
        args.bootstrap_samples, args.bootstrap_seed + 1,
    )
    report = {"schema": 1, "summary": summary, "fold": fold_reports, "cases": all_rows}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out:
        write_markdown(report, args.md_out)
    print(
        f"[odistill-bench] pooled folds={summary['folds']} n={summary['cases']} "
        f"accept_delta={_fmt(summary['mean_accept_rate_delta'])} "
        f"speedup={_fmt(summary['mean_speedup'], 3)}x"
    )
    parity_ok = summary["exact_output_matches"] == summary["cases"]
    if summary["target_reference_matches"] is not None:
        parity_ok = parity_ok and summary["target_reference_matches"] == summary["cases"]
    return 0 if parity_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create deterministic stratified K-fold prompt splits")
    prepare.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--suites", default="he,gsm,math")
    prepare.add_argument("--folds", type=int, default=3)
    prepare.add_argument("--seed", default="odistill-v1")
    prepare.set_defaults(func=cmd_prepare)

    compare = sub.add_parser("compare", help="Compare frozen base and adapted held-out reports")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument(
        "--target-reference", type=Path,
        help="Optional target-only AR bench report over the identical held-out cases",
    )
    compare.add_argument("--json-out", type=Path, required=True)
    compare.add_argument("--md-out", type=Path)
    compare.add_argument("--bootstrap-samples", type=int, default=5000)
    compare.add_argument("--bootstrap-seed", type=int, default=1)
    compare.set_defaults(func=cmd_compare)

    repeat = sub.add_parser(
        "repeat", help="Check deterministic outputs and acceptance across duplicate frozen runs")
    repeat.add_argument("--first", type=Path, required=True)
    repeat.add_argument("--second", type=Path, required=True)
    repeat.add_argument("--json-out", type=Path, required=True)
    repeat.set_defaults(func=cmd_repeat)

    pool = sub.add_parser("pool", help="Pool disjoint held-out fold comparisons")
    pool.add_argument("--baseline", type=Path, action="append", required=True)
    pool.add_argument("--candidate", type=Path, action="append", required=True)
    pool.add_argument("--target-reference", type=Path, action="append")
    pool.add_argument("--json-out", type=Path, required=True)
    pool.add_argument("--md-out", type=Path)
    pool.add_argument("--bootstrap-samples", type=int, default=5000)
    pool.add_argument("--bootstrap-seed", type=int, default=1)
    pool.set_defaults(func=cmd_pool)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[odistill-bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
