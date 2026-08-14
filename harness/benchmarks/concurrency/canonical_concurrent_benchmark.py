#!/usr/bin/env python3
"""Run a complete repository benchmark suite in fixed-concurrency waves."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "concurrent_benchmark", HERE / "concurrent_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


CONCURRENCY_METRICS_MARKER = re.compile(r"\[concurrency-metrics\]\s+(\{.*\})\s*$")
SERVER_DONE_MARKER = re.compile(r"\[server\] chat DONE\s+(\S+)")


def retired_response_ids(text: str) -> set[str]:
    retired: set[str] = set()
    for line in text.splitlines():
        marker = CONCURRENCY_METRICS_MARKER.search(line)
        if marker:
            try:
                value = json.loads(marker.group(1))
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                response_id = value.get("response_id") or value.get("request_id")
                if isinstance(response_id, str) and response_id:
                    retired.add(response_id)
        done = SERVER_DONE_MARKER.search(line)
        if done:
            retired.add(done.group(1))
    return retired


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{path}:{line_no}: missing or duplicate string id")
        if isinstance(record.get("prompt"), str) and record["prompt"]:
            prompt = record["prompt"]
        else:
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_no}: 'messages' must be an array")
            prompt = base.prompt_messages(messages)
        seen.add(case_id)
        cases.append({"id": case_id, "prompt": prompt})
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def wait_for_retirement(path: Path, response_ids: list[str], timeout: float) -> float:
    started = time.perf_counter()
    pending = set(response_ids)
    deadline = started + timeout
    while pending and time.perf_counter() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            pending -= retired_response_ids(text)
        if pending:
            time.sleep(0.05)
    if pending:
        raise TimeoutError(f"scheduler did not retire responses: {sorted(pending)}")
    return time.perf_counter() - started


def aggregate_waves(clients: int, waves: list[dict[str, Any]]) -> dict[str, Any]:
    details = [record for wave in waves for record in wave["requests_detail"]]
    ok = [record for record in details if record["error"] is None]
    completion = [record["completion_tokens"] for record in ok]
    prompts = [record["prompt_tokens"] for record in ok]
    rates = [record["request_decode_tok_s"] for record in ok]
    ttfts = [record["ttft_s"] for record in ok]
    completion_complete = bool(ok) and all(isinstance(value, int) for value in completion)
    prompt_complete = bool(ok) and all(isinstance(value, int) for value in prompts)
    wall = sum(wave["wall_s"] for wave in waves)
    output_window_values = [wave.get("output_window_s") for wave in waves]
    output_window = (
        sum(output_window_values)
        if all(isinstance(value, (int, float)) for value in output_window_values)
        else None
    )
    prompt_window_values = [wave.get("prompt_to_first_token_s") for wave in waves]
    prompt_window = (
        sum(prompt_window_values)
        if all(isinstance(value, (int, float)) for value in prompt_window_values)
        else None
    )
    failures = sum(wave["failures"] for wave in waves)
    return {
        "clients": clients,
        "waves": len(waves),
        "requests": len(details),
        "requests_ok": len(ok),
        "failures": failures,
        "wall_s": wall,
        "completion_tokens_total": sum(completion) if completion_complete else None,
        "token_count_complete": completion_complete,
        "prompt_tokens_total": sum(prompts) if prompt_complete else None,
        "prompt_tokens_min": min(prompts) if prompt_complete else None,
        "prompt_tokens_max": max(prompts) if prompt_complete else None,
        "prompt_token_count_complete": prompt_complete,
        "prompt_to_first_token_s": prompt_window,
        "prompt_tokens_per_s_to_first_token": (
            sum(prompts) / prompt_window
            if prompt_complete and prompt_window is not None and prompt_window > 0 else None
        ),
        "aggregate_tok_s": sum(completion) / wall if completion_complete and wall > 0 else None,
        "output_window_tok_s": (
            sum(completion) / output_window
            if completion_complete and output_window is not None and output_window > 0 else None
        ),
        "request_decode_tok_s_median": (
            statistics.median(rates)
            if len(rates) == len(ok) and ok else None
        ),
        "ttft_median_s": statistics.median(ttfts) if len(ttfts) == len(ok) and ok else None,
        "ttft_max_s": max(ttfts) if len(ttfts) == len(ok) and ok else None,
        "wave_results": waves,
    }


def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.prompt_file)
    if args.case_limit is not None:
        if args.case_limit < 1 or args.case_limit > len(cases):
            raise ValueError(
                f"--case-limit must be between 1 and the suite size {len(cases)}"
            )
        cases = cases[:args.case_limit]
    if args.clients < 1 or len(cases) % args.clients:
        raise ValueError(
            f"suite size {len(cases)} must be divisible by --clients={args.clients}; "
            "refusing a lower-concurrency tail wave"
        )
    waves = []
    for offset in range(0, len(cases), args.clients):
        selected = cases[offset:offset + args.clients]
        wave = base.run_level(args.clients, args, [case["prompt"] for case in selected], 0)
        for case, detail in zip(selected, wave["requests_detail"], strict=True):
            detail["case_id"] = case["id"]
        if args.retire_log and wave["failures"] == 0:
            response_ids = [
                detail["response_id"] for detail in wave["requests_detail"]
                if isinstance(detail.get("response_id"), str)
            ]
            if len(response_ids) != args.clients:
                raise ValueError("successful wave is missing response IDs")
            wait_s = wait_for_retirement(args.retire_log, response_ids, args.timeout)
            wave["retirement_wait_s"] = wait_s
            wave["wall_s"] += wait_s
        waves.append(wave)
    level = aggregate_waves(args.clients, waves)
    level["fixed_token_workload_valid"] = (
        level["failures"] == 0
        and level["requests_ok"] == len(cases)
        and all(wave["fixed_token_workload_valid"] is True for wave in waves)
    ) if args.ignore_eos else None
    metadata = (
        json.loads(args.server_metadata_json.read_text(encoding="utf-8"))
        if args.server_metadata_json else {}
    )
    report = {
        "schema_version": 1,
        "label": args.label,
        "suite": args.suite,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
        "case_limit": args.case_limit,
        "prompt_file_sha256": hashlib.sha256(args.prompt_file.read_bytes()).hexdigest(),
        "server_metadata": metadata,
        "levels": [level],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(base.markdown(report), end="")
    return 1 if base.level_failed(level, args.ignore_eos) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="luce-dflash")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--server-metadata-json", type=Path)
    parser.add_argument("--retire-log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"[canonical-bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
