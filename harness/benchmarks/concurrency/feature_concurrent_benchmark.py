#!/usr/bin/env python3
"""Concurrency client with request IDs and effective-prompt telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import concurrent_benchmark as base


CLIENT_SCRIPT = Path(__file__).resolve()


def client_provenance(argv: list[str] | None = None) -> dict[str, Any]:
    """Return the literal process argv and the exact client source digest."""
    process_argv = list(sys.orig_argv if argv is None else argv)
    if not process_argv or not all(isinstance(value, str) for value in process_argv):
        raise ValueError("client process argv must be a non-empty string array")
    return {
        "client_argv": process_argv,
        "client_script": str(CLIENT_SCRIPT),
        "client_script_sha256": hashlib.sha256(CLIENT_SCRIPT.read_bytes()).hexdigest(),
    }


def stream_request(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    first = None
    request_id = None
    content: list[str] = []
    reasoning: list[str] = []
    completion_tokens = None
    prompt_tokens = None
    finish_reason = None
    done_received = False
    timings: dict[str, Any] = {}
    wire_metrics: dict[str, Any] = {}
    error = None
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for data in base.iter_sse_data(response):
                if data == "[DONE]":
                    done_received = True
                    break
                event = json.loads(data)
                if isinstance(event.get("id"), str):
                    request_id = event["id"]
                usage = event.get("usage") or {}
                if type(usage.get("completion_tokens")) is int:
                    completion_tokens = usage["completion_tokens"]
                if type(usage.get("prompt_tokens")) is int:
                    prompt_tokens = usage["prompt_tokens"]
                if isinstance(usage.get("timings"), dict):
                    timings = dict(usage["timings"])
                # Log telemetry is authoritative, but retaining a future wire
                # copy makes reports forward-compatible without weakening proof.
                if isinstance(usage.get("concurrency_metrics"), dict):
                    wire_metrics = dict(usage["concurrency_metrics"])
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    thought = delta.get("reasoning_content")
                    if isinstance(piece, str) and piece:
                        first = first or time.perf_counter()
                        content.append(piece)
                    if isinstance(thought, str) and thought:
                        first = first or time.perf_counter()
                        reasoning.append(thought)
    except Exception as exc:  # retain partial data for diagnosis
        error = f"{type(exc).__name__}: {exc}"
    if error is None and not done_received:
        error = "ProtocolError: stream ended before [DONE]"
    elif error is None and finish_reason is None:
        error = "ProtocolError: stream ended without a terminal finish_reason"
    ended = time.perf_counter()
    output = "".join(content)
    reasoning_output = "".join(reasoning)
    decode_duration = ended - first if first is not None and ended > first else None
    request_decode_tok_s = (
        (completion_tokens - 1) / decode_duration
        if type(completion_tokens) is int and completion_tokens > 0
        and decode_duration is not None else None
    )
    return {
        "request_id": request_id,
        "t_start": started, "t_first": first, "t_end": ended,
        "duration_s": ended - started,
        "ttft_s": first - started if first is not None else None,
        "decode_duration_s": decode_duration,
        "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens,
        "effective_prompt_tokens": timings.get("effective_prompt_tokens"),
        "prefilled_tokens": timings.get("prefilled_tokens"),
        "cached_prefix_tokens": timings.get("cached_prefix_tokens"),
        "cache_hit": timings.get("cache_hit"),
        "server_prefill_ms": timings.get("prefill_ms"),
        "server_decode_ms": timings.get("decode_ms"),
        "server_decode_tokens_per_sec": timings.get("decode_tokens_per_sec"),
        "server_timings": timings,
        "wire_concurrency_metrics": wire_metrics,
        "finish_reason": finish_reason, "done_received": done_received, "error": error,
        "content_sha256": base.sha256_text(output),
        "reasoning_content_sha256": base.sha256_text(reasoning_output),
        "content_chars": len(output), "reasoning_content_chars": len(reasoning_output),
        "request_output_tok_s": (
            completion_tokens / (ended - started)
            if completion_tokens is not None and ended > started else None
        ),
        "request_decode_tok_s": request_decode_tok_s,
    }


# The base client owns the concurrency/barrier/accounting implementation. Its
# module-global hook is intentional: this process runs one benchmark at a time.
base.stream_request = stream_request


def enrich_level(level: dict[str, Any]) -> None:
    ok = [r for r in level["requests_detail"] if r.get("error") is None]
    effective = [r.get("effective_prompt_tokens") for r in ok]
    effective_complete = bool(ok) and all(type(v) is int for v in effective)
    request_ids = [r.get("request_id") for r in ok]
    request_ids_complete = (
        bool(ok) and all(isinstance(v, str) and v for v in request_ids)
        and len(set(request_ids)) == len(request_ids)
    )
    level.update({
        "request_ids_complete": request_ids_complete,
        "effective_prompt_token_count_complete": effective_complete,
        "effective_prompt_tokens_total": sum(effective) if effective_complete else None,
        "effective_prompt_tokens_min": min(effective) if effective_complete else None,
        "effective_prompt_tokens_max": max(effective) if effective_complete else None,
        "effective_to_wire_prompt_ratio": (
            sum(effective) / level["prompt_tokens_total"]
            if effective_complete and level.get("prompt_tokens_total") else None
        ),
    })
    for key in ("server_prefill_ms", "server_decode_ms", "server_decode_tokens_per_sec"):
        values = [r.get(key) for r in ok if type(r.get(key)) in (int, float)]
        level[f"{key}_median"] = statistics.median(values) if len(values) == len(ok) and ok else None


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Concurrent feature benchmark — {report['label']}", "",
        "| C | Ok | Output goodput tok/s | Output-window tok/s | "
        "Request decode tok/s | Wire prompt range | Effective prompt range | "
        "Effective/wire | TTFT max s |",
        "| ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: |",
    ]
    for level in report["levels"]:
        lines.append(
            f"| {level['clients']} | {level['requests_ok']}/{level['requests']} | "
            f"{base.fmt(level['aggregate_tok_s'])} | "
            f"{base.fmt(level['output_window_tok_s'])} | "
            f"{base.fmt(level['request_decode_tok_s_median'])} | "
            f"{base.fmt(level['prompt_tokens_min'], '.0f')}–{base.fmt(level['prompt_tokens_max'], '.0f')} | "
            f"{base.fmt(level['effective_prompt_tokens_min'], '.0f')}–"
            f"{base.fmt(level['effective_prompt_tokens_max'], '.0f')} | "
            f"{base.fmt(level['effective_to_wire_prompt_ratio'], '.3f')} | "
            f"{base.fmt(level['ttft_max_s'], '.3f')} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--require-effective-prompt-telemetry", action="store_true",
        help="fail when usage.timings.effective_prompt_tokens is absent",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    levels = args.client_levels or [1, 4, 8, 16]
    if any(level < 1 for level in levels):
        raise ValueError("--clients must be positive")
    if args.prompt_offset < 0 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("invalid offset, max-tokens, or timeout")
    prompts = base.load_prompts(args.prompt_file)
    results = []
    offset = args.prompt_offset
    for index, clients in enumerate(levels):
        if index and args.cooldown > 0:
            time.sleep(args.cooldown)
        print(f"[bench] C={clients} max_tokens={args.max_tokens}", flush=True)
        level = base.run_level(clients, args, prompts, offset)
        enrich_level(level)
        results.append(level)
        offset += clients
    metadata = (
        json.loads(args.server_metadata_json.read_text(encoding="utf-8"))
        if args.server_metadata_json else {}
    )
    report = {
        "schema_version": 3, "label": args.label, "base_url": args.base_url,
        "model": args.model, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "seed": args.seed,
        "ignore_eos": args.ignore_eos, "prompt_offset": args.prompt_offset,
        "prompt_file_sha256": hashlib.sha256(args.prompt_file.read_bytes()).hexdigest(),
        "server_metadata": metadata, "levels": results,
        **client_provenance(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown(report), end="")
    bad = any(
        base.level_failed(level, args.ignore_eos)
        or not level["request_ids_complete"]
        or (args.require_effective_prompt_telemetry
            and not level["effective_prompt_token_count_complete"])
        for level in results
    )
    return 1 if bad else 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"[bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
