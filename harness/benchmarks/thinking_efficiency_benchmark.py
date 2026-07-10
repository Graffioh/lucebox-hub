#!/usr/bin/env python3
"""Measure reasoning-token efficiency and surface possible doom loops.

The benchmark targets Lucebox's OpenAI-compatible chat endpoint. It opts into
the thinking-budget envelope, stores one JSON object per request, and keeps
loop detection separate from broader failure signals such as hitting the
output cap or producing no visible answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generation_benchmark import load_cases, messages_for_case, score_gold_answer

WORD_RE = re.compile(r"[\w]+(?:['’-][\w]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def normalized_words(text: str) -> list[str]:
    return [word.casefold() for word in WORD_RE.findall(text)]


def normalize_span(text: str) -> str:
    return " ".join(normalized_words(text))


def non_overlapping_positions(positions: list[int], width: int) -> list[int]:
    """Greedily retain non-overlapping occurrences from sorted positions."""
    if not positions:
        return []
    selected = [positions[0]]
    next_allowed = positions[0] + width
    for position in positions[1:]:
        if position >= next_allowed:
            selected.append(position)
            next_allowed = position + width
    return selected


def ngram_positions(words: list[str], width: int) -> dict[tuple[str, ...], list[int]]:
    positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(words) - width + 1):
        positions[tuple(words[index : index + width])].append(index)
    return positions


def detect_repeated_ngram(
    text: str,
    ngram_size: int = 12,
    min_repeats: int = 3,
) -> dict[str, Any] | None:
    """Return the strongest exact repeated n-gram using non-overlapping hits."""
    words = normalized_words(text)
    if len(words) < ngram_size * min_repeats:
        return None

    candidates = ngram_positions(words, ngram_size)
    best: dict[str, Any] | None = None
    for ngram, raw_positions in candidates.items():
        if len(raw_positions) < min_repeats:
            continue
        positions = non_overlapping_positions(raw_positions, ngram_size)
        if len(positions) < min_repeats:
            continue
        candidate = {
            "kind": "repeated_ngram",
            "count": len(positions),
            "ngram_size": ngram_size,
            "positions": positions,
            "sample": " ".join(ngram),
        }
        if best is None or candidate["count"] > best["count"]:
            best = candidate
    return best


def detect_repeated_span(
    text: str,
    min_words: int = 8,
    min_repeats: int = 3,
) -> dict[str, Any] | None:
    """Detect repeated sentence-like spans after whitespace/punctuation cleanup."""
    spans = [normalize_span(span) for span in SENTENCE_RE.split(text)]
    spans = [span for span in spans if len(span.split()) >= min_words]
    counts = Counter(spans)
    repeated = [(span, count) for span, count in counts.items() if count >= min_repeats]
    if not repeated:
        return None
    span, count = max(repeated, key=lambda item: (item[1], len(item[0])))
    return {
        "kind": "repeated_span",
        "count": count,
        "span_words": len(span.split()),
        "sample": span,
    }


def detect_loop(
    reasoning: str,
    ngram_size: int = 12,
    min_repeats: int = 3,
    span_min_words: int = 8,
) -> dict[str, Any]:
    signals = []
    ngram = detect_repeated_ngram(reasoning, ngram_size, min_repeats)
    if ngram:
        signals.append(ngram)
    span = detect_repeated_span(reasoning, span_min_words, min_repeats)
    if span:
        signals.append(span)
    return {"detected": bool(signals), "signals": signals}


def approx_token_count(text: str) -> int:
    # Used only when the endpoint omits server-side token accounting.
    words = normalized_words(text)
    return math.ceil(len(words) * 1.3) if words else 0


def extract_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content = message.get("content")
    if not isinstance(content, str):
        content = ""

    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""
        details = message.get("reasoning_details") or []
        if isinstance(details, list):
            reasoning = "".join(
                detail.get("text", "")
                for detail in details
                if isinstance(detail, dict) and isinstance(detail.get("text"), str)
            )

    # Compatibility fallback for endpoints that leave reasoning inline.
    if not reasoning and "<think>" in content:
        before, _, rest = content.partition("<think>")
        inline_reasoning, close, after = rest.partition("</think>")
        reasoning = inline_reasoning.strip()
        content = (before + (after if close else "")).strip()

    usage = response.get("usage") or {}
    finish_details = choice.get("finish_details") or {}
    token_details = usage.get("completion_tokens_details") or {}
    thinking_tokens = finish_details.get("thinking_tokens")
    if not isinstance(thinking_tokens, int):
        thinking_tokens = token_details.get("reasoning_tokens")
    thinking_token_source = "server"
    if not isinstance(thinking_tokens, int):
        thinking_tokens = approx_token_count(reasoning)
        thinking_token_source = "approx_words"

    completion_tokens = usage.get("completion_tokens")
    completion_token_source = "server"
    if not isinstance(completion_tokens, int):
        completion_tokens = approx_token_count(reasoning + " " + content)
        completion_token_source = "approx_words"

    content_tokens = finish_details.get("content_tokens")
    if not isinstance(content_tokens, int):
        content_tokens = max(0, completion_tokens - thinking_tokens)

    return {
        "reasoning_text": reasoning.strip(),
        "content": content.strip(),
        "thinking_tokens": thinking_tokens,
        "content_tokens": content_tokens,
        "completion_tokens": completion_tokens,
        "thinking_token_source": thinking_token_source,
        "completion_token_source": completion_token_source,
        "finish_reason": choice.get("finish_reason"),
        "close_kind": finish_details.get("close_kind"),
        "degenerate_decode": bool(finish_details.get("degenerate_decode", False)),
        "usage": usage,
        "finish_details": finish_details,
    }


def post_chat(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def parse_temperatures(value: str) -> list[float]:
    temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not temperatures:
        raise argparse.ArgumentTypeError("provide at least one temperature")
    if any(temperature < 0 for temperature in temperatures):
        raise argparse.ArgumentTypeError("temperatures must be non-negative")
    return temperatures


def completed_run_keys(path: Path) -> set[tuple[str, float, int]]:
    keys: set[tuple[str, float, int]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                keys.add((row["prompt_id"], float(row["temperature"]), int(row["repeat_index"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return keys


def run_one(
    case: dict[str, Any],
    temperature: float,
    repeat_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages_for_case(case),
        "max_tokens": args.max_tokens,
        "temperature": temperature,
        "stream": False,
        "thinking": {
            "type": "enabled",
            "budget_tokens": args.thinking_budget,
            "reply_budget": args.reply_budget,
        },
    }
    if args.seed is not None:
        payload["seed"] = args.seed + repeat_index

    started = time.perf_counter()
    response = post_chat(args.url, args.api_key, payload, args.timeout)
    elapsed_s = time.perf_counter() - started
    extracted = extract_response(response)
    loop = detect_loop(
        extracted["reasoning_text"],
        ngram_size=args.ngram_size,
        min_repeats=args.min_repeats,
        span_min_words=args.span_min_words,
    )
    thinking_tokens = extracted["thinking_tokens"]
    threshold = min(args.overthinking_tokens, math.ceil(args.max_tokens * args.overthinking_ratio))
    hit_max_tokens = extracted["finish_reason"] == "length"
    has_final_answer = bool(extracted["content"])
    gold_correct, gold_detail = score_gold_answer(case, extracted["content"])

    if loop["detected"] and (hit_max_tokens or not has_final_answer):
        loop_confidence = "high"
    elif loop["detected"]:
        loop_confidence = "medium"
    else:
        loop_confidence = "none"

    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "prompt_id": case["id"],
        "category": case.get("category", case.get("suite", "uncategorized")),
        "difficulty": case.get("difficulty"),
        "description": case.get("description", ""),
        "model": args.model,
        "temperature": temperature,
        "repeat_index": repeat_index,
        "seed": args.seed + repeat_index if args.seed is not None else None,
        "max_tokens": args.max_tokens,
        "thinking_budget": args.thinking_budget,
        "reply_budget": args.reply_budget,
        "elapsed_s": elapsed_s,
        **extracted,
        "has_final_answer": has_final_answer,
        "hit_max_tokens": hit_max_tokens,
        "overthinking_threshold": threshold,
        "overthinking_detected": thinking_tokens >= threshold,
        "doom_loop_detected": loop["detected"],
        "loop_confidence": loop_confidence,
        "loop_signals": loop["signals"],
        "gold_answer": case.get("gold_answer"),
        "gold_correct": gold_correct,
        "gold_detail": gold_detail,
    }


def cmd_run(args: argparse.Namespace) -> int:
    if args.max_tokens <= 0 or args.thinking_budget <= 0 or args.reply_budget <= 0:
        raise ValueError("token budgets must be positive")
    if args.thinking_budget + args.reply_budget > args.max_tokens:
        raise ValueError("thinking_budget + reply_budget must not exceed max_tokens")
    if not 0 < args.overthinking_ratio <= 1:
        raise ValueError("overthinking_ratio must be in (0, 1]")
    cases = load_cases(Path(args.prompts))
    if args.limit:
        cases = cases[: args.limit]
    output_path = Path(args.jsonl_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = completed_run_keys(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    failures = 0

    with output_path.open(mode, encoding="utf-8") as output:
        for temperature in args.temperatures:
            for case in cases:
                for repeat_index in range(args.repeats):
                    key = (case["id"], temperature, repeat_index)
                    if key in existing:
                        print(f"[thinking-bench] skip {case['id']} temp={temperature:g} repeat={repeat_index}")
                        continue
                    print(
                        f"[thinking-bench] run {case['id']} temp={temperature:g} "
                        f"repeat={repeat_index}",
                        flush=True,
                    )
                    try:
                        row = run_one(case, temperature, repeat_index, args)
                        status = "LOOP" if row["doom_loop_detected"] else "ok"
                        print(
                            f"[thinking-bench] {status} think={row['thinking_tokens']} "
                            f"finish={row['finish_reason']} elapsed={row['elapsed_s']:.1f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        failures += 1
                        row = {
                            "schema_version": 1,
                            "created_at": dt.datetime.now(dt.UTC).isoformat(),
                            "prompt_id": case["id"],
                            "category": case.get("category", case.get("suite", "uncategorized")),
                            "model": args.model,
                            "temperature": temperature,
                            "repeat_index": repeat_index,
                            "error": str(exc),
                        }
                        print(f"[thinking-bench] error: {exc}", file=sys.stderr, flush=True)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()

    print(f"[thinking-bench] wrote {output_path}")
    return 1 if failures else 0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    thinking_tokens = [float(row.get("thinking_tokens", 0)) for row in valid]
    scored = [row for row in valid if row.get("gold_correct") is not None]

    def count(field: str) -> int:
        return sum(1 for row in valid if row.get(field))

    summary = {
        "requests": len(rows),
        "successful_requests": len(valid),
        "errors": len(rows) - len(valid),
        "doom_loops": count("doom_loop_detected"),
        "doom_loop_rate": count("doom_loop_detected") / len(valid) if valid else 0.0,
        "overthinking": count("overthinking_detected"),
        "overthinking_rate": count("overthinking_detected") / len(valid) if valid else 0.0,
        "hit_max_tokens": count("hit_max_tokens"),
        "hit_max_tokens_rate": count("hit_max_tokens") / len(valid) if valid else 0.0,
        "final_answers": count("has_final_answer"),
        "final_answer_rate": count("has_final_answer") / len(valid) if valid else 0.0,
        "mean_thinking_tokens": statistics.mean(thinking_tokens) if thinking_tokens else 0.0,
        "median_thinking_tokens": statistics.median(thinking_tokens) if thinking_tokens else 0.0,
        "p95_thinking_tokens": percentile(thinking_tokens, 0.95),
        "mean_elapsed_s": (
            statistics.mean(float(row.get("elapsed_s", 0)) for row in valid) if valid else 0.0
        ),
        "gold_correct": sum(1 for row in scored if row.get("gold_correct")),
        "gold_scored": len(scored),
    }
    return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Thinking Efficiency Baseline",
        "",
        f"Source: `{report['source']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Successful requests | {summary['successful_requests']}/{summary['requests']} |",
        f"| Doom-loop rate | {format_rate(summary['doom_loop_rate'])} |",
        f"| Overthinking rate | {format_rate(summary['overthinking_rate'])} |",
        f"| Max-token hit rate | {format_rate(summary['hit_max_tokens_rate'])} |",
        f"| Final-answer rate | {format_rate(summary['final_answer_rate'])} |",
        f"| Mean thinking tokens | {summary['mean_thinking_tokens']:.1f} |",
        f"| Median thinking tokens | {summary['median_thinking_tokens']:.1f} |",
        f"| p95 thinking tokens | {summary['p95_thinking_tokens']:.0f} |",
        f"| Mean latency | {summary['mean_elapsed_s']:.2f}s |",
        f"| Gold correctness | {summary['gold_correct']}/{summary['gold_scored']} |",
        "",
        "## By temperature",
        "",
        "| Temperature | Requests | Loop rate | Mean thinking tokens | Max-token rate | Final-answer rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for temperature, group in report["by_temperature"].items():
        lines.append(
            f"| {temperature} | {group['successful_requests']} | "
            f"{format_rate(group['doom_loop_rate'])} | {group['mean_thinking_tokens']:.1f} | "
            f"{format_rate(group['hit_max_tokens_rate'])} | "
            f"{format_rate(group['final_answer_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Cases to inspect",
            "",
            "| Prompt | Temp | Thinking tokens | Loop | Finish | Final answer |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in report["inspection_cases"]:
        lines.append(
            f"| `{row['prompt_id']}` | {row['temperature']:g} | {row['thinking_tokens']} | "
            f"{'yes' if row['doom_loop_detected'] else 'no'} | "
            f"{row.get('finish_reason') or 'n/a'} | "
            f"{'yes' if row['has_final_answer'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def cmd_summarize(args: argparse.Namespace) -> int:
    source = Path(args.jsonl)
    rows = load_jsonl(source)
    temperatures = sorted(
        {float(row["temperature"]) for row in rows if "temperature" in row and not row.get("error")}
    )
    by_temperature = {
        f"{temperature:g}": summarize_rows(
            [row for row in rows if float(row.get("temperature", -1)) == temperature]
        )
        for temperature in temperatures
    }
    valid = [row for row in rows if not row.get("error")]
    inspection_cases = sorted(
        valid,
        key=lambda row: (
            bool(row.get("doom_loop_detected")),
            bool(row.get("hit_max_tokens")),
            int(row.get("thinking_tokens", 0)),
        ),
        reverse=True,
    )[: args.top]
    report = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": str(source),
        "summary": summarize_rows(rows),
        "by_temperature": by_temperature,
        "inspection_cases": inspection_cases,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"[thinking-bench] wrote {json_out}")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown_report(report), encoding="utf-8")
        print(f"[thinking-bench] wrote {md_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the thinking baseline against one endpoint")
    run.add_argument("--url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--api-key", default="")
    run.add_argument("--model", default="deepseek-v4-flash")
    run.add_argument(
        "--prompts",
        default=str(Path(__file__).with_name("prompts") / "thinking_stress.jsonl"),
    )
    run.add_argument("--jsonl-out", required=True)
    run.add_argument("--temperatures", type=parse_temperatures, default=[0.0, 0.2, 0.6])
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--thinking-budget", type=int, default=7168)
    run.add_argument("--reply-budget", type=int, default=1024)
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--ngram-size", type=int, default=12)
    run.add_argument("--min-repeats", type=int, default=3)
    run.add_argument("--span-min-words", type=int, default=8)
    run.add_argument("--overthinking-tokens", type=int, default=4096)
    run.add_argument("--overthinking-ratio", type=float, default=0.75)
    run.set_defaults(func=cmd_run)

    summarize = subparsers.add_parser("summarize", help="Summarize a JSONL baseline")
    summarize.add_argument("--jsonl", required=True)
    summarize.add_argument("--json-out", required=True)
    summarize.add_argument("--md-out", default="")
    summarize.add_argument("--top", type=int, default=10)
    summarize.set_defaults(func=cmd_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"[thinking-bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
