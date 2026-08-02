#!/usr/bin/env python3
"""Measure Lucebox throughput and latency under concurrent streaming load.

generation_benchmark.py answers "how fast is one request". This file answers
"what do per-stream tok/s, TTFT, and aggregate throughput look like with
4/8/16 chat completions streaming at once" — the load shape served by
`--max-concurrency N` with the paged-attention slot engine.

For each client-load level N it releases N threads from a barrier, each of
which POSTs a streaming /v1/chat/completions request and times every SSE
content chunk. Levels run back to back against the same server process with a
cool-down in between, results accumulate into one JSON report, and a compact
markdown table prints at the end. Stdlib only; point it at a running server.

Terminology: `--clients` controls only the offered benchmark load. Server
capacity is configured independently when dflash_server starts with
`--max-concurrency`; this benchmark neither repeats nor changes that value,
and may intentionally offer more clients to measure queueing.

Example:
    python3 harness/benchmarks/concurrent_benchmark.py \
        --base-url http://127.0.0.1:18080/v1 --model luce-dflash \
        --clients 1 --clients 4 --clients 8 --clients 16 \
        --label paged-parallel --out /tmp/concurrent.json

Metric definitions:
- TTFT: request start to the first SSE chunk carrying a content or
  reasoning_content delta (the role-only chunk does not count).
- Per-stream decode tok/s: completion_tokens / (stream end - first token).
- Aggregate tok/s: sum of completion_tokens across all successful requests
  in the level divided by the level's wall-clock (first request start to
  last request end).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Eight distinct base prompts, each a few hundred words. Runs with more than
# eight requests prepend a per-request discriminator before each reused base,
# preventing duplicate prompts and long shared-prefix cache hits.
BUILTIN_PROMPTS = [
    (
        "I am preparing a lecture for second-year systems students on modern CPU "
        "cache hierarchies and I want a thorough written explanation I can adapt. "
        "Please explain how the L1, L2, and L3 caches cooperate on a current "
        "desktop processor: what a cache line is and why 64 bytes became the "
        "common size, how set associativity works and what conflict misses look "
        "like in practice, and how the inclusive versus exclusive design choice "
        "changes eviction behavior. Then cover coherence: walk through the MESI "
        "states with a concrete two-core example where both cores read and then "
        "one writes, explain what false sharing is and how a programmer would "
        "detect it with performance counters, and describe how hardware "
        "prefetchers recognize stride patterns. Close with practical guidance: "
        "three code-level habits that keep hot loops cache-friendly, and one "
        "worked example estimating the latency difference between an L1 hit and "
        "a main-memory access for a pointer-chasing workload. Use headings and "
        "keep the tone precise but approachable."
    ),
    (
        "Write a detailed engineering explainer on TCP congestion control for a "
        "backend team that keeps hitting throughput cliffs on long-fat links. "
        "Start from first principles: why the network needs congestion control "
        "at all, what the congestion window is, and how slow start and "
        "congestion avoidance interact with the receiver window. Explain fast "
        "retransmit and fast recovery with a concrete packet-loss timeline, "
        "then compare the classic AIMD approach with CUBIC and with BBR: what "
        "signal each algorithm treats as evidence of congestion, how they "
        "behave on a 100 ms transatlantic path with shallow buffers, and why "
        "loss-based algorithms suffer from bufferbloat while model-based ones "
        "can underuse capacity when competing flows appear. Include the "
        "bandwidth-delay product arithmetic for a 1 Gbit/s, 80 ms path and what "
        "socket buffer sizes that implies, and finish with a checklist the team "
        "can use to diagnose whether a slow transfer is limited by the "
        "application, the kernel, or the network."
    ),
    (
        "I am choosing a storage engine for a new service and want a careful "
        "comparison of B-tree and LSM-tree designs written for an experienced "
        "programmer who has never built either. Describe how a B+-tree keeps "
        "keys sorted in fixed-size pages, what a split and a merge look like, "
        "and why reads are cheap but random writes cause write amplification "
        "through page rewrites. Then describe the LSM path: memtables, "
        "write-ahead logs, sorted runs flushed to disk, leveled versus tiered "
        "compaction, and how bloom filters and fence pointers keep point reads "
        "from touching every run. Quantify the trade-offs: rough read, write, "
        "and space amplification for each design, how deletes behave "
        "(tombstones versus in-place removal), and what happens under a "
        "scan-heavy workload. Conclude with concrete selection guidance: three "
        "workload profiles where the B-tree wins, three where the LSM wins, "
        "and the operational costs (compaction stalls, page fragmentation) "
        "that each choice signs you up for."
    ),
    (
        "Explain the Raft consensus algorithm end to end for an engineer who "
        "needs to operate an etcd-like cluster and wants to reason about "
        "failures rather than treat it as a black box. Cover the three roles "
        "and the term concept, then walk through leader election in detail: "
        "randomized timeouts, what a candidate's RequestVote carries, and why "
        "the up-to-date log check prevents a stale node from winning. Next "
        "cover log replication: what AppendEntries contains, how the leader "
        "backs up nextIndex after a mismatch, when an entry becomes committed, "
        "and why a leader may only commit entries from its own term directly. "
        "Give a concrete five-node scenario where a partition isolates the "
        "leader with one follower and show step by step how the majority side "
        "elects a new leader and how the old leader's uncommitted entries are "
        "overwritten after the partition heals. Finish with snapshotting and "
        "log compaction, the single-server membership-change rule, and three "
        "practical symptoms of a misconfigured election timeout."
    ),
    (
        "Write a thorough explanation of virtual memory on a modern 64-bit "
        "operating system, aimed at a systems programmer moving down the stack "
        "from application work. Explain why processes get private address "
        "spaces, how a four-level page table translates a virtual address, and "
        "what the TLB caches; include the arithmetic for how many entries a "
        "4 KiB-page walk touches. Describe the life of a page fault from the "
        "hardware trap through the kernel's fault handler for three cases: a "
        "demand-zero page on first touch, a copy-on-write fault after fork, "
        "and a file-backed page mapped with mmap that has been evicted. Cover "
        "the page cache and how read and write system calls interact with it, "
        "what dirty writeback means, and when huge pages help or hurt. Close "
        "with a section on observability: how to read the relevant fields of "
        "/proc/<pid>/smaps, what major versus minor fault counts indicate, and "
        "two common performance bugs (TLB thrashing and accidental page-cache "
        "eviction) with the symptoms an engineer would actually see."
    ),
    (
        "Describe, in detail, how a GPU executes a dense matrix multiplication, "
        "written for a performance engineer who knows CPUs well but is new to "
        "CUDA-style programming. Begin with the execution model: grids, thread "
        "blocks, warps, and why divergence within a warp serializes work. Then "
        "build up the classic tiled GEMM: how a block stages tiles of the two "
        "input matrices in shared memory, why the tile size trades occupancy "
        "against reuse, what bank conflicts are, and how memory coalescing "
        "determines effective DRAM bandwidth. Explain the roofline model with "
        "arithmetic-intensity numbers for a 4096-square multiply in fp16, and "
        "show why the naive kernel is bandwidth-bound while the tiled kernel "
        "approaches the compute roof. Cover tensor cores: what shape of "
        "multiply they consume, why accumulation happens in fp32, and how a "
        "kernel keeps them fed with double buffering and asynchronous copies. "
        "End with a tuning checklist ordered by expected payoff and the three "
        "profiler counters that most quickly explain a slow kernel."
    ),
    (
        "Give a complete narrative of what happens when a C source file becomes "
        "a running executable, suitable for a curious engineer who has only "
        "ever typed 'make'. Follow one translation unit through preprocessing "
        "(include expansion, macro pitfalls), lexing and parsing into an AST, "
        "semantic analysis, and lowering to an SSA intermediate representation; "
        "explain what SSA form buys the optimizer with a small example of "
        "constant propagation plus dead-code elimination. Survey the mid-level "
        "passes that matter most in practice — inlining, loop-invariant code "
        "motion, vectorization — and what actually blocks them (aliasing, "
        "escaping pointers). Then cover the backend: instruction selection, "
        "register allocation by graph coloring and what a spill costs, and "
        "instruction scheduling. Finish with linking: relocations, symbol "
        "resolution rules for static versus dynamic libraries, what the "
        "dynamic loader does at startup, and how link-time optimization "
        "changes the picture. Include one concrete bug story per stage that "
        "shows why understanding that stage pays off."
    ),
    (
        "Explain the key-value cache in transformer inference deeply enough "
        "that a reader could size a serving deployment, assuming they know how "
        "attention works mathematically but have never run a server. Derive "
        "why autoregressive decoding without a cache is quadratic and how "
        "caching keys and values makes each step linear in context length. "
        "Work the memory arithmetic for a concrete model: given hidden size, "
        "head count, layer count, and fp16 storage, show the bytes per token "
        "per sequence and what a 32k-token context costs. Explain why naive "
        "per-request contiguous allocation fragments memory under concurrent "
        "load and how paged attention fixes it with fixed-size blocks and a "
        "block table, including the analogy to virtual memory and what new "
        "overheads paging introduces. Cover continuous batching: why decode "
        "steps from different requests can share one kernel launch, what "
        "determines the batch's arithmetic intensity, and where time-to-first-"
        "token and per-token latency trade against throughput. Close with the "
        "three metrics a serving benchmark must report and why each one can "
        "mislead in isolation."
    ),
]


def load_prompts(path: Path | None) -> list[str]:
    """Load prompts: one per line, blank lines and # comments skipped.

    Lines starting with '{' are parsed as JSONL objects with a "prompt" field
    (compatible with the files in harness/benchmarks/prompts/).
    """
    if path is None:
        return list(BUILTIN_PROMPTS)
    prompts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                text = obj.get("prompt")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"{path}:{line_no}: JSONL line missing string 'prompt'")
                prompts.append(text)
            else:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def request_prompts(prompts: list[str], count: int) -> list[str]:
    """Return `count` distinct prompts for one concurrency level."""
    width = len(str(count))
    return [
        prompts[i] if i < len(prompts) else
        f"Benchmark request {i + 1:0{width}d}.\n\n{prompts[i % len(prompts)]}"
        for i in range(count)
    ]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = min(len(xs) - 1, max(0, math.ceil(pct / 100.0 * len(xs)) - 1))
    return xs[k]


def stream_request(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    """POST one streaming chat completion and timestamp every content chunk.

    Returns a raw record with absolute perf_counter times (t_start, t_first,
    t_end) and chunk timestamps; finalize_request() converts these to offsets
    and derived metrics. Never raises — failures land in record["error"].
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    record: dict[str, Any] = {
        "t_start": time.perf_counter(),
        "t_first": None,
        "t_end": None,
        "chunk_times": [],
        "delta_chunks": 0,
        "completion_tokens": None,
        "prompt_tokens": None,
        "token_count_source": None,
        "server_decode_tok_s": None,
        "finish_reason": None,
        "done_seen": False,
        "error": None,
    }
    deadline = record["t_start"] + timeout

    def handle_event(event_type: str, data: str) -> bool:
        """Process one SSE event; returns True when the stream is done."""
        if data == "[DONE]":
            record["done_seen"] = True
            return True
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return False
        if event_type == "error" or (isinstance(obj, dict) and obj.get("error")):
            err = obj.get("error") if isinstance(obj, dict) else obj
            raise RuntimeError(f"server error event: {err}")
        if not isinstance(obj, dict):
            return False
        choices = obj.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        if delta.get("content") or delta.get("reasoning_content"):
            now = time.perf_counter()
            if record["t_first"] is None:
                record["t_first"] = now
            record["chunk_times"].append(now)
            record["delta_chunks"] += 1
        if choices and choices[0].get("finish_reason"):
            record["finish_reason"] = choices[0]["finish_reason"]
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("completion_tokens"), int):
                record["completion_tokens"] = usage["completion_tokens"]
                record["token_count_source"] = "usage"
            if isinstance(usage.get("prompt_tokens"), int):
                record["prompt_tokens"] = usage["prompt_tokens"]
            timings = usage.get("timings")
            if isinstance(timings, dict):
                record["server_decode_tok_s"] = timings.get("decode_tokens_per_sec")
        return False

    try:
        # The server closes each response (Connection: close, no keep-alive),
        # so every request opens a fresh connection; urllib handles the
        # chunked transfer coding and yields the body line by line.
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            event_type = ""
            data_lines: list[str] = []
            for raw in resp:
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"stream exceeded {timeout:.0f}s")
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    # Blank line terminates one SSE event.
                    if data_lines:
                        done = handle_event(event_type, "\n".join(data_lines))
                        event_type = ""
                        data_lines = []
                        if done:
                            break
                    continue
                if line.startswith(":"):
                    continue  # SSE comment / keep-alive
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())
            if data_lines:  # stream closed without a trailing blank line
                handle_event(event_type, "\n".join(data_lines))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        record["error"] = f"HTTP {e.code}: {detail}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["t_end"] = time.perf_counter()

    if record["error"] is None and not record["done_seen"]:
        record["error"] = "stream ended without [DONE]"
    if record["error"] is None and record["completion_tokens"] is None:
        # Fall back to counting received delta chunks (holdback batching in
        # the emitter can merge tokens, so this undercounts slightly).
        record["completion_tokens"] = record["delta_chunks"]
        record["token_count_source"] = "delta_chunks"
    return record


def finalize_request(record: dict[str, Any], level_start: float) -> dict[str, Any]:
    ttft = None
    decode_tok_s = None
    if record["t_first"] is not None:
        ttft = record["t_first"] - record["t_start"]
        decode_s = record["t_end"] - record["t_first"]
        ct = record["completion_tokens"]
        # From first-token arrival to stream end there are ct - 1 decode
        # intervals. Counting the already-arrived first token overstates the
        # rate, especially for short completions.
        if record["error"] is None and ct and ct > 1 and decode_s > 0:
            decode_tok_s = (ct - 1) / decode_s
    times = record["chunk_times"]
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {
        "prompt_index": record["prompt_index"],
        "start_offset_s": record["t_start"] - level_start,
        "elapsed_s": record["t_end"] - record["t_start"],
        "ttft_s": ttft,
        "decode_tok_s": decode_tok_s,
        "completion_tokens": record["completion_tokens"],
        "prompt_tokens": record["prompt_tokens"],
        "token_count_source": record["token_count_source"],
        "delta_chunks": record["delta_chunks"],
        "itl_mean_ms": statistics.mean(gaps) * 1000.0 if gaps else None,
        "itl_p95_ms": percentile(gaps, 95.0) * 1000.0 if gaps else None,
        "server_decode_tok_s": record["server_decode_tok_s"],
        "finish_reason": record["finish_reason"],
        "error": record["error"],
    }


def run_level(n: int, args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    barrier = threading.Barrier(n)
    stream_records: list[list[dict[str, Any]] | None] = [None] * n
    level_prompts = request_prompts(prompts, n * args.requests_per_stream)

    def worker(stream_index: int) -> None:
        barrier.wait()
        records: list[dict[str, Any]] = []
        for j in range(args.requests_per_stream):
            prompt_index = stream_index * args.requests_per_stream + j
            record = stream_request(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=level_prompts[prompt_index],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            record["prompt_index"] = prompt_index
            records.append(record)
            if record["error"] is not None:
                break  # a broken stream stops issuing follow-up requests
        stream_records[stream_index] = records

    threads = [
        threading.Thread(target=worker, args=(i,), name=f"stream-{i}", daemon=True)
        for i in range(n)
    ]
    for t in threads:
        t.start()
    # Per-request socket timeouts bound each read, so this cap only trips if
    # a thread is truly wedged; daemon threads let the process exit anyway.
    join_deadline = time.monotonic() + args.timeout * args.requests_per_stream + 30.0
    for t in threads:
        t.join(timeout=max(0.0, join_deadline - time.monotonic()))
    hung = sum(1 for t in threads if t.is_alive())

    raw = [r for records in stream_records if records for r in records]
    level_start = min((r["t_start"] for r in raw), default=time.perf_counter())
    level_end = max((r["t_end"] for r in raw), default=level_start)
    wall = level_end - level_start

    failures = hung
    streams: list[dict[str, Any]] = []
    for i, records in enumerate(stream_records):
        if records is None:
            streams.append({"stream_index": i, "completion_tokens": 0,
                            "error": "stream did not finish", "requests": []})
            continue
        requests = [finalize_request(r, level_start) for r in records]
        failures += sum(1 for r in requests if r["error"] is not None)
        streams.append({
            "stream_index": i,
            "completion_tokens": sum(
                r["completion_tokens"] or 0 for r in requests if r["error"] is None),
            "requests": requests,
        })

    ok = [r for s in streams for r in s["requests"] if r["error"] is None]
    total_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tok_s = [r["decode_tok_s"] for r in ok if r["decode_tok_s"] is not None]
    return {
        "clients": n,
        "requests_per_stream": args.requests_per_stream,
        "requests": n * args.requests_per_stream,
        "requests_ok": len(ok),
        "failures": failures,
        "wall_s": wall,
        "completion_tokens_total": total_tokens,
        "aggregate_tok_s": total_tokens / wall if wall > 0 else 0.0,
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p95_s": percentile(ttfts, 95.0),
        "stream_tok_s_mean": statistics.mean(tok_s) if tok_s else None,
        "stream_tok_s_median": statistics.median(tok_s) if tok_s else None,
        "stream_tok_s_min": min(tok_s) if tok_s else None,
        "stream_tok_s_max": max(tok_s) if tok_s else None,
        "stream_completion_tokens": [s["completion_tokens"] for s in streams],
        "streams": streams,
    }


def fmt(value: Any, spec: str = ".2f") -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return format(value, spec)


def markdown_lines(report: dict[str, Any]) -> list[str]:
    title = "# Concurrent Serving Benchmark"
    if report["label"]:
        title += f" — {report['label']}"
    lines = [
        title,
        "",
        f"Server: `{report['base_url']}` model `{report['model']}` "
        f"max_tokens={report['max_tokens']} temperature={report['temperature']}",
        "",
        "| Clients | Ok/Req | Agg tok/s | Stream tok/s mean | Stream tok/s median "
        "| TTFT mean s | TTFT median s | TTFT p95 s | Wall s | Tokens |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in report["levels"]:
        lines.append(
            f"| {level['clients']} "
            f"| {level['requests_ok']}/{level['requests']} "
            f"| {fmt(level['aggregate_tok_s'])} "
            f"| {fmt(level['stream_tok_s_mean'])} "
            f"| {fmt(level['stream_tok_s_median'])} "
            f"| {fmt(level['ttft_mean_s'], '.3f')} "
            f"| {fmt(level['ttft_median_s'], '.3f')} "
            f"| {fmt(level['ttft_p95_s'], '.3f')} "
            f"| {fmt(level['wall_s'])} "
            f"| {level['completion_tokens_total']} |"
        )
    lines.append("")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1",
                        help="Base URL ending in /v1 (default %(default)s)")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="luce-dflash")
    parser.add_argument("--clients", dest="client_levels", type=int,
                        action="append", metavar="N",
                        help="Simultaneous clients; repeat for several load "
                             "levels (default: 1 4 8 16).")
    parser.add_argument("--requests-per-stream", type=int, default=1,
                        help="Sequential requests each stream issues (default 1)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-file", default="",
                        help="Prompt file, one prompt per line (or JSONL with a "
                             "'prompt' field); default: 8 built-in prompts")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-request timeout in seconds (default 600)")
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Seconds to sleep between levels (default 2)")
    parser.add_argument("--out", default="", help="Write the JSON report here")
    parser.add_argument("--label", default="", help="Free-form run label for the report")
    return parser


def run(args: argparse.Namespace) -> int:
    levels_arg = args.client_levels or [1, 4, 8, 16]
    for n in levels_arg:
        if n < 1:
            raise ValueError(f"--clients must be >= 1, got {n}")
    if args.requests_per_stream < 1:
        raise ValueError("--requests-per-stream must be >= 1")
    prompts = load_prompts(Path(args.prompt_file) if args.prompt_file else None)

    levels: list[dict[str, Any]] = []
    for idx, n in enumerate(levels_arg):
        if idx > 0 and args.cooldown > 0:
            time.sleep(args.cooldown)
        print(f"[bench] level N={n}: {n} stream(s) x {args.requests_per_stream} "
              f"request(s), max_tokens={args.max_tokens}", flush=True)
        level = run_level(n, args, prompts)
        levels.append(level)
        print(f"[bench] level N={n}: {level['requests_ok']}/{level['requests']} ok, "
              f"agg {fmt(level['aggregate_tok_s'])} tok/s, "
              f"stream mean {fmt(level['stream_tok_s_mean'])} tok/s, "
              f"ttft p95 {fmt(level['ttft_p95_s'], '.3f')}s, "
              f"wall {fmt(level['wall_s'])}s", flush=True)

    report = {
        "label": args.label,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "requests_per_stream": args.requests_per_stream,
        "timeout_s": args.timeout,
        "prompt_source": args.prompt_file or f"builtin ({len(BUILTIN_PROMPTS)} prompts)",
        "levels": levels,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[bench] wrote {out}", flush=True)

    print()
    print("\n".join(markdown_lines(report)), flush=True)

    total_failures = sum(level["failures"] for level in levels)
    if total_failures:
        print(f"[bench] {total_failures} stream request(s) failed", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
