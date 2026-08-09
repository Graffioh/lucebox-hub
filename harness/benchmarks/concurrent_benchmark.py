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
- Per-request stream tok/s: server-reported completion_tokens divided by the
  full request interval (request start to stream end). It includes queueing,
  prefill, and decode; no SSE chunk is treated as one tokenizer token.
- Aggregate tok/s: sum of completion_tokens across all successful requests
  in the level divided by the level's wall-clock (first request start to
  last request end). This metric is unavailable unless every successful
  request has a server-reported completion-token count.

SSE content deltas are transport chunks, not necessarily tokenizer tokens.
The harness records their count and timing separately and never substitutes
them for completion_tokens.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import subprocess
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


def request_prompts(
    prompts: list[str], count: int, offset: int = 0,
) -> list[str]:
    """Return `count` distinct prompts starting at the global request offset."""
    width = len(str(offset + count))
    return [
        prompts[offset + i] if offset + i < len(prompts) else
        (f"Benchmark request {offset + i + 1:0{width}d}.\n\n"
         f"{prompts[(offset + i) % len(prompts)]}")
        for i in range(count)
    ]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = min(len(xs) - 1, max(0, math.ceil(pct / 100.0 * len(xs)) - 1))
    return xs[k]


def percentile_if_sufficient(
    values: list[float], pct: float, min_samples: int,
) -> float | None:
    """Return a nearest-rank percentile only for an adequate sample."""
    if len(values) < min_samples:
        return None
    return percentile(values, pct)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_identity(text: str) -> dict[str, Any]:
    """Return a chunk-boundary-independent identity for generated text.

    Character length is Python's Unicode code-point count. Byte length and
    SHA-256 are over the exact UTF-8 encoding used by the JSON/SSE protocol.
    """
    encoded = text.encode("utf-8")
    return {
        "sha256": sha256_bytes(encoded),
        "char_length": len(text),
        "byte_length": len(encoded),
    }


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def prompt_content_sha256(prompts: list[str]) -> str:
    """Hash parsed prompt content independently of its file formatting."""
    canonical = json.dumps(
        prompts, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(canonical)


def sanitize_argv(argv: list[str]) -> list[str]:
    """Copy argv while redacting API-key values from reproducibility data."""
    sanitized: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
        elif arg == "--api-key":
            sanitized.append(arg)
            redact_next = True
        elif arg.startswith("--api-key="):
            sanitized.append("--api-key=<redacted>")
        else:
            sanitized.append(arg)
    return sanitized


def git_provenance(path: Path) -> dict[str, Any] | None:
    """Capture the revision and dirty-state digest without requiring git."""
    try:
        root_result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, timeout=10,
        )
        root = Path(root_result.stdout.decode("utf-8", errors="replace").strip())

        def git(*args: str) -> bytes:
            result = subprocess.run(
                ["git", "-C", str(root), *args], check=True,
                capture_output=True, timeout=30,
            )
            return result.stdout

        head = git("rev-parse", "HEAD").decode().strip()
        branch = git("branch", "--show-current").decode().strip() or None
        status = git("status", "--short", "--untracked-files=normal").decode(
            "utf-8", errors="replace",
        ).splitlines()
        tracked_diff = git("diff", "--no-ext-diff", "--full-index", "HEAD")
        return {
            "root": str(root),
            "head": head,
            "branch": branch,
            "dirty": bool(status),
            "status_short": status,
            "tracked_diff_sha256": sha256_bytes(tracked_diff),
        }
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def load_metadata_json(
    path: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load optional launcher-supplied server/build metadata."""
    if path is None:
        return None, None
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: server metadata must be a JSON object")
    return value, {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
    }


def server_props_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/props"


def fetch_server_props(
    base_url: str, api_key: str, timeout: float,
) -> dict[str, Any]:
    """Snapshot a compatible server's read-only /props endpoint if present."""
    url = server_props_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    result: dict[str, Any] = {"url": url, "available": False}
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 10.0)) as resp:
            raw = resp.read(4 * 1024 * 1024 + 1)
            result["http_status"] = getattr(resp, "status", None)
            result["http_server_header"] = resp.headers.get("Server")
        if len(raw) > 4 * 1024 * 1024:
            result["error"] = "response exceeded 4 MiB capture limit"
            return result
        result["body_sha256"] = sha256_bytes(raw)
        body = json.loads(raw)
        result["body"] = body
        result["available"] = True
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = f"HTTP {exc.code}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def finalize_token_accounting(
    record: dict[str, Any], max_tokens: int, fixed_tokens_requested: bool,
) -> None:
    """Validate server-reported counts without treating SSE deltas as tokens."""
    completion_tokens = record.get("completion_tokens")
    if (
        not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens < 0
    ):
        completion_tokens = None

    record["completion_tokens"] = completion_tokens
    record["token_count_available"] = completion_tokens is not None

    if fixed_tokens_requested:
        validated = (
            record.get("error") is None
            and completion_tokens == max_tokens
        )
        record["fixed_token_count_validated"] = validated
        if not validated:
            if completion_tokens is None:
                record["token_count_warning"] = (
                    "fixed-token request has no exact completion-token count"
                )
            else:
                record["token_count_warning"] = (
                    f"fixed-token request returned {completion_tokens} tokens; "
                    f"expected {max_tokens}"
                )
    else:
        record["fixed_token_count_validated"] = None

    if completion_tokens is None:
        record["token_count_source"] = "unavailable"


def stream_request(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    ignore_eos: bool = False,
    request_stream_usage: bool = True,
    capture_output_text: bool = False,
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
    if request_stream_usage:
        payload["stream_options"] = {"include_usage": True}
    if ignore_eos:
        payload["ignore_eos"] = True
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
        "token_count_available": False,
        "fixed_token_count_validated": None,
        "token_count_warning": None,
        "server_decode_tok_s": None,
        "response_model": None,
        "system_fingerprint": None,
        "http_server_header": None,
        "finish_reason": None,
        "done_seen": False,
        "error": None,
        "_content_parts": [],
        "_reasoning_content_parts": [],
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
        if isinstance(obj.get("model"), str):
            record["response_model"] = obj["model"]
        if isinstance(obj.get("system_fingerprint"), str):
            record["system_fingerprint"] = obj["system_fingerprint"]
        choices = obj.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        content_delta = delta.get("content")
        reasoning_delta = delta.get("reasoning_content")
        if isinstance(content_delta, str):
            record["_content_parts"].append(content_delta)
        if isinstance(reasoning_delta, str):
            record["_reasoning_content_parts"].append(reasoning_delta)
        if (
            (isinstance(content_delta, str) and content_delta)
            or (isinstance(reasoning_delta, str) and reasoning_delta)
        ):
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
            record["http_server_header"] = resp.headers.get("Server")
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
    for field in ("content", "reasoning_content"):
        parts = record.pop(f"_{field}_parts")
        text = "".join(parts)
        identity = text_identity(text)
        record[f"{field}_sha256"] = identity["sha256"]
        record[f"{field}_char_length"] = identity["char_length"]
        record[f"{field}_byte_length"] = identity["byte_length"]
        if capture_output_text:
            record[f"{field}_text"] = text
    finalize_token_accounting(record, max_tokens, ignore_eos)
    return record


def finalize_request(
    record: dict[str, Any], level_start: float, min_percentile_samples: int = 100,
) -> dict[str, Any]:
    ttft = None
    request_tok_s = None
    elapsed_s = record["t_end"] - record["t_start"]
    if record["t_first"] is not None:
        ttft = record["t_first"] - record["t_start"]
    ct = record["completion_tokens"]
    if (
        record["error"] is None
        and record.get("token_count_source") == "usage"
        and ct is not None
        and elapsed_s > 0
    ):
        request_tok_s = ct / elapsed_s
    times = record["chunk_times"]
    gaps = [b - a for a, b in zip(times, times[1:])]
    gap_p95 = percentile_if_sufficient(gaps, 95.0, min_percentile_samples)
    finalized = {
        "prompt_index": record["prompt_index"],
        "global_prompt_index": record["global_prompt_index"],
        "prompt_sha256": record["prompt_sha256"],
        "start_offset_s": record["t_start"] - level_start,
        "elapsed_s": elapsed_s,
        "ttft_s": ttft,
        "request_tok_s": request_tok_s,
        "request_tok_s_interval": "request_start_to_stream_end",
        "request_tok_s_source": (
            "usage.completion_tokens" if request_tok_s is not None else None
        ),
        "decode_tok_s": None,
        "decode_tok_s_status": "unavailable_from_sse_chunks",
        "completion_tokens": record["completion_tokens"],
        "prompt_tokens": record["prompt_tokens"],
        "token_count_source": record["token_count_source"],
        "token_count_available": record["token_count_available"],
        "fixed_token_count_validated": record["fixed_token_count_validated"],
        "token_count_warning": record["token_count_warning"],
        "delta_chunks": record["delta_chunks"],
        "content_sha256": record.get("content_sha256"),
        "content_char_length": record.get("content_char_length"),
        "content_byte_length": record.get("content_byte_length"),
        "reasoning_content_sha256": record.get("reasoning_content_sha256"),
        "reasoning_content_char_length": record.get(
            "reasoning_content_char_length",
        ),
        "reasoning_content_byte_length": record.get(
            "reasoning_content_byte_length",
        ),
        "sse_delta_gap_mean_ms": statistics.mean(gaps) * 1000.0 if gaps else None,
        "sse_delta_gap_p95_ms": gap_p95 * 1000.0 if gap_p95 is not None else None,
        "sse_delta_gap_max_ms": max(gaps) * 1000.0 if gaps else None,
        "sse_delta_gap_sample_count": len(gaps),
        "sse_delta_gap_p95_min_samples": min_percentile_samples,
        "sse_delta_gap_p95_status": (
            "reported" if gap_p95 is not None else "insufficient_samples"
        ),
        # SSE emitters may split or merge token text arbitrarily, so these
        # legacy token-latency names cannot be populated from chunk timing.
        "itl_mean_ms": None,
        "itl_p95_ms": None,
        "itl_status": "unavailable_from_sse_chunks",
        "server_decode_tok_s": record["server_decode_tok_s"],
        "response_model": record["response_model"],
        "system_fingerprint": record["system_fingerprint"],
        "http_server_header": record["http_server_header"],
        "finish_reason": record["finish_reason"],
        "error": record["error"],
    }
    for field in ("content", "reasoning_content"):
        if f"{field}_text" in record:
            finalized[f"{field}_text"] = record[f"{field}_text"]
    return finalized


def run_level(
    n: int, args: argparse.Namespace, prompts: list[str], prompt_offset: int = 0,
) -> dict[str, Any]:
    level_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    barrier = threading.Barrier(n)
    stream_records: list[list[dict[str, Any]] | None] = [None] * n
    level_prompts = request_prompts(
        prompts, n * args.requests_per_stream, prompt_offset,
    )

    def worker(stream_index: int) -> None:
        barrier.wait()
        records: list[dict[str, Any]] = []
        for j in range(args.requests_per_stream):
            prompt_index = stream_index * args.requests_per_stream + j
            prompt = level_prompts[prompt_index]
            record = stream_request(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                ignore_eos=args.ignore_eos,
                request_stream_usage=args.request_stream_usage,
                capture_output_text=getattr(
                    args, "capture_output_text", False,
                ),
            )
            record["prompt_index"] = prompt_index
            record["global_prompt_index"] = prompt_offset + prompt_index
            record["prompt_sha256"] = sha256_bytes(prompt.encode("utf-8"))
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
    level_finished_at = dt.datetime.now(dt.timezone.utc).isoformat()

    raw = [r for records in stream_records if records for r in records]
    level_start = min((r["t_start"] for r in raw), default=time.perf_counter())
    level_end = max((r["t_end"] for r in raw), default=level_start)
    wall = level_end - level_start

    failures = hung
    streams: list[dict[str, Any]] = []
    for i, records in enumerate(stream_records):
        if records is None:
            streams.append({"stream_index": i, "completion_tokens": None,
                            "error": "stream did not finish", "requests": []})
            continue
        requests = [
            finalize_request(r, level_start, args.min_percentile_samples)
            for r in records
        ]
        failures += sum(1 for r in requests if r["error"] is not None)
        successful = [r for r in requests if r["error"] is None]
        stream_counts = [
            r["completion_tokens"] for r in successful
            if r["completion_tokens"] is not None
        ]
        stream_count_complete = len(stream_counts) == len(successful)
        streams.append({
            "stream_index": i,
            "completion_tokens": (
                sum(stream_counts) if stream_count_complete and successful else None
            ),
            "token_count_complete": stream_count_complete and bool(successful),
            "requests": requests,
        })

    ok = [r for s in streams for r in s["requests"] if r["error"] is None]
    known_counts = [
        r["completion_tokens"] for r in ok if r["completion_tokens"] is not None
    ]
    token_count_complete = bool(ok) and len(known_counts) == len(ok)
    total_tokens = sum(known_counts) if token_count_complete else None
    delta_chunks_total = sum(r["delta_chunks"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tok_s = [r["request_tok_s"] for r in ok if r["request_tok_s"] is not None]
    p95 = percentile_if_sufficient(
        ttfts, 95.0, args.min_percentile_samples,
    )
    start_times = [r["t_start"] for r in raw]
    fixed_token_workload_valid = None
    if args.ignore_eos:
        fixed_token_workload_valid = (
            failures == 0
            and len(ok) == n * args.requests_per_stream
            and all(r["fixed_token_count_validated"] is True for r in ok)
        )

    def distinct(field: str) -> list[str]:
        return sorted({
            value for value in (r.get(field) for r in ok)
            if isinstance(value, str) and value
        })

    return {
        "clients": n,
        "started_at": level_started_at,
        "finished_at": level_finished_at,
        "requests_per_stream": args.requests_per_stream,
        "requests": n * args.requests_per_stream,
        "requests_ok": len(ok),
        "failures": failures,
        "wall_s": wall,
        "start_skew_s": (
            max(start_times) - min(start_times) if start_times else None
        ),
        "completion_tokens_total": total_tokens,
        "completion_tokens_known_total": sum(known_counts),
        "completion_tokens_known_requests": len(known_counts),
        "token_count_complete": token_count_complete,
        "fixed_token_workload_valid": fixed_token_workload_valid,
        "aggregate_tok_s": (
            total_tokens / wall
            if total_tokens is not None and wall > 0 else None
        ),
        "sse_delta_chunks_total": delta_chunks_total,
        "aggregate_sse_delta_chunks_s": (
            delta_chunks_total / wall if wall > 0 else None
        ),
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "ttft_sample_count": len(ttfts),
        "ttft_p95_s": p95,
        "ttft_p95_min_samples": args.min_percentile_samples,
        "ttft_p95_status": (
            "reported" if p95 is not None else "insufficient_samples"
        ),
        "stream_tok_s_interval": "request_start_to_stream_end",
        "stream_tok_s_mean": statistics.mean(tok_s) if tok_s else None,
        "stream_tok_s_median": statistics.median(tok_s) if tok_s else None,
        "stream_tok_s_min": min(tok_s) if tok_s else None,
        "stream_tok_s_max": max(tok_s) if tok_s else None,
        "stream_completion_tokens": [s["completion_tokens"] for s in streams],
        "response_models": distinct("response_model"),
        "system_fingerprints": distinct("system_fingerprint"),
        "http_server_headers": distinct("http_server_header"),
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
        f"TTFT p95 uses nearest-rank and is withheld below "
        f"{report['min_percentile_samples']} samples; TTFT max and sample count "
        f"remain visible.",
        "",
        "| Clients | Ok/Req | Agg tok/s | Request tok/s mean | Request tok/s median "
        "| TTFT mean s | TTFT median s | TTFT p95 s | TTFT max s | TTFT n "
        "| Wall s | Tokens | Token counts |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | :--- |",
    ]
    for level in report["levels"]:
        fixed_status = level.get("fixed_token_workload_valid")
        if fixed_status is True:
            count_status = "fixed valid"
        elif fixed_status is False:
            count_status = "fixed mismatch"
        elif level["token_count_complete"]:
            count_status = "complete"
        else:
            count_status = "missing"
        lines.append(
            f"| {level['clients']} "
            f"| {level['requests_ok']}/{level['requests']} "
            f"| {fmt(level['aggregate_tok_s'])} "
            f"| {fmt(level['stream_tok_s_mean'])} "
            f"| {fmt(level['stream_tok_s_median'])} "
            f"| {fmt(level['ttft_mean_s'], '.3f')} "
            f"| {fmt(level['ttft_median_s'], '.3f')} "
            f"| {fmt(level['ttft_p95_s'], '.3f')} "
            f"| {fmt(level['ttft_max_s'], '.3f')} "
            f"| {level['ttft_sample_count']} "
            f"| {fmt(level['wall_s'])} "
            f"| {fmt(level['completion_tokens_total'], '.0f')} "
            f"| {count_status} |"
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
    parser.add_argument("--disjoint-level-prompts", action="store_true",
                        help="Do not reuse prompts between client levels; avoids "
                             "cross-level slot/prefix-cache hits")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Request fixed-length generation by suppressing EOS "
                             "until max_tokens (server support required)")
    parser.add_argument(
        "--no-stream-usage", dest="request_stream_usage", action="store_false",
        help="Do not request stream_options.include_usage; exact counts then "
             "depend on unsolicited server usage",
    )
    parser.set_defaults(request_stream_usage=True)
    parser.add_argument(
        "--capture-output-text", action="store_true",
        help="Store generated content and reasoning text in the JSON report; "
             "off by default because hashes and lengths usually suffice",
    )

    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-request timeout in seconds (default 600)")
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Seconds to sleep between levels (default 2)")
    parser.add_argument(
        "--min-percentile-samples", type=int, default=100,
        help="Withhold p95 below this many observations (default 100)",
    )
    parser.add_argument(
        "--server-metadata-json", default="",
        help="Optional JSON object with exact server command, build/model "
             "hashes, and hardware metadata supplied by the launcher",
    )
    parser.add_argument("--out", default="", help="Write the JSON report here")
    parser.add_argument("--label", default="", help="Free-form run label for the report")
    return parser


def run(args: argparse.Namespace) -> int:
    run_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    levels_arg = args.client_levels or [1, 4, 8, 16]
    for n in levels_arg:
        if n < 1:
            raise ValueError(f"--clients must be >= 1, got {n}")
    if args.requests_per_stream < 1:
        raise ValueError("--requests-per-stream must be >= 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.cooldown < 0:
        raise ValueError("--cooldown must be >= 0")
    if args.min_percentile_samples < 1:
        raise ValueError("--min-percentile-samples must be >= 1")

    prompt_path = Path(args.prompt_file) if args.prompt_file else None
    prompts = load_prompts(prompt_path)
    metadata_path = (
        Path(args.server_metadata_json) if args.server_metadata_json else None
    )
    server_metadata, server_metadata_source = load_metadata_json(metadata_path)
    server_props = fetch_server_props(args.base_url, args.api_key, args.timeout)

    levels: list[dict[str, Any]] = []
    prompt_offset = 0
    for idx, n in enumerate(levels_arg):
        if idx > 0 and args.cooldown > 0:
            time.sleep(args.cooldown)
        print(f"[bench] level N={n}: {n} stream(s) x {args.requests_per_stream} "
              f"request(s), max_tokens={args.max_tokens}", flush=True)
        level = run_level(n, args, prompts, prompt_offset)
        levels.append(level)
        if args.disjoint_level_prompts:
            prompt_offset += n * args.requests_per_stream
        if level["ttft_p95_s"] is not None:
            ttft_summary = f"p95 {fmt(level['ttft_p95_s'], '.3f')}s"
        else:
            ttft_summary = (
                f"max {fmt(level['ttft_max_s'], '.3f')}s "
                f"(n={level['ttft_sample_count']}; p95 withheld)"
            )
        print(
            f"[bench] level N={n}: {level['requests_ok']}/{level['requests']} ok, "
            f"agg {fmt(level['aggregate_tok_s'])} tok/s, "
            f"request mean {fmt(level['stream_tok_s_mean'])} tok/s, "
            f"ttft {ttft_summary}, wall {fmt(level['wall_s'])}s",
            flush=True,
        )

    run_finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    script_path = Path(__file__).resolve()
    resolved_prompt_path = prompt_path.resolve() if prompt_path else None
    observed_server = {
        "response_models": sorted({
            value for level in levels for value in level["response_models"]
        }),
        "system_fingerprints": sorted({
            value for level in levels for value in level["system_fingerprints"]
        }),
        "http_server_headers": sorted({
            value for level in levels for value in level["http_server_headers"]
        }),
    }
    report = {
        "schema_version": 2,
        "label": args.label,
        "timestamp": run_started_at,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "client_levels": levels_arg,
        "requests_per_stream": args.requests_per_stream,
        "timeout_s": args.timeout,
        "cooldown_s": args.cooldown,
        "prompt_source": args.prompt_file or f"builtin ({len(BUILTIN_PROMPTS)} prompts)",
        "prompt_metadata": {
            "source": "file" if prompt_path else "builtin",
            "path": str(resolved_prompt_path) if resolved_prompt_path else None,
            "file_sha256": (
                sha256_file(resolved_prompt_path) if resolved_prompt_path else None
            ),
            "content_sha256": prompt_content_sha256(prompts),
            "prompt_count": len(prompts),
        },
        "disjoint_level_prompts": args.disjoint_level_prompts,
        "ignore_eos": args.ignore_eos,
        "request_stream_usage": args.request_stream_usage,
        "capture_output_text": args.capture_output_text,
        "token_count_policy": {
            "preferred": "usage.completion_tokens",
            "exact_fallback": None,
            "sse_delta_chunks_are_tokens": False,
            "per_request_tok_s_interval": "request_start_to_stream_end",
            "per_request_tok_s_includes": ["queueing", "prefill", "decode"],
            "decode_tok_s_from_sse": False,
        },
        "percentile_method": "nearest_rank",
        "min_percentile_samples": args.min_percentile_samples,
        "client_provenance": {
            "argv": sanitize_argv([sys.executable, *sys.argv]),
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "git": git_provenance(script_path.parent),
        },
        "server_metadata": server_metadata,
        "server_metadata_source": server_metadata_source,
        "server_props_snapshot": server_props,
        "observed_server": observed_server,
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
    missing_counts = [
        level["clients"] for level in levels if not level["token_count_complete"]
    ]
    if missing_counts:
        print(
            f"[bench] exact completion-token counts unavailable at client "
            f"level(s): {missing_counts}",
            file=sys.stderr,
        )
        return 1
    invalid_fixed = [
        level["clients"] for level in levels
        if level["fixed_token_workload_valid"] is False
    ]
    if invalid_fixed:
        print(
            f"[bench] fixed-token workload validation failed at client "
            f"level(s): {invalid_fixed}",
            file=sys.stderr,
        )
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
