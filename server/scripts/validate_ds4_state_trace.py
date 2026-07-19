#!/usr/bin/env python3
"""Validate opt-in DeepSeek4 shard-state trace invariants."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


STATE_RE = re.compile(
    r"\[deepseek4-state\] pid=(?P<pid>\d+) role=(?P<role>\S+) "
    r"phase=(?P<phase>\S+) shard=(?P<shard>\d+) device=(?P<device>-?\d+) "
    r"layers=\[(?P<begin>\d+),(?P<end>\d+)\) tokens=(?P<tokens>\d+) "
    r"elements=(?P<elements>\d+) nonfinite=(?P<nonfinite>\d+) "
    r"hash=(?P<hash>[0-9a-f]+).* alias_hc=(?P<alias>[01])"
)
CACHE_RE = re.compile(
    r"\[deepseek4-cache-state\] pid=(?P<pid>\d+) event=(?P<event>\S+) "
    r"cache_owner=(?P<owner>\S+) runtime=(?P<runtime>\S+) .* "
    r"device=(?P<device>-?\d+) layers=\[(?P<begin>\d+),(?P<end>\d+)\) "
    r"owns_output=(?P<output>[01])"
)
SCRATCH_RE = re.compile(
    r"\[deepseek4-scratch-state\] pid=(?P<pid>\d+) event=(?P<event>\S+) "
    r"device=(?P<device>-?\d+) slot=(?P<slot>\S+) owner_device=(?P<owner>-?\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = "\n".join(path.read_text(errors="replace") for path in args.logs)
    states = [match.groupdict() for match in STATE_RE.finditer(text)]
    caches = [match.groupdict() for match in CACHE_RE.finditer(text)]
    scratches = [match.groupdict() for match in SCRATCH_RE.finditer(text)]
    errors: list[str] = []

    if not states:
        errors.append("no DeepSeek4 state records found")
    if not caches:
        errors.append("no DeepSeek4 cache records found")

    for state in states:
        if int(state["nonfinite"]) != 0:
            errors.append(
                f"non-finite HC state: pid={state['pid']} role={state['role']} "
                f"phase={state['phase']} shard={state['shard']}"
            )

    latest_local_output: dict[tuple[str, int, int], dict[str, str]] = {}
    local_pairs = 0
    for state in states:
        key = (state["pid"], int(state["tokens"]), int(state["shard"]))
        if state["role"] == "local" and state["phase"] == "output-hc":
            latest_local_output[key] = state
        if state["role"] == "local" and state["phase"] == "input-hc":
            producer_key = (
                state["pid"],
                int(state["tokens"]),
                int(state["shard"]) - 1,
            )
            producer = latest_local_output.get(producer_key)
            if not producer:
                errors.append(f"missing local HC producer for consumer {key}")
                continue
            local_pairs += 1
            if state["alias"] != "1":
                errors.append(f"local HC consumer is not aliased: {key}")
            if (state["elements"], state["hash"]) != (
                producer["elements"],
                producer["hash"],
            ):
                errors.append(f"local HC producer/consumer mismatch: {producer_key} -> {key}")

    sends = [
        state for state in states
        if state["role"] == "ipc-parent" and state["phase"] == "send-hc"
    ]
    receives = [
        state for state in states
        if state["role"] == "ipc-daemon" and state["phase"] == "receive-hc"
    ]
    ipc_pairs = 0
    if sends or receives:
        if len(sends) != len(receives):
            errors.append(f"IPC state count mismatch: sends={len(sends)} receives={len(receives)}")
        for index, (send, receive) in enumerate(zip(sends, receives)):
            ipc_pairs += 1
            send_key = (send["tokens"], send["elements"], send["hash"])
            receive_key = (receive["tokens"], receive["elements"], receive["hash"])
            if send_key != receive_key:
                errors.append(f"IPC HC mismatch at forward {index}: {send_key} != {receive_key}")

    cache_by_owner: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    runtimes_by_pid: dict[str, dict[str, str]] = {}
    for cache in caches:
        if cache["event"] == "reinitialize":
            errors.append(
                f"cache reinitialized: pid={cache['pid']} owner={cache['owner']} "
                f"layers=[{cache['begin']},{cache['end']})"
            )
        key = (cache["pid"], cache["owner"])
        value = (
            cache["runtime"], cache["device"], cache["begin"],
            cache["end"], cache["output"],
        )
        previous = cache_by_owner.setdefault(key, value)
        if previous != value:
            errors.append(f"cache identity or metadata changed: {key}")
        owners = runtimes_by_pid.setdefault(cache["pid"], {})
        other_owner = owners.get(cache["runtime"])
        if other_owner is not None and other_owner != cache["owner"]:
            errors.append(
                f"runtime cache shared by two owners in pid={cache['pid']}: "
                f"{other_owner}, {cache['owner']}"
            )
        owners[cache["runtime"]] = cache["owner"]

    scratch_by_device: dict[tuple[str, str], str] = {}
    slots_by_pid: dict[str, dict[str, str]] = {}
    for scratch in scratches:
        if scratch["device"] != scratch["owner"]:
            errors.append(f"scratch owner mismatch: {scratch}")
        key = (scratch["pid"], scratch["device"])
        previous = scratch_by_device.setdefault(key, scratch["slot"])
        if previous != scratch["slot"]:
            errors.append(f"scratch slot changed for pid/device {key}")
        slots = slots_by_pid.setdefault(scratch["pid"], {})
        other_device = slots.get(scratch["slot"])
        if other_device is not None and other_device != scratch["device"]:
            errors.append(
                f"scratch slot shared by devices in pid={scratch['pid']}: "
                f"{other_device}, {scratch['device']}"
            )
        slots[scratch["slot"]] = scratch["device"]

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "OK: "
        f"states={len(states)} local_handoffs={local_pairs} ipc_handoffs={ipc_pairs} "
        f"cache_records={len(caches)} cache_owners={len(cache_by_owner)} "
        f"scratch_records={len(scratches)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
