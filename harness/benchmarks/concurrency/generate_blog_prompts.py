#!/usr/bin/env python3
"""Write the exact raw HumanEval-style prompts used by scripts/bench_he.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "server" / "scripts"))
from bench_he import PROMPTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": f"he_raw_{index:02d}", "suite": "he-raw", "name": name,
         "prompt": prompt, "max_tokens": 128}
        for index, (name, prompt) in enumerate(PROMPTS, 1)
    ]
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
