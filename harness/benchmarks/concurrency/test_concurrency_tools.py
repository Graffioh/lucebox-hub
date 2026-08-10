#!/usr/bin/env python3
"""Unit tests for the deterministic prompt generator and compact summarizer."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


HERE = Path(__file__).parent


def load(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load("generate_ragged_prompts")
summarizer = load("summarize_concurrency")


class PromptGeneratorTests(unittest.TestCase):
    def test_cohorts_are_disjoint_ragged_and_mean_matched(self) -> None:
        records = generator.build_records("short")
        self.assertEqual(len(records), 29)
        self.assertEqual(
            [row["cohort"] for row in records],
            ["c1"] + ["c4"] * 4 + ["c8"] * 8 + ["c16"] * 16,
        )
        self.assertEqual(len({row["prompt"] for row in records}), 29)
        by_cohort = {
            cohort: [row for row in records if row["cohort"] == cohort]
            for cohort in ("c1", "c4", "c8", "c16")
        }
        means = {
            cohort: sum(row["target_words"] for row in rows) / len(rows)
            for cohort, rows in by_cohort.items()
        }
        self.assertEqual(len(set(means.values())), 1)
        for cohort in ("c4", "c8", "c16"):
            self.assertEqual(len({row["target_words"] for row in by_cohort[cohort]}), 4)
        for row in records:
            self.assertEqual(len(row["prompt"].split()), row["target_words"])


class SummarizerTests(unittest.TestCase):
    @staticmethod
    def item(variant: str, goodput: float, output_window: float | None = None) -> dict:
        return {
            "meta": {"workload": "short", "variant": variant},
            "level": {
                "clients": 8,
                "aggregate_tok_s": goodput,
                "output_window_tok_s": output_window if output_window is not None else goodput,
                "request_decode_tok_s_median": goodput / 8,
                "prompt_tokens_per_s_to_first_token": 100.0,
                "ttft_max_s": 2.0,
                "selected_prompt_set_sha256": "same-prompts",
                "selected_output_set_sha256": "same-outputs",
            },
        }

    def test_summary_reports_product_and_packing_deltas(self) -> None:
        text = summarizer.summarize([
            self.item("luce-k8", 20.0),
            self.item("luce-k1", 10.0),
            self.item("llama", 8.0),
        ])
        self.assertIn("+150.0%", text)
        self.assertIn("+100.0%", text)
        self.assertIn("Decode vs llama", text)


if __name__ == "__main__":
    unittest.main()
