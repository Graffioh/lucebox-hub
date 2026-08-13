#!/usr/bin/env python3
"""Unit tests for the deterministic prompt generator and compact summarizer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
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


class RunnerShellTests(unittest.TestCase):
    def test_runner_guards_case_identity_and_records_launch_environment(self) -> None:
        runner = (HERE / "run_qwen36_concurrency.sh").read_text(
            encoding="utf-8",
        )
        self.assertIn(
            'reject_duplicates CLIENTS "${client_list[@]}"', runner,
        )
        self.assertIn(
            'reject_duplicates VARIANTS "${variant_list[@]}"', runner,
        )
        self.assertIn('printf \'env \' > "$case_dir/server-command.txt"', runner)
        self.assertIn('"${launch_env[@]}" "${command[@]}"', runner)
        self.assertIn("port_is_available || return 1", runner)
        self.assertIn('wait_health "$model_id"', runner)


class SummarizerTests(unittest.TestCase):
    @staticmethod
    def item(
        variant: str,
        goodput: float,
        output_window: float | None = None,
        *,
        repeat: int = 1,
        output_hash: str | None = "same-outputs",
    ) -> dict:
        return {
            "report": {
                "max_tokens": 256, "ignore_eos": True,
                "temperature": 0.0, "seed": 1,
            },
            "meta": {
                "workload": "short", "variant": variant, "repeat": repeat,
                "model_sha256": "a" * 64,
            },
            "level": {
                "clients": 8,
                "aggregate_tok_s": goodput,
                "output_window_tok_s": output_window if output_window is not None else goodput,
                "request_decode_tok_s_median": goodput / 8,
                "prompt_tokens_per_s_to_first_token": 100.0,
                "ttft_max_s": 2.0,
                "selected_prompt_set_sha256": "same-prompts",
                "selected_output_set_sha256": output_hash,
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

    def test_summary_uses_same_repeat_ratios(self) -> None:
        reports = []
        for repeat, luce, llama in (
            (1, 10.0, 1.0),
            (2, 20.0, 90.0),
            (3, 100.0, 50.0),
        ):
            reports.extend([
                self.item("luce-k8", luce, repeat=repeat),
                self.item("llama", llama, repeat=repeat),
            ])
        text = summarizer.summarize(reports)
        luce_row = next(line for line in text.splitlines() if "| luce-k8 |" in line)
        self.assertIn("+100.0%", luce_row)
        self.assertNotIn("-60.0%", luce_row)

    def test_summary_rejects_mismatched_repeat_sets(self) -> None:
        reports = [
            self.item("luce-k8", 20.0, repeat=1),
            self.item("luce-k8", 22.0, repeat=2),
            self.item("llama", 10.0, repeat=1),
        ]
        with self.assertRaisesRegex(ValueError, "repeat sets differ"):
            summarizer.summarize(reports)

    def test_single_repeat_does_not_claim_stability(self) -> None:
        text = summarizer.summarize([self.item("llama", 8.0)])
        row = next(line for line in text.splitlines() if "| llama |" in line)
        self.assertEqual(row.split("|")[10].strip(), "n/a")

    def test_multiple_repeats_report_output_stability(self) -> None:
        stable = summarizer.summarize([
            self.item("llama", 8.0, repeat=1),
            self.item("llama", 9.0, repeat=2),
        ])
        stable_row = next(line for line in stable.splitlines() if "| llama |" in line)
        self.assertEqual(stable_row.split("|")[10].strip(), "yes")

        unstable = summarizer.summarize([
            self.item("llama", 8.0, repeat=1, output_hash="first"),
            self.item("llama", 9.0, repeat=2, output_hash="second"),
        ])
        unstable_row = next(line for line in unstable.splitlines() if "| llama |" in line)
        self.assertEqual(unstable_row.split("|")[10].strip(), "NO")

    def test_unstable_current_variant_suppresses_deltas(self) -> None:
        reports = [
            self.item("llama", 10.0, repeat=1),
            self.item("llama", 10.0, repeat=2),
            self.item("luce-k8", 20.0, repeat=1, output_hash="first"),
            self.item("luce-k8", 22.0, repeat=2, output_hash="second"),
        ]
        text = summarizer.summarize(reports)
        row = next(line for line in text.splitlines() if "| luce-k8 |" in line)
        self.assertEqual(row.split("|")[10].strip(), "NO")
        self.assertEqual(row.split("|")[11].strip(), "n/a")
        self.assertEqual(row.split("|")[12].strip(), "n/a")

    def test_unstable_peer_variant_suppresses_deltas(self) -> None:
        reports = [
            self.item(
                "llama", 10.0, repeat=1, output_hash="first",
            ),
            self.item(
                "llama", 10.0, repeat=2, output_hash="second",
            ),
            self.item("luce-k8", 20.0, repeat=1),
            self.item("luce-k8", 22.0, repeat=2),
        ]
        text = summarizer.summarize(reports)
        row = next(line for line in text.splitlines() if "| luce-k8 |" in line)
        self.assertEqual(row.split("|")[10].strip(), "yes")
        self.assertEqual(row.split("|")[11].strip(), "n/a")
        self.assertEqual(row.split("|")[12].strip(), "n/a")


    def test_missing_output_digest_does_not_claim_stability(self) -> None:
        reports = [
            self.item("llama", 8.0, repeat=1, output_hash=None),
            self.item("llama", 9.0, repeat=2, output_hash=None),
        ]
        text = summarizer.summarize(reports)
        row = next(line for line in text.splitlines() if "| llama |" in line)
        self.assertEqual(row.split("|")[10].strip(), "n/a")

    def test_incomplete_repeat_metrics_are_reported_as_na(self) -> None:
        first = self.item("llama", 8.0, repeat=1)
        second = self.item("llama", 9.0, repeat=2)
        for metric in (
            "output_window_tok_s", "request_decode_tok_s_median",
            "prompt_tokens_per_s_to_first_token", "ttft_max_s",
        ):
            second["level"][metric] = None
        text = summarizer.summarize([first, second])
        row = next(line for line in text.splitlines() if "| llama |" in line)
        self.assertEqual(
            [row.split("|")[column].strip() for column in range(6, 10)],
            ["n/a"] * 4,
        )

    def test_comparison_rejects_incompatible_run_metadata(self) -> None:
        fields = (
            ("report", "max_tokens", 64),
            ("report", "ignore_eos", False),
            ("report", "temperature", 0.5),
            ("report", "seed", 2),
            ("meta", "model_sha256", "b" * 64),
        )
        for container, key, value in fields:
            with self.subTest(key=key):
                luce = self.item("luce-k8", 20.0)
                llama = self.item("llama", 10.0)
                luce[container][key] = value
                with self.assertRaisesRegex(ValueError, "run metadata differs"):
                    summarizer.summarize([luce, llama])

    def test_summary_rejects_incomplete_run_metadata(self) -> None:
        item = self.item("llama", 10.0)
        del item["report"]["max_tokens"]
        with self.assertRaisesRegex(ValueError, "incomplete run metadata"):
            summarizer.summarize([item])

    def test_summary_reports_descriptive_errors_for_missing_fields(self) -> None:
        fields = (
            ("meta", "workload", "workload"),
            ("level", "clients", "clients"),
            ("level", "aggregate_tok_s", "aggregate token rate"),
            ("level", "selected_prompt_set_sha256", "prompt set hash"),
        )
        for container, key, message in fields:
            with self.subTest(key=key):
                item = self.item("llama", 10.0)
                del item[container][key]
                with self.assertRaisesRegex(ValueError, message):
                    summarizer.summarize([item])

    def test_load_reports_rejects_missing_prompt_usage(self) -> None:
        report = {
            "ignore_eos": True,
            "server_metadata": {"workload": "short", "variant": "llama", "repeat": 1},
            "levels": [{
                "failures": 0,
                "token_count_complete": True,
                "prompt_token_count_complete": False,
                "fixed_token_workload_valid": True,
            }],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bench.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete token accounting"):
                summarizer.load_reports(Path(root))


if __name__ == "__main__":
    unittest.main()
