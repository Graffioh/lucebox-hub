#!/usr/bin/env python3
"""Tests for canonical_concurrent_benchmark.py and blog prompt parity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "canonical_concurrent_benchmark", HERE / "canonical_concurrent_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class CanonicalBenchmarkTests(unittest.TestCase):
    def test_loads_raw_and_multi_message_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                json.dumps({"id": "raw", "prompt": "code"}) + "\n"
                + json.dumps({"id": "chat", "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ]}) + "\n",
                encoding="utf-8",
            )
            cases = benchmark.load_cases(path)
        self.assertEqual(cases[0]["prompt"], "code")
        self.assertEqual(cases[1]["prompt"][0]["role"], "system")

    def test_rejects_non_array_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps({"id": "bad", "messages": {"role": "user"}}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "messages.*array"):
                benchmark.load_cases(path)

    def test_full_suite_waves_have_no_tail_or_reuse(self) -> None:
        cases = [
            json.dumps({"id": f"p{i}", "prompt": f"prompt {i}"})
            for i in range(10)
        ]
        seen = []

        def fake_level(clients, args, prompts, offset):
            self.assertEqual(clients, 5)
            self.assertEqual(offset, 0)
            seen.extend(prompts)
            details = [{
                "error": None, "completion_tokens": 8, "prompt_tokens": 4,
                "request_decode_tok_s": 2.0, "ttft_s": 0.1,
            } for _ in prompts]
            return {
                "requests_detail": details, "failures": 0, "wall_s": 4.0,
                "output_window_s": 3.0, "fixed_token_workload_valid": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text("\n".join(cases) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                clients=5, prompt_file=prompt_file, suite="he", ignore_eos=True,
                server_metadata_json=None, label="test", base_url="x", model="m",
                max_tokens=8, temperature=0.0, seed=1, out=root / "report.json",
                retire_log=None, case_limit=None,
            )
            with mock.patch.object(benchmark.base, "run_level", side_effect=fake_level):
                self.assertEqual(benchmark.run(args), 0)
            report = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(seen, [f"prompt {i}" for i in range(10)])
        self.assertEqual(report["levels"][0]["waves"], 2)
        self.assertEqual(report["levels"][0]["requests"], 10)

    def test_rejects_partial_tail_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text("".join(
                json.dumps({"id": f"p{i}", "prompt": "x"}) + "\n"
                for i in range(10)
            ), encoding="utf-8")
            args = argparse.Namespace(clients=4, prompt_file=prompt_file, case_limit=None)
            with self.assertRaisesRegex(ValueError, "lower-concurrency tail"):
                benchmark.run(args)

    def test_case_limit_supports_three_full_c3_waves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text("".join(
                json.dumps({"id": f"p{i}", "prompt": f"x{i}"}) + "\n"
                for i in range(10)
            ), encoding="utf-8")
            args = argparse.Namespace(
                clients=3, case_limit=9, prompt_file=prompt_file, suite="he-raw",
                ignore_eos=True, server_metadata_json=None, label="c3", base_url="x",
                model="m", max_tokens=8, temperature=0.0, seed=1,
                out=root / "report.json", retire_log=None,
            )
            seen = []

            def fake_level(clients, _args, prompts, offset):
                self.assertEqual((clients, offset, len(prompts)), (3, 0, 3))
                seen.extend(prompts)
                return {
                    "requests_detail": [{
                        "error": None, "completion_tokens": 8, "prompt_tokens": 4,
                        "request_decode_tok_s": 2.0, "ttft_s": 0.1,
                    } for _ in prompts],
                    "failures": 0, "wall_s": 2.0, "output_window_s": 1.5,
                    "fixed_token_workload_valid": True,
                }

            with mock.patch.object(benchmark.base, "run_level", side_effect=fake_level):
                self.assertEqual(benchmark.run(args), 0)
            report = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(seen, [f"x{i}" for i in range(9)])
        self.assertEqual(report["case_limit"], 9)
        self.assertEqual(report["levels"][0]["waves"], 3)

    def test_retirement_wait_matches_every_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                '[concurrency-metrics] {"ddtree_steps": 1, "request_id": "a"}\n'
                '[server] chat DONE b ok=true\n',
                encoding="utf-8",
            )
            elapsed = benchmark.wait_for_retirement(log, ["a", "b"], 0.1)
            self.assertGreaterEqual(elapsed, 0)
            with self.assertRaisesRegex(TimeoutError, "did not retire"):
                benchmark.wait_for_retirement(log, ["missing"], 0.01)

    def test_blog_generator_matches_bench_he_source(self) -> None:
        generator_spec = importlib.util.spec_from_file_location(
            "generate_blog_prompts", HERE / "generate_blog_prompts.py"
        )
        assert generator_spec is not None and generator_spec.loader is not None
        generator = importlib.util.module_from_spec(generator_spec)
        generator_spec.loader.exec_module(generator)
        self.assertEqual(len(generator.PROMPTS), 10)
        source_spec = importlib.util.spec_from_file_location(
            "bench_he_source", HERE.parents[2] / "server" / "scripts" / "bench_he.py"
        )
        assert source_spec is not None and source_spec.loader is not None
        source = importlib.util.module_from_spec(source_spec)
        source_spec.loader.exec_module(source)
        self.assertEqual(generator.PROMPTS, source.PROMPTS)
        self.assertEqual(
            (HERE / "raw_prompt_identity.jinja").read_text(encoding="utf-8"),
            "{%- for message in messages -%}{{ message.content }}{%- endfor -%}\n",
        )

    def test_summary_keeps_suites_separate_and_reports_acceptance(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_canonical", HERE / "summarize_canonical_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite, variant, acceptance in (("he-raw", "blog-ddtree", 0.35), ("gsm", "ar", None)):
                path = root / suite / "c1" / "r1" / variant / "bench.json"
                path.parent.mkdir(parents=True)
                report = {
                    "suite": suite, "case_limit": None,
                    "server_metadata": {"variant": variant, "repeat": 1},
                    "levels": [{
                        "clients": 1, "failures": 0, "fixed_token_workload_valid": True,
                        "requests": 1,
                        "aggregate_tok_s": 10.0, "output_window_tok_s": 11.0,
                        "prompt_tokens_per_s_to_first_token": 13.0,
                        "request_decode_tok_s_median": 12.0,
                        "ttft_median_s": 0.1, "ttft_max_s": 0.2,
                    }],
                }
                if acceptance is not None:
                    report["ddtree_proof"] = {
                        "ddtree_steps": 1, "requests_proven": 1,
                        "mean_accepted_length": 5.6, "acceptance_rate": acceptance,
                    }
                path.write_text(json.dumps(report), encoding="utf-8")
            text = summary.summarize(root)
        self.assertIn("| he-raw | 1 | 1 | blog-ddtree", text)
        self.assertIn("5.60 | 35.0%", text)
        self.assertIn("| gsm | 1 | 1 | ar", text)

    def test_summary_reports_output_stability_by_case_id(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_canonical_stability", HERE / "summarize_canonical_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat, content_hash in ((1, "same"), (2, "changed")):
                path = root / "gsm" / "c1" / f"r{repeat}" / "ar" / "bench.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "suite": "gsm", "case_limit": None,
                    "server_metadata": {"variant": "ar", "repeat": repeat},
                    "levels": [{
                        "clients": 1, "requests": 1, "failures": 0,
                        "fixed_token_workload_valid": True, "aggregate_tok_s": 10.0,
                        "output_window_tok_s": 11.0,
                        "prompt_tokens_per_s_to_first_token": 13.0,
                        "request_decode_tok_s_median": 12.0,
                        "ttft_median_s": 0.1, "ttft_max_s": 0.2,
                        "wave_results": [{"requests_detail": [{
                            "case_id": "gsm_01", "content_sha256": content_hash,
                            "reasoning_content_sha256": "reasoning",
                        }]}],
                    }],
                }), encoding="utf-8")
            text = summary.summarize(root)
        self.assertIn("| n/a | n/a | NO |", text)


if __name__ == "__main__":
    unittest.main()
