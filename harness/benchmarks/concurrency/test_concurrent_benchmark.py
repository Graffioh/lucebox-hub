#!/usr/bin/env python3
"""Focused tests for concurrent_benchmark.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("concurrent_benchmark.py")
SPEC = importlib.util.spec_from_file_location("concurrent_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_sse_parser_handles_events_and_done(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n', b"\n",
            b"data: [DONE]\n", b"\n",
        ]
        self.assertEqual(
            list(benchmark.iter_sse_data(lines)),
            ['{"choices":[{"delta":{"content":"hi"}}]}', "[DONE]"],
        )

    def test_prompt_selection_never_wraps(self) -> None:
        self.assertEqual(benchmark.request_prompts(["a", "b", "c"], 2, 1), ["b", "c"])
        with self.assertRaisesRegex(ValueError, "refusing to reuse"):
            benchmark.request_prompts(["a", "b"], 2, 1)

    def test_level_uses_exact_usage_and_first_token_window(self) -> None:
        def fake_request(_args: argparse.Namespace, prompt: str) -> dict:
            start, first, end, prompt_tokens = {
                "first": (10.0, 12.0, 14.0, 10),
                "second": (10.25, 11.25, 15.0, 30),
            }[prompt]
            return {
                "t_start": start, "t_first": first, "t_end": end,
                "duration_s": end - start, "ttft_s": first - start,
                "decode_duration_s": end - first,
                "completion_tokens": 8, "prompt_tokens": prompt_tokens,
                "finish_reason": "length", "error": None,
                "content_sha256": benchmark.sha256_text(prompt + " output"),
                "reasoning_content_sha256": benchmark.sha256_text(""),
                "content_chars": 6, "reasoning_content_chars": 0,
                "request_output_tok_s": 8 / (end - start),
                "request_decode_tok_s": 7 / (end - first),
            }

        args = argparse.Namespace(max_tokens=8, ignore_eos=True, timeout=2.0)
        with mock.patch.object(benchmark, "stream_request", side_effect=fake_request):
            level = benchmark.run_level(2, args, ["first", "second"], 0)
        self.assertEqual(level["completion_tokens_total"], 16)
        self.assertEqual(level["prompt_tokens_total"], 40)
        self.assertTrue(level["fixed_token_workload_valid"])
        # Assert independently known windows before their derived rates.
        self.assertEqual(level["wall_s"], 5.0)
        self.assertEqual(level["output_window_s"], 3.75)
        self.assertEqual(level["prompt_to_first_token_s"], 2.0)
        self.assertAlmostEqual(level["aggregate_tok_s"], 3.2)
        self.assertAlmostEqual(
            level["output_window_tok_s"], 16 / 3.75,
        )
        self.assertAlmostEqual(
            level["request_decode_tok_s_median"], (3.5 + 7 / 3.75) / 2,
        )
        self.assertAlmostEqual(level["prompt_tokens_per_s_to_first_token"], 20.0)

    def test_stream_request_keeps_usage_separate_from_sse_chunks(self) -> None:
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"one chunk"}}]}\n', b"\n",
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n', b"\n",
                    b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":64}}\n', b"\n",
                    b"data: [DONE]\n", b"\n",
                ])

        args = argparse.Namespace(
            model="m", max_tokens=64, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=Response()):
            record = benchmark.stream_request(args, "prompt")
        self.assertEqual(record["completion_tokens"], 64)
        self.assertEqual(record["prompt_tokens"], 12)
        self.assertTrue(record["done_received"])
        self.assertIsNone(record["error"])
        self.assertIsNotNone(record["request_decode_tok_s"])
        self.assertEqual(record["content_sha256"], benchmark.sha256_text("one chunk"))

    def test_stream_request_rejects_clean_eof_without_done(self) -> None:
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"partial"}}]}\n', b"\n",
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n', b"\n",
                    b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":64}}\n', b"\n",
                ])

        args = argparse.Namespace(
            model="m", max_tokens=64, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=Response()):
            record = benchmark.stream_request(args, "prompt")
        self.assertFalse(record["done_received"])
        self.assertIn("before [DONE]", record["error"])

    def test_missing_prompt_usage_fails_level(self) -> None:
        level = {
            "failures": 0,
            "token_count_complete": True,
            "prompt_token_count_complete": False,
            "fixed_token_workload_valid": True,
        }
        self.assertTrue(benchmark.level_failed(level, ignore_eos=True))

    def test_hung_worker_aborts_level_instead_of_overlapping_next(self) -> None:
        thread = mock.Mock()
        thread.is_alive.return_value = True
        args = argparse.Namespace(timeout=1.0)
        with (
            mock.patch.object(benchmark.threading, "Thread", return_value=thread),
            mock.patch.object(
                benchmark.time, "monotonic", side_effect=(10.0, 50.0),
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "exceeded the level deadline"):
                benchmark.run_level(1, args, ["prompt"], 0)
        thread.start.assert_called_once_with()
        thread.join.assert_called_once_with(0.0)



if __name__ == "__main__":
    unittest.main()
