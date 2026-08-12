#!/usr/bin/env python3
"""Focused tests for request-correlated feature benchmark telemetry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SCRIPT = HERE / "feature_concurrent_benchmark.py"
SPEC = importlib.util.spec_from_file_location("feature_concurrent_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FeatureBenchmarkTests(unittest.TestCase):
    def test_stream_request_captures_id_and_effective_prompt(self) -> None:
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"id":"chatcmpl-42","choices":[{"delta":{"content":"x"}}]}\n', b"\n",
                    b'data: {"id":"chatcmpl-42","choices":[{"delta":{},'
                    b'"finish_reason":"length"}]}\n', b"\n",
                    b'data: {"id":"chatcmpl-42","choices":[],"usage":'
                    b'{"prompt_tokens":40000,"completion_tokens":8,"timings":'
                    b'{"effective_prompt_tokens":2000,"prefilled_tokens":2000,'
                    b'"cached_prefix_tokens":0,"cache_hit":false,"prefill_ms":12.5,'
                    b'"decode_ms":20.0,"decode_tokens_per_sec":400.0}}}\n', b"\n",
                    b"data: [DONE]\n", b"\n",
                ])

        args = argparse.Namespace(
            model="m", max_tokens=8, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=Response()):
            row = benchmark.stream_request(args, "prompt")
        self.assertEqual(row["request_id"], "chatcmpl-42")
        self.assertEqual(row["prompt_tokens"], 40000)
        self.assertEqual(row["effective_prompt_tokens"], 2000)
        self.assertEqual(row["server_prefill_ms"], 12.5)
        self.assertTrue(row["done_received"])
        self.assertIsNone(row["error"])

    def test_clean_eof_without_done_is_rejected(self) -> None:
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"id":"chatcmpl-cut","choices":[{"delta":{"content":"x"},'
                    b'"finish_reason":"length"}]}\n', b"\n",
                ])

        args = argparse.Namespace(
            model="m", max_tokens=8, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=Response()):
            row = benchmark.stream_request(args, "prompt")
        self.assertIn("before [DONE]", row["error"])

    def test_enrich_level_reports_compression_ratio(self) -> None:
        level = {
            "prompt_tokens_total": 40000,
            "requests_detail": [{
                "request_id": "r1", "error": None,
                "effective_prompt_tokens": 2000,
                "server_prefill_ms": 10.0, "server_decode_ms": 20.0,
                "server_decode_tokens_per_sec": 100.0,
            }],
        }
        benchmark.enrich_level(level)
        self.assertTrue(level["request_ids_complete"])
        self.assertTrue(level["effective_prompt_token_count_complete"])
        self.assertEqual(level["effective_prompt_tokens_total"], 2000)
        self.assertEqual(level["effective_to_wire_prompt_ratio"], 0.05)

    def test_duplicate_request_ids_are_not_complete(self) -> None:
        level = {
            "prompt_tokens_total": 20,
            "requests_detail": [
                {"request_id": "same", "error": None, "effective_prompt_tokens": 10},
                {"request_id": "same", "error": None, "effective_prompt_tokens": 10},
            ],
        }
        benchmark.enrich_level(level)
        self.assertFalse(level["request_ids_complete"])

    def test_boolean_wire_counts_are_not_accepted_as_integers(self) -> None:
        level = {
            "prompt_tokens_total": 1,
            "requests_detail": [{
                "request_id": "r1", "error": None,
                "effective_prompt_tokens": True,
            }],
        }
        benchmark.enrich_level(level)
        self.assertFalse(level["effective_prompt_token_count_complete"])

    def test_client_provenance_records_exact_argv_and_source_digest(self) -> None:
        argv = ["python3", "feature_concurrent_benchmark.py", "--clients", "4"]
        result = benchmark.client_provenance(argv)
        self.assertEqual(result["client_argv"], argv)
        self.assertEqual(result["client_script"], str(SCRIPT.resolve()))
        self.assertEqual(
            result["client_script_sha256"],
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
