#!/usr/bin/env python3
"""Focused correctness tests for concurrent_benchmark.py."""

from __future__ import annotations

import importlib.util
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).with_name("concurrent_benchmark.py")
SPEC = importlib.util.spec_from_file_location("concurrent_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FakeStreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines
        self.headers = {"Server": "test-server/1"}

    def __enter__(self) -> "FakeStreamingResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class FakePropsResponse:
    status = 200
    headers = {"Server": "test-server/1"}

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakePropsResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.body


class TokenAccountingTests(unittest.TestCase):
    def test_usage_is_authoritative_and_validates_fixed_workload(self) -> None:
        record = {
            "completion_tokens": 64,
            "token_count_source": "usage",
            "finish_reason": "stop",
            "error": None,
        }

        benchmark.finalize_token_accounting(record, 64, True)

        self.assertEqual(record["completion_tokens"], 64)
        self.assertEqual(record["token_count_source"], "usage")
        self.assertTrue(record["token_count_available"])
        self.assertTrue(record["fixed_token_count_validated"])

    def test_length_finish_without_usage_is_still_unknown(self) -> None:
        record = {
            "completion_tokens": None,
            "token_count_source": None,
            "delta_chunks": 7,
            "finish_reason": "length",
            "error": None,
        }

        benchmark.finalize_token_accounting(record, 64, True)

        self.assertIsNone(record["completion_tokens"])
        self.assertEqual(record["token_count_source"], "unavailable")
        self.assertFalse(record["token_count_available"])
        self.assertFalse(record["fixed_token_count_validated"])

    def test_sse_chunks_are_never_used_as_tokens(self) -> None:
        record = {
            "completion_tokens": None,
            "token_count_source": None,
            "delta_chunks": 63,
            "finish_reason": "stop",
            "error": None,
        }

        benchmark.finalize_token_accounting(record, 64, True)

        self.assertIsNone(record["completion_tokens"])
        self.assertEqual(record["token_count_source"], "unavailable")
        self.assertFalse(record["token_count_available"])
        self.assertFalse(record["fixed_token_count_validated"])

    def test_stream_requests_usage_and_keeps_chunk_count_separate(self) -> None:
        events = [
            b'data: {"model":"served-model","choices":[{"delta":{"content":"hi"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n',
            b"\n",
            b'data: {"choices":[],"usage":{"completion_tokens":64}}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        response = FakeStreamingResponse(events)
        with mock.patch.object(
            benchmark.urllib.request, "urlopen", return_value=response,
        ) as urlopen:
            record = benchmark.stream_request(
                base_url="http://127.0.0.1:1/v1",
                api_key="",
                model="requested-model",
                prompt="hello",
                max_tokens=64,
                temperature=0.0,
                timeout=5.0,
                ignore_eos=True,
                capture_output_text=True,
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(record["delta_chunks"], 1)
        self.assertEqual(record["completion_tokens"], 64)
        self.assertEqual(record["response_model"], "served-model")
        self.assertEqual(record["http_server_header"], "test-server/1")
        self.assertEqual(
            record["content_sha256"], benchmark.sha256_bytes(b"hi"),
        )
        self.assertEqual(record["content_char_length"], 2)
        self.assertEqual(record["content_byte_length"], 2)
        self.assertEqual(record["content_text"], "hi")
        self.assertEqual(record["reasoning_content_text"], "")

    def test_output_identity_is_independent_of_sse_chunk_boundaries(self) -> None:
        def response(deltas: list[dict[str, str]]) -> FakeStreamingResponse:
            lines: list[bytes] = []
            for delta in deltas:
                event = {"choices": [{"delta": delta}]}
                lines.extend([
                    f"data: {json.dumps(event, ensure_ascii=False)}\n".encode(),
                    b"\n",
                ])
            lines.extend([
                b'data: {"choices":[],"usage":{"completion_tokens":64}}\n',
                b"\n",
                b"data: [DONE]\n",
                b"\n",
            ])
            return FakeStreamingResponse(lines)

        split = response([
            {"content": "alpha ", "reasoning_content": "思"},
            {"content": "β", "reasoning_content": "考"},
        ])
        merged = response([
            {"content": "alpha β", "reasoning_content": "思考"},
        ])
        with mock.patch.object(
            benchmark.urllib.request, "urlopen", side_effect=[split, merged],
        ):
            records = [
                benchmark.stream_request(
                    base_url="http://127.0.0.1:1/v1",
                    api_key="",
                    model="model",
                    prompt="prompt",
                    max_tokens=64,
                    temperature=0.0,
                    timeout=5.0,
                    ignore_eos=True,
                )
                for _ in range(2)
            ]

        identity_fields = (
            "content_sha256",
            "content_char_length",
            "content_byte_length",
            "reasoning_content_sha256",
            "reasoning_content_char_length",
            "reasoning_content_byte_length",
        )
        self.assertEqual(
            {field: records[0][field] for field in identity_fields},
            {field: records[1][field] for field in identity_fields},
        )
        self.assertEqual(records[0]["content_char_length"], len("alpha β"))
        self.assertEqual(
            records[0]["content_byte_length"], len("alpha β".encode("utf-8")),
        )
        self.assertEqual(records[0]["reasoning_content_char_length"], 2)
        self.assertNotIn("content_text", records[0])
        self.assertNotIn("reasoning_content_text", records[0])
        self.assertEqual(records[0]["reasoning_content_byte_length"], 6)


class ReportingTests(unittest.TestCase):
    def test_server_props_snapshot_records_body_and_hash(self) -> None:
        body = b'{"server":{"version":"test"},"runtime":{"backend":"hip"}}'
        with mock.patch.object(
            benchmark.urllib.request, "urlopen",
            return_value=FakePropsResponse(body),
        ) as urlopen:
            snapshot = benchmark.fetch_server_props(
                "http://127.0.0.1:18080/v1", "secret", 5.0,
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:18080/props")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["body"]["runtime"]["backend"], "hip")
        self.assertEqual(snapshot["body_sha256"], benchmark.sha256_bytes(body))

    def test_missing_counts_suppress_token_throughput_not_sse_diagnostics(
        self,
    ) -> None:
        def fake_stream_request(**_kwargs: object) -> dict[str, object]:
            start = time.perf_counter()
            return {
                "t_start": start,
                "t_first": start + 0.1,
                "t_end": start + 1.0,
                "chunk_times": [start + 0.1, start + 0.2],
                "delta_chunks": 2,
                "completion_tokens": None,
                "prompt_tokens": None,
                "token_count_source": "unavailable",
                "token_count_available": False,
                "fixed_token_count_validated": False,
                "token_count_warning": "missing",
                "server_decode_tok_s": None,
                "response_model": "model",
                "system_fingerprint": None,
                "http_server_header": "server",
                "finish_reason": "length",
                "error": None,
            }

        args = SimpleNamespace(
            requests_per_stream=1,
            base_url="http://127.0.0.1:1/v1",
            api_key="",
            model="model",
            max_tokens=64,
            temperature=0.0,
            timeout=5.0,
            ignore_eos=True,
            request_stream_usage=True,
            min_percentile_samples=100,
        )
        with mock.patch.object(
            benchmark, "stream_request", side_effect=fake_stream_request,
        ):
            level = benchmark.run_level(1, args, ["prompt"])

        self.assertIsNone(level["completion_tokens_total"])
        self.assertIsNone(level["aggregate_tok_s"])
        self.assertFalse(level["token_count_complete"])
        self.assertEqual(level["sse_delta_chunks_total"], 2)
        self.assertAlmostEqual(level["aggregate_sse_delta_chunks_s"], 2.0)
        request = level["streams"][0]["requests"][0]
        self.assertIsNone(request["itl_mean_ms"])
        self.assertAlmostEqual(request["sse_delta_gap_mean_ms"], 100.0)
        self.assertIsNone(request["sse_delta_gap_p95_ms"])
        self.assertEqual(
            request["sse_delta_gap_p95_status"], "insufficient_samples",
        )

    def test_request_tok_s_uses_exact_count_over_full_request_interval(
        self,
    ) -> None:
        record = {
            "prompt_index": 0,
            "global_prompt_index": 0,
            "prompt_sha256": "prompt-hash",
            "t_start": 100.0,
            "t_first": 107.5,
            "t_end": 108.0,
            "chunk_times": [107.5, 107.75],
            "delta_chunks": 2,
            "completion_tokens": 64,
            "prompt_tokens": 10,
            "token_count_source": "usage",
            "token_count_available": True,
            "fixed_token_count_validated": True,
            "token_count_warning": None,
            "server_decode_tok_s": None,
            "response_model": "model",
            "system_fingerprint": None,
            "http_server_header": "server",
            "finish_reason": "length",
            "error": None,
        }

        request = benchmark.finalize_request(record, level_start=100.0)

        self.assertEqual(request["request_tok_s"], 8.0)
        self.assertEqual(
            request["request_tok_s_source"], "usage.completion_tokens",
        )
        self.assertEqual(
            request["request_tok_s_interval"], "request_start_to_stream_end",
        )
        self.assertIsNone(request["decode_tok_s"])
        self.assertEqual(
            request["decode_tok_s_status"], "unavailable_from_sse_chunks",
        )

        # Moving the first transport delta changes TTFT, but not an exact
        # full-request token rate.
        earlier_first = dict(record, t_first=101.0, chunk_times=[101.0, 107.75])
        earlier = benchmark.finalize_request(earlier_first, level_start=100.0)
        self.assertEqual(earlier["request_tok_s"], 8.0)
        self.assertNotEqual(earlier["ttft_s"], request["ttft_s"])

        inferred = benchmark.finalize_request(
            dict(record, token_count_source="delta_chunks"), level_start=100.0,
        )
        self.assertIsNone(inferred["request_tok_s"])
        self.assertIsNone(inferred["request_tok_s_source"])

    def test_p95_is_withheld_below_sample_floor(self) -> None:
        values = [float(i) for i in range(100)]

        self.assertIsNone(
            benchmark.percentile_if_sufficient(values[:99], 95.0, 100),
        )
        self.assertEqual(
            benchmark.percentile_if_sufficient(values, 95.0, 100), 94.0,
        )

    def test_api_key_is_redacted_from_recorded_argv(self) -> None:
        argv = [
            "benchmark.py", "--api-key", "secret-one",
            "--api-key=secret-two", "--clients", "8",
        ]

        sanitized = benchmark.sanitize_argv(argv)

        self.assertEqual(
            sanitized,
            [
                "benchmark.py", "--api-key", "<redacted>",
                "--api-key=<redacted>", "--clients", "8",
            ],
        )
        self.assertNotIn("secret-one", " ".join(sanitized))
        self.assertNotIn("secret-two", " ".join(sanitized))

    def test_parser_defaults_to_usage_and_conservative_p95(self) -> None:
        args = benchmark.build_parser().parse_args([])

        self.assertTrue(args.request_stream_usage)
        self.assertEqual(args.min_percentile_samples, 100)


if __name__ == "__main__":
    unittest.main()
