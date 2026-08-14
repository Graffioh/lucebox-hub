#!/usr/bin/env python3
"""Tests for attach_ddtree_metrics.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("attach_ddtree_metrics.py")
SPEC = importlib.util.spec_from_file_location("attach_ddtree_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)


class DDTreeProofTests(unittest.TestCase):
    def test_attaches_acceptance_from_matched_requests(self) -> None:
        report = {"levels": [{"wave_results": [{"requests_detail": [
            {"response_id": "a", "error": None},
            {"response_id": "b", "error": None},
        ]}]}]}
        metrics = {
            "a": {"ddtree_steps": 2, "ddtree_accepted_tokens": 8, "target_forwards": 4},
            "b": {"ddtree_steps": 3, "ddtree_accepted_tokens": 12, "target_forwards": 6},
        }
        proof.attach(report, metrics)
        self.assertEqual(report["ddtree_proof"]["ddtree_steps"], 5)
        self.assertEqual(report["ddtree_proof"]["mean_accepted_length"], 5.0)
        self.assertEqual(report["ddtree_proof"]["acceptance_rate"], 5 / 16)

    def test_missing_or_zero_step_proof_fails_closed(self) -> None:
        report = {"levels": [{"wave_results": [{"requests_detail": [
            {"response_id": "a", "error": None},
        ]}]}]}
        with self.assertRaisesRegex(ValueError, "missing concurrency metric"):
            proof.attach(report, {})
        metrics = {"a": {
            "ddtree_steps": 0, "ddtree_accepted_tokens": 0, "target_forwards": 1,
        }}
        with self.assertRaisesRegex(ValueError, "ddtree_steps must be positive"):
            proof.attach(report, metrics)


    def test_boolean_counters_are_not_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("[concurrency-metrics] " + json.dumps({
                "response_id": "a", "ddtree_steps": True,
                "ddtree_accepted_tokens": 1, "target_forwards": 1,
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid ddtree_steps"):
                proof.load_metrics(path)


if __name__ == "__main__":
    unittest.main()
