#!/usr/bin/env python3
"""Tests for the Qwen3.8 fixed-work replicate comparator."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = HERE / "compare_qwen38_fixed.py"
SPEC = importlib.util.spec_from_file_location("compare_qwen38_fixed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write_replicate(root: Path, name: str, rates: dict[str, float], prompt_hash: str = "prompt") -> Path:
    replicate = root / name
    for variant, rate in rates.items():
        case = replicate / variant
        case.mkdir(parents=True)
        report = {
            "max_tokens": 256,
            "prompt_file_sha256": "fixture",
            "server_metadata": {
                "model_sha256": "model",
                "server_binary_sha256": "binary",
            },
            "levels": [{
                "aggregate_tok_s": rate,
                "clients": 29,
                "requests": 29,
                "requests_ok": 29,
                "failures": 0,
                "fixed_token_workload_valid": True,
                "completion_tokens_total": 7424,
                "selected_prompt_set_sha256": prompt_hash,
            }],
        }
        (case / "bench.json").write_text(json.dumps(report), encoding="utf-8")
        (case / "runtime-metadata.json").write_text("{}\n", encoding="utf-8")
        (case / "gpu-identity.txt").write_text("active_gpu_arch=gfx1201\n", encoding="utf-8")
        metric = {
            "engine_request_id": 1,
            "output_tokens": 256,
            "spec_accepted_tokens": 2 if variant != "ar" else 0,
            "spec_steps": 1 if variant != "ar" else 0,
            "spec_service_ar_steps": 0,
            "target_forwards": 255 if variant != "ar" else 256,
        }
        lines = [
            "[concurrency-metrics] " + json.dumps(metric | {"engine_request_id": request_id})
            for request_id in range(1, 30)
        ]
        if variant == "adaptive":
            lines.append(
                "[spec-profile] context=4096 "
                "context_method=synthetic-zero-kv-zero-features-v1 reps=5"
            )
            lines.append("[spec-epoch] " + json.dumps({
                "profiled_cost_us": 100.0,
                "predicted_cost_us": 110.0,
                "requests": [{
                    "request_id": 1,
                    "route": "speculation",
                    "reason": "selected_by_joint_goodput",
                }],
            }))
        (case / "server.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return replicate


class Qwen38FixedComparisonTests(unittest.TestCase):
    def test_accepts_stable_adaptive_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                write_replicate(root, "r1", {"ar": 100, "speculation": 80, "adaptive": 110}),
                write_replicate(root, "r2", {"ar": 101, "speculation": 81, "adaptive": 111}),
            ]
            result = comparison.compare(runs, max_relative_range=0.05, min_adaptive_ratio=1.0)
            self.assertEqual(result["measurement_status"], "valid")
            self.assertEqual(result["performance_observation"], "adaptive_above_ar")
            self.assertAlmostEqual(result["paired"]["adaptive_vs_ar_mean"], 1.0995, places=3)
            profile = result["profiles"]["adaptive"][0]
            self.assertEqual(profile["spec_accepted_tokens"], 58)
            self.assertEqual(profile["spec_steps"], 29)
            self.assertEqual(profile["selector_routes"], {"speculation": 1})
            self.assertEqual(
                profile["profile_context_method"],
                "synthetic-zero-kv-zero-features-v1",
            )

    def test_rejects_adaptive_run_without_profile_context_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_replicate(
                Path(tmp),
                "r1",
                {"ar": 100, "speculation": 80, "adaptive": 110},
            )
            log = run / "adaptive" / "server.log"
            lines = [
                line for line in log.read_text(encoding="utf-8").splitlines()
                if not line.startswith("[spec-profile] ")
            ]
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profile context method"):
                comparison.load_case(run, "adaptive")

    def test_reports_inconclusive_variance_before_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                write_replicate(root, "r1", {"ar": 100, "speculation": 80, "adaptive": 90}),
                write_replicate(root, "r2", {"ar": 100, "speculation": 80, "adaptive": 110}),
            ]
            result = comparison.compare(runs, max_relative_range=0.05, min_adaptive_ratio=1.0)
            self.assertEqual(result["measurement_status"], "valid")
            self.assertEqual(result["performance_observation"], "adaptive_variable")
            self.assertFalse(result["stability"]["adaptive"])

    def test_rejects_nonidentical_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                write_replicate(root, "r1", {"ar": 100, "speculation": 80, "adaptive": 90}),
                write_replicate(
                    root, "r2", {"ar": 100, "speculation": 80, "adaptive": 90},
                    prompt_hash="different",
                ),
            ]
            with self.assertRaisesRegex(ValueError, "workload fingerprint"):
                comparison.compare(runs, max_relative_range=0.05, min_adaptive_ratio=1.0)

    def test_rejects_failed_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = write_replicate(
                root, "r1", {"ar": 100, "speculation": 80, "adaptive": 90},
            )
            path = run / "adaptive" / "bench.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["levels"][0]["failures"] = 1
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed or incomplete requests"):
                comparison.compare([run], max_relative_range=0.05, min_adaptive_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
