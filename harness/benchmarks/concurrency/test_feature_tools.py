#!/usr/bin/env python3
"""Tests for pressure prompts, activation proof, and feature summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def load(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proof = load("verify_feature_metrics")


def report(
    variant: str = "full", workload: str = "compression",
    effective: tuple[int, int] = (2000, 2100),
) -> dict:
    return {
        "server_metadata": {
            "variant": variant,
            "workload": workload,
            "feature_config": {
                "prefill_compression": "auto",
                "prefill_threshold": 32000,
                "kvflash": "auto",
                "kvflash_max_pool_tokens": 8192,
                "kvflash_scorer_drafter": "/models/Qwen3-0.6B-BF16.gguf",
                "kvflash_scorer_drafter_sha256": "a" * 64,
            },
            "runtime_observed": {
                "kvflash_active": True,
                "physical_kv_pool_tokens": 8192,
            },
        },
        "levels": [{
            "requests_detail": [
                {"request_id": "r1", "error": None,
                 "prompt_tokens": 40000,
                 "effective_prompt_tokens": effective[0]},
                {"request_id": "r2", "error": None,
                 "prompt_tokens": 40000,
                 "effective_prompt_tokens": effective[1]},
            ],
        }],
    }


def metric(request_id: str, effective: int, page_outs: int = 1) -> dict:
    return {
        "request_id": request_id,
        "effective_prompt_tokens": effective,
        "ddtree_steps": 3,
        "ddtree_suspensions": 0,
        "ddtree_accepted_tokens": 0,
        "spec_steps": 0,
        "spec_accepted_tokens": 0,
        "target_forwards": 3,
        "kvflash_page_ins": 0,
        "kvflash_page_outs": page_outs,
        "kvflash_resident_blocks": 8,
        "kvflash_reselects": 1,
        "pflash_applied": True,
        "pflash_input_tokens": 40000,
        "pflash_output_tokens": effective,
    }


class FeatureProofTests(unittest.TestCase):
    def test_full_below_pool_passes_without_page_traffic(self) -> None:
        rows = [
            metric("warmup", 50),
            metric("r1", 2000, page_outs=0),
            metric("r2", 2100, page_outs=0),
        ]
        result = proof.verify(
            report(), rows, {"ddtree", "pflash", "kvflash"},
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertFalse(result["kvflash_page_traffic_required"])
        self.assertEqual(
            result["kvflash_page_traffic_reason"],
            "compressed-prompt-fits-pool",
        )
        self.assertEqual(result["aggregate"]["ddtree_accepted_tokens"], 0)
        self.assertEqual(result["aggregate"]["ddtree_suspensions"], 0)
        self.assertEqual(result["matched_metric_count"], 2)

    def test_ddtree_suspensions_required_and_aggregated(self) -> None:
        missing = metric("r1", 2000)
        del missing["ddtree_suspensions"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(
                proof.PREFIX + json.dumps(missing) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "missing telemetry keys.*ddtree_suspensions",
            ):
                proof.parse_markers(path)

        rows = [metric("r1", 2000), metric("r2", 2100)]
        rows[1]["ddtree_suspensions"] = 1
        result = proof.verify(
            report(), rows, {"ddtree", "pflash", "kvflash"},
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["aggregate"]["ddtree_suspensions"], 1)
        self.assertEqual(
            [row["ddtree_suspensions"] for row in result["requests"]],
            [0, 1],
        )

    def test_ddtree_suspensions_must_be_binary_per_request(self) -> None:
        raw_two = metric("r1", 2000)
        raw_two["ddtree_suspensions"] = 2
        with self.assertRaisesRegex(
            ValueError, "ddtree_suspensions must be 0 or 1 per request",
        ):
            proof.aggregate_rows([raw_two])

    def test_duplicate_terminal_telemetry_is_rejected(self) -> None:
        row = metric("r1", 2000)
        with self.assertRaisesRegex(ValueError, "duplicate telemetry request ID r1"):
            proof.aggregate_rows([row, dict(row)])

    def test_boolean_telemetry_counts_are_rejected(self) -> None:
        integer_fields = (*proof.COUNTERS, "effective_prompt_tokens",
                          "kvflash_resident_blocks", "pflash_input_tokens",
                          "pflash_output_tokens")
        for key in integer_fields:
            with self.subTest(key=key):
                row = metric("r1", 2000)
                row[key] = True
                with self.assertRaisesRegex(ValueError, key):
                    proof.aggregate_rows([row])

    def test_boolean_wire_and_metadata_counts_do_not_prove_features(self) -> None:
        rows = [metric("r1", 2000), metric("r2", 2100)]
        wire = report()
        wire["levels"][0]["requests_detail"][0]["effective_prompt_tokens"] = True
        result = proof.verify(wire, rows, set())
        self.assertFalse(result["valid"])
        self.assertIn("missing usage.timings.effective_prompt_tokens", result["errors"][0])

        threshold = report()
        threshold["server_metadata"]["feature_config"]["prefill_threshold"] = True
        result = proof.verify(threshold, rows, {"pflash"})
        self.assertFalse(result["valid"])
        self.assertIn("positive recorded token threshold", "\n".join(result["errors"]))

        pool = report()
        pool["server_metadata"]["runtime_observed"]["physical_kv_pool_tokens"] = True
        result = proof.verify(pool, rows, {"kvflash"})
        self.assertFalse(result["valid"])
        self.assertIn("positive startup-observed physical pool", "\n".join(result["errors"]))

        requested = report()
        requested["server_metadata"]["feature_config"]["kvflash_max_pool_tokens"] = True
        result = proof.verify(requested, rows, {"kvflash"})
        self.assertFalse(result["valid"])
        self.assertIn("recorded pool-token cap", "\n".join(result["errors"]))

    def test_full_above_pool_requires_page_traffic(self) -> None:
        rows = [
            metric("r1", 9000, page_outs=0),
            metric("r2", 9100, page_outs=0),
        ]
        result = proof.verify(
            report(effective=(9000, 9100)), rows,
            {"ddtree", "pflash", "kvflash"},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(result["kvflash_page_traffic_required"])
        self.assertIn(
            "no page-in/page-out", "\n".join(result["errors"])
        )

    def test_full_aggregate_effective_demand_requires_page_traffic(self) -> None:
        rows = [
            metric("r1", 5000, page_outs=0),
            metric("r2", 5000, page_outs=0),
        ]
        result = proof.verify(
            report(effective=(5000, 5000)), rows,
            {"ddtree", "pflash", "kvflash"},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(result["kvflash_page_traffic_required"])
        self.assertEqual(
            result["kvflash_page_traffic_reason"],
            "effective-prompt-exceeds-pool",
        )
        self.assertIn("no page-in/page-out", "\n".join(result["errors"]))

    def test_full_effective_demand_is_evaluated_per_client_level(self) -> None:
        input_report = report(effective=(5000, 5000))
        requests = input_report["levels"][0]["requests_detail"]
        input_report["levels"] = [
            {"requests_detail": [requests[0]]},
            {"requests_detail": [requests[1]]},
        ]
        rows = [
            metric("r1", 5000, page_outs=0),
            metric("r2", 5000, page_outs=0),
        ]
        result = proof.verify(
            input_report, rows, {"ddtree", "pflash", "kvflash"},
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertFalse(result["kvflash_page_traffic_required"])
        self.assertEqual(
            result["kvflash_page_traffic_reason"],
            "compressed-prompt-fits-pool",
        )

    def test_unknown_kvflash_variant_fails_closed(self) -> None:
        rows = [
            metric("r1", 2000, page_outs=0),
            metric("r2", 2100, page_outs=0),
        ]
        result = proof.verify(
            report(variant="typo"), rows, {"kvflash"},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(result["kvflash_page_traffic_required"])
        self.assertEqual(result["kvflash_page_traffic_reason"], "unknown-variant")
        self.assertIn("no page-in/page-out", "\n".join(result["errors"]))

    def test_pflash_auto_requires_measured_input_above_threshold(self) -> None:
        rows = [metric("r1", 2000), metric("r2", 2100)]
        rows[0]["pflash_input_tokens"] = 31999
        result = proof.verify(report(), rows, {"pflash"})
        self.assertFalse(result["valid"])
        self.assertIn(
            "did not reach its recorded token threshold",
            "\n".join(result["errors"]),
        )

    def test_kvflash_only_cannot_succeed_without_page_traffic(self) -> None:
        rows = [
            metric("r1", 2000, page_outs=0),
            metric("r2", 2100, page_outs=0),
        ]
        result = proof.verify(
            report(variant="kvflash"), rows, {"kvflash"},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(result["kvflash_page_traffic_required"])
        self.assertEqual(
            result["kvflash_page_traffic_reason"],
            "kvflash-only-ablation",
        )
        self.assertIn(
            "no page-in/page-out", "\n".join(result["errors"])
        )

    def test_kvflash_requires_explicit_hashed_scorer(self) -> None:
        input_report = report(variant="kvflash")
        config = input_report["server_metadata"]["feature_config"]
        config["kvflash_scorer_drafter"] = None
        config["kvflash_scorer_drafter_sha256"] = None
        rows = [metric("r1", 9000), metric("r2", 9100)]
        result = proof.verify(input_report, rows, {"kvflash"})
        self.assertFalse(result["valid"])
        text = "\n".join(result["errors"])
        self.assertIn("no explicit scorer drafter", text)
        self.assertIn("not a valid SHA-256 digest", text)

    def test_kvflash_rejects_malformed_scorer_hash(self) -> None:
        rows = [metric("r1", 9000), metric("r2", 9100)]
        for digest in ("not-a-hash", "g" * 64, "a" * 63, "a" * 65):
            with self.subTest(digest=digest):
                input_report = report(variant="kvflash")
                input_report["server_metadata"]["feature_config"][
                    "kvflash_scorer_drafter_sha256"
                ] = digest
                result = proof.verify(input_report, rows, {"kvflash"})
                self.assertFalse(result["valid"])
                self.assertIn(
                    "not a valid SHA-256 digest", "\n".join(result["errors"])
                )

    def test_kvflash_requires_startup_observed_pool(self) -> None:
        input_report = report(variant="kvflash")
        input_report["server_metadata"]["runtime_observed"] = None
        rows = [metric("r1", 9000), metric("r2", 9100)]
        result = proof.verify(input_report, rows, {"kvflash"})
        self.assertFalse(result["valid"])
        text = "\n".join(result["errors"])
        self.assertIn("startup marker was not recorded", text)
        self.assertIn("no positive startup-observed physical pool", text)

    def test_requested_features_cannot_succeed_silently(self) -> None:
        rows = [metric("r1", 9000, page_outs=0), metric("r2", 9100, page_outs=0)]
        rows[0]["ddtree_steps"] = 0
        rows[1]["pflash_applied"] = False
        result = proof.verify(
            report(effective=(9000, 9100)), rows,
            {"ddtree", "pflash", "kvflash"},
        )
        self.assertFalse(result["valid"])
        text = "\n".join(result["errors"])
        self.assertIn("ddtree_steps is zero", text)
        self.assertIn("pflash_applied is false", text)
        self.assertIn("no page-in/page-out", text)

    def test_pflash_input_mismatch_with_wire_tokens_fails(self) -> None:
        rows = [metric("r1", 2000), metric("r2", 2100)]
        rows[0]["pflash_input_tokens"] = 39999
        result = proof.verify(report(), rows, {"pflash"})
        self.assertFalse(result["valid"])
        self.assertIn(
            "pflash_input_tokens=39999 does not match wire value 40000",
            "\n".join(result["errors"]),
        )

    def test_effective_prompt_mismatch_fails(self) -> None:
        rows = [metric("r1", 1999), metric("r2", 2100)]
        result = proof.verify(report(), rows, set())
        self.assertFalse(result["valid"])
        self.assertIn("does not match wire value", result["errors"][0])

    def test_measured_request_requires_explicit_error_status(self) -> None:
        input_report = report()
        del input_report["levels"][0]["requests_detail"][0]["error"]
        with self.assertRaisesRegex(ValueError, "explicit error status"):
            proof.measured_requests(input_report)

    def test_forced_chain_requires_steps_and_matching_startup_proof(self) -> None:
        input_report = report(variant="speculation")
        input_report["server_metadata"]["feature_config"]["decode_mode"] = (
            "speculation"
        )
        rows = [metric("r1", 2000), metric("r2", 2100)]
        for row in rows:
            row["ddtree_steps"] = 0
            row["ddtree_accepted_tokens"] = 0
            row["spec_steps"] = 3
            row["spec_accepted_tokens"] = 2
        startup = (
            "[parallel-chain] speculator=test-score "
            "decode_mode=speculation draft=q4-mix-compatible"
        )
        result = proof.verify(input_report, rows, {"chain"}, startup)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["decode_mode"], "speculation")
        self.assertEqual(result["aggregate"]["spec_steps"], 6)
        self.assertEqual(result["aggregate"]["spec_accepted_tokens"], 4)

        rows[0]["spec_steps"] = 0
        rows[0]["spec_accepted_tokens"] = 0
        result = proof.verify(input_report, rows, {"chain"}, startup)
        self.assertFalse(result["valid"])
        self.assertIn("spec_steps is zero", "\n".join(result["errors"]))

        result = proof.verify(input_report, rows[1:], {"chain"}, "")
        self.assertFalse(result["valid"])
        self.assertIn("startup proof is missing", "\n".join(result["errors"]))

    def test_adaptive_chain_allows_ar_argmax_but_requires_profile(self) -> None:
        input_report = report(variant="adaptive-on")
        input_report["server_metadata"]["feature_config"]["decode_mode"] = (
            "adaptive"
        )
        rows = [metric("r1", 2000), metric("r2", 2100)]
        for row in rows:
            row["ddtree_steps"] = 0
            row["ddtree_accepted_tokens"] = 0
        startup = "\n".join((
            "[spec-profile] context=4096 reps=5 mode=batched-draft",
            "[parallel-chain] speculator=test-score "
            "",
        ))
        result = proof.verify(input_report, rows, {"chain"}, startup)
        self.assertTrue(result["valid"], result["errors"])

        no_profile = startup.splitlines()[1]
        result = proof.verify(input_report, rows, {"chain"}, no_profile)
        self.assertFalse(result["valid"])
        self.assertIn("cost-profile proof is missing", "\n".join(result["errors"]))

    def test_ar_decode_mode_rejects_speculation_activity(self) -> None:
        input_report = report(variant="ar")
        input_report["server_metadata"]["feature_config"]["decode_mode"] = "ar"
        rows = [metric("r1", 2000), metric("r2", 2100)]
        rows[0]["spec_steps"] = 1
        result = proof.verify(input_report, rows, set())
        self.assertFalse(result["valid"])
        self.assertIn(
            "AR decode_mode emitted chain speculation", "\n".join(result["errors"])
        )

    def test_spec_acceptance_without_step_is_rejected(self) -> None:
        row = metric("r1", 2000)
        row["spec_accepted_tokens"] = 1
        with self.assertRaisesRegex(
            ValueError, "spec_accepted_tokens requires positive spec_steps",
        ):
            proof.aggregate_rows([row])


if __name__ == "__main__":
    unittest.main()
