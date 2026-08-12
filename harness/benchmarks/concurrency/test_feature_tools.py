#!/usr/bin/env python3
"""Tests for pressure prompts, activation proof, and feature summaries."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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


generator = load("generate_feature_prompts")
proof = load("verify_feature_metrics")
summary = load("summarize_feature_matrix")


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
        # Zero acceptance is legitimate and must not invalidate execution proof.
        "ddtree_accepted_tokens": 0,
        "target_forwards": 3,
        "kvflash_page_ins": 0,
        "kvflash_page_outs": page_outs,
        "kvflash_resident_blocks": 8,
        "kvflash_reselects": 1,
        "pflash_applied": True,
        "pflash_input_tokens": 40000,
        "pflash_output_tokens": effective,
    }


class FeaturePromptTests(unittest.TestCase):
    def test_activation_profiles_are_disjoint_and_above_thresholds(self) -> None:
        compression = generator.build_records("compression")
        pressure = generator.build_records("kv-pressure")
        self.assertEqual(len(compression), 29)
        self.assertEqual(len(pressure), 29)
        self.assertEqual(len({row["prompt"] for row in compression}), 29)
        self.assertEqual(len({row["prompt"] for row in pressure}), 29)
        self.assertTrue(
            {row["prompt"] for row in compression}.isdisjoint(
                row["prompt"] for row in pressure
            )
        )
        self.assertGreaterEqual(min(row["target_words"] for row in compression), 34000)
        self.assertGreaterEqual(min(row["target_words"] for row in pressure), 12000)
        self.assertTrue(all(row["activation_target"] == "pflash-auto" for row in compression))


class FeatureRunnerShellTests(unittest.TestCase):
    def run_invalid_matrix(
        self, tmp: str, **overrides: str,
    ) -> subprocess.CompletedProcess[str]:
        model = Path(tmp) / "model.gguf"
        model.touch()
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("GGML_", "DFLASH_", "LUCE_", "HIP_", "ROCR_", "HSA_"))
            and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH")
        }
        env.update({
            "MODEL": str(model),
            "LUCE_SERVER_BIN": "/bin/true",
            "OUT": str(Path(tmp) / "out"),
            "VARIANTS": "ar",
            **overrides,
        })
        return subprocess.run(
            ["bash", str(HERE / "run_qwen36_feature_matrix.sh")],
            env=env, capture_output=True, text=True, check=False,
        )

    def test_client_and_proof_invocations_are_array_backed(self) -> None:
        runner = (HERE / "run_qwen36_feature_matrix.sh").read_text(
            encoding="utf-8",
        )
        client_lines = [
            line.strip() for line in runner.splitlines()
            if 'python3 "$CLIENT"' in line
        ]
        self.assertEqual(client_lines, [
            'python3 "$CLIENT" "${common_client[@]}"',
            'python3 "$CLIENT" "${common_client[@]}"',
        ])
        self.assertIn('local -a warmup_cmd=(', runner)
        self.assertIn('local -a benchmark_cmd=(', runner)
        self.assertIn(
            '"${warmup_cmd[@]}" > "$case_dir/warmup.txt"',
            runner,
        )
        self.assertIn(
            '"${benchmark_cmd[@]}" | tee "$case_dir/bench.txt"',
            runner,
        )

        proof_lines = [
            line.strip() for line in runner.splitlines()
            if 'python3 "$PROOF_TOOL"' in line
        ]
        self.assertEqual(proof_lines, ['python3 "$PROOF_TOOL"'])
        self.assertIn('local -a proof_cmd=(', runner)
        self.assertIn('"${proof_cmd[@]}"', runner)

    def test_signals_exit_and_launch_environment_is_recorded(self) -> None:
        runner = (HERE / "run_qwen36_feature_matrix.sh").read_text(encoding="utf-8")
        self.assertIn("trap stop_server EXIT", runner)
        self.assertIn("trap 'exit 130' INT", runner)
        self.assertIn("trap 'exit 143' TERM", runner)
        self.assertNotIn("trap stop_server EXIT INT TERM", runner)
        self.assertIn("'env ' > \"$case_dir/server-command.txt\"", runner)
        self.assertIn('"${launch_env[@]}" "${command[@]}"', runner)

    def test_duplicate_clients_are_rejected_before_artifacts_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_invalid_matrix(tmp, CLIENTS="4,4")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("CLIENTS contains duplicate entry: 4", result.stderr)
            self.assertFalse((Path(tmp) / "out").exists())

    def test_duplicate_variants_are_rejected_before_artifacts_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_invalid_matrix(tmp, VARIANTS="ar,ar")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("VARIANTS contains duplicate entry: ar", result.stderr)
            self.assertFalse((Path(tmp) / "out").exists())


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


class FeatureSummaryTests(unittest.TestCase):
    @staticmethod
    def item(
        variant: str, goodput: float, *, repeat: int = 1,
        output_hash: str | None = "same-output",
    ) -> dict:
        return {
            "report": {"max_tokens": 256, "ignore_eos": True},
            "meta": {
                "workload": "compression", "variant": variant, "repeat": repeat,
                "model_sha256": "a" * 64,
            },
            "level": {
                "clients": 8,
                "aggregate_tok_s": goodput,
                "output_window_tok_s": goodput,
                "effective_to_wire_prompt_ratio": 0.05 if variant == "full" else 1.0,
                "ttft_max_s": 2.0,
                "selected_prompt_set_sha256": "same-prompts",
                "selected_output_set_sha256": output_hash,
            },
            "proof": {
                "aggregate": {
                    "ddtree_steps": 4 if variant == "full" else 0,
                    "ddtree_suspensions": 1 if variant == "full" else 0,
                    "ddtree_accepted_tokens": 8 if variant == "full" else 0,
                    "target_forwards": 4,
                    "kvflash_page_ins": 2 if variant == "full" else 0,
                    "kvflash_page_outs": 3 if variant == "full" else 0,
                    "pflash_applied_requests": 8 if variant == "full" else 0,
                },
            },
        }

    def test_summary_compares_feature_row_to_ar(self) -> None:
        text = summary.summarize([self.item("ar", 10.0), self.item("full", 12.0)])
        self.assertIn("+20.0%", text)
        self.assertIn("2.00", text)
        self.assertIn("DDTree steps/susp.", text)
        self.assertIn("| 4/1 | 4 | 2/3 |", text)

    def test_feature_row_without_ar_control_reports_na(self) -> None:
        text = summary.summarize([self.item("full", 12.0)])
        row = next(line for line in text.splitlines() if "| full |" in line)
        self.assertEqual(row.split("|")[7].strip(), "n/a")

    def test_missing_output_digest_does_not_claim_stability(self) -> None:
        first = self.item("full", 12.0, output_hash=None)
        second = self.item("full", 13.0, repeat=2, output_hash=None)
        text = summary.summarize([first, second])
        row = next(line for line in text.splitlines() if "| full |" in line)
        self.assertEqual(row.split("|")[15].strip(), "n/a")

    def test_incomplete_repeat_metrics_are_reported_as_na(self) -> None:
        first = self.item("full", 12.0)
        second = self.item("full", 13.0, repeat=2)
        second["level"]["output_window_tok_s"] = None
        second["level"]["effective_to_wire_prompt_ratio"] = None
        text = summary.summarize([first, second])
        row = next(line for line in text.splitlines() if "| full |" in line)
        self.assertEqual(row.split("|")[6].strip(), "n/a")
        self.assertEqual(row.split("|")[8].strip(), "n/a")

    def test_incomplete_repeat_ttft_is_reported_as_na(self) -> None:
        first = self.item("full", 12.0)
        second = self.item("full", 13.0, repeat=2)
        second["level"]["ttft_max_s"] = None
        text = summary.summarize([first, second])
        row = next(line for line in text.splitlines() if "| full |" in line)
        self.assertEqual(row.split("|")[14].strip(), "n/a")

    def test_summary_rejects_incompatible_repeat_metadata(self) -> None:
        first = self.item("full", 12.0)
        second = self.item("full", 13.0, repeat=2)
        second["report"]["max_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "incompatible run metadata"):
            summary.summarize([first, second])

    def test_summary_rejects_incompatible_ar_control_metadata(self) -> None:
        ar = self.item("ar", 10.0)
        feature = self.item("full", 12.0)
        feature["meta"]["model_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "run metadata differs"):
            summary.summarize([ar, feature])

    def test_unstable_feature_row_suppresses_ar_delta(self) -> None:
        reports = [
            self.item("ar", 10.0, repeat=1),
            self.item("ar", 10.0, repeat=2),
            self.item("full", 12.0, repeat=1, output_hash="first"),
            self.item("full", 13.0, repeat=2, output_hash="second"),
        ]
        text = summary.summarize(reports)
        row = next(line for line in text.splitlines() if "| full |" in line)
        self.assertEqual(row.split("|")[7].strip(), "n/a")
        self.assertEqual(row.split("|")[15].strip(), "NO")

    def test_unstable_ar_control_suppresses_feature_delta(self) -> None:
        reports = [
            self.item("ar", 10.0, repeat=1, output_hash="first"),
            self.item("ar", 10.0, repeat=2, output_hash="second"),
            self.item("full", 12.0, repeat=1),
            self.item("full", 13.0, repeat=2),
        ]
        text = summary.summarize(reports)
        row = next(line for line in text.splitlines() if "| full |" in line)
        self.assertEqual(row.split("|")[7].strip(), "n/a")


if __name__ == "__main__":
    unittest.main()
