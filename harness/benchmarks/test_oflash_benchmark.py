import json
import sys
import tempfile
import unittest
from pathlib import Path

import oflash_benchmark as bench

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import client_test_runner as client_bench


class OFlashBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _report(phase, generation, case_id, accept, tps):
        return {
            "command": "bench",
            "model": "test-model",
            "oflash_phase": phase,
            "oflash_before": {"trainer_alive": False, "adapter_generation": generation},
            "oflash_after": {"trainer_alive": False, "adapter_generation": generation},
            "suites": {
                "he": {
                    "prompt_fingerprint": f"fingerprint-{case_id}",
                    "results": [{
                        "id": case_id, "text": "same", "correct": True,
                        "accept_rate": accept, "ttft_s": 0.1,
                        "output_tok_s": tps,
                        "server_timings": {"decode_tokens_per_sec": tps},
                    }]
                }
            },
        }

    def test_prepare_folds_has_no_leakage_and_full_heldout_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts"
            prompts.mkdir()
            for suite in ("he", "gsm"):
                rows = [
                    {"id": f"{suite}_{i}", "messages": [{"role": "user", "content": str(i)}]}
                    for i in range(6)
                ]
                (prompts / bench.SUITE_FILES[suite]).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )

            manifest = bench.prepare_folds(
                prompts, root / "out", ["he", "gsm"], folds=3, seed="test"
            )
            self.assertEqual(manifest["total_cases"], 12)
            seen = set()
            for fold in manifest["fold"]:
                for suite, split in fold["suites"].items():
                    adapt = set(split["adapt_ids"])
                    heldout = set(split["heldout_ids"])
                    self.assertFalse(adapt & heldout)
                    self.assertEqual(len(adapt | heldout), 6)
                    seen.update(f"{suite}:{case_id}" for case_id in heldout)
            self.assertEqual(len(seen), 12)

    def test_compare_uses_frozen_paired_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base.json"
            candidate_path = root / "candidate.json"
            base_path.write_text(json.dumps(self._report("heldout-base", 0, "he_1", 0.5, 80)))
            candidate_path.write_text(json.dumps(self._report("heldout-adapted", 3, "he_1", 0.6, 100)))
            result = bench.compare_reports(
                base_path, candidate_path, None, bootstrap_samples=100, bootstrap_seed=1
            )
            self.assertAlmostEqual(result["summary"]["mean_accept_rate_delta"], 0.1)
            self.assertAlmostEqual(result["summary"]["mean_speedup"], 1.25)
            self.assertEqual(result["summary"]["exact_output_matches"], 1)

    def test_pool_rejects_duplicate_heldout_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(2):
                base = root / f"base-{index}.json"
                candidate = root / f"candidate-{index}.json"
                base.write_text(json.dumps(self._report("heldout-base", 0, "same-id", 0.5, 80)))
                candidate.write_text(json.dumps(self._report("heldout-adapted", index + 1, "same-id", 0.6, 100)))
                paths.append((base, candidate))
            args = type("Args", (), {
                "baseline": [pair[0] for pair in paths],
                "candidate": [pair[1] for pair in paths],
                "target_reference": None,
                "json_out": root / "pool.json",
                "md_out": None,
                "bootstrap_samples": 10,
                "bootstrap_seed": 1,
            })()
            with self.assertRaisesRegex(ValueError, "multiple folds"):
                bench.cmd_pool(args)

    def test_phase_transition_requires_capture_without_drops(self):
        before = {
            "records_written": 10,
            "records_dropped": 0,
            "adapter_generation": 2,
        }
        after = {
            "records_written": 20,
            "records_dropped": 0,
            "adapter_generation": 2,
        }
        client_bench._validate_oflash_bench_transition(
            "heldout-adapted", before, after
        )
        with self.assertRaisesRegex(SystemExit, "no OFlash capture records"):
            client_bench._validate_oflash_bench_transition(
                "heldout-adapted", before, {**after, "records_written": 10}
            )
        with self.assertRaisesRegex(SystemExit, "dropped capture records"):
            client_bench._validate_oflash_bench_transition(
                "heldout-adapted", before, {**after, "records_dropped": 1}
            )

    def test_hierarchical_bootstrap_is_deterministic(self):
        folds = [
            [{"baseline_accept_rate": 0.4, "candidate_accept_rate": 0.5}],
            [{"baseline_accept_rate": 0.6, "candidate_accept_rate": 0.8}],
        ]
        first = bench._hierarchical_bootstrap_ci(
            folds, lambda rows: bench._mean_delta(rows, "accept_rate"), 100, 7
        )
        second = bench._hierarchical_bootstrap_ci(
            folds, lambda rows: bench._mean_delta(rows, "accept_rate"), 100, 7
        )
        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        self.assertLessEqual(first[0], 0.15)
        self.assertGreaterEqual(first[1], 0.15)


if __name__ == "__main__":
    unittest.main()
