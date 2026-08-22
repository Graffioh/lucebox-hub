#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import profile_payload
import profile_view

HERE = Path(__file__).parent


class ProfileViewTest(unittest.TestCase):
    def records(self) -> list[dict]:
        return [
            {
                "type": "metadata",
                "schema": "lucebox.concurrency.v1",
                "git_sha": "current123",
                "model_name": "Qwen3.6-27B-Q4_K_M.gguf",
                "model_path": "/models/Qwen3.6-27B-Q4_K_M.gguf",
                "draft_path": "/models/unlisted-draft.gguf",
                "arch": "qwen35",
                "runtime_backend": "hip",
                "max_concurrency": 2,
                "started_unix_ns": 1_700_000_000_000_000_000,
                "retention_policy": "keep_first",
                "label": "safe </script><script>alert(1)</script>",
            },
            {
                "type": "step",
                "round_id": 1,
                "started_ns": 1_000,
                "duration_ns": 100,
                "path": "packed",
                "live_slots": 1,
                "target_rows": 10,
                "target_padding_rows": 2,
                "max_kv_len": 20,
                "lanes": [
                    {"kind": "decode", "scheduler_consumed_tokens": 2}
                ],
                "phases": [
                    {
                        "phase": "target_compute",
                        "start_offset_ns": 10,
                        "duration_ns": 50,
                    },
                    {
                        "phase": "readback_sync",
                        "start_offset_ns": 40,
                        "duration_ns": 40,
                    },
                ],
                "proposed_by_position": [],
                "accepted_by_position": [],
            },
            {
                "type": "step",
                "round_id": 2,
                "started_ns": 1_200,
                "duration_ns": 200,
                "path": "speculative",
                "live_slots": 2,
                "target_rows": 20,
                "target_padding_rows": 0,
                "draft_rows": 8,
                "draft_padding_rows": 2,
                "max_kv_len": 30,
                "spec_eligible_lanes": 2,
                "spec_reserved_lanes": 2,
                "spec_attempted_lanes": 2,
                "spec_proposed_draft_tokens": 4,
                "spec_verified_draft_tokens": 4,
                "spec_accepted_draft_tokens": 3,
                "spec_durable_draft_tokens": 3,
                "spec_scheduler_consumed_tokens": 3,
                "lanes": [
                    {"kind": "decode", "scheduler_consumed_tokens": 4}
                ],
                "phases": [
                    {
                        "phase": "target_compute",
                        "start_offset_ns": 0,
                        "duration_ns": 100,
                    },
                    {
                        "phase": "draft_compute",
                        "start_offset_ns": 100,
                        "duration_ns": 50,
                    },
                ],
                "proposed_by_position": [2, 2],
                "accepted_by_position": [2, 1],
            },
            {
                "type": "request",
                "request_id": 1,
                "ok": True,
                "queued_ns": 900,
                "admitted_ns": 920,
                "prefill_completed_ns": 950,
                "first_token_ns": 1_000,
                "completed_ns": 1_350,
                "prompt_tokens": 12,
                "output_tokens": 6,
            },
            {
                "type": "request",
                "request_id": 2,
                "ok": None,
                "queued_ns": 1_100,
                "admitted_ns": 1_150,
                "prefill_completed_ns": 0,
                "first_token_ns": 0,
                "completed_ns": 0,
            },
            {
                "type": "footer",
                "complete": True,
                "dropped_steps": 0,
                "dropped_requests": 0,
                "dropped_token_bursts": 0,
            },
        ]

    def specs(self, device: str = "bandwidth") -> dict:
        return {
            "selected_device": device,
            "devices": {
                "bandwidth": {
                    "name": "Bandwidth fixture",
                    "mem_bw_gbps": 1000,
                    "fp16_tflops": 100,
                },
                "compute": {
                    "name": "Compute fixture",
                    "mem_bw_gbps": 1000,
                    "fp16_tflops": 0.01,
                },
            },
            "models": {
                "Qwen3.6-27B-Q4_K_M.gguf": {
                    "weight_bytes": 100,
                    "active_params": 100,
                    "kv_bytes_per_token_per_seq": 0,
                }
            },
        }

    def test_checked_in_device_and_model_facts(self) -> None:
        specs = profile_view.load_device_specs()

        self.assertEqual(specs["devices"]["gfx1201"]["mem_bw_gbps"], 640)
        self.assertEqual(specs["devices"]["gfx1201"]["fp16_tflops"], 95.7)
        self.assertEqual(specs["devices"]["gfx1151"]["mem_bw_gbps"], 256)
        self.assertEqual(specs["devices"]["gfx1151"]["fp16_tflops"], 60)
        model = specs["models"]["Qwen3.6-27B-Q4_K_M.gguf"]
        self.assertEqual(
            (
                model["weight_bytes"],
                model["active_params"],
                model["kv_bytes_per_token_per_seq"],
            ),
            (16_800_000_000, 27_000_000_000, 18_432),
        )

    def embedded_payload(self, html: str) -> dict:
        match = re.search(
            r'<script type="application/json" id="data">(.*?)</script>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

    def test_embedded_aggregates_round_trip(self) -> None:
        html = profile_view.build_html(self.records(), device_specs=self.specs())
        payload = self.embedded_payload(html)

        self.assertEqual(html.count('type="application/json" id="data"'), 1)
        self.assertIn("<title>LuceGraph</title>", html)
        self.assertIn("<h1>LuceGraph</h1>", html)
        self.assertNotIn("</script><script>alert", html)
        self.assertTrue(payload["current"]["mixed_run_cohorts"])
        self.assertEqual(
            payload["current"]["capture_bounds"],
            {
                "earliest_steady_ns": 900,
                "latest_steady_ns": 1400,
                "duration_ns": 500,
            },
        )
        self.assertIn("configured C=${current.run.max_concurrency", html)
        self.assertIn("arch ${current.run.arch", html)
        self.assertIn("started ${isoStart(current)}", html)
        self.assertIn("steady bounds", html)
        self.assertEqual(payload["current"]["cohorts"], [1, 2])
        c1 = payload["current"]["phase_groups"]["all"][0]
        phases = {phase["phase"]: phase for phase in c1["phases"]}
        self.assertEqual(c1["rounds"], 1)
        self.assertEqual(c1["durable_tokens"], 2)
        self.assertEqual(phases["target_compute"]["total_ns"], 30)
        self.assertEqual(phases["target_compute"]["ns_per_token"], 15)
        self.assertEqual(phases["target_compute"]["ns_per_round"], 30)
        self.assertEqual(phases["target_compute"]["wall_share"], 0.3)
        self.assertEqual(phases["target_compute"]["p50_ns"], 30)
        self.assertEqual(phases["overlap(readback_sync+target_compute)"]["total_ns"], 20)
        self.assertEqual(phases["unattributed"]["total_ns"], 30)
        self.assertEqual(
            payload["current"]["phase_groups"]["inter_round_idle"]["total_ns"],
            100,
        )
        self.assertTrue(payload["current"]["requests"]["rows"][1]["open_ended"])

    def test_capture_bounds_handle_missing_and_token_burst_timestamps(self) -> None:
        records = [
            {"type": "metadata", "schema": "lucebox.concurrency.v1"},
            {"type": "footer", "complete": True},
        ]
        missing = profile_payload.build_run_payload(records)
        self.assertEqual(
            missing["capture_bounds"],
            {
                "earliest_steady_ns": None,
                "latest_steady_ns": None,
                "duration_ns": None,
            },
        )

        records.insert(1, {
            "type": "token_burst",
            "request_id": 1,
            "round_id": 1,
            "ready_ns": 500,
            "token_count": 1,
        })
        burst = profile_payload.build_run_payload(records)
        self.assertEqual(
            burst["capture_bounds"],
            {
                "earliest_steady_ns": 500,
                "latest_steady_ns": 500,
                "duration_ns": 0,
            },
        )

    def test_roofline_classifies_both_sides_and_unknown_device(self) -> None:
        bandwidth = profile_payload.build_run_payload(
            self.records(), device_specs=self.specs(), device_key="bandwidth"
        )
        compute = profile_payload.build_run_payload(
            self.records(), device_specs=self.specs(), device_key="compute"
        )
        unknown = profile_payload.build_run_payload(
            self.records(), device_specs=self.specs(), device_key="missing"
        )

        def target_class(run: dict) -> dict:
            phases = run["phase_groups"]["all"][0]["phases"]
            return next(
                phase["classification"]
                for phase in phases
                if phase["phase"] == "target_compute"
            )

        self.assertEqual(target_class(bandwidth)["class"], "bandwidth")
        self.assertEqual(target_class(compute)["class"], "compute")
        self.assertEqual(target_class(unknown)["class"], "neutral")
        self.assertIn("No device spec", unknown["notices"][0]["message"])
        self.assertEqual(target_class(bandwidth)["arithmetic_intensity_flops_per_byte"], 20)
        self.assertEqual(target_class(bandwidth)["machine_balance_flops_per_byte"], 100)

    def test_diff_reports_mismatch_and_numeric_delta(self) -> None:
        current = self.records()
        baseline = copy.deepcopy(current)
        baseline[0]["model_name"] = "different-model"
        baseline[1]["phases"][0]["duration_ns"] = 20
        specs = self.specs()
        specs["baseline_device"] = "compute"
        html = profile_view.build_html(
            current, baseline, device_specs=specs
        )
        payload = self.embedded_payload(html)

        self.assertTrue(any("model mismatch" in warning for warning in payload["diff"]["warnings"]))
        self.assertTrue(
            any("device mismatch" in warning for warning in payload["diff"]["warnings"])
        )
        self.assertIn("mismatch warning", html)
        self.assertIn(
            "Baseline ${runLabel(baseline)} · Current ${runLabel(current)}", html
        )
        target = next(
            row for row in payload["diff"]["rows"]
            if row["path"] == "all"
            and row["cohort"] == 1
            and row["phase"] == "target_compute"
        )
        self.assertEqual(target["baseline_ns_per_token"], 10)
        self.assertEqual(target["current_ns_per_token"], 15)
        self.assertEqual(target["delta_ns_per_token"], 5)
        self.assertEqual(target["delta_percent"], 0.5)

    @unittest.skipUnless(shutil.which("node"), "node is required for JavaScript syntax checks")
    def test_generated_javascript_is_valid(self) -> None:
        html = profile_view.build_html(self.records(), device_specs=self.specs())
        scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
        javascript = [script for script in scripts if not script.lstrip().startswith("{")]
        self.assertEqual(len(javascript), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lucegraph.js"
            path.write_text(javascript[0], encoding="utf-8")
            result = subprocess.run(
                ["node", "--check", str(path)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_writes_one_offline_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture.jsonl"
            report = root / "report.html"
            capture.write_text(
                "".join(json.dumps(record) + "\n" for record in self.records()),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "profile_report.py"),
                    str(capture),
                    "--html",
                    str(report),
                    "--device",
                    "gfx1201",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            html = report.read_text(encoding="utf-8") if report.exists() else ""

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIsNone(re.search(r"https?://", html))
        self.assertEqual(self.embedded_payload(html)["current"]["device"]["key"], "gfx1201")

    def test_c1_only_without_speculation_stays_renderable(self) -> None:
        records = [
            record for record in self.records()
            if record.get("type") not in {"step", "request"}
            or record.get("round_id") == 1
            or record.get("request_id") == 1
        ]
        html = profile_view.build_html(records)
        payload = self.embedded_payload(html)

        self.assertEqual(payload["current"]["cohorts"], [1])
        self.assertFalse(payload["current"]["mixed_run_cohorts"])
        self.assertEqual(payload["current"]["speculation"]["spec_attempted_lanes"], 0)


if __name__ == "__main__":
    unittest.main()
