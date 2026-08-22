#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import profile_report


class ProfileReportTest(unittest.TestCase):
    def records(self):
        return [
            {"type": "metadata", "schema": "lucebox.concurrency.v1",
             "git_sha": "abc123", "max_concurrency": 4},
            {
                "type": "step", "round_id": 1, "started_ns": 1_000_000,
                "duration_ns": 2_000_000, "path": "speculative",
                "live_slots": 4, "target_rows": 20,
                "spec_tree_width": 8,
                "target_padding_rows": 4, "draft_rows": 16,
                "draft_padding_rows": 0, "spec_eligible_lanes": 4,
                "spec_reserved_lanes": 4, "spec_attempted_lanes": 4,
                "spec_proposed_draft_tokens": 12,
                "spec_verified_draft_tokens": 12,
                "spec_accepted_draft_tokens": 8,
                "spec_durable_draft_tokens": 8,
                "spec_scheduler_consumed_tokens": 7,
                "lanes": [{"kind": "decode", "spec": "selected"}],
                "phases": [{"phase": "target_compute",
                            "start_offset_ns": 100, "duration_ns": 1000}],
            },
            {
                "type": "request", "request_id": 9, "ok": True,
                "queued_ns": 100, "admitted_ns": 200,
                "prefill_completed_ns": 500, "first_token_ns": 500,
                "completed_ns": 1000,
            },
            {"type": "token_burst", "request_id": 9, "round_id": 1,
             "ready_ns": 500, "token_count": 1},
            {"type": "token_burst", "request_id": 9, "round_id": 2,
             "ready_ns": 900, "token_count": 2},
            {"type": "footer", "complete": True, "dropped_steps": 0,
             "dropped_requests": 0, "dropped_token_bursts": 0},
        ]

    def test_markdown_and_perfetto_share_the_records(self):
        records = self.records()
        markdown = profile_report.build_markdown(records)
        self.assertIn("## Speculation funnel", markdown)
        self.assertIn("| 4 | 1 | 2.00 ms | 20.0% | 66.7%", markdown)
        trace = profile_report.build_perfetto(records)
        names = [event["name"] for event in trace["traceEvents"]]
        self.assertIn("target_compute", names)
        self.assertIn("queue", names)
        self.assertIn("tokens_ready", names)

        summary = profile_report.build_summary(records)
        self.assertEqual(summary["schema"],
                         "lucebox.concurrency.summary.v1")
        self.assertEqual(summary["run"]["git_sha"], "abc123")
        self.assertTrue(summary["capture"]["complete"])
        self.assertEqual(summary["cohorts"]["4"]["rounds"], 1)
        self.assertEqual(
            summary["speculation"]["spec_accepted_draft_tokens"], 8)
        self.assertEqual(summary["speculation"]["tree_widths"], [8])
        self.assertAlmostEqual(
            summary["padding"]["target_padding_ratio"], 0.2)

    def test_loader_rejects_a_different_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text(json.dumps({"type": "metadata", "schema": "bad"}) + "\n")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                profile_report.load_records(path)

    def test_failure_and_durability_gap_are_actionable(self):
        records = self.records()
        records[1]["spec_durable_draft_tokens"] = 0
        records[2]["ok"] = False

        markdown = profile_report.build_markdown(records)

        self.assertIn("1/1 captured requests failed", markdown)
        self.assertIn("Accepted and durable draft token counts differ", markdown)


if __name__ == "__main__":
    unittest.main()
