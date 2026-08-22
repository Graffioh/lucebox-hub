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
                "lanes": [{"kind": "decode", "spec": "selected",
                           "scheduler_consumed_tokens": 2}],
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
        markdown = profile_report.build_markdown(
            profile_report.build_summary(records))
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

        markdown = profile_report.build_markdown(
            profile_report.build_summary(records))

        self.assertIn("1/1 captured requests failed", markdown)
        self.assertIn("Accepted and durable draft token counts differ", markdown)

    def test_incomplete_request_is_not_failed(self):
        records = self.records()
        records.insert(-1, {
            "type": "request", "request_id": 10, "ok": None,
            "queued_ns": 1000, "admitted_ns": 1100,
            "completed_ns": 0,
        })

        summary = profile_report.build_summary(records)

        self.assertEqual(summary["capture"]["requests"], 2)
        self.assertEqual(summary["capture"]["failed_requests"], 0)

    def test_folded_views_merge_frames_and_normalize_by_group(self):
        records = self.records()
        records.insert(2, {
            "type": "step", "round_id": 2, "started_ns": 4_000_000,
            "duration_ns": 1_000, "path": "speculative",
            "live_slots": 4,
            "lanes": [{"kind": "decode", "scheduler_consumed_tokens": 3}],
            "phases": [{"phase": "target_compute",
                        "start_offset_ns": 0, "duration_ns": 1_000}],
        })

        wall = profile_report.build_folded(records)
        per_token = profile_report.build_folded(records, per_token=True)

        self.assertIn("speculative;C=4;target_compute 2000\n", wall)
        self.assertIn("speculative;C=4;unattributed 1999000\n", wall)
        self.assertIn("idle;inter_round 1000000\n", wall)
        self.assertEqual(
            per_token,
            "speculative;C=4;target_compute 400\n"
            "speculative;C=4;unattributed 399800\n",
        )

    def test_folded_coverage_partitions_overlap(self):
        step = {
            "type": "step", "duration_ns": 100, "path": "packed",
            "live_slots": 2,
            "phases": [
                {"phase": "a", "start_offset_ns": 10, "duration_ns": 50},
                {"phase": "b", "start_offset_ns": 40, "duration_ns": 40},
            ],
        }

        buckets = profile_report.step_phase_buckets(step)

        self.assertEqual(buckets, {
            "unattributed": 30,
            "a": 30,
            "overlap(a+b)": 20,
            "b": 20,
        })
        self.assertEqual(sum(buckets.values()), step["duration_ns"])

        step["phases"] = [
            {"phase": "a", "start_offset_ns": -10, "duration_ns": 60},
            {"phase": "a", "start_offset_ns": 40, "duration_ns": 80},
        ]
        self.assertEqual(
            profile_report.step_phase_buckets(step), {"a": 100}
        )

    def test_folded_stack_order_is_validated(self):
        stack = profile_report.parse_folded_stack("cohort,path,phase")
        folded = profile_report.build_folded(self.records(), stack=stack)
        self.assertIn("C=4;speculative;target_compute 1000\n", folded)

        with self.assertRaisesRegex(ValueError, "permutation"):
            profile_report.build_folded(
                self.records(), stack=("path", "path", "phase")
            )
        with self.assertRaisesRegex(Exception, "permutation"):
            profile_report.parse_folded_stack("path,cohort,kind")

    def test_folded_per_token_omits_groups_without_decode_tokens(self):
        records = self.records()
        records[1]["lanes"] = [
            {"kind": "prefill", "scheduler_consumed_tokens": 1}
        ]

        self.assertEqual(
            profile_report.build_folded(records, per_token=True), ""
        )


if __name__ == "__main__":
    unittest.main()
