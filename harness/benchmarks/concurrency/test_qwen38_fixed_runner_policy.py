#!/usr/bin/env python3
"""Policy checks for the PR #626 fixed-work acceptance runner."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("run_qwen38_dflash2_fixed.sh")
MEASUREMENT_RUNNER = Path(__file__).with_name("run_qwen38_dflash2_measurement.sh")


def runner_env(**updates: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GGML_", "DFLASH_", "LUCE_", "HIP_", "ROCR_", "HSA_"))
        and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH")
    }
    env.update(
        MODEL="/dev/null",
        DRAFT_MODEL="/dev/null",
        SERVER_BIN="/bin/true",
        **updates,
    )
    return env


class Qwen38FixedRunnerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = RUNNER.read_text(encoding="utf-8")

    def test_server_requires_two_slots_but_allows_one_client(self) -> None:
        rejected = subprocess.run(
            [str(RUNNER)],
            env=runner_env(CLIENTS="1", SLOTS="1"),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("SLOTS must be at least 2", rejected.stderr)

        accepted = subprocess.run(
            [str(RUNNER)],
            env=runner_env(CLIENTS="1", SLOTS="2"),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotIn("SLOTS must be", accepted.stderr)
        self.assertIn("HIP code object", accepted.stderr)

    def test_smoke_precedes_full_concurrency_warmup(self) -> None:
        self.assertIn('SMOKE_TOKENS="${SMOKE_TOKENS:-8}"', self.script)
        smoke = self.script.index('--clients 1 --prompt-file "$PROMPT_FILE"')
        warmup = self.script.index('--clients "$CLIENTS" --prompt-file "$PROMPT_FILE"')
        self.assertLess(smoke, warmup)
        self.assertIn('--out "$case_dir/smoke.json"', self.script)

    def test_binary_code_object_must_match_the_active_gpu(self) -> None:
        self.assertIn('readelf -p .hip_fatbin "$SERVER_BIN"', self.script)
        self.assertIn('server binary has no HIP code object', self.script)
        self.assertIn('server binary HIP code objects do not include active GPU', self.script)
        self.assertIn('> "$case_dir/gpu-identity.txt"', self.script)

    def test_ar_does_not_receive_speculation_environment(self) -> None:
        conditional = self.script.index('if [[ "$variant" != "ar" ]]')
        batched = self.script.index('DFLASH_SPEC_BATCHED_DRAFT=1', conditional)
        depth = self.script.index('DFLASH_SPEC_CHAIN_DEPTH=$SPEC_DEPTH', conditional)
        command = self.script.index('printf \'env \'', depth)
        self.assertLess(conditional, batched)
        self.assertLess(batched, command)
        self.assertLess(depth, command)

    def test_measurement_suite_uses_three_rotated_replicates(self) -> None:
        script = MEASUREMENT_RUNNER.read_text(encoding="utf-8")
        self.assertIn('REPLICATES="${REPLICATES:-3}"', script)
        self.assertIn('"ar,speculation,adaptive"', script)
        self.assertIn('"adaptive,ar,speculation"', script)
        self.assertIn('"speculation,adaptive,ar"', script)
        self.assertIn('comparison.json', script)


if __name__ == "__main__":
    unittest.main()
