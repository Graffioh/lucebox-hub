#!/usr/bin/env python3
"""Tests for literal feature flags and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).parent
SCRIPT = HERE / "write_feature_metadata.py"
SPEC = importlib.util.spec_from_file_location("write_feature_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)
RUNTIME_SCRIPT = HERE / "record_feature_runtime.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "record_feature_runtime", RUNTIME_SCRIPT,
)
assert RUNTIME_SPEC is not None and RUNTIME_SPEC.loader is not None
runtime_metadata = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime_metadata)


class FeatureMetadataTests(unittest.TestCase):
    def test_full_row_records_literal_screenshot_flags_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.gguf"
            draft = root / "draft.gguf"
            prefill = root / "prefill.gguf"
            prompts = root / "prompts.jsonl"
            command = root / "command.txt"
            out = root / "metadata.json"
            target.write_bytes(b"target")
            draft.write_bytes(b"draft")
            prefill.write_bytes(b"prefill")
            draft_sha = hashlib.sha256(draft.read_bytes()).hexdigest()
            prefill_sha = hashlib.sha256(prefill.read_bytes()).hexdigest()
            prompts.write_text('{"prompt":"p"}\n', encoding="utf-8")
            command.write_text("server --target-device hip:0\n", encoding="utf-8")
            argv = [
                str(SCRIPT), "--out", str(out), "--variant", "full",
                "--workload", "compression", "--clients", "4", "--repeat", "1",
                "--binary", "/bin/true", "--model", str(target),
                "--model-sha256", hashlib.sha256(target.read_bytes()).hexdigest(),
                "--prompt-file", str(prompts), "--command-file", str(command),
                "--repo", str(HERE.parents[2]), "--max-concurrent-prefills", "8",
                "--target-device", "hip:0", "--draft-device", "hip:0",
                "--draft-model", str(draft), "--draft-model-sha256", draft_sha,
                "--ddtree", "--ddtree-budget", "22", "--fast-rollback",
                "--prefill-compression", "auto", "--prefill-threshold", "32000",
                "--prefill-keep-ratio", "0.05", "--prefill-drafter", str(prefill),
                "--prefill-drafter-sha256", prefill_sha,
                "--draft-residency", "persistent", "--kvflash", "auto",
                "--kvflash-max-pool-tokens", "8192",
                "--kvflash-scorer-drafter", str(prefill),
                "--kvflash-scorer-drafter-sha256", prefill_sha,
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(metadata.main(), 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["model_sha256"], hashlib.sha256(b"target").hexdigest())
            self.assertTrue(result["server_binary_sha256"])
            self.assertTrue(result["git_head"])
            self.assertEqual(result["feature_config"]["draft_model_sha256"], draft_sha)
            self.assertEqual(result["feature_config"]["prefill_drafter_sha256"], prefill_sha)
            self.assertEqual(
                result["feature_config"]["kvflash_scorer_drafter"],
                str(prefill.resolve()),
            )
            self.assertEqual(
                result["feature_config"]["kvflash_scorer_drafter_sha256"],
                prefill_sha,
            )
            self.assertEqual(result["literal_screenshot_flags"], [
                "--target-device", "hip:0",
                "--draft-device", "hip:0",
                "--ddtree",
                "--ddtree-budget", "22",
                "--fast-rollback",
                "--draft-residency", "persistent",
                "--prefill-compression", "auto",
                "--prefill-drafter", str(prefill.resolve()),
                "--kvflash", "auto",
            ])
            self.assertIsNone(result["runtime_observed"])

    def test_model_digest_claim_must_match_referenced_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"model")
            expected = hashlib.sha256(b"model").hexdigest()
            cache: dict[Path, str] = {}
            self.assertEqual(
                metadata.validated_digest(model, expected, "model", cache),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                metadata.validated_digest(model, "stale", "model", cache)
            with self.assertRaisesRegex(ValueError, "is required"):
                metadata.validated_digest(model, None, "model", cache)
            with self.assertRaisesRegex(ValueError, "without a model file"):
                metadata.validated_digest(None, expected, "model", cache)


    def test_ldd_failure_is_fatal(self) -> None:
        failed = mock.Mock(returncode=1, stdout="", stderr="not a dynamic executable")
        with mock.patch.object(metadata.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "ldd failed"):
                metadata.resolved_libraries(Path("/tmp/server"))

    def test_unresolved_shared_library_is_fatal(self) -> None:
        unresolved = mock.Mock(
            returncode=0, stdout="libmissing.so => not found\n", stderr="",
        )
        with mock.patch.object(metadata.subprocess, "run", return_value=unresolved):
            with self.assertRaisesRegex(RuntimeError, "unresolved libraries"):
                metadata.resolved_libraries(Path("/tmp/server"))

    def test_git_revision_failure_and_empty_output_are_fatal(self) -> None:
        for result, message in (
            (mock.Mock(returncode=128, stdout="", stderr="not a repository"),
             "git rev-parse failed"),
            (mock.Mock(returncode=0, stdout="\n", stderr=""),
             "empty revision"),
        ):
            with self.subTest(message=message):
                with mock.patch.object(metadata.subprocess, "run", return_value=result):
                    with self.assertRaisesRegex(RuntimeError, message):
                        metadata.repository_head(Path("/tmp/repo"))

    def test_runtime_records_actual_kvflash_pool_from_startup(self) -> None:
        original = {
            "schema_version": 3,
            "feature_config": {
                "kvflash": "auto",
                "kvflash_max_pool_tokens": 16384,
            },
        }
        log = "\n".join((
            "[parallel-kvflash] physical resident pool 8192 tokens; "
            "logical per-slot cap 65536 across 16 slots "
            "(--kv-pool-tokens does not expand resident VRAM)",
            "[paged-attention] 512 physical blocks x 16 tokens "
            "(8192 pool tokens, per-sequence max_ctx 65536)",
        ))
        result = runtime_metadata.update_metadata(original, log)
        observed = result["runtime_observed"]
        self.assertTrue(observed["kvflash_active"])
        self.assertEqual(observed["physical_kv_pool_tokens"], 8192)
        self.assertEqual(observed["physical_kv_pool_blocks"], 512)
        self.assertEqual(observed["kv_block_size_tokens"], 16)
        self.assertEqual(observed["logical_per_slot_max_ctx"], 65536)
        self.assertEqual(observed["configured_slots"], 16)

    def test_runtime_rejects_enabled_kvflash_without_marker(self) -> None:
        original = {"feature_config": {"kvflash": "auto"}}
        with self.assertRaisesRegex(ValueError, "startup marker is missing"):
            runtime_metadata.update_metadata(
                original,
                "[paged-attention] 512 physical blocks x 16 tokens "
                "(8192 pool tokens, per-sequence max_ctx 65536)",
            )

    def test_runtime_rejects_kvflash_marker_without_paged_marker(self) -> None:
        original = {"feature_config": {"kvflash": "auto"}}
        log = (
            "[parallel-kvflash] physical resident pool 8192 tokens; "
            "logical per-slot cap 65536 across 16 slots "
            "(--kv-pool-tokens does not expand resident VRAM)"
        )
        with self.assertRaisesRegex(ValueError, "paged physical-pool"):
            runtime_metadata.update_metadata(original, log)


if __name__ == "__main__":
    unittest.main()
