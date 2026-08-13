"""CLI defaults that bound the first online-training run."""

import pytest

from oflash.cli import parse_args
from oflash.identity import (fnv1a64, require_sha256,
                             resolved_drafter_semantics, sha256_file)


def test_conservative_training_defaults():
    args = parse_args([
        "draft.gguf",
        "--ring-name=/lucebox-test",
        "--out-dir=/tmp/oflash-test",
    ])
    assert args.dtype == "auto"
    assert args.batch_rows == 128
    assert args.train_ctx == 128
    assert args.reservoir_rows == 10_000
    assert args.seed == 0
    assert args.resolved_rope_theta is None
    assert args.resolved_swa_window is None
    assert args.resolved_swa_pattern is None
    assert args.resolved_mask_token_id is None
    assert args.drafter_semantics == ""
    assert args.drafter_sha256 is None
    assert args.target_sha256 is None


def test_resolved_engine_metadata_parses_strictly():
    args = parse_args([
        "draft.gguf",
        "--ring-name=/lucebox-test",
        "--out-dir=/tmp/oflash-test",
        "--resolved-rope-theta=1000000",
        "--resolved-swa-window=2048",
        "--resolved-swa-pattern=1,1,1,1,0",
        "--resolved-mask-token-id=248070",
    ])
    assert args.resolved_rope_theta == 1_000_000.0
    assert args.resolved_swa_window == 2048
    assert args.resolved_swa_pattern == (True, True, True, True, False)
    assert args.resolved_mask_token_id == 248070
    assert args.drafter_semantics == (
        "v1;rope=49742400;swa=2048;pattern=11110;mask=248070")


def test_cross_language_semantics_tag_golden():
    semantics = resolved_drafter_semantics(
        1_000_000.0, 2048, (True, False), 248070)
    assert semantics == "v1;rope=49742400;swa=2048;pattern=10;mask=248070"
    assert fnv1a64(semantics) == 0x1EC7F022739C0547


@pytest.mark.parametrize("flag", [
    "--resolved-rope-theta=nan",
    "--resolved-rope-theta=0",
    "--resolved-swa-window=-1",
    "--resolved-mask-token-id=-1",
    "--resolved-swa-pattern=1,true,0",
])
def test_invalid_resolved_engine_metadata_is_rejected(flag):
    with pytest.raises(SystemExit):
        parse_args([
            "draft.gguf",
            "--ring-name=/lucebox-test",
            "--out-dir=/tmp/oflash-test",
            flag,
        ])


@pytest.mark.parametrize("flag", [
    "--drafter-sha256=" + "a" * 16,
    "--target-sha256=" + "A" * 64,
    "--target-sha256=not-a-hash",
])
def test_partial_or_noncanonical_model_hashes_are_rejected(flag):
    with pytest.raises(SystemExit):
        parse_args([
            "draft.gguf",
            "--ring-name=/lucebox-test",
            "--out-dir=/tmp/oflash-test",
            flag,
        ])


@pytest.mark.parametrize("flag", [
    "--resolved-swa-window=2048",
    "--resolved-swa-pattern=1,1,1,1,0",
])
def test_partial_swa_override_is_rejected(flag):
    with pytest.raises(SystemExit):
        parse_args([
            "draft.gguf",
            "--ring-name=/lucebox-test",
            "--out-dir=/tmp/oflash-test",
            flag,
        ])


def test_direct_mode_hash_helper_is_exact(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"oflash identity\n")
    digest = sha256_file(str(model))
    assert digest == "7d4bc553f5c8bee036582edd46c3761b419d56087dcaf72035dfd3cd74a500c4"
    assert require_sha256(digest) == digest
