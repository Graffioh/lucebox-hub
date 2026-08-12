"""CLI defaults that bound the first online-training run."""

from oflash.cli import parse_args


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
