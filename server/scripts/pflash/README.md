# PFlash development tools

The three tools here cover request-probe capture, export and parity fixtures.
They preserve the existing probe schemas, capture configuration and seeded
fixtures. NumPy is required; GGUF commands use the vendored `gguf-py` package.

## Capture and validate

`request_probe_capture.py` combines the former capture and capture-validation
scripts. Its two modes retain the existing arguments, shard format, checks and
`CAPTURE.json` manifest:

```sh
python server/scripts/pflash/request_probe_capture.py capture \
  --daemon server/build/pflash_daemon --drafter /path/to/drafter.gguf \
  --scoring-head /path/to/head.gguf --segment-probe /path/to/segment.gguf \
  --request-probe /path/to/request.gguf --input conversations.jsonl --out capture

python server/scripts/pflash/request_probe_capture.py validate capture \
  --probe /path/to/request.gguf --recapture repeated-capture --trunk-convs 20
```

Capture runs one fresh drafter session per conversation and stores user/tool
feature rows and runtime logits. Validation checks finite values, NumPy trunk
parity, labelled-message coverage and optional byte-identical recapture. Its
exit status is nonzero when the checks fail. Capture also retains partitioning,
shard sizing and remote transfer options; see `capture --help`.

The old top-level `pflash_request_probe_capture.py` and
`pflash_request_probe_capture_validate.py` paths have been replaced by these
subcommands.

## Export the probe

```sh
python server/scripts/pflash/request_probe_gguf.py request.gguf \
  --drafter-sha256 HEX --tau THRESHOLD --weights probe.npz
```

`--random-seed` remains available for runtime plumbing tests. Those weights are
not a trained probe. The exporter retains the request-probe tensor shapes,
schema and extraction-version metadata.

## Regenerate parity fixtures

```sh
python server/scripts/pflash/gen_request_probe_v2_fixtures.py TRAINER_SRC_DIR \
  server/test/pflash/fixtures/request_probe_v2_parity.inc
```

The generator imports the trainer's offline rule, checks its candidate-rule
contract and emits the same 80 seed-0 cases. Keeping that offline implementation
as the reference makes the fixtures independent of the C++ rule under test.
