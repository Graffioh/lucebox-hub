# PFlash configuration

This reference separates built-in defaults from the measured release profile.
Fold, query-grid, stage-2 and oracle experiments remain available. With prefill
compression enabled, selection, query, admission and chat settings are checked
before model loading. Invalid settings stop startup and name the offending
variable. Model schemas and replay files receive their loader checks.

The [development tools](../scripts/pflash/README.md) document request-probe
capture, validation and export; [PFlash tests](../test/pflash/README.md) describe
the fixtures and replay executables.

Serving booleans accept `0` or `1`; unset or empty serving overrides retain their
defaults. Selection and query overrides reject empty values unless explicitly
allowed below. Valid integer overrides retain their existing upper clamps.

## Serving defaults

| Setting | Built-in default | Values / behavior |
|---|---|---|
| `--prefill-compression` | `off` | `off`, `auto`, `always` |
| `--prefill-threshold` | `32000` | Auto admission threshold, in droppable tokens |
| `--prefill-keep-ratio` | `0.05` | Flat ratio; `--prefill-curve tokens:ratio ...` overrides it |
| `PFLASH_ADMISSION` | `length` | `length` or `cache`; cache admission accounts for restorable target KV |
| `PFLASH_AGENT_PASTE` | `0` | With cache admission in auto mode, `1` permits a large newest user paste in a tool session; earlier history stays raw |
| `PFLASH_CHAT_VIEW`, `PFLASH_CHAT_RECALL` | `1`, `1` | Continue the served view; recall omitted material for a new question |
| `PFLASH_CHAT_COMPRESS_NEW_TOKENS` | `16384` | Integer >= 1; smaller additions stay verbatim; capped at 2^30 |
| `PFLASH_CHAT_SKELETON_TOKENS` | `256` | Integer >= 0; capped at 2^20 |
| `PFLASH_CHAT_HISTORY_QUERIES` | `3` | Integer >= 0; capped at 8 earlier user turns |
| `PFLASH_CHAT_REBUILD_RATIO` | `2` | Finite number >= 0; `0` rebuilds only at the context limit |
| `PFLASH_CHAT_REBUILD_MIN_TOKENS` | `0` | Integer >= 0; smallest view eligible for drift rebuild; capped at 2^30 |
| `PFLASH_ARRIVE` | `0` | Score only a large newly arriving region |
| `PFLASH_SELECT_PARAGRAPH_JOIN` | `1` | Insert paragraph breaks between nonadjacent kept spans |
| `PFLASH_FALLBACK_SPREAD` | off | Positive finite threshold; unset, empty, `0` or `off` disables it. Applies to whole-prompt selections, not region plans |
| `PFLASH_FAIL_CLOSED` | `0` | `0` serves raw after compression failure when it fits; `1` returns the error. Keep-history admission still serves raw |
| `PFLASH_DRAFTER_SESSIONS` | `2` | Integer >= 0; capped at 16; `0` scores from scratch. Stage 2 uses its own rows and ignores this setting |
| `PFLASH_DRAFTER_TRIM_POOL` | `1` | Release cached temporary allocations from a resident drafter's pool |
| `PFLASH_SESSION_OFFLOAD_PINNED` | `0` | Plain host memory; `1` opts into pinned host allocations |

## Selection and model assets

| Variable | Built-in default | Values / requirements |
|---|---|---|
| `PFLASH_SCORING_HEAD_GGUF` | unset | Trained head file; otherwise built-in attention-mass scorer |
| `PFLASH_SEGMENT_PROBE_GGUF` | unset | Segment probe file; otherwise fixed chunks |
| `PFLASH_REQUEST_PROBE_GGUF` | unset | Request probe file; required for query-probe selection outside the test stub |
| `PFLASH_SELECT_MODE` | unset: legacy | `budget_only`, `top_p`, `top_k`; setting a mode enables strict selection |
| `PFLASH_SELECT_CHUNK_SIZE` | scheduled in strict mode | Positive integer; 128 below 500 input tokens, 512 below 3000, otherwise 1024 |
| `PFLASH_SELECT_QUERY_TOKENS` | `8` | Integer in [1, 512] |
| `PFLASH_SELECT_QUERY_PARSER` | `arbitrary_tail` | `arbitrary_tail` or `latest_user`; default uses the latest user turn in chat |
| `PFLASH_SELECT_QUERY_HEAD` | `0` | `1` adds prose around pasted material; needs strict selection |
| `PFLASH_SELECT_QUERY_HEAD_TOKENS`, `PFLASH_SELECT_QUERY_TAIL_TOKENS` | `256` each | Integers in [2, 512] |
| `PFLASH_SELECT_QUERY_PROBE` | `0` | `1` replaces scoring windows with request-probe query tokens; needs strict head selection |
| `PFLASH_SELECT_SEGMENTS` | `auto` | `auto`, `fixed`, `probe`; auto uses probe segments when loaded |
| `PFLASH_SELECT_SCORE` | `auto` | `auto`, `sum`, `density`; auto uses density for probe segments, sum for fixed chunks |
| `PFLASH_SELECT_SCORER`, `PFLASH_SELECT_SPLIT` | `head`, `0.5` | `head`, `legacy`, `split`; split fraction finite and in (0, 1) |
| `PFLASH_SELECT_TOP_P`, `PFLASH_SELECT_TOPK` | `0.95`, unset | Top-p finite in (0, 1]; top-k positive integer, required in `top_k` mode |
| `PFLASH_SELECT_STRATEGY` | `segments` | `segments` or `fold`; fold needs `budget_only` and the head scorer |
| `PFLASH_SELECT_FOLD_K`, `PFLASH_SELECT_FOLD_CAP`, `PFLASH_SELECT_FOLD_HEAD` | `20`, `1500`, `150` | K and cap >= 1, head >= 0; with fold, head cannot exceed cap |
| `PFLASH_SELECT_FOLD_PARAGRAPHS` | `1` | Paragraph completion as fold's last fallback |

Query-probe, identifier and spread switches modify selection; they do not
enable strict selection on their own.

## Relative queries and retained experiments

| Variable | Built-in default | Values / requirements |
|---|---|---|
| `PFLASH_QUERY_CANDIDATES`, `PFLASH_QUERY_PIN` | `0`, `0` | Candidates in [0, 8], pin tokens in [0, 4096]; positive enables the feature |
| `PFLASH_QUERY_UNIT_SCORE`, `PFLASH_QUERY_AGG` | `max`, `softmax` | `max` or `mean`; `softmax` or `wmax` |
| `PFLASH_QUERY_TEMP`, `PFLASH_QUERY_NULL_MARGIN` | `1`, `0` | Finite temperature > 0; finite null margin >= 0 |
| `PFLASH_QUERY_EXTENT`, `PFLASH_QUERY_PIN_MARGIN` | `gap`, `gap` | `gap` or finite number >= 0; pin margin also accepts `inf` |
| `PFLASH_QUERY_TW`, `PFLASH_QUERY_END` | `0.5`, `0.25` | Explicit shares in (0, 1), sum < 1; TW also accepts `score` |
| `PFLASH_REQUEST_PROBE_POLICY`, `PFLASH_SELECT_QUERY_PROBE_RECENCY_PIN` | GGUF policy | JSON policy file; recency pin in [0, 4096]; checked by the loader |
| `PFLASH_REQUEST_STAGE2_GGUF` | unset | Experimental look-back combiner; needs compatible request probe and trained head |
| `PFLASH_EVAL_REF_IDENT_QUERY` | `off` | `off` or `heur`; heuristic evaluation arm needs strict head selection and excludes query-probe selection |
| `PFLASH_QUERY_GRID`, `PFLASH_QUERY_GRID_OUT` | unset | Replay input and append-only result file; grid needs query-probe selection and an output path |
| `PFLASH_QUERY_ORACLE_DIR` | unset | Oracle evaluation output during grid replay |
| `PFLASH_QWEN35_LEGACY_SCORER` | `0` | Force the original Qwen3.5 scorer |
| `PFLASH_SELECT_QUERY_PROBE_STUB` | `0` | Test-only request-probe stub |
| `PFLASH_COMPRESS_ANCHOR_RADIUS`, `PFLASH_COMPRESS_MAX_ANCHOR_HITS` | tier default | Legacy anchor controls |
| `PFLASH_FREEZE_HOT_WINDOW` | `2` | FlowKV's recent uncompressed message window |

Query tuning overrides require positive candidates or pinning. TW/END blend
overrides additionally require candidates. Candidates, pinning and grid replay
need the segment probe and cannot combine with request stage 2.

`PFLASH_TRACE_PATH` and `PFLASH_VIEW_TRACE_PATH` append selection and served-view
JSONL. `PFLASH_TRACE_REQUEST_FEATURES`, `PFLASH_TRACE_REQUEST_LOGITS` and
`PFLASH_TRACE_STAGE2_FEATURES` retain the feature/logit diagnostics.

## Measured release profile

The PFlash-v2 measurements used explicit overrides rather than the flat-ratio
and legacy-selection defaults. Preserve these when reproducing those results:

```sh
export PFLASH_SCORING_HEAD_GGUF=/path/to/head.gguf
export PFLASH_SEGMENT_PROBE_GGUF=/path/to/segment-probe.gguf
export PFLASH_REQUEST_PROBE_GGUF=/path/to/request-probe.gguf
export PFLASH_SELECT_MODE=budget_only
export PFLASH_SELECT_SEGMENTS=probe
export PFLASH_SELECT_SCORE=density
export PFLASH_SELECT_CHUNK_SIZE=256
export PFLASH_SELECT_QUERY_TOKENS=128
export PFLASH_SELECT_QUERY_HEAD=1
export PFLASH_SELECT_QUERY_PROBE=1
export PFLASH_SELECT_PARAGRAPH_JOIN=0
export PFLASH_QUERY_CANDIDATES=1
export PFLASH_QUERY_PIN=512
export PFLASH_ADMISSION=cache
export PFLASH_ARRIVE=1
export PFLASH_CHAT_RECALL=1
export PFLASH_CHAT_REBUILD_RATIO=2
export PFLASH_CHAT_REBUILD_MIN_TOKENS=65536
export PFLASH_FALLBACK_SPREAD=0.11

luce_server /path/to/target.gguf \
  --prefill-drafter /path/to/Qwen3.5-0.8B.gguf \
  --prefill-compression auto --prefill-threshold 16384 \
  --prefill-curve 16384:0.315 32768:0.270 65536:0.270 120000:0.225
```

Use the same trained assets and request-probe policy as the measurement being
reproduced. The 7.5% recall threshold belongs to that policy, rather than the
target keep-ratio curve. Agent-paste compression stays off by default; the
paste-specific experiments explicitly set `PFLASH_AGENT_PASTE=1`.

The obsolete `PFLASH_COLD_HISTORY` and `PFLASH_AGENT_HISTORY` names have been
removed: cache admission keeps tool history raw, with paste compression behind
`PFLASH_AGENT_PASTE`. The active experimental families above remain intact.
