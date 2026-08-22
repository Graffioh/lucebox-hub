#!/usr/bin/env bash
set -euo pipefail

# Run a Lucebox server under a delayed rocprofv3 collection window while also
# keeping the high-level concurrency capture in the same artifact directory.

PROFILED_SERVER_BIN="${PROFILED_SERVER_BIN:?set PROFILED_SERVER_BIN}"
ROCPROF_OUTPUT_DIR="${ROCPROF_OUTPUT_DIR:?set ROCPROF_OUTPUT_DIR}"
ROCPROF_START_SECONDS="${ROCPROF_START_SECONDS:-180}"
ROCPROF_DURATION_SECONDS="${ROCPROF_DURATION_SECONDS:-90}"

if [[ ! "$ROCPROF_START_SECONDS" =~ ^[0-9]+$ ]] ||
   [[ ! "$ROCPROF_DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROCPROF_START_SECONDS must be non-negative and " \
         "ROCPROF_DURATION_SECONDS must be positive integer seconds" >&2
    exit 2
fi

if [[ ! -x "$PROFILED_SERVER_BIN" ]]; then
    echo "PROFILED_SERVER_BIN is not executable: $PROFILED_SERVER_BIN" >&2
    exit 2
fi

if [[ -n "${ROCPROF_BIN:-}" ]]; then
    rocprof_bin="$ROCPROF_BIN"
elif rocprof_bin="$(command -v rocprofv3 2>/dev/null)" &&
     [[ -n "$rocprof_bin" ]]; then
    :
else
    rocprof_bin="${ROCM_PATH:-/opt/rocm}/bin/rocprofv3"
fi
if [[ ! -x "$rocprof_bin" ]]; then
    echo "rocprofv3 is not executable: $rocprof_bin" >&2
    exit 2
fi

mkdir -p "$ROCPROF_OUTPUT_DIR"
export DFLASH_PROF=concurrency
export DFLASH_PROF_OUT="${DFLASH_PROF_OUT:-$ROCPROF_OUTPUT_DIR/profile.jsonl}"
export DFLASH_QWEN35_ROCTX=1

exec "$rocprof_bin" \
    --marker-trace \
    --kernel-trace \
    --memory-copy-trace \
    --hip-runtime-trace \
    --group-by-queue true \
    --stats \
    --summary \
    --summary-output-file "$ROCPROF_OUTPUT_DIR/summary.txt" \
    --collection-period \
        "${ROCPROF_START_SECONDS}:${ROCPROF_DURATION_SECONDS}:1" \
    --output-format csv pftrace \
    --output-directory "$ROCPROF_OUTPUT_DIR" \
    --output-file trace \
    -- "$PROFILED_SERVER_BIN" "$@"
