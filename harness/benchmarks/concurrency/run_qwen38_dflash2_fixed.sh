#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}"
CLIENT="${CLIENT:-$SCRIPT_DIR/concurrent_benchmark.py}"
METADATA_TOOL="${METADATA_TOOL:-$SCRIPT_DIR/write_feature_metadata.py}"
RUNTIME_METADATA_TOOL="${RUNTIME_METADATA_TOOL:-$SCRIPT_DIR/record_feature_runtime.py}"
MODEL="${MODEL:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
SERVER_BIN="${SERVER_BIN:-$REPO/server/build-hip/dflash_server}"
PROMPT_FILE="${PROMPT_FILE:-$REPO/harness/benchmarks/prompts/qwen38_dflash2_fixed_c29.jsonl}"
OUT="${OUT:-$REPO/.harness-runs/qwen38-dflash2-fixed-$(date -u +%Y%m%dT%H%M%SZ)}"
VARIANTS="${VARIANTS:-ar,speculation,adaptive}"
CLIENTS="${CLIENTS:-29}"
SLOTS="${SLOTS:-16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
SMOKE_TOKENS="${SMOKE_TOKENS:-8}"
SPEC_DEPTH="${SPEC_DEPTH:-8}"
MAX_CTX="${MAX_CTX:-4096}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
FA_WINDOW="${FA_WINDOW:-0}"
TARGET_DEVICE="${TARGET_DEVICE:-hip:0}"
DRAFT_DEVICE="${DRAFT_DEVICE:-hip:0}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
PORT="${PORT:-18141}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-3}"

usage() {
  cat <<'EOF'
Usage:
  MODEL=/path/Qwen3.8-27B-target.gguf \
  DRAFT_MODEL=/path/Qwen3.8-27B-DFlash2-q8_0.gguf \
  harness/benchmarks/concurrency/run_qwen38_dflash2_fixed.sh

Fixed-work paired protocol: all CLIENTS requests launch simultaneously
(barrier-start), the server admits at most SLOTS of them concurrently, and
every request decodes MAX_TOKENS output tokens with ignore-eos. Each variant
(ar, speculation, adaptive) runs against its own fresh server process over the
same committed prompt fixture, so variants are directly comparable and every
case directory carries server-command.txt, server-metadata.json,
runtime-metadata.json, gpu-identity.txt, smoke.json, warmup.json and bench.json.

Adaptive vs AR on this artifact is the primary PR #626 performance observation.
EOF
}

if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi
for cmd in python3 curl sha256sum ldd readelf; do command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }; done
[[ -r "$MODEL" ]] || { echo "set MODEL to a readable target GGUF" >&2; exit 2; }
[[ -r "$DRAFT_MODEL" ]] || { echo "set DRAFT_MODEL to a readable DFlash2 drafter GGUF" >&2; exit 2; }
[[ -x "$SERVER_BIN" ]] || { echo "missing server: $SERVER_BIN" >&2; exit 2; }
[[ -r "$PROMPT_FILE" ]] || { echo "missing prompt fixture: $PROMPT_FILE" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }

IFS=, read -r -a variant_list <<< "$VARIANTS"
for variant in "${variant_list[@]}"; do
  [[ "$variant" =~ ^(ar|speculation|adaptive)$ ]] || { echo "unknown variant $variant" >&2; exit 2; }
done
[[ "$CLIENTS" =~ ^[0-9]+$ ]] && (( CLIENTS >= 1 )) || { echo "CLIENTS must be positive" >&2; exit 2; }
[[ "$SLOTS" =~ ^[0-9]+$ ]] && (( SLOTS >= 2 )) || { echo "SLOTS must be at least 2" >&2; exit 2; }
for value_name in MAX_TOKENS WARMUP_TOKENS SMOKE_TOKENS SPEC_DEPTH MAX_CTX; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] && (( value >= 1 )) || {
    echo "$value_name must be positive" >&2
    exit 2
  }
done
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || { echo "PORT must be in 1..65535" >&2; exit 2; }
[[ "$TARGET_DEVICE" =~ ^hip:[0-9]+$ ]] || { echo "TARGET_DEVICE must match hip:N" >&2; exit 2; }
[[ "$DRAFT_DEVICE" =~ ^hip:[0-9]+$ ]] || { echo "DRAFT_DEVICE must match hip:N" >&2; exit 2; }

binary_gpu_arches="$(readelf -p .hip_fatbin "$SERVER_BIN" 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u || true)"
[[ -n "$binary_gpu_arches" ]] || { echo "server binary has no HIP code object" >&2; exit 2; }

ambient_tuning="$(env | grep -E '^(GGML_|DFLASH_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD=|LD_LIBRARY_PATH=)' || true)"
if [[ -n "$ambient_tuning" ]]; then
  echo "refusing ambient GPU/backend tuning variables:" >&2
  echo "$ambient_tuning" >&2
  exit 2
fi

capacity=$((SLOTS * MAX_CTX))
model_id="$(basename "$MODEL" .gguf)"

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT
trap 'exit 130' INT TERM

wait_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

run_variant() {
  local variant="$1"
  local case_dir="$OUT/$variant"
  mkdir -p "$case_dir"

  local -a command launch_env launch_env_args
  command=("$SERVER_BIN" "$MODEL" --draft "$DRAFT_MODEL"
    --target-device "$TARGET_DEVICE" --draft-device "$DRAFT_DEVICE"
    --paged-attention --max-concurrency "$SLOTS"
    --kv-pool-tokens "$capacity" --max-ctx "$MAX_CTX"
    --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V"
    --fa-window "$FA_WINDOW" --prefix-cache-slots 0 --prefill-cache-slots 0
    --admission-coalesce-ms 20 --draft-residency persistent
    --decode-mode "$variant" --host 127.0.0.1 --port "$PORT"
    --model-name "$model_id")
  launch_env=(
    "HIP_VISIBLE_DEVICES=$VISIBLE_DEVICES"
    "DFLASH_MAX_CONCURRENT_PREFILLS=$SLOTS"
  )
  if [[ "$variant" != "ar" ]]; then
    launch_env+=(
      "DFLASH_SPEC_BATCHED_DRAFT=1"
      "DFLASH_SPEC_CHAIN_DEPTH=$SPEC_DEPTH"
    )
  fi
  local assignment
  for assignment in "${launch_env[@]}"; do
    launch_env_args+=(--launch-env "$assignment")
  done
  printf 'env ' > "$case_dir/server-command.txt"
  printf '%q ' "${launch_env[@]}" "${command[@]}" >> "$case_dir/server-command.txt"
  printf '\n' >> "$case_dir/server-command.txt"

  local -a decode_args=()
  [[ "$variant" != "ar" ]] && decode_args+=(--confidence on)
  python3 "$METADATA_TOOL" \
    --out "$case_dir/server-metadata.json" \
    --variant "$variant" --workload fixed-backlog \
    --clients "$CLIENTS" --repeat 1 \
    --binary "$SERVER_BIN" --model "$MODEL" \
    --model-sha256 "$(sha256sum "$MODEL" | cut -d' ' -f1)" \
    --draft-model "$DRAFT_MODEL" \
    --draft-model-sha256 "$(sha256sum "$DRAFT_MODEL" | cut -d' ' -f1)" \
    --prompt-file "$PROMPT_FILE" \
    --command-file "$case_dir/server-command.txt" \
    --repo "$REPO" \
    --max-concurrent-prefills "$SLOTS" \
    --target-device "$TARGET_DEVICE" --draft-device "$DRAFT_DEVICE" \
    --decode-mode "$variant" \
    --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V" \
    --fa-window "$FA_WINDOW" \
    --draft-residency persistent \
    ${decode_args+"${decode_args[@]}"} \
    "${launch_env_args[@]}"

  echo "[fixed] $variant C=$CLIENTS slots=$SLOTS max_tokens=$MAX_TOKENS"
  env "${launch_env[@]}" stdbuf -oL -eL "${command[@]}" \
    > "$case_dir/server.log" 2>&1 &
  server_pid=$!
  if ! wait_health; then
    tail -n 80 "$case_dir/server.log" >&2 || true
    stop_server
    return 1
  fi

  local target_device_index="${TARGET_DEVICE#hip:}"
  local active_gpu_arch
  active_gpu_arch="$(grep -E "Device ${target_device_index}:.*gfx" "$case_dir/server.log" | grep -oE 'gfx[0-9a-f]+' | head -n 1 || true)"
  [[ -n "$active_gpu_arch" ]] || {
    echo "server log does not identify active GPU" >&2
    tail -n 80 "$case_dir/server.log" >&2 || true
    stop_server
    return 1
  }
  if ! grep -Fxq "$active_gpu_arch" <<< "$binary_gpu_arches"; then
    echo "server binary HIP code objects do not include active GPU" >&2
    printf 'binary_gpu_arches=%s\nactive_gpu_arch=%s\ntarget_device=%s\n' \
      "$binary_gpu_arches" "$active_gpu_arch" "$TARGET_DEVICE" \
      > "$case_dir/gpu-identity.txt"
    stop_server
    return 1
  fi
  printf 'binary_gpu_arches=%s\nactive_gpu_arch=%s\ntarget_device=%s\n' \
    "$binary_gpu_arches" "$active_gpu_arch" "$TARGET_DEVICE" \
    > "$case_dir/gpu-identity.txt"

  local status=0
  python3 "$CLIENT" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
    --clients 1 --prompt-file "$PROMPT_FILE" \
    --temperature 0 --seed 1 --ignore-eos --timeout 1800 \
    --max-tokens "$SMOKE_TOKENS" \
    --out "$case_dir/smoke.json" \
    --label "$variant smoke" > "$case_dir/smoke.txt" || status=1

  if (( status == 0 )); then
    python3 "$CLIENT" \
      --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
      --clients "$CLIENTS" --prompt-file "$PROMPT_FILE" \
      --temperature 0 --seed 1 --ignore-eos --timeout 1800 \
      --max-tokens "$WARMUP_TOKENS" \
      --out "$case_dir/warmup.json" \
      --label "$variant warmup" > "$case_dir/warmup.txt" || status=1
  fi

  if (( status == 0 )); then
    python3 "$CLIENT" \
      --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
      --clients "$CLIENTS" --prompt-file "$PROMPT_FILE" \
      --temperature 0 --seed 1 --ignore-eos --timeout 1800 \
      --max-tokens "$MAX_TOKENS" \
      --server-metadata-json "$case_dir/server-metadata.json" \
      --out "$case_dir/bench.json" \
      --label "$variant fixed backlog C=$CLIENTS slots=$SLOTS t$MAX_TOKENS" \
      | tee "$case_dir/bench.txt" || status=1
  fi

  (( status == 0 )) || tail -n 80 "$case_dir/server.log" >&2 || true
  stop_server
  if (( status == 0 )); then
    python3 "$RUNTIME_METADATA_TOOL" \
      --metadata "$case_dir/server-metadata.json" \
      --server-log "$case_dir/server.log" \
      --out "$case_dir/runtime-metadata.json" || status=1
  fi
  sleep "$COOLDOWN_SECONDS"
  return "$status"
}

failures=0
for variant in "${variant_list[@]}"; do
  run_variant "$variant" || failures=$((failures + 1))
done
(( failures == 0 )) || { echo "$failures variant(s) failed" >&2; exit 1; }
echo "[fixed] complete: $OUT"
