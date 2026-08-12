#!/usr/bin/env bash
# Fresh-process Qwen3.6 concurrent feature ablations with fail-closed proof.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}"
CLIENT="${CLIENT:-$SCRIPT_DIR/feature_concurrent_benchmark.py}"
GENERATOR="${GENERATOR:-$SCRIPT_DIR/generate_feature_prompts.py}"
SUMMARIZER="${SUMMARIZER:-$SCRIPT_DIR/summarize_feature_matrix.py}"
PROOF_TOOL="${PROOF_TOOL:-$SCRIPT_DIR/verify_feature_metrics.py}"
METADATA_TOOL="${METADATA_TOOL:-$SCRIPT_DIR/write_feature_metadata.py}"
RUNTIME_METADATA_TOOL="${RUNTIME_METADATA_TOOL:-$SCRIPT_DIR/record_feature_runtime.py}"

MODEL="${MODEL:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
PREFILL_DRAFTER="${PREFILL_DRAFTER:-}"
LUCE_SERVER_BIN="${LUCE_SERVER_BIN:-$REPO/server/build-hip/dflash_server}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server 2>/dev/null || true)}"
OUT="${OUT:-$REPO/.harness-runs/qwen36-feature-matrix-$(date -u +%Y%m%dT%H%M%SZ)}"
REPEATS="${REPEATS:-1}"
WORKLOADS="${WORKLOADS:-short,compression}"
VARIANTS="${VARIANTS:-ar,ddtree,pflash,kvflash,full}"
CLIENTS="${CLIENTS:-4}"
PORT="${PORT:-18114}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-3}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
MAX_TOKENS="${MAX_TOKENS:-64}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
SLOTS="${SLOTS:-16}"
MAX_CONCURRENT_PREFILLS="${MAX_CONCURRENT_PREFILLS:-8}"

# The requested Strix Halo configuration. Every value is serialized into case
# metadata; no performance-affecting DFLASH variable is inherited implicitly.
TARGET_DEVICE="${TARGET_DEVICE:-hip:0}"
DRAFT_DEVICE="${DRAFT_DEVICE:-hip:0}"
DDTREE_BUDGET="${DDTREE_BUDGET:-22}"
DRAFT_RESIDENCY="${DRAFT_RESIDENCY:-persistent}"
PREFILL_COMPRESSION="${PREFILL_COMPRESSION:-auto}"
PREFILL_THRESHOLD="${PREFILL_THRESHOLD:-32000}"
PREFILL_KEEP_RATIO="${PREFILL_KEEP_RATIO:-0.05}"
KVFLASH_MODE="${KVFLASH_MODE:-auto}"
KVFLASH_MAX_POOL_TOKENS="${KVFLASH_MAX_POOL_TOKENS:-8192}"

usage() {
  cat <<'EOF'
Usage:
  MODEL=/path/Qwen3.6-27B-Q4_K_M.gguf \
  DRAFT_MODEL=/path/dflash-draft-3.6-q4_k_m.gguf \
  PREFILL_DRAFTER=/path/Qwen3-0.6B-BF16.gguf \
  harness/benchmarks/concurrency/run_qwen36_feature_matrix.sh

The default is a bounded C4 smoke matrix with independently selectable
ar, ddtree, pflash, kvflash, and full rows. The full row is the requested
Strix Halo configuration: target/draft hip:0, DDTree
budget 22, persistent PFlash auto, and KVFlash auto. The long-context profiles
are intended to cross the recorded 32K PFlash and 8K KV-residency thresholds;
word count is not treated as proof. Per-request wire/log token counts and
activation telemetry fail the case if an "auto" feature did not execute.

llama is optional: include it explicitly with VARIANTS=ar,ddtree,llama and set
LLAMA_SERVER_BIN. For publication, set CLIENTS=1,4,8,16 and REPEATS=5.
OUT must not already exist.
EOF
}

if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi
for cmd in python3 curl sha256sum awk; do
  command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }
done
[[ -r "$MODEL" ]] || { echo "set MODEL to a readable target GGUF" >&2; exit 2; }
[[ -x "$LUCE_SERVER_BIN" ]] || { echo "missing Lucebox server: $LUCE_SERVER_BIN" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }
[[ "$SLOTS" =~ ^[1-9][0-9]*$ ]] || { echo "SLOTS must be positive" >&2; exit 2; }
[[ "$MAX_CONCURRENT_PREFILLS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CONCURRENT_PREFILLS must be positive" >&2; exit 2; }
[[ "$DDTREE_BUDGET" =~ ^[1-9][0-9]*$ ]] || { echo "DDTREE_BUDGET must be positive" >&2; exit 2; }
[[ "$PREFILL_THRESHOLD" =~ ^[1-9][0-9]*$ ]] || { echo "PREFILL_THRESHOLD must be positive" >&2; exit 2; }
[[ "$KVFLASH_MAX_POOL_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "KVFLASH_MAX_POOL_TOKENS must be positive" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }

ambient_tuning="$(env | grep -E '^(GGML_|DFLASH_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD=|LD_LIBRARY_PATH=)' \
  | grep -v '^LUCE_SERVER_BIN=' || true)"
if [[ -n "$ambient_tuning" ]]; then
  echo "refusing ambient GPU/backend tuning variables:" >&2
  echo "$ambient_tuning" >&2
  exit 2
fi

IFS=, read -r -a workload_list <<< "$WORKLOADS"
IFS=, read -r -a variant_list <<< "$VARIANTS"
IFS=, read -r -a client_list <<< "$CLIENTS"
reject_duplicates() {
  local list_name="$1" value
  shift
  local -A seen=()
  for value in "$@"; do
    if [[ -n "${seen[$value]+yes}" ]]; then
      echo "$list_name contains duplicate entry: $value" >&2
      return 1
    fi
    seen["$value"]=1
  done
}
reject_duplicates CLIENTS "${client_list[@]}" || exit 2
reject_duplicates VARIANTS "${variant_list[@]}" || exit 2
declare -A prompt_offsets=([1]=0 [4]=1 [8]=5 [16]=13)
for c in "${client_list[@]}"; do
  [[ -n "${prompt_offsets[$c]+yes}" ]] || { echo "supported CLIENTS are 1,4,8,16" >&2; exit 2; }
  (( c <= SLOTS )) || { echo "CLIENTS=$c exceeds SLOTS=$SLOTS" >&2; exit 2; }
done
for v in "${variant_list[@]}"; do
  case "$v" in
    ar|ddtree|pflash|kvflash|full|llama) ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
done

contains_variant() {
  local needle="$1" value
  for value in "${variant_list[@]}"; do [[ "$value" == "$needle" ]] && return 0; done
  return 1
}
if contains_variant ddtree || contains_variant full; then
  [[ -r "$DRAFT_MODEL" ]] || { echo "DRAFT_MODEL is required for DDTree/full" >&2; exit 2; }
fi
if contains_variant pflash || contains_variant kvflash || contains_variant full; then
  [[ -r "$PREFILL_DRAFTER" ]] || {
    echo "PREFILL_DRAFTER is required for PFlash and drafter-scored KVFlash" >&2
    exit 2
  }
fi
if contains_variant llama; then
  [[ -x "$LLAMA_SERVER_BIN" ]] || { echo "llama requested but LLAMA_SERVER_BIN is missing" >&2; exit 2; }
fi

MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
DRAFT_MODEL_SHA256=""
PREFILL_DRAFTER_SHA256=""
[[ -n "$DRAFT_MODEL" ]] && DRAFT_MODEL_SHA256="$(sha256sum "$DRAFT_MODEL" | awk '{print $1}')"
[[ -n "$PREFILL_DRAFTER" ]] && PREFILL_DRAFTER_SHA256="$(sha256sum "$PREFILL_DRAFTER" | awk '{print $1}')"

mkdir -p "$OUT/prompts"
for workload in "${workload_list[@]}"; do
  python3 "$GENERATOR" --profile "$workload" --out "$OUT/prompts/$workload.jsonl"
done

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

port_is_available() {
  python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        print(f"PORT {port} is unavailable: {exc}", file=sys.stderr)
        sys.exit(1)
PY
}

# Auto modes cannot be proven on sub-threshold inputs. These skips are listed
# explicitly and are not emitted as successful benchmark rows.
case_applicable() {
  local variant="$1" workload="$2"
  case "$variant" in
    pflash|full) [[ "$workload" == compression ]] ;;
    kvflash) [[ "$workload" == compression || "$workload" == kv-pressure ]] ;;
    *) return 0 ;;
  esac
}

workload_limits() {
  case "$1" in
    short) echo "4096 1200" ;;
    medium) echo "8192 1800" ;;
    long) echo "16384 2400" ;;
    kv-pressure) echo "32768 3600" ;;
    compression) echo "65536 5400" ;;
    *) echo "unknown workload $1" >&2; return 1 ;;
  esac
}

run_case() {
  local repeat="$1" workload="$2" clients="$3" variant="$4"
  local max_ctx timeout
  read -r max_ctx timeout <<< "$(workload_limits "$workload")"
  local capacity=$((SLOTS * max_ctx))
  local case_dir="$OUT/$workload/c$clients/r$repeat/$variant"
  mkdir -p "$case_dir"

  local binary model_id max_prefills
  local -a command launch_env metadata expected
  if [[ "$variant" == llama ]]; then
    binary="$LLAMA_SERVER_BIN"; model_id=qwen36-llama; max_prefills=0
    command=("$binary" -m "$MODEL" -ngl all --parallel "$SLOTS" -c "$capacity"
      -b 2048 -ub 512 --cont-batching --no-context-shift --no-mmap -fa on
      -ctk q4_0 -ctv q4_0 --no-cache-prompt --host 127.0.0.1 --port "$PORT" --alias "$model_id")
    launch_env=()
  else
    binary="$LUCE_SERVER_BIN"; model_id=qwen36-luce; max_prefills="$MAX_CONCURRENT_PREFILLS"
    command=("$binary" "$MODEL" --target-device "$TARGET_DEVICE" --paged-attention
      --max-concurrency "$SLOTS" --kv-pool-tokens "$capacity" --max-ctx "$max_ctx"
      --cache-type-k q4_0 --cache-type-v q4_0 --fa-window 0
      --prefix-cache-slots 0 --prefill-cache-slots 0 --admission-coalesce-ms 5
      --host 127.0.0.1 --port "$PORT" --model-name "$model_id")
    launch_env=("DFLASH_MIN_TOKENS=$WARMUP_TOKENS" "DFLASH_MAX_CONCURRENT_PREFILLS=$max_prefills")
    if [[ "$variant" == ddtree || "$variant" == full ]]; then
      command+=(--draft "$DRAFT_MODEL" --draft-device "$DRAFT_DEVICE"
        --ddtree --ddtree-budget "$DDTREE_BUDGET")
      expected+=(--expect ddtree)
    fi
    if [[ "$variant" == pflash || "$variant" == full ]]; then
      if [[ "$variant" == pflash ]]; then command+=(--draft-device "$DRAFT_DEVICE"); fi
      command+=(--prefill-compression "$PREFILL_COMPRESSION"
        --prefill-threshold "$PREFILL_THRESHOLD"
        --prefill-keep-ratio "$PREFILL_KEEP_RATIO"
        --prefill-drafter "$PREFILL_DRAFTER"
        --draft-residency "$DRAFT_RESIDENCY")
      expected+=(--expect pflash)
    fi
    if [[ "$variant" == kvflash || "$variant" == full ]]; then
      # --prefill-drafter also selects the default drafter-scored KVFlash
      # residency policy; it does not enable PFlash when compression is off.
      if [[ "$variant" == kvflash ]]; then
        command+=(--prefill-drafter "$PREFILL_DRAFTER")
      fi
      command+=(--kvflash "$KVFLASH_MODE")
      launch_env+=("DFLASH_KVFLASH_MAX_POOL=$KVFLASH_MAX_POOL_TOKENS")
      expected+=(--expect kvflash)
    fi
  fi

  if ((${#launch_env[@]})); then
    printf 'env ' > "$case_dir/server-command.txt"
    printf '%q ' "${launch_env[@]}" "${command[@]}" >> "$case_dir/server-command.txt"
  else
    printf '%q ' "${command[@]}" > "$case_dir/server-command.txt"
  fi
  printf '\n' >> "$case_dir/server-command.txt"

  metadata=(python3 "$METADATA_TOOL" --out "$case_dir/server-metadata.json"
    --variant "$variant" --workload "$workload" --clients "$clients" --repeat "$repeat"
    --binary "$binary" --model "$MODEL" --model-sha256 "$MODEL_SHA256"
    --prompt-file "$OUT/prompts/$workload.jsonl"
    --command-file "$case_dir/server-command.txt" --repo "$REPO"
    --max-concurrent-prefills "$max_prefills")
  if [[ "$variant" != llama ]]; then
    metadata+=(--target-device "$TARGET_DEVICE")
    local item
    for item in "${launch_env[@]}"; do metadata+=(--launch-env "$item"); done
  fi
  if [[ "$variant" == ddtree || "$variant" == full ]]; then
    metadata+=(--draft-device "$DRAFT_DEVICE" --draft-model "$DRAFT_MODEL"
      --draft-model-sha256 "$DRAFT_MODEL_SHA256" --ddtree --ddtree-budget "$DDTREE_BUDGET")
  fi
  if [[ "$variant" == pflash || "$variant" == full ]]; then
    metadata+=(--draft-device "$DRAFT_DEVICE" --prefill-compression "$PREFILL_COMPRESSION"
      --prefill-threshold "$PREFILL_THRESHOLD" --prefill-keep-ratio "$PREFILL_KEEP_RATIO"
      --prefill-drafter "$PREFILL_DRAFTER"
      --prefill-drafter-sha256 "$PREFILL_DRAFTER_SHA256"
      --draft-residency "$DRAFT_RESIDENCY")
  fi
  if [[ "$variant" == kvflash || "$variant" == full ]]; then
    metadata+=(--kvflash "$KVFLASH_MODE"
      --kvflash-max-pool-tokens "$KVFLASH_MAX_POOL_TOKENS"
      --kvflash-scorer-drafter "$PREFILL_DRAFTER"
      --kvflash-scorer-drafter-sha256 "$PREFILL_DRAFTER_SHA256")
    if [[ "$variant" == kvflash ]]; then
      # Record the literal server flag too, while keeping compression off.
      metadata+=(--prefill-drafter "$PREFILL_DRAFTER"
        --prefill-drafter-sha256 "$PREFILL_DRAFTER_SHA256")
    fi
  fi
  "${metadata[@]}"

  echo "[run] $workload C=$clients repeat=$repeat variant=$variant"
  port_is_available || return 1
  if ((${#launch_env[@]})); then
    env "${launch_env[@]}" "${command[@]}" > "$case_dir/server.log" 2>&1 &
  else
    "${command[@]}" > "$case_dir/server.log" 2>&1 &
  fi
  server_pid=$!
  if ! wait_health; then tail -n 120 "$case_dir/server.log" >&2 || true; return 1; fi
  if [[ "$variant" != llama ]]; then
    python3 "$RUNTIME_METADATA_TOOL" --metadata "$case_dir/server-metadata.json" \
      --server-log "$case_dir/server.log"
  fi

  local offset="${prompt_offsets[$clients]}" prompts="$OUT/prompts/$workload.jsonl"
  local -a common_client=(--base-url "http://127.0.0.1:$PORT/v1" --model "$model_id"
    --clients "$clients" --prompt-file "$prompts" --prompt-offset "$offset"
    --require-distinct-prompts --temperature 0 --ignore-eos --timeout "$timeout" --cooldown 0)
  local -a telemetry_arg=()
  [[ "$variant" != llama ]] && telemetry_arg+=(--require-effective-prompt-telemetry)
  python3 "$CLIENT" "${common_client[@]}" --max-tokens "$WARMUP_TOKENS"
    "${telemetry_arg[@]}" --out "$case_dir/warmup.json"
    --label "$variant $workload C=$clients warmup" > "$case_dir/warmup.txt"
  python3 "$CLIENT" "${common_client[@]}" --max-tokens "$MAX_TOKENS"
    "${telemetry_arg[@]}" --server-metadata-json "$case_dir/server-metadata.json"
    --out "$case_dir/bench.json" --label "$variant $workload C=$clients repeat=$repeat" \
    | tee "$case_dir/bench.txt"
  stop_server

  if [[ "$variant" != llama ]]; then
    python3 "$PROOF_TOOL" --bench "$case_dir/bench.json" --server-log "$case_dir/server.log"
      "${expected[@]}" --out "$case_dir/feature-proof.json"
  fi
  sleep "$COOLDOWN_SECONDS"
}

active_cases=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  for workload in "${workload_list[@]}"; do
    for c_index in "${!client_list[@]}"; do
      clients="${client_list[$c_index]}"
      shift_by=$(((repeat + c_index) % ${#variant_list[@]}))
      for ((i=0; i<${#variant_list[@]}; i++)); do
        variant="${variant_list[$(((i + shift_by) % ${#variant_list[@]}))]}"
        if ! case_applicable "$variant" "$workload"; then
          echo "[skip] $variant requires an activation workload; workload=$workload"
          continue
        fi
        active_cases=$((active_cases + 1))
        run_case "$repeat" "$workload" "$clients" "$variant"
      done
    done
  done
done
(( active_cases > 0 )) || { echo "no applicable benchmark cases" >&2; exit 2; }

python3 "$SUMMARIZER" "$OUT" --out "$OUT/summary.md"
echo "[run] complete: $OUT"
