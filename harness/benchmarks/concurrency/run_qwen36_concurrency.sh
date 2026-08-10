#!/usr/bin/env bash
# Paired, fresh-process Qwen3.6 concurrency benchmark for Lucebox and llama.cpp.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}"
CLIENT="${CLIENT:-$SCRIPT_DIR/concurrent_benchmark.py}"
GENERATOR="${GENERATOR:-$SCRIPT_DIR/generate_ragged_prompts.py}"
SUMMARIZER="${SUMMARIZER:-$SCRIPT_DIR/summarize_concurrency.py}"

MODEL="${MODEL:-}"
LUCE_SERVER_BIN="${LUCE_SERVER_BIN:-$REPO/server/build-hip/dflash_server}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server 2>/dev/null || true)}"
OUT="${OUT:-$REPO/.harness-runs/qwen36-concurrency-$(date -u +%Y%m%dT%H%M%SZ)}"
REPEATS="${REPEATS:-1}"
WORKLOADS="${WORKLOADS:-short,medium}"
VARIANTS="${VARIANTS:-luce-k8,luce-k1,llama}"
CLIENTS="${CLIENTS:-1,4,8,16}"
PORT="${PORT:-18114}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-3}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
MAX_TOKENS="${MAX_TOKENS:-64}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
SLOTS=16

usage() {
  cat <<'EOF'
Usage: MODEL=/path/model.gguf [REPEATS=5] run_qwen36_concurrency.sh

Runs fresh-server, same-concurrency warmup + measurement cases for luce-k8,
luce-k1, and llama at C=1/4/8/16. Defaults to one repeat for screening; use at
least five paired repeats for publication. OUT must not already exist.
EOF
}

if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi
for cmd in python3 curl sha256sum; do command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }; done
[[ -r "$MODEL" ]] || { echo "set MODEL to a readable GGUF" >&2; exit 2; }
[[ -x "$LUCE_SERVER_BIN" ]] || { echo "missing Lucebox server: $LUCE_SERVER_BIN" >&2; exit 2; }
[[ -x "$LLAMA_SERVER_BIN" ]] || { echo "missing llama.cpp server: $LLAMA_SERVER_BIN" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
ambient_tuning="$(env | grep -E '^(GGML_|DFLASH_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD=|LD_LIBRARY_PATH=)' \
  | grep -v '^LUCE_SERVER_BIN=' || true)"
if [[ -n "$ambient_tuning" ]]; then
  echo "refusing ambient GPU/backend tuning variables:" >&2
  echo "$ambient_tuning" >&2
  exit 2
fi
MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"

IFS=, read -r -a workload_list <<< "$WORKLOADS"
IFS=, read -r -a variant_list <<< "$VARIANTS"
IFS=, read -r -a client_list <<< "$CLIENTS"
declare -A prompt_offsets=([1]=0 [4]=1 [8]=5 [16]=13)
for c in "${client_list[@]}"; do
  [[ -n "${prompt_offsets[$c]+yes}" ]] || { echo "supported CLIENTS are 1,4,8,16" >&2; exit 2; }
done
for v in "${variant_list[@]}"; do
  [[ "$v" == luce-k8 || "$v" == luce-k1 || "$v" == llama ]] || { echo "unknown variant $v" >&2; exit 2; }
done

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
trap stop_server EXIT INT TERM

wait_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

write_metadata() {
  local path="$1" variant="$2" workload="$3" clients="$4" repeat="$5" binary="$6" max_prefills="$7" command_file="$8"
  python3 -c 'import hashlib,json,pathlib,subprocess,sys
p,variant,workload,clients,repeat,binary,max_prefills,cmd_file,model_sha,prompts,repo=sys.argv[1:]
digest=lambda x: hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest()
libs={}
for line in subprocess.run(["ldd",binary],text=True,capture_output=True).stdout.splitlines():
    fields=line.replace("=>"," ").split()
    paths=[x for x in fields if x.startswith("/") and pathlib.Path(x).is_file()]
    for lib in paths: libs[str(pathlib.Path(lib).resolve())]=digest(lib)
git_head=subprocess.run(["git","-C",repo,"rev-parse","HEAD"],text=True,capture_output=True).stdout.strip() or None
obj={"variant":variant,"workload":workload,"clients":int(clients),"repeat":int(repeat),
     "max_concurrent_prefills":int(max_prefills),"server_binary":str(pathlib.Path(binary).resolve()),
     "server_binary_sha256":digest(binary),"model_sha256":model_sha,
     "prompt_file_sha256":digest(prompts),"server_command":pathlib.Path(cmd_file).read_text().strip(),
     "resolved_shared_library_sha256":libs,"git_head":git_head}
pathlib.Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")' \
    "$path" "$variant" "$workload" "$clients" "$repeat" "$binary" "$max_prefills" "$command_file" "$MODEL_SHA256" "$OUT/prompts/$workload.jsonl" "$REPO"
}

run_case() {
  local repeat="$1" workload="$2" clients="$3" variant="$4"
  local max_ctx timeout capacity max_prefills binary model_id
  if [[ "$workload" == short ]]; then max_ctx=4096; timeout=900; else max_ctx=8192; timeout=1200; fi
  capacity=$((SLOTS * max_ctx))
  local case_dir="$OUT/$workload/c$clients/r$repeat/$variant"
  mkdir -p "$case_dir"
  local -a command
  if [[ "$variant" == llama ]]; then
    binary="$LLAMA_SERVER_BIN"; model_id=qwen36-llama; max_prefills=0
    command=("$binary" -m "$MODEL" -ngl all --parallel "$SLOTS" -c "$capacity"
      -b 2048 -ub 512 --cont-batching --no-context-shift --no-mmap -fa on
      -ctk q4_0 -ctv q4_0 --no-cache-prompt --host 127.0.0.1 --port "$PORT" --alias "$model_id")
  else
    binary="$LUCE_SERVER_BIN"; model_id=qwen36-luce
    [[ "$variant" == luce-k8 ]] && max_prefills=8 || max_prefills=1
    command=("$binary" "$MODEL" --target-device hip:0 --paged-attention
      --max-concurrency "$SLOTS" --kv-pool-tokens "$capacity" --max-ctx "$max_ctx"
      --cache-type-k q4_0 --cache-type-v q4_0 --fa-window 0
      --prefix-cache-slots 0 --prefill-cache-slots 0 --admission-coalesce-ms 5
      --prefill-quantum 512 --prefill-token-budget 4096
      --mixed-prefill-token-budget 2048 --host 127.0.0.1 --port "$PORT" --model-name "$model_id")
  fi
  printf '%q ' "${command[@]}" > "$case_dir/server-command.txt"; printf '\n' >> "$case_dir/server-command.txt"
  write_metadata "$case_dir/server-metadata.json" "$variant" "$workload" "$clients" "$repeat" "$binary" "$max_prefills" "$case_dir/server-command.txt"

  echo "[run] $workload C=$clients repeat=$repeat variant=$variant"
  if [[ "$variant" == llama ]]; then
    "${command[@]}" > "$case_dir/server.log" 2>&1 &
  else
    env DFLASH_MIN_TOKENS="$WARMUP_TOKENS" DFLASH_MAX_CONCURRENT_PREFILLS="$max_prefills" \
      "${command[@]}" > "$case_dir/server.log" 2>&1 &
  fi
  server_pid=$!
  if ! wait_health; then tail -n 80 "$case_dir/server.log" >&2 || true; return 1; fi

  local offset="${prompt_offsets[$clients]}" prompts="$OUT/prompts/$workload.jsonl"
  python3 "$CLIENT" --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
    --clients "$clients" --prompt-file "$prompts" --prompt-offset "$offset" \
    --require-distinct-prompts --max-tokens "$WARMUP_TOKENS" --temperature 0 \
    --ignore-eos --timeout "$timeout" --cooldown 0 --out "$case_dir/warmup.json" \
    --label "$variant $workload C=$clients warmup" > "$case_dir/warmup.txt"
  python3 "$CLIENT" --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
    --clients "$clients" --prompt-file "$prompts" --prompt-offset "$offset" \
    --require-distinct-prompts --max-tokens "$MAX_TOKENS" --temperature 0 \
    --ignore-eos --timeout "$timeout" --cooldown 0 \
    --server-metadata-json "$case_dir/server-metadata.json" --out "$case_dir/bench.json" \
    --label "$variant $workload C=$clients repeat=$repeat" | tee "$case_dir/bench.txt"
  stop_server
  sleep "$COOLDOWN_SECONDS"
}

for ((repeat=1; repeat<=REPEATS; repeat++)); do
  for workload in "${workload_list[@]}"; do
    for c_index in "${!client_list[@]}"; do
      clients="${client_list[$c_index]}"
      # Rotate start variant by case so one engine is not always hot or cold.
      shift_by=$(((repeat + c_index) % ${#variant_list[@]}))
      for ((i=0; i<${#variant_list[@]}; i++)); do
        variant="${variant_list[$(((i + shift_by) % ${#variant_list[@]}))]}"
        run_case "$repeat" "$workload" "$clients" "$variant"
      done
    done
  done
done

python3 "$SUMMARIZER" "$OUT" --out "$OUT/summary.md"
echo "[run] complete: $OUT"
