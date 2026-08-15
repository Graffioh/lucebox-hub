#!/usr/bin/env bash

set -euo pipefail

RUN_OMP="${1:?usage: test_run_omp_config.sh <run_omp.sh>}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

FAKE_BIN="$TMP_DIR/bin"
FAKE_TARGET="$TMP_DIR/model.gguf"
FAKE_DRAFT="$TMP_DIR/draft.gguf"
FAKE_SERVER="$TMP_DIR/dflash_server"
FAKE_OMP="$TMP_DIR/omp"
mkdir -p "$FAKE_BIN"
touch "$FAKE_TARGET" "$FAKE_DRAFT"

cat >"$FAKE_SERVER" <<'EOF'
#!/usr/bin/env bash
exec sleep 600
EOF

cat >"$FAKE_OMP" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >"${OMP_ARGS_CAPTURE:?}"
printf 'HOME=%s\n' "$HOME" >>"${OMP_ARGS_CAPTURE}"
printf 'PI_CODING_AGENT_DIR=%s\n' "$PI_CODING_AGENT_DIR" >>"${OMP_ARGS_CAPTURE}"
printf 'OMP_PROFILE=%s\n' "$OMP_PROFILE" >>"${OMP_ARGS_CAPTURE}"
printf 'PI_PROFILE=%s\n' "$PI_PROFILE" >>"${OMP_ARGS_CAPTURE}"
EOF

cat >"$FAKE_BIN/curl" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$FAKE_BIN/timeout" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$1" >"${TIMEOUT_CAPTURE:?}"
shift
exec "$@"
EOF

chmod +x "$FAKE_SERVER" "$FAKE_OMP" "$FAKE_BIN"/*

ARGS_CAPTURE="$TMP_DIR/omp-args"
TIMEOUT_CAPTURE="$TMP_DIR/omp-timeout"
env \
  PATH="$FAKE_BIN:$PATH" \
  RUN_DIR="$TMP_DIR/runs" \
  STAMP=omp-contract \
  TARGET="$FAKE_TARGET" \
  DRAFT="$FAKE_DRAFT" \
  FA_WINDOW=1 \
  DFLASH_SERVER_BIN="$FAKE_SERVER" \
  OMP_BIN="$FAKE_OMP" \
  AUTO_INSTALL_CLIENTS=0 \
  OMP_ARGS_CAPTURE="$ARGS_CAPTURE" \
  TIMEOUT_CAPTURE="$TIMEOUT_CAPTURE" \
  MODEL_ID=omp-test-model \
  MAX_CTX=32768 \
  MAX_TOKENS=768 \
  OMP_STREAM_IDLE_TIMEOUT_MS=720000 \
  OMP_PROFILE=inherited-profile \
  PI_PROFILE=inherited-profile \
  PROMPT=omp-contract-prompt \
  bash "$RUN_OMP" >/dev/null

MODELS="$TMP_DIR/runs/omp-contract/omp-home/agent/models.yml"
grep -Fq 'baseUrl: "http://127.0.0.1:18080/v1"' "$MODELS"
grep -Fq 'auth: none' "$MODELS"
grep -Fq 'api: openai-responses' "$MODELS"
grep -Fq 'supportsDeveloperRole: false' "$MODELS"
grep -Fq 'supportsReasoningEffort: false' "$MODELS"
grep -Fq 'maxTokensField: max_tokens' "$MODELS"
grep -Fq 'streamIdleTimeoutMs: 720000' "$MODELS"
grep -Fq 'id: "omp-test-model"' "$MODELS"
grep -Fq 'contextWindow: 32768' "$MODELS"
grep -Fq 'maxTokens: 768' "$MODELS"

grep -Fq -- '--model lucebox/omp-test-model' "$ARGS_CAPTURE"
grep -Fq -- '--print --mode json' "$ARGS_CAPTURE"
grep -Fq -- '--tools read,grep,glob' "$ARGS_CAPTURE"
grep -Fq -- '--no-session --no-extensions --no-skills --no-rules --no-title' "$ARGS_CAPTURE"
grep -Fq -- 'omp-contract-prompt' "$ARGS_CAPTURE"
grep -Fq "PI_CODING_AGENT_DIR=$TMP_DIR/runs/omp-contract/omp-home/agent" "$ARGS_CAPTURE"
grep -Fxq 'OMP_PROFILE=' "$ARGS_CAPTURE"
grep -Fxq 'PI_PROFILE=' "$ARGS_CAPTURE"
grep -Fxq '3600s' "$TIMEOUT_CAPTURE"

INTERACTIVE_ARGS_CAPTURE="$TMP_DIR/omp-interactive-args"
env -u OMP_TOOLS \
  PATH="$FAKE_BIN:$PATH" \
  CLIENT_WORK_DIR="$TMP_DIR/client-work" \
  RUN_DIR="$TMP_DIR/runs" \
  STAMP=omp-interactive-contract \
  TARGET="$FAKE_TARGET" \
  DRAFT="$FAKE_DRAFT" \
  FA_WINDOW=1 \
  DFLASH_SERVER_BIN="$FAKE_SERVER" \
  OMP_BIN="$FAKE_OMP" \
  AUTO_INSTALL_CLIENTS=0 \
  OMP_ARGS_CAPTURE="$INTERACTIVE_ARGS_CAPTURE" \
  OMP_INTERACTIVE=1 \
  OMP_INITIAL_PROMPT=interactive-contract-prompt \
  MODEL_ID=omp-test-model \
  MAX_CTX=32768 \
  MAX_TOKENS=768 \
  bash "$RUN_OMP" >/dev/null

grep -Fq -- '--model lucebox/omp-test-model' "$INTERACTIVE_ARGS_CAPTURE"
grep -Fq -- 'interactive-contract-prompt' "$INTERACTIVE_ARGS_CAPTURE"
grep -Fq "PI_CODING_AGENT_DIR=$TMP_DIR/client-work/omp-home/agent" "$INTERACTIVE_ARGS_CAPTURE"
if grep -Eq -- '(^| )--print( |$)|(^| )--mode json( |$)|(^| )--no-session( |$)|(^| )--tools( |$)' \
  "$INTERACTIVE_ARGS_CAPTURE"; then
  echo "interactive OMP must keep the TUI/session and default tool set" >&2
  exit 1
fi

if [[ -n "${REAL_OMP_BIN:-}" ]]; then
  real_models="$(
    env \
      HOME="$TMP_DIR/runs/omp-contract/omp-home" \
      PI_CODING_AGENT_DIR="$TMP_DIR/runs/omp-contract/omp-home/agent" \
      PI_CODING_AGENT_SESSION_DIR="$TMP_DIR/runs/omp-contract/omp-home/sessions" \
      "$REAL_OMP_BIN" models
  )"
  grep -Fq 'lucebox (1)' <<<"$real_models"
  grep -Fq 'omp-test-model' <<<"$real_models"
fi

set +e
invalid_output="$(
  OMP_BIN="$FAKE_OMP" OMP_STREAM_IDLE_TIMEOUT_MS=invalid \
    AUTO_INSTALL_CLIENTS=0 bash "$RUN_OMP" 2>&1
)"
invalid_rc=$?
set -e
if [[ "$invalid_rc" -ne 2 ]] ||
   ! grep -Fq 'OMP_STREAM_IDLE_TIMEOUT_MS must be a non-negative integer (0 disables OMP'\''s stream watchdog)' <<<"$invalid_output"; then
  echo "invalid OMP stream timeout must fail with a useful error" >&2
  exit 1
fi

set +e
invalid_output="$(
  OMP_BIN="$FAKE_OMP" OMP_TIMEOUT=invalid \
    AUTO_INSTALL_CLIENTS=0 bash "$RUN_OMP" 2>&1
)"
invalid_rc=$?
set -e
if [[ "$invalid_rc" -ne 2 ]] ||
   ! grep -Fq 'OMP_TIMEOUT must be a non-negative integer (seconds; 0 disables it)' <<<"$invalid_output"; then
  echo "invalid OMP_TIMEOUT must fail with a useful error before startup" >&2
  exit 1
fi

set +e
invalid_output="$(
  OMP_BIN="$FAKE_OMP" OMP_INTERACTIVE=invalid \
    AUTO_INSTALL_CLIENTS=0 bash "$RUN_OMP" 2>&1
)"
invalid_rc=$?
set -e
if [[ "$invalid_rc" -ne 2 ]] ||
   ! grep -Fq 'OMP_INTERACTIVE must be 0 or 1' <<<"$invalid_output"; then
  echo "invalid OMP_INTERACTIVE must fail with a useful error before startup" >&2
  exit 1
fi

echo "OMP launcher configuration: PASS"
