#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENTS="$REPO_ROOT/harness/clients"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

FAKE_BIN="$TMP_DIR/bin"
TIMEOUT_CAPTURE="$TMP_DIR/timeout-duration"
mkdir -p "$FAKE_BIN"
cat >"$FAKE_BIN/timeout" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$1" >"${TIMEOUT_CAPTURE:?}"
shift
exec "$@"
EOF
chmod +x "$FAKE_BIN/timeout"
export PATH="$FAKE_BIN:$PATH"
export TIMEOUT_CAPTURE

SCRIPT_DIR="$CLIENTS"
CLIENT_WORK_DIR="$TMP_DIR/work"
RUN_DIR="$TMP_DIR/runs"
STAMP="timeout-policy"
source "$CLIENTS/common.sh"

if [[ "$(run_with_timeout 5 printf '%s' bounded)" != "bounded" ]]; then
  echo "bounded client timeout did not execute the command" >&2
  exit 1
fi
if ! grep -Fxq '5s' "$TIMEOUT_CAPTURE"; then
  echo "bounded client timeout used the wrong duration" >&2
  exit 1
fi
rm -f "$TIMEOUT_CAPTURE"
if [[ "$(run_with_timeout 0 printf '%s' unlimited)" != "unlimited" ]]; then
  echo "zero client timeout must execute without a deadline" >&2
  exit 1
fi
if [[ -e "$TIMEOUT_CAPTURE" ]]; then
  echo "zero client timeout must bypass the timeout command" >&2
  exit 1
fi

set +e
invalid_output="$(run_with_timeout invalid true 2>&1)"
invalid_rc=$?
set -e
if [[ "$invalid_rc" -ne 2 ]] ||
   ! grep -Fq 'client timeout must be a non-negative integer' <<<"$invalid_output"; then
  echo "invalid client timeout must fail with a useful error" >&2
  exit 1
fi

grep -Fq ': "${OPENCODE_TIMEOUT:=3600}"' "$CLIENTS/run_opencode.sh"
grep -Fq ': "${OPENCODE_REQUEST_TIMEOUT_MS:=3600000}"' "$CLIENTS/run_opencode.sh"
grep -Fq ': "${OPENCODE_CHUNK_TIMEOUT_MS:=3600000}"' "$CLIENTS/run_opencode.sh"
grep -Fq 'run_with_timeout "$OPENCODE_TIMEOUT"' "$CLIENTS/run_opencode.sh"
grep -Fq ': "${CODEX_TIMEOUT:=3600}"' "$CLIENTS/run_codex.sh"
grep -Fq 'run_with_timeout "$CODEX_TIMEOUT"' "$CLIENTS/run_codex.sh"
grep -Fq ': "${HERMES_TIMEOUT:=3600}"' "$CLIENTS/run_hermes.sh"
grep -Fq 'run_with_timeout "$HERMES_TIMEOUT"' "$CLIENTS/run_hermes.sh"
grep -Fq ': "${CLAUDE_TIMEOUT:=3600}"' "$CLIENTS/run_claude_code.sh"
grep -Fq 'run_with_timeout "$CLAUDE_TIMEOUT"' "$CLIENTS/run_claude_code.sh"
grep -Fq ': "${OPENCLAW_TIMEOUT:=3600}"' "$CLIENTS/run_openclaw.sh"
grep -Fq 'run_with_timeout "$OPENCLAW_TIMEOUT"' "$CLIENTS/run_openclaw.sh"
grep -Fq 'timeout_args=(--timeout "$OPENCLAW_TIMEOUT")' "$CLIENTS/run_openclaw.sh"
grep -Fq 'CURL_MAX_TIME="${CURL_MAX_TIME:-3600}"' "$CLIENTS/run_openwebui.sh"
grep -Fq 'CURL_MAX_TIME="${CURL_MAX_TIME:-3600}"' "$CLIENTS/run_openwebui_tools.sh"

if grep -REq 'timeout (300|420)s|--timeout 300|CURL_MAX_TIME:-300' "$CLIENTS"/*.sh; then
  echo "a short hard-coded real-client timeout remains" >&2
  exit 1
fi

echo "Client launcher timeout policy: PASS"
