#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_qwen38_dflash2_fixed.sh}"
COMPARATOR="${COMPARATOR:-$SCRIPT_DIR/compare_qwen38_fixed.py}"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}"
OUT_ROOT="${OUT_ROOT:-$REPO/.harness-runs/qwen38-dflash2-measurement-$(date -u +%Y%m%dT%H%M%SZ)}"
REPLICATES="${REPLICATES:-3}"
REQUIRE_STABLE="${REQUIRE_STABLE:-0}"

[[ "$REPLICATES" =~ ^[0-9]+$ ]] && (( REPLICATES >= 2 )) || {
  echo "REPLICATES must be at least 2" >&2
  exit 2
}
[[ "$REQUIRE_STABLE" =~ ^[01]$ ]] || {
  echo "REQUIRE_STABLE must be 0 or 1" >&2
  exit 2
}
[[ ! -e "$OUT_ROOT" ]] || {
  echo "refusing to overwrite $OUT_ROOT" >&2
  exit 2
}
mkdir -p "$OUT_ROOT"

variant_orders=(
  "ar,speculation,adaptive"
  "adaptive,ar,speculation"
  "speculation,adaptive,ar"
)
replicate_paths=()
for ((replicate = 1; replicate <= REPLICATES; replicate++)); do
  out="$OUT_ROOT/replicate-$replicate"
  order_index=$(((replicate - 1) % ${#variant_orders[@]}))
  VARIANTS="${variant_orders[$order_index]}" OUT="$out" "$RUNNER"
  replicate_paths+=("$out")
done

compare_args=(
  --out "$OUT_ROOT/comparison.json"
  "${replicate_paths[@]}"
)
if (( REQUIRE_STABLE == 1 )); then
  compare_args=(--require-stable "${compare_args[@]}")
fi
python3 "$COMPARATOR" "${compare_args[@]}" | tee "$OUT_ROOT/comparison.md"
