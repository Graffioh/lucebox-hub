#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

TARGET_SOURCE="${TARGET_SOURCE:?set TARGET_SOURCE to bartowski Qwen3.8-27B IQ4_XS GGUF}"
DRAFT_SOURCE="${DRAFT_SOURCE:?set DRAFT_SOURCE to RadixArk Qwen3.8-27B-DSpark model.safetensors}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:?set LLAMA_QUANTIZE to a llama-quantize binary}"
PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-$REPO/models/.lucebox/qwen38-pr625}"
DRAFT_SCHEME="${DRAFT_SCHEME:-q8_0}"

[[ -r "$TARGET_SOURCE" ]] || { echo "unreadable TARGET_SOURCE: $TARGET_SOURCE" >&2; exit 2; }
[[ -r "$DRAFT_SOURCE" ]] || { echo "unreadable DRAFT_SOURCE: $DRAFT_SOURCE" >&2; exit 2; }
[[ -x "$LLAMA_QUANTIZE" ]] || { echo "LLAMA_QUANTIZE is not executable: $LLAMA_QUANTIZE" >&2; exit 2; }
command -v "$PYTHON" >/dev/null || { echo "PYTHON is unavailable: $PYTHON" >&2; exit 2; }
[[ "$DRAFT_SCHEME" == f16 || "$DRAFT_SCHEME" == q8_0 || "$DRAFT_SCHEME" == q4-mix ]] || {
  echo "DRAFT_SCHEME must be f16, q8_0, or q4-mix" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
work_dir="$(mktemp -d "$OUT_DIR/.prepare.XXXXXX")"
cleanup() { rm -rf -- "$work_dir"; }
trap cleanup EXIT

draft_f16="$work_dir/Qwen3.8-27B-DSpark-RadixArk-no-yarn-f16.gguf"
draft_final="$work_dir/Qwen3.8-27B-DSpark-RadixArk-no-yarn-$DRAFT_SCHEME.gguf"
target_final="$work_dir/Qwen3.8-27B-PR625-IQ4_XS.gguf"

"$PYTHON" "$SCRIPT_DIR/convert_dflash_to_gguf.py" \
  "$DRAFT_SOURCE" "$draft_f16" --no-yarn
if [[ "$DRAFT_SCHEME" != f16 ]]; then
  "$PYTHON" "$SCRIPT_DIR/quantize_dflash_draft.py" \
    "$draft_f16" "$draft_final" --scheme "$DRAFT_SCHEME"
fi

"$LLAMA_QUANTIZE" \
  --allow-requantize --pure \
  --output-tensor-type q5_k \
  --tensor-type ssm_out=q6_k \
  --tensor-type attn_v=q6_k \
  "$TARGET_SOURCE" "$target_final" iq4_xs

"$PYTHON" "$SCRIPT_DIR/validate_qwen38_pr625_models.py" \
  --target "$target_final" --draft "$draft_final" --draft-scheme "$DRAFT_SCHEME"

mv -- "$target_final" "$OUT_DIR/Qwen3.8-27B-PR625-IQ4_XS.gguf"
mv -- "$draft_final" "$OUT_DIR/Qwen3.8-27B-DSpark-RadixArk-no-yarn-$DRAFT_SCHEME.gguf"

echo "PR #625 model pair ready in $OUT_DIR"
