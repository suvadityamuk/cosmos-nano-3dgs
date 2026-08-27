#!/usr/bin/env bash
set -euo pipefail

IMAGE=""
PROMPT_FILE=""
OUTPUT_DIR=""
PROFILE="full"
MASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --mask) MASK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$IMAGE" || -z "$PROMPT_FILE" || -z "$OUTPUT_DIR" ]]; then
  echo "--image, --prompt-file, and --output-dir are required" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/cosmos3-gsplat-venv}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/cosmos3-torch-extensions}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$OUTPUT_DIR" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$OUTPUT_DIR/job.log") 2>&1

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --no-cache-dir uv
fi

echo "GPU:"
nvidia-smi
echo "Installing locked project dependencies..."
uv sync --project /workspace --frozen --extra gpu

echo "Precompiling gsplat CUDA kernels for ${TORCH_CUDA_ARCH_LIST:-the detected GPU}..."
if [[ -d "$OUTPUT_DIR/.torch_extensions" ]]; then
  cp -a "$OUTPUT_DIR/.torch_extensions/." "$TORCH_EXTENSIONS_DIR/"
fi
VERBOSE=1 uv run --project /workspace python - <<'PY'
import gsplat
import torch
from gsplat.cuda._backend import _C

has_3dgs = getattr(gsplat, "has_3dgs", None)
available = bool(has_3dgs()) if callable(has_3dgs) else _C is not None
if not torch.cuda.is_available() or not available or _C is None:
    raise RuntimeError("gsplat CUDA preflight failed")
print("gsplat CUDA preflight complete:", torch.cuda.get_device_name(0), torch.version.cuda)
PY
rm -rf "$OUTPUT_DIR/.torch_extensions"
cp -a "$TORCH_EXTENSIONS_DIR" "$OUTPUT_DIR/.torch_extensions"

PROMPT="$(<"$PROMPT_FILE")"
ARGS=(
  run
  --image "$IMAGE"
  --prompt "$PROMPT"
  --output-dir "$OUTPUT_DIR"
  --profile "$PROFILE"
)
if [[ -n "$MASK" ]]; then
  ARGS+=(--mask "$MASK")
fi

uv run --project /workspace cosmos3-gsplat "${ARGS[@]}"
