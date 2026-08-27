#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/cosmos3-gsplat-venv}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" /artifacts
exec > >(tee -a /artifacts/job.log) 2>&1

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --no-cache-dir uv
fi

nvidia-smi
uv sync --project /workspace --frozen --extra gpu
uv run --project /workspace python -m cosmos3_gsplat.camera_diagnostics \
  --control-image /inputs/lighthouse_720.png \
  --control-prompt-file /inputs/lighthouse.txt \
  --control-actions /inputs/camera_action.json \
  --chair-image /inputs/chair.png \
  --chair-mask /inputs/chair_mask.png \
  --output-dir /artifacts
