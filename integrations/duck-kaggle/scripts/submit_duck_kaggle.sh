#!/usr/bin/env bash
# Push a parallel Duck harness kernel (does NOT overwrite SmartRouter).
# Run from Linux/WSL with uv + make available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INF="$ROOT/ARC3-Inference"
cd "$INF"

KAGGLE_USERNAME="${KAGGLE_USERNAME:-liaojiaxuan}"
RUN_NAME="${RUN_NAME:-lingjing-duck-harness-v1}"
KAGGLE_KERNEL_SLUG="${KAGGLE_KERNEL_SLUG:-lingjing-duck-harness-v1}"
KAGGLE_DATASET_REF="${KAGGLE_DATASET_REF:-${KAGGLE_USERNAME}/taaf-kaggle-source-${RUN_NAME}}"
KAGGLE_ACCELERATOR="${KAGGLE_ACCELERATOR:-nvidiaRtx6000}"
# Conservative concurrency for first RTX6000 pass (avoid OOM / 9h kill).
CONCURRENT_JOBS="${CONCURRENT_JOBS:-8}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-90}"
MAX_EXPERIMENT_RUNTIME_MINUTES="${MAX_EXPERIMENT_RUNTIME_MINUTES:-520}"
DRY_RUN="${KAGGLE_DRY_RUN:-false}"

# Prefer token from SmartRouter starter if present.
STARTER_TOKEN="$(cd "$ROOT/.." && pwd)/ARC-AGI-3-Kaggle-Starter/.kaggle/access_token"
if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "$STARTER_TOKEN" ]]; then
  export KAGGLE_API_TOKEN
  KAGGLE_API_TOKEN="$(tr -d '\r\n' < "$STARTER_TOKEN")"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> make install (base + server extras may be heavy)"
make install

echo "==> make kaggle-duck RUN_NAME=$RUN_NAME SLUG=$KAGGLE_KERNEL_SLUG ACCEL=$KAGGLE_ACCELERATOR"
make kaggle-duck \
  RUN_NAME="$RUN_NAME" \
  KAGGLE_KERNEL_SLUG="$KAGGLE_KERNEL_SLUG" \
  KAGGLE_DATASET_REF="$KAGGLE_DATASET_REF" \
  KAGGLE_USERNAME="$KAGGLE_USERNAME" \
  KAGGLE_ACCELERATOR="$KAGGLE_ACCELERATOR" \
  CONCURRENT_JOBS="$CONCURRENT_JOBS" \
  MAX_RUNTIME_MINUTES="$MAX_RUNTIME_MINUTES" \
  MAX_EXPERIMENT_RUNTIME_MINUTES="$MAX_EXPERIMENT_RUNTIME_MINUTES" \
  KAGGLE_DRY_RUN="$DRY_RUN" \
  KAGGLE_RUN_AS_SUBMISSION=true

echo "Done. Track kernel: ${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
