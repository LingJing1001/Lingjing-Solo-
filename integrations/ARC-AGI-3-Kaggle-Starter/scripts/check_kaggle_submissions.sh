#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN_FILE="$ROOT/.kaggle/access_token"
KAGGLE_BIN="$ROOT/.venv/bin/kaggle"

if [[ ! -r "$TOKEN_FILE" ]]; then
  echo "Missing readable token file: $TOKEN_FILE" >&2
  exit 1
fi
if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Missing Kaggle CLI: $KAGGLE_BIN" >&2
  exit 1
fi

export KAGGLE_API_TOKEN="$(< "$TOKEN_FILE")"
exec "$KAGGLE_BIN" competitions submissions -c arc-prize-2026-arc-agi-3
