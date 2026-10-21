#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${ARC_API_KEY:-}" ]]; then
  printf '%s\n' 'ARC_API_KEY is missing. Export it in this shell before running targeted collection.' >&2
  exit 2
fi

OUT="${1:?usage: run_targeted_collection.sh OUT_DIR [GROUPS_PER_GAME] [FOCUS_ACTIONS]}"
GROUPS="${2:-6}"
FOCUS="${3:-ACTION6,ACTION7}"
GAMES="${4:-dc22,ls20,re86,vc33}"
PREFIXES="${5:-10,20,30}"
ACTION6_X="${6:-}"
ACTION6_Y="${7:-}"
cd "$ROOT"
RUN_ID="${AOP_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-$$}"
ARGS=(
  --out "$OUT"
  --groups-per-game "$GROUPS"
  --focus-actions "$FOCUS"
  --games "$GAMES"
  --prefix-lengths "$PREFIXES"
  --run-id "$RUN_ID"
)
if [[ -n "$ACTION6_X" || -n "$ACTION6_Y" ]]; then
  if [[ -z "$ACTION6_X" || -z "$ACTION6_Y" ]]; then
    printf '%s\n' 'ACTION6 requires both X and Y payload coordinates.' >&2
    exit 2
  fi
  ARGS+=(--action6-x "$ACTION6_X" --action6-y "$ACTION6_Y")
fi
exec python3 tools/aop/counterfactual_collect.py "${ARGS[@]}"
