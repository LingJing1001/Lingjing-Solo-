#!/usr/bin/env bash
set -euo pipefail

# Sync the versioned ARC reproduction bundle into an existing ARC checkout.
# Usage: bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh [ARC_DIR] [--with-recording-patch]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINGJING_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARC_DIR="${1:-$PWD}"
WITH_RECORDING_PATCH=0

if [[ "${2:-}" == "--with-recording-patch" || "${1:-}" == "--with-recording-patch" ]]; then
  if [[ "${1:-}" == "--with-recording-patch" ]]; then ARC_DIR="$PWD"; fi
  WITH_RECORDING_PATCH=1
fi

ARC_DIR="$(cd -- "$ARC_DIR" && pwd)"
cd "$ARC_DIR"

git rev-parse --show-toplevel >/dev/null
test -f main.py

mkdir -p agents/templates tests/unit tools
cp "$SCRIPT_DIR/agents/templates/lingjing_solo_agent.py" agents/templates/lingjing_solo_agent.py
cp "$SCRIPT_DIR/agents/__init__.py" agents/__init__.py
cp "$SCRIPT_DIR/tests/test_lingjing_solo_agent.py" tests/unit/test_lingjing_solo_agent.py
cp "$SCRIPT_DIR/tests/test_action_recording.py" tests/unit/test_action_recording.py
cp "$SCRIPT_DIR/tools/ls20_single_action_probe.py" tools/ls20_single_action_probe.py

if (( WITH_RECORDING_PATCH )); then
  git apply --check "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  git apply "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  echo "recording_patch=APPLIED"
else
  echo "recording_patch=NOT_APPLIED (optional; use --with-recording-patch)"
fi

printf 'arc_root=%s\n' "$ARC_DIR"
printf 'arc_commit=%s\n' "$(git rev-parse HEAD)"
printf 'synced_files:\n'
printf '%s\n' \
  agents/templates/lingjing_solo_agent.py \
  agents/__init__.py \
  tests/unit/test_lingjing_solo_agent.py \
  tests/unit/test_action_recording.py \
  tools/ls20_single_action_probe.py

git diff --check
