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

mkdir -p agents/templates agents/strategies tests/unit tools
cp "$SCRIPT_DIR/agents/templates/lingjing_solo_agent.py" agents/templates/lingjing_solo_agent.py
cp -R "$SCRIPT_DIR/agents/strategies/." agents/strategies/
cp "$SCRIPT_DIR/tests/test_lingjing_solo_agent.py" tests/unit/test_lingjing_solo_agent.py
cp "$SCRIPT_DIR/tests/test_action_recording.py" tests/unit/test_action_recording.py
cp "$SCRIPT_DIR/tests/test_r11l_probe.py" tests/unit/test_r11l_probe.py
# 统一 R3 plan 契约（§3.2 九字段）：只依赖 stdlib + editable 装的 lingjing_solo，
# 在 ARC 里跑它才算"契约在线上包内可用"，在本仓库跑只证明它自己能 import。
cp "$LINGJING_ROOT/tests/test_plan_contract.py" tests/unit/test_plan_contract.py
cp "$SCRIPT_DIR/tools/ls20_single_action_probe.py" tools/ls20_single_action_probe.py
cp "$SCRIPT_DIR/tools/r11l_single_action_probe.py" tools/r11l_single_action_probe.py

if (( WITH_RECORDING_PATCH )); then
  git apply --check --unidiff-zero "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  git apply --unidiff-zero "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  echo "recording_patch=APPLIED"
else
  echo "recording_patch=NOT_APPLIED (optional; use --with-recording-patch)"
fi

printf 'arc_root=%s\n' "$ARC_DIR"
printf 'arc_commit=%s\n' "$(git rev-parse HEAD)"
printf 'synced_files:\n'
printf '%s\n' \
  agents/templates/lingjing_solo_agent.py \
  agents/strategies/ \
  tests/unit/test_lingjing_solo_agent.py \
  tests/unit/test_action_recording.py \
  tests/unit/test_r11l_probe.py \
  tests/unit/test_plan_contract.py \
  tools/ls20_single_action_probe.py \
  tools/r11l_single_action_probe.py
# agents/__init__.py **不在**上面：规则 2 要求不覆盖 ARC 原生 exports，
# 策略注册是 ARC checkout 里的手工步骤（改了要单独 git diff 复核，别当成同步产物）。
printf 'agents/__init__.py=NOT_SYNCED (register strategies manually, see docs/ARC-AGI3-adapter-architecture.md)\n'

git diff --check
