#!/usr/bin/env bash
set -euo pipefail

# Sync the versioned ARC reproduction bundle into an existing ARC checkout.
# Usage: bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh [ARC_DIR] [--check-only] [--with-recording-patch]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINGJING_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARC_DIR="${1:-$PWD}"
WITH_RECORDING_PATCH=0
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --with-recording-patch) WITH_RECORDING_PATCH=1 ;;
    --check-only) CHECK_ONLY=1 ;;
  esac
done
if [[ "${1:-}" == --* ]]; then ARC_DIR="$PWD"; fi

ARC_DIR="$(cd -- "$ARC_DIR" && pwd)"
cd "$ARC_DIR"

git rev-parse --show-toplevel >/dev/null
test -f main.py

mkdir -p agents/templates agents/strategies tests/unit tools
declare -a SOURCE_FILES=(
  agents/templates/lingjing_solo_agent.py
  agents/strategies
  tests/unit/test_lingjing_solo_agent.py
  tests/unit/test_action_recording.py
  tests/unit/test_r11l_probe.py
  tests/unit/test_plan_contract.py
  tests/unit/test_ar25_plan_contract.py
  tests/unit/test_ar25_recording.py
  tools/ls20_single_action_probe.py
  tools/r11l_single_action_probe.py
)

if (( ! CHECK_ONLY )); then
  cp "$SCRIPT_DIR/agents/templates/lingjing_solo_agent.py" agents/templates/lingjing_solo_agent.py
  cp -R "$SCRIPT_DIR/agents/strategies/." agents/strategies/
  cp "$SCRIPT_DIR/tests/test_lingjing_solo_agent.py" tests/unit/test_lingjing_solo_agent.py
  cp "$SCRIPT_DIR/tests/test_action_recording.py" tests/unit/test_action_recording.py
  cp "$SCRIPT_DIR/tests/test_r11l_probe.py" tests/unit/test_r11l_probe.py
  # 统一 R3 plan 契约（§3.2 九字段）：只依赖 stdlib + editable 装的 lingjing_solo，
  # 在 ARC 里跑它才算"契约在线上包内可用"，在本仓库跑只证明它自己能 import。
  cp "$LINGJING_ROOT/tests/test_plan_contract.py" tests/unit/test_plan_contract.py
  # AR25 运行器的 §3.2 出口回归（设计文档 §8.8）。在 ARC 里它是 skip 不是 assert。
  cp "$LINGJING_ROOT/tests/test_ar25_plan_contract.py" tests/unit/test_ar25_plan_contract.py
  # AR25 逐 tick 证据层（设计文档 §8 ③）。在 ARC 里是模块级 skip。
  cp "$LINGJING_ROOT/tests/test_ar25_recording.py" tests/unit/test_ar25_recording.py
  cp "$SCRIPT_DIR/tools/ls20_single_action_probe.py" tools/ls20_single_action_probe.py
  cp "$SCRIPT_DIR/tools/r11l_single_action_probe.py" tools/r11l_single_action_probe.py
fi

if (( WITH_RECORDING_PATCH )); then
  (( CHECK_ONLY )) && { echo "recording_patch=CHECK_ONLY"; exit 0; }
  git apply --check --unidiff-zero "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  git apply --unidiff-zero "$SCRIPT_DIR/patches/arc-agent-recording.patch"
  echo "recording_patch=APPLIED"
else
  echo "recording_patch=NOT_APPLIED (optional; use --with-recording-patch)"
fi

printf 'arc_root=%s\n' "$ARC_DIR"
printf 'arc_commit=%s\n' "$(git rev-parse HEAD)"
printf 'synced_files:\n'
for rel in "${SOURCE_FILES[@]}"; do
  if [[ -d "$rel" ]]; then
    while IFS= read -r source_file; do
      [[ "$source_file" == *__pycache__/* || "$source_file" == *.pyc ]] && continue
      file="${source_file#$SCRIPT_DIR/}"
      target_file="$ARC_DIR/$file"
      source_hash="$(sha256sum "$source_file" | cut -d' ' -f1)"
      target_hash="$(sha256sum "$target_file" | cut -d' ' -f1)"
      [[ "$source_hash" == "$target_hash" ]] || { echo "manifest_mismatch=$file" >&2; exit 1; }
      printf '%s  %s\n' "$target_hash" "$file"
    done < <(find "$SCRIPT_DIR/$rel" -type f -print | sort)
  else
    source_file="$SCRIPT_DIR/$rel"
    if [[ ! -f "$source_file" && "$rel" == tests/unit/* ]]; then
      source_file="$SCRIPT_DIR/tests/${rel##*/}"
    fi
    source_hash="$(sha256sum "$source_file" | cut -d' ' -f1)"
    target_hash="$(sha256sum "$rel" | cut -d' ' -f1)"
    [[ "$source_hash" == "$target_hash" ]] || { echo "manifest_mismatch=$rel" >&2; exit 1; }
    printf '%s  %s\n' "$target_hash" "$rel"
  fi
done

official_init_hash="$(sha256sum agents/__init__.py | cut -d' ' -f1)"
printf 'official_agents_init_sha256=%s\n' "$official_init_hash"
# agents/__init__.py 不在同步白名单中：不覆盖 ARC 原生 exports。
printf 'agents/__init__.py=NOT_SYNCED (register strategies manually, see docs/ARC-AGI3-adapter-architecture.md)\n'

git diff --check
