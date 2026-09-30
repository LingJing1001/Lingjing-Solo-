#!/usr/bin/env bash
set -u
ROOT=/srv/agent-platform/projects/ARC-AGI-3-Agents
OUT=/srv/agent-platform/projects/Lingjing-Solo-/state/r11l-level1-background-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$(dirname "$OUT")"
exec > >(tee -a "$OUT") 2>&1
printf 'CHECKPOINT start utc=%s pid=%s\n' "$(date -u +%FT%TZ)" "$$"
cd "$ROOT" || exit 2
printf 'CHECKPOINT branch='; git branch --show-current
printf 'CHECKPOINT status='; git status --short | head -20
printf 'CHECKPOINT targeted-tests-start utc=%s\n' "$(date -u +%FT%TZ)"
uv run pytest -q tests/unit/test_lingjing_solo_agent.py tests/unit/test_r11l_probe.py
rc=$?
printf 'CHECKPOINT targeted-tests-end utc=%s rc=%s\n' "$(date -u +%FT%TZ)" "$rc"
printf 'HEARTBEAT probe-start utc=%s\n' "$(date -u +%FT%TZ)"
if [ -f /tmp/r11l_route_probe.py ]; then
  uv run python /tmp/r11l_route_probe.py
  prc=$?
else
  printf 'probe-missing\n'
  prc=3
fi
printf 'CHECKPOINT probe-end utc=%s rc=%s\n' "$(date -u +%FT%TZ)" "$prc"
printf 'COMPLETION utc=%s tests_rc=%s probe_rc=%s report=%s\n' "$(date -u +%FT%TZ)" "$rc" "$prc" "$OUT"
[ "$rc" -eq 0 ] && [ "$prc" -eq 0 ]
