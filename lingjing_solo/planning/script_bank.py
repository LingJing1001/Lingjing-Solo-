"""Unified per-game offline script bank (ls20 / ar25 / …).

JSON layout (lingjing_solo/planning/data/<game>_scripts.json)::

    {
      "game_id": "ar25",
      "levels": {
        "0": ["ACTION3", ...],
        "1": [...]
      },
      "note": "..."
    }

Level key = levels_completed when the script should start (0 = L1).
Complex actions may be objects: {"action": "ACTION6", "x": 12, "y": 34}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

ActionSpec = Union[str, dict[str, Any]]

_DATA = Path(__file__).resolve().parent / "data"
_CACHE: dict[str, dict[str, list[ActionSpec]]] = {}


def scripts_path(game_id: str = "ls20") -> Path:
    gid = (game_id or "ls20").split("-")[0].lower()
    return _DATA / f"{gid}_scripts.json"


def _normalize_action(item: Any) -> Optional[ActionSpec]:
    if isinstance(item, str) and item.strip():
        return item.strip().upper()
    if isinstance(item, dict):
        name = str(item.get("action") or item.get("name") or "").strip().upper()
        if not name:
            return None
        out = {"action": name}
        for k in ("x", "y", "data"):
            if k in item:
                out[k] = item[k]
        return out
    return None


def load_level_scripts(game_id: str = "ls20", *, reload: bool = False) -> dict[str, list[ActionSpec]]:
    gid = (game_id or "ls20").split("-")[0].lower()
    if not reload and gid in _CACHE:
        return _CACHE[gid]
    path = scripts_path(gid)
    if not path.is_file():
        _CACHE[gid] = {}
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CACHE[gid] = {}
        return {}
    levels = data.get("levels") or {}
    out: dict[str, list[ActionSpec]] = {}
    for k, v in levels.items():
        if not isinstance(v, list) or not v:
            continue
        seq = [_normalize_action(a) for a in v]
        seq = [a for a in seq if a is not None]
        if seq:
            out[str(k)] = seq
    _CACHE[gid] = out
    return out


def script_for_level(
    levels_completed: int, game_id: str = "ls20"
) -> Optional[list[ActionSpec]]:
    """Return script for the *current* level index (= levels_completed)."""
    scripts = load_level_scripts(game_id)
    return scripts.get(str(int(levels_completed)))


def has_scripts(game_id: str) -> bool:
    return bool(load_level_scripts(game_id))


def list_scripted_games() -> list[str]:
    if not _DATA.is_dir():
        return []
    return sorted(p.stem.replace("_scripts", "") for p in _DATA.glob("*_scripts.json"))


def flatten_plan(game_id: str, max_level: Optional[int] = None) -> list[ActionSpec]:
    """Concatenate level scripts 0..N-1 into one plan (legacy LS20 route style)."""
    scripts = load_level_scripts(game_id)
    keys = sorted(scripts.keys(), key=lambda k: int(k))
    plan: list[ActionSpec] = []
    for k in keys:
        if max_level is not None and int(k) >= max_level:
            break
        plan.extend(scripts[k])
    return plan


class ScriptPlayer:
    """Consume per-level scripts; reload when levels_completed advances."""

    def __init__(self, game_id: str = "") -> None:
        self.game_id = (game_id or "").split("-")[0].lower()
        self._level = -1
        self._queue: list[ActionSpec] = []
        self._idx = 0
        self._armed = False

    def reset(self, game_id: Optional[str] = None) -> None:
        if game_id is not None:
            self.game_id = game_id.split("-")[0].lower()
        self._level = -1
        self._queue = []
        self._idx = 0
        self._armed = False

    def arm(self, levels_completed: int = 0) -> bool:
        """Load script for current level. Returns True if a script exists."""
        if not self.game_id:
            return False
        # Allow env override: LINGJING_SCRIPT_PLAN=ACTION1,ACTION2,...
        env_plan = os.getenv("LINGJING_SCRIPT_PLAN", "").strip()
        if env_plan and not self._armed:
            self._queue = [a.strip().upper() for a in env_plan.split(",") if a.strip()]
            self._idx = 0
            self._level = int(levels_completed)
            self._armed = True
            return bool(self._queue)

        lv = int(levels_completed)
        if self._armed and lv == self._level and self._idx < len(self._queue):
            return True
        script = script_for_level(lv, self.game_id)
        if not script:
            self._queue = []
            self._idx = 0
            self._level = lv
            self._armed = False
            return False
        self._queue = list(script)
        self._idx = 0
        self._level = lv
        self._armed = True
        return True

    def remaining(self) -> int:
        return max(0, len(self._queue) - self._idx)

    def next(self, levels_completed: int = 0) -> Optional[ActionSpec]:
        """Return next action spec, or None if exhausted / no script."""
        if not self.arm(levels_completed):
            return None
        if self._idx >= len(self._queue):
            # Level advanced mid-script? try reload
            if int(levels_completed) != self._level:
                if not self.arm(levels_completed):
                    return None
            else:
                return None
        if self._idx >= len(self._queue):
            return None
        item = self._queue[self._idx]
        self._idx += 1
        return item

    def peek_name(self) -> Optional[str]:
        if self._idx >= len(self._queue):
            return None
        item = self._queue[self._idx]
        if isinstance(item, dict):
            return str(item.get("action") or "")
        return str(item) if item else None
