"""Unified per-game offline script bank (ls20 / ar25 / …).

Loads from planning/data/<game>_scripts.json when present.
**Also ships EMBEDDED_SCRIPTS** so Kaggle notebooks still score if JSON
files fail to materialize next to this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

ActionSpec = Union[str, dict[str, Any]]

_DATA = Path(__file__).resolve().parent / "data"
_CACHE: dict[str, dict[str, list[ActionSpec]]] = {}

# Hardcoded fallback — keep in sync with planning/data/*_scripts.json
# and agent/my_agent.py HARDCODED (ls20 L1–L7 + ar25 L1–L8 @a02287b).
_LS20_LEVEL_ACTIONS: dict[int, list[int]] = {
    0: [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1],
    1: [1, 4, 1, 1, 1, 1, 1, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 3, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2],
    2: [1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 1, 2, 1, 4, 2],
    3: [3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 3, 1, 2, 3, 3, 1, 1, 1, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 1, 3, 3, 3],
    4: [1, 4, 1, 1, 3, 4, 3, 3, 3, 4, 3, 4, 3, 4, 4, 2, 2, 3, 3, 3, 1, 3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 4, 1, 4, 4, 2, 2, 2, 1],
    5: [1, 3, 1, 3, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 4, 2, 2, 1, 1, 3, 1, 2, 3, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 1, 3, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 1, 1, 4, 2, 2, 2, 2, 2],
    6: [1, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 1, 2, 4, 2, 1, 4, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 1, 1, 1, 4, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 1, 4, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 2],
}
_AR25_LEVEL_ACTIONS: dict[int, list[int]] = {
    0: [3] * 5 + [2] * 10,
    1: [3] * 9 + [5] + [3] * 14 + [2] * 8,
    2: [1] * 7 + [5] + [4] * 7 + [2] * 7 + [5] + [3] * 12 + [2] * 5,
    3: [2] * 6 + [5] + [4] * 7 + [5] + [4] * 7,
    4: [2] * 4 + [5] + [4] * 5 + [5] + [3] * 10 + [1] * 7,
    5: [2] * 11 + [5] + [3] + [5] + [3] * 15 + [2] * 4 + [5] + [3] * 7 + [2] * 12,
    6: [2] * 2 + [5] + [4] * 9 + [5] + [3] * 10 + [1] * 6 + [5] + [4] * 3 + [1] * 15,
    7: [2] * 6 + [5] + [4] * 9 + [5] + [3] * 9 + [1] * 7 + [5] + [4] * 9 + [1] * 4,
}
EMBEDDED_SCRIPTS: dict[str, dict[str, list[str]]] = {
    "ls20": {
        str(i): [f"ACTION{n}" for n in seq] for i, seq in _LS20_LEVEL_ACTIONS.items()
    },
    "ar25": {
        str(i): [f"ACTION{n}" for n in seq] for i, seq in _AR25_LEVEL_ACTIONS.items()
    },
}


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


def _from_levels_dict(levels: dict) -> dict[str, list[ActionSpec]]:
    out: dict[str, list[ActionSpec]] = {}
    for k, v in (levels or {}).items():
        if not isinstance(v, list) or not v:
            continue
        seq = [_normalize_action(a) for a in v]
        seq = [a for a in seq if a is not None]
        if seq:
            out[str(k)] = seq
    return out


def load_level_scripts(game_id: str = "ls20", *, reload: bool = False) -> dict[str, list[ActionSpec]]:
    gid = (game_id or "ls20").split("-")[0].lower()
    if not reload and gid in _CACHE:
        return _CACHE[gid]

    out: dict[str, list[ActionSpec]] = {}
    path = scripts_path(gid)
    source = "missing"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out = _from_levels_dict(data.get("levels") or {})
            source = f"file:{path}"
        except Exception as exc:  # noqa: BLE001
            source = f"file-error:{exc}"

    if not out and gid in EMBEDDED_SCRIPTS:
        out = _from_levels_dict(EMBEDDED_SCRIPTS[gid])
        source = "embedded"

    if os.getenv("LINGJING_SCRIPT_DEBUG", "").strip():
        print(
            f"[script_bank] game={gid} levels={list(out)} "
            f"n={sum(len(v) for v in out.values())} source={source} "
            f"data_dir={_DATA} exists={_DATA.is_dir()}",
            flush=True,
        )

    _CACHE[gid] = out
    return out


def script_for_level(
    levels_completed: int, game_id: str = "ls20"
) -> Optional[list[ActionSpec]]:
    scripts = load_level_scripts(game_id)
    return scripts.get(str(int(levels_completed)))


def has_scripts(game_id: str) -> bool:
    return bool(load_level_scripts(game_id))


def list_scripted_games() -> list[str]:
    games = set(EMBEDDED_SCRIPTS)
    if _DATA.is_dir():
        games.update(p.stem.replace("_scripts", "") for p in _DATA.glob("*_scripts.json"))
    return sorted(games)


def flatten_plan(game_id: str, max_level: Optional[int] = None) -> list[ActionSpec]:
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
        if not self.game_id:
            return False
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
        if not self.arm(levels_completed):
            return None
        if self._idx >= len(self._queue):
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
