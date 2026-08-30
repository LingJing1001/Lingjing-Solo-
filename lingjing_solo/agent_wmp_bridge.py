"""Optional WMP/Planner bridge (v10) for ARC action suggestions.

Enabled when LINGJING_USE_WMP=1. Builds a coarse SymbolTable from color blobs
and asks v10 Planner for a short plan. Falls back silently if v10 is missing
or no useful suggestion is available.

Action map: up/down/left/right ↔ ACTION1/2/3/4
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_TEAM = Path(__file__).resolve().parents[2]
_V10 = _TEAM / "lingjing_solo_v10"
if _V10.is_dir() and str(_V10) not in sys.path:
    sys.path.append(str(_V10))

_DIR_TO_ACTION = {
    "up": "ACTION1",
    "down": "ACTION2",
    "left": "ACTION3",
    "right": "ACTION4",
}


def _blobs(grid: np.ndarray, max_objs: int = 12) -> dict:
    """Very coarse connected-component symbols (color > 0)."""
    from collections import deque

    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    objs: dict = {}
    oid = 0
    for y in range(h):
        for x in range(w):
            c = int(grid[y, x])
            if c <= 0 or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            cells = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == c:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            role = "avatar" if c == 1 and len(cells) <= 16 else ("goal" if c in (9, 10, 11, 12) else "block")
            objs[f"o{oid}"] = {
                "x": int(sum(xs) / len(xs)),
                "y": int(sum(ys) / len(ys)),
                "color": c,
                "role": role,
                "size": len(cells),
            }
            oid += 1
            if oid >= max_objs:
                return objs
    return objs


class WmpBridge:
    """Lazy-load v10 Planner; suggest one ACTION* name or None."""

    def __init__(self) -> None:
        self._planner = None
        self._ready = False
        self._init_err: Optional[str] = None
        try:
            from lingjing_solo.planning.planner import Planner  # type: ignore

            self._planner = Planner(max_depth=4, adaptive_depth=True)
            self._ready = True
        except Exception as exc:  # noqa: BLE001
            self._init_err = str(exc)

    def reset(self) -> None:
        if self._planner is not None and hasattr(self._planner, "reset"):
            try:
                self._planner.reset()
            except Exception:  # noqa: BLE001
                pass

    def suggest(
        self,
        grid: np.ndarray,
        legal: list[str],
        levels_completed: int = 0,
    ) -> Optional[str]:
        if not self._ready or self._planner is None:
            return None
        objs = _blobs(np.asarray(grid))
        if not objs:
            return None
        avatar = next((k for k, v in objs.items() if v.get("role") == "avatar"), None)
        state = {"objects": objs, "avatar_id": avatar, "extras": {"level": levels_completed}}

        def goal_eval(s):
            # Prefer moving avatar toward any goal-like blob.
            oo = s.get("objects") or {}
            aid = s.get("avatar_id")
            if not aid or aid not in oo:
                return 0.0
            ax, ay = oo[aid]["x"], oo[aid]["y"]
            goals = [o for o in oo.values() if o.get("role") == "goal"]
            if not goals:
                return 0.0
            g = goals[0]
            return -abs(ax - g["x"]) - abs(ay - g["y"])

        try:
            self._planner.goal_evaluator = goal_eval
            plan = self._planner.plan(state, budget_hint=4)
        except Exception:  # noqa: BLE001
            return None
        if not plan:
            return None
        # plan may be list of dirs or single dir
        first = plan[0] if isinstance(plan, (list, tuple)) else plan
        name = _DIR_TO_ACTION.get(str(first).lower())
        if name and name in legal:
            return name
        return None
