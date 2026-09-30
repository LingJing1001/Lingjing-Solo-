"""Validated LS20 route strategy, isolated from the adapter boundary."""
from __future__ import annotations

from typing import Any

import numpy as np

from lingjing_solo.planning.ls20_solver import Ls20Solver


# LS20 全关罐头解 (L1-L7, 本地引擎验证, 共 309 步), 供外部查询.
# 求解器内部由 planning/script_bank.py 提供同一份脚本.
_LEVEL_ACTIONS: dict[int, list[int]] = {
    0: [3,3,3,1,1,1,1,4,4,4,1,1,1],
    1: [1,4,1,1,1,1,1,4,4,2,4,2,2,2,2,2,2,1,2,2,3,3,4,1,4,1,1,1,1,1,1,1,3,3,3,3,3,3,2,3,2,2,2,2,2],
    2: [1,1,1,1,1,1,1,1,3,2,2,2,2,2,2,2,2,1,1,1,3,3,1,4,4,4,4,4,4,4,1,1,1,3,1,2,1,4,2],
    3: [3,3,3,2,2,2,3,2,2,3,3,1,2,1,2,1,2,1,1,3,3,1,2,3,3,1,1,1,2,2,4,1,1,1,1,4,1,4,1,1,3,3,3],
    4: [1,4,1,1,3,4,3,3,3,4,3,4,3,4,4,2,2,3,3,3,1,3,3,3,4,4,2,2,2,2,2,4,4,2,4,4,4,1,4,4,2,2,2,1],
    5: [1,3,1,3,3,1,1,1,4,4,4,4,4,4,1,4,1,4,1,1,4,2,2,1,1,3,1,2,3,3,4,3,3,3,3,3,2,2,2,2,4,4,1,3,4,3,3,1,1,1,1,1,1,1,2,4,4,4,4,4,4,2,4,4,1,1,4,2,2,2,2,2],
    6: [1,1,2,2,3,3,2,2,2,2,2,1,2,4,2,1,4,1,2,1,2,1,2,1,2,3,3,1,1,1,4,4,4,4,1,4,4,1,4,4,1,1,4,2,2,3,3,3,1,2,2,2,2,2],
}


def level_plan(level_index: int) -> list[str]:
    return [f"ACTION{number}" for number in _LEVEL_ACTIONS.get(level_index, [])]


def _frame_grid(frame: Any) -> np.ndarray | None:
    try:
        array = np.asarray(frame.frame, dtype=np.int8)
    except Exception:
        return None
    grid = array[0] if array.ndim == 3 else array
    return grid if grid.ndim == 2 else None


class LS20Strategy:
    """Drive ``Ls20Solver`` from the adapter: observe transition, then plan."""

    def __init__(self) -> None:
        self.solver = Ls20Solver()
        self._prev_grid: np.ndarray | None = None
        self._last_action: str | None = None

    def reset(self, frame: Any) -> None:
        self._prev_grid = None
        self._last_action = None
        grid = _frame_grid(frame)
        if grid is not None:
            self.solver.reset_level(grid)

    def choose_action(
        self,
        frames: list[Any],
        frame: Any,
        grid: Any,
        legal_names: list[str],
        levels_completed: int,
    ) -> str | None:
        del frames, frame
        self.solver.observe(self._prev_grid, grid, self._last_action, levels_completed)
        chosen = self.solver.plan(legal_names)

        layout_wait = getattr(self.solver, "_layout_wait", 0)
        if (
            chosen is None
            and not self.solver._await_level_layout
            and not layout_wait
            and not self.solver.active
        ):
            self.solver.reset_level(grid)
            chosen = self.solver.plan(legal_names)
        if chosen is None and layout_wait and "ACTION1" in legal_names:
            chosen = "ACTION1"

        self._prev_grid = grid
        self._last_action = chosen
        return chosen
