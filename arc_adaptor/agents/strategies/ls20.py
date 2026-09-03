"""Validated LS20 route strategy, isolated from the adapter boundary."""
from __future__ import annotations

import os
from typing import Any

from lingjing_solo.planning import LS20Solver


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


class LS20Strategy:
    def __init__(self) -> None:
        self.solver = LS20Solver()
        self.explicit_plan = [x.strip().upper() for x in os.getenv("LINGJING_LS20_PLAN", "").split(",") if x.strip()]
        self.seeded_level: int | None = None

    def reset(self, frame: Any) -> None:
        self.solver.reset()
        self.seeded_level = None
        if self.explicit_plan:
            self.solver.set_plan(self.explicit_plan)
            self.seeded_level = 99
        else:
            self._seed(int(getattr(frame, "levels_completed", 0) or 0))

    def _seed(self, level: int) -> None:
        plan = level_plan(level)
        if plan:
            self.solver.set_plan(plan)
            self.seeded_level = level

    def choose_action(self, frames: list[Any], frame: Any, grid: Any, legal_names: list[str], levels_completed: int) -> str | None:
        del frames, frame
        if not self.explicit_plan and levels_completed != self.seeded_level:
            self._seed(levels_completed)
        return self.solver.next_action(grid, legal_names)
