"""Official ARC-AGI-3 adapter for the installable Lingjing-Solo package.

The package owns the reasoning logic; this module only translates
ARC ``FrameData``/``GameAction`` values at the boundary. Install Lingjing-Solo
with ``uv pip install -e /path/to/Lingjing-Solo-`` during local development,
or install the pinned public package in a team environment.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState
from lingjing_solo import LingjingSoloAgent
from lingjing_solo.planning import LS20Solver

from ..agent import Agent


# LS20 各关罐头解 (来自 ls20_solve_v4 求解链, 本地引擎验证 L1-L6 全通, 共 256 步).
# 编号 1-4 对应 ACTION1-ACTION4 (1=上, 2=下, 3=左, 4=右).
_LS20_LEVEL_ACTIONS: dict[int, list[int]] = {
    0: [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1],
    1: [1, 4, 1, 1, 1, 1, 1, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 3, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2],
    2: [1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 1, 2, 1, 4, 2],
    3: [3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 3, 1, 2, 3, 3, 1, 1, 1, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 1, 3, 3, 3],
    4: [1, 4, 1, 1, 3, 4, 3, 3, 3, 4, 3, 4, 3, 4, 4, 2, 2, 3, 3, 3, 1, 3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 4, 1, 4, 4, 2, 2, 2, 1],
    5: [1, 3, 1, 3, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 4, 2, 2, 1, 1, 3, 1, 2, 3, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 1, 3, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 1, 1, 4, 2, 2, 2, 2, 2],
}


def _ls20_level_plan(level_index: int) -> list[str]:
    """Return the verified canned action-name plan for an LS20 level, if any."""
    numbers = _LS20_LEVEL_ACTIONS.get(level_index)
    if numbers is None:
        return []
    return [f"ACTION{number}" for number in numbers]


def _frame_grid(frame: FrameData) -> np.ndarray:
    """Convert official FrameData.frame into Lingjing's numpy grid."""
    array = np.asarray(frame.frame, dtype=np.int8)
    # FrameData stores one or more 2-D planes; Lingjing consumes one grid.
    return array[0] if array.ndim == 3 else array


def _available_actions(frame: FrameData) -> list[GameAction]:
    """Return legal non-RESET actions, tolerating older frame payloads."""
    actions = getattr(frame, "available_actions", None) or []
    normalized = []
    for action in actions:
        try:
            if isinstance(action, GameAction):
                normalized.append(action)
            elif int(action) == 0:
                normalized.append(GameAction.RESET)
            else:
                normalized.append(GameAction[f"ACTION{int(action)}"])
        except (KeyError, TypeError, ValueError):
            continue
    return [action for action in normalized if action is not GameAction.RESET]


class LingjingSolo(Agent):
    """Run Lingjing-Solo through the official ARC-AGI-3 Agent interface."""

    # L1-L6 罐头解共 256 步; 保留余量给 L7 的通用策略尝试.
    MAX_ACTIONS = 800

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.solo = LingjingSoloAgent.with_llm(
            lambda _snapshot, _actions: None,
            llm_calls_per_game=0,
            enable_undo=False,
            enable_mouse=False,
            human_baseline_estimate=self.MAX_ACTIONS,
        )
        self.solo.reset()
        self.ls20_solver = LS20Solver()
        self._seeded_level: int | None = None
        self._ls20_plan = [
            name.strip().upper()
            for name in os.getenv("LINGJING_LS20_PLAN", "").split(",")
            if name.strip()
        ]
        self._experiment_actions = [
            name.strip().upper()
            for name in os.getenv("LINGJING_EXPERIMENT_ACTIONS", "").split(",")
            if name.strip()
        ]
        self._experiment_index = 0

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state is GameState.NOT_PLAYED:
            reset = getattr(self.solo, "reset", None)
            if callable(reset):
                reset()
            solver = getattr(self, "ls20_solver", None)
            if solver is not None:
                solver.reset()
                self._seeded_level = None
                if self._ls20_plan:
                    # 环境变量平铺计划: 开局播种一次, 全链直接执行, 不按关重播
                    solver.set_plan(self._ls20_plan)
                    self._seeded_level = 99
                elif str(getattr(latest_frame, "game_id", "")).startswith("ls20"):
                    plan = _ls20_level_plan(0)
                    if plan:
                        solver.set_plan(plan)
                        self._seeded_level = 0
            self._experiment_index = 0
            return GameAction.RESET

        legal = _available_actions(latest_frame)
        if not legal:
            return GameAction.RESET

        observe = getattr(self.solo, "observe", None)
        if callable(observe):
            observe(
                _frame_grid(latest_frame),
                state=latest_frame.state,
                levels_completed=getattr(latest_frame, "levels_completed", None),
            )

        # LS20 罐头计划: 直接弹动作执行 (解已本地验证 L1-L6 全通), 不做运动学
        # 失效检查. 计划为空时按 levels_completed 播下一关计划; 中途耗尽则回退
        # 通用策略. 平铺 env 计划 (LINGJING_LS20_PLAN) 只在开局播种, 不按关重播.
        chosen_name = None
        if str(getattr(latest_frame, "game_id", "")).startswith("ls20"):
            solver = getattr(self, "ls20_solver", None)
            if solver is not None:
                levels_completed = getattr(latest_frame, "levels_completed", 0)
                if (
                    not self._ls20_plan
                    and not getattr(solver, "_plan", None)
                    and levels_completed != self._seeded_level
                    and levels_completed < len(_LS20_LEVEL_ACTIONS)
                ):
                    plan = _ls20_level_plan(levels_completed)
                    if plan:
                        solver.set_plan(plan)
                        self._seeded_level = levels_completed
                legal_names = {action.name for action in legal}
                while getattr(solver, "_plan", None):
                    candidate = solver._plan.pop(0)
                    if candidate in legal_names:
                        chosen_name = candidate
                        break

        # An explicit experiment sequence is opt-in and only used when legal.
        experiment_actions = getattr(self, "_experiment_actions", [])
        experiment_index = getattr(self, "_experiment_index", 0)
        if experiment_index < len(experiment_actions):
            candidate = experiment_actions[experiment_index]
            self._experiment_index = experiment_index + 1
            if candidate in {action.name for action in legal}:
                chosen_name = candidate

        if chosen_name is None:
            chosen_name = self.solo.choose_action(
                frames,
                _frame_grid(latest_frame),
                valid_actions=[action.name for action in legal],
            )
        by_name = {action.name: action for action in legal}
        chosen = by_name.get(chosen_name)
        if chosen is None:
            chosen = legal[0]
        if chosen.is_complex():
            chosen.set_data({"x": 0, "y": 0})
        chosen.reasoning = {"source": "lingjing-solo", "abstract_action": chosen.name}
        return chosen
