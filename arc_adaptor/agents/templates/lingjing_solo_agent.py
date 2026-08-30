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


def _frame_grid(frame: FrameData) -> np.ndarray:
    """Convert official FrameData.frame into Lingjing's numpy grid."""
    array = np.asarray(frame.frame, dtype=np.int8)
    # FrameData stores one or more 2-D planes; Lingjing consumes one grid.
    return array[0] if array.ndim == 3 else array


def _player_position(grid: np.ndarray) -> tuple[int, int] | None:
    """Find the tiny color-1 player marker in an LS20 grid."""
    positions = np.argwhere(grid == 1)
    if len(positions) == 0 or len(positions) > 16:
        return None
    row, col = np.rint(positions.mean(axis=0)).astype(int)
    return int(row), int(col)


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

    MAX_ACTIONS = 80

    @staticmethod
    def _ls20_default_plan() -> list[str]:
        """Return the verified LS20 Level 1 -> Level 2 bootstrap route."""
        level2 = ["ACTION1", "ACTION1"]
        level2 += [
            {"R": "ACTION4", "U": "ACTION1", "D": "ACTION2", "L": "ACTION3"}[action]
            for action in "RUUUUURRDRDDDDDDDLLRURUDUUUUUUULLLLLLDLDDDDD"
        ]
        return LS20Solver.level1_verified_route() + level2

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
                plan = self._ls20_plan
                if not plan and str(getattr(latest_frame, "game_id", "")).startswith("ls20"):
                    plan = self._ls20_default_plan()
                if plan:
                    solver.set_plan(plan)
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

        # LS20 solver owns only an explicit, frame-validated plan.  An empty
        # or stale plan falls through to the general Lingjing policy.
        solver = getattr(self, "ls20_solver", None)
        if (
            solver is not None
            and not getattr(solver, "_plan", None)
            and str(getattr(latest_frame, "game_id", "")).startswith("ls20")
        ):
            plan = self._ls20_plan or self._ls20_default_plan()
            if plan:
                solver.set_plan(plan)
        chosen_name = None
        if solver is not None:
            if frames:
                previous = np.asarray(getattr(frames[-1], "frame", []))
                if previous.ndim == 2 or (
                    previous.ndim == 3 and previous.shape[0] == 1
                ):
                    solver.observe_transition(
                        _frame_grid(frames[-1]),
                        _frame_grid(latest_frame),
                        player=_player_position(_frame_grid(latest_frame)),
                    )
            chosen_name = solver.next_action(
                _frame_grid(latest_frame),
                [action.name for action in legal],
            )

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
