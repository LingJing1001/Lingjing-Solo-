"""Official ARC-AGI-3 boundary adapter for Lingjing-Solo."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState
from lingjing_solo import LingjingSoloAgent

from ..agent import Agent
from ..strategies import GameStrategyRegistry, level_plan


def _ls20_level_plan(level_index: int) -> list[str]:
    """Compatibility export; route data is owned by ``strategies/ls20.py``."""
    return level_plan(level_index)


def _frame_grid(frame: FrameData) -> np.ndarray:
    """Convert official FrameData.frame into Lingjing's numpy grid."""
    array = np.asarray(frame.frame, dtype=np.int8)
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
        self.strategies = GameStrategyRegistry(self.solo)
        self._strategy = None
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

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        game_id = str(getattr(latest_frame, "game_id", ""))
        strategy = self.strategies.resolve(game_id)
        if latest_frame.state is GameState.NOT_PLAYED:
            reset = getattr(self.solo, "reset", None)
            if callable(reset):
                reset()
            strategy.reset(latest_frame)
            self._strategy = strategy
            self._experiment_index = 0
            return GameAction.RESET

        legal = _available_actions(latest_frame)
        if not legal:
            return GameAction.RESET

        observe = getattr(self.solo, "observe", None)
        grid = _frame_grid(latest_frame)
        if callable(observe):
            observe(grid, state=latest_frame.state, levels_completed=getattr(latest_frame, "levels_completed", None))

        if self._strategy is not strategy:
            self._strategy = strategy
            strategy.reset(latest_frame)

        legal_names = [action.name for action in legal]
        chosen_name = strategy.choose_action(
            frames, latest_frame, grid, legal_names,
            int(getattr(latest_frame, "levels_completed", 0) or 0),
        )

        if self._experiment_index < len(self._experiment_actions):
            candidate = self._experiment_actions[self._experiment_index]
            self._experiment_index += 1
            if candidate in set(legal_names):
                chosen_name = candidate

        by_name = {action.name: action for action in legal}
        chosen = by_name.get(chosen_name) or legal[0]
        if chosen.is_complex():
            chosen.set_data({"x": 0, "y": 0})
        chosen.reasoning = {"source": "lingjing-solo", "strategy": type(strategy).__name__, "abstract_action": chosen.name}
        return chosen
