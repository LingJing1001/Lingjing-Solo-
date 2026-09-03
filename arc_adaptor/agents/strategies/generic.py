"""Default strategy for games without a validated route."""
from __future__ import annotations

from typing import Any


class GenericStrategy:
    def __init__(self, solo: Any) -> None:
        self.solo = solo

    def reset(self, frame: Any) -> None:
        del frame
        self.solo.reset()

    def choose_action(
        self,
        frames: list[Any],
        frame: Any,
        grid: Any,
        legal_names: list[str],
        levels_completed: int,
    ) -> str | None:
        del frame, levels_completed
        return self.solo.choose_action(frames, grid, valid_actions=legal_names)
