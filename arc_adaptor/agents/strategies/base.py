"""Strategy interface at the ARC adapter boundary."""
from __future__ import annotations

from typing import Any, Protocol


class GameStrategy(Protocol):
    """Per-game policy hook; the core Lingjing agent remains game-agnostic."""

    def reset(self, frame: Any) -> None: ...

    def choose_action(
        self,
        frames: list[Any],
        frame: Any,
        grid: Any,
        legal_names: list[str],
        levels_completed: int,
    ) -> str | None: ...
