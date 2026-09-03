"""Resolve a game id to a strategy without branching in the adapter."""
from __future__ import annotations

from typing import Any

from .generic import GenericStrategy
from .ls20 import LS20Strategy


class GameStrategyRegistry:
    def __init__(self, solo: Any) -> None:
        self._solo = solo
        self._generic = GenericStrategy(solo)
        self._ls20 = LS20Strategy()

    def resolve(self, game_id: str) -> Any:
        if game_id.lower().startswith("ls20"):
            return self._ls20
        return self._generic
