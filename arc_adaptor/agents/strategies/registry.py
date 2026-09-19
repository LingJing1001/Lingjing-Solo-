"""Resolve a game id to a strategy without branching in the adapter."""
from __future__ import annotations

from typing import Any

from .ar25 import AR25Strategy
from .generic import GenericStrategy
from .ls20 import LS20Strategy
from .r11l import R11LStrategy


class GameStrategyRegistry:
    def __init__(self, solo: Any) -> None:
        self._solo = solo
        self._generic = GenericStrategy(solo)
        self._ls20 = LS20Strategy()
        self._ar25 = AR25Strategy()
        self._r11l = R11LStrategy()

    def resolve(self, game_id: str) -> Any:
        normalized = game_id.lower()
        if normalized.startswith("ls20"):
            return self._ls20
        if normalized.startswith("ar25"):
            return self._ar25
        if normalized.startswith("r11l"):
            return self._r11l
        return self._generic
