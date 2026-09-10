"""Resolve a game id to a strategy without branching in the adapter."""
from __future__ import annotations

from typing import Any

from .ar25 import AR25Strategy
from .generic import GenericStrategy
from .ls20 import LS20Strategy


class GameStrategyRegistry:
    def __init__(self, solo: Any) -> None:
        self._solo = solo
        self._generic = GenericStrategy(solo)
        self._ls20 = LS20Strategy()
        self._ar25 = AR25Strategy()

    def resolve(self, game_id: str) -> Any:
        gid = game_id.lower()
        if gid.startswith("ls20"):
            return self._ls20
        if gid.startswith("ar25"):
            return self._ar25
        return self._generic
