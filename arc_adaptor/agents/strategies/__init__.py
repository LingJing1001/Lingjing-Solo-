"""Game strategy plugins for the Lingjing ARC adapter."""

from .base import GameStrategy
from .generic import GenericStrategy
from .ls20 import LS20Strategy, level_plan
from .registry import GameStrategyRegistry

__all__ = [
    "GameStrategy",
    "GenericStrategy",
    "LS20Strategy",
    "GameStrategyRegistry",
    "level_plan",
]
