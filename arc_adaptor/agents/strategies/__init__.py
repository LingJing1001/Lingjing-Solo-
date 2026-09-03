"""Game strategy plugins for the Lingjing ARC adapter."""

from .ar25 import AR25Strategy
from .ar25 import level_plan as ar25_level_plan
from .base import GameStrategy
from .generic import GenericStrategy
from .ls20 import LS20Strategy, level_plan
from .registry import GameStrategyRegistry

__all__ = [
    "GameStrategy",
    "GenericStrategy",
    "LS20Strategy",
    "AR25Strategy",
    "GameStrategyRegistry",
    "level_plan",
    "ar25_level_plan",
]
