"""游戏状态模拟器模块。

移植自 C:\\newtask-pi\\ar25_solver\\simulator.py。
提供不依赖官方引擎的纯 Python 逆向实现。
"""
from .ar25_simulator import (
    AR25Simulator,
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_SWITCH,
    AXIS_TAG, AXIS_TAG_HORIZONTAL, AXIS_TAG_VERTICAL,
    IMMOVABLE_TAG, MOVABLE_TAG,
    BOARD_SIZE,
)

__all__ = [
    "AR25Simulator",
    "ACT_UP", "ACT_DOWN", "ACT_LEFT", "ACT_RIGHT", "ACT_SWITCH",
    "AXIS_TAG", "AXIS_TAG_HORIZONTAL", "AXIS_TAG_VERTICAL",
    "IMMOVABLE_TAG", "MOVABLE_TAG",
    "BOARD_SIZE",
]
