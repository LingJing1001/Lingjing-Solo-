"""Lingjing-Solo: A World-Model-Field Agent for Interactive Reasoning Benchmarks.

重构自"灵境引擎"的统一信息场思想：
    多 Agent 共享介质  →  单 Agent 内部世界模型场 (Φ)
    泡壁局部高精度      →  ROI 驱动差分感知
    版本化因果链        →  转移表驱动的假设验证
"""
from .core import SoloConfig
from .agent import LingjingSoloAgent

__version__ = "0.1.0"
__all__ = ["SoloConfig", "LingjingSoloAgent"]
