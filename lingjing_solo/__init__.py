"""Lingjing-Solo: 钱学森灵境引擎的 ARC-AGI-3 Solo 化身。

统一信息场 Φ · 泡壁面积律 · 版本化因果 · 人机结合（LLM 仅战略层）。
"""
from .core import SoloConfig, MANIFESTO, protocol_status_summary
from .agent import LingjingSoloAgent

__version__ = "0.5.0"
__all__ = ["SoloConfig", "LingjingSoloAgent", "MANIFESTO", "protocol_status_summary"]
