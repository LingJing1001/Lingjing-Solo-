"""Lingjing-Solo: 钱学森灵境引擎的 ARC-AGI-3 Solo 化身。

统一信息场 Φ · 泡壁面积律 · 版本化因果 · 人机结合（LLM 仅战略层）。
"""
from .core import SoloConfig, MANIFESTO, protocol_status_summary
from .agent import LingjingSoloAgent

__version__ = "0.5.0"
__all__ = ["SoloConfig", "LingjingSoloAgent", "MANIFESTO", "protocol_status_summary"]

from .agents import Agent, AgentState, EvacuationAgent
from .arc_transition import TransitionDiagnostics, V14ArcTransition
from .engine import Engine, EngineConfig, TickRecord
from .scenarios import EvacuationScenario
from .core import Field, FieldConfig, Scenario, bubble_mask_spherical, build_bubble_laplacian_csr, laplacian_7point
from .core.writeback import AgentWriteback, validate_writeback

__all__ += [
    "Agent", "AgentState", "AgentWriteback", "Engine", "EngineConfig",
    "EvacuationAgent", "EvacuationScenario", "Field", "FieldConfig",
    "Scenario", "TickRecord", "TransitionDiagnostics", "V14ArcTransition",
    "bubble_mask_spherical", "build_bubble_laplacian_csr", "laplacian_7point",
    "validate_writeback",
]
