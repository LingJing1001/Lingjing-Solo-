"""Lingjing physical-field engine V14.2.

The package exposes the stable field, writeback, scenario, agent, and engine
contracts through a normal ``lingjing_solo.v14`` namespace.  Importing it has
no path or runtime side effects.
"""

from .agents import Agent, AgentState, EvacuationAgent
from .core import (
    Field,
    FieldConfig,
    Scenario,
    bubble_mask_spherical,
    build_bubble_laplacian_csr,
    laplacian_7point,
)
from .core.writeback import AgentWriteback, validate_writeback
from .engine import Engine, EngineConfig, TickRecord
from .scenarios import EvacuationScenario

__all__ = [
    "Agent",
    "AgentState",
    "AgentWriteback",
    "Engine",
    "EngineConfig",
    "EvacuationAgent",
    "EvacuationScenario",
    "Field",
    "FieldConfig",
    "Scenario",
    "TickRecord",
    "bubble_mask_spherical",
    "build_bubble_laplacian_csr",
    "laplacian_7point",
    "validate_writeback",
]
