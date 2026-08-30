"""Lingjing-Solo: 单 Agent 世界模型场框架（ARC-AGI-3）。"""
__version__ = "1.0.0"

from .world_model.symbols import SymbolTable, GameObject, Predicate
from .world_model.relations import (
    RelationGraph, RelationalFact, Relation,
    build_relation_graph, neighbors, invert_direction,
)
from .world_model.induction import RelationalInducer, TransitionEvidence
from .world_model.codegen import CodeGenerator, DynamicsProgram, FakeLLM, UnsafeCodeError
from .world_model.program import WorldModelProgram, WMPEvidence
from .planning import Planner, make_box_goal_evaluator, state_key
from .telemetry import Telemetry
from .agent import LingjingAgent, GameAction
