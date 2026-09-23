"""SICA safety primitives for controlled self-improvement."""

from .candidate import CandidateModification, CandidateRegistry, GateResult, SafetyGate
from .holdout import HoldoutGate, HoldoutResult
from .evolution_controls import (
    EvolutionController,
    PerformanceDecision,
    PerformanceMonitor,
    RollbackManager,
    SnapshotStore,
    VersionSnapshot,
)
from .rule_lifecycle import (
    ComplexityBudget,
    ComplexityProfile,
    Evidence,
    RuleRecord,
    RuleRegistry,
    RuleState,
    measure_complexity,
)
from .immutable_core import DEFAULT_IMMUTABLE_PATHS, ImmutablePathError, pre_commit_check
from .tabu_store import (
    TabuEntryError,
    TabuStore,
    WriterCapability,
)

__all__ = [
    "CandidateModification",
    "DEFAULT_IMMUTABLE_PATHS",
    "GateResult",
    "ImmutablePathError",
    "SafetyGate",
    "TabuEntryError",
    "TabuStore",
    "WriterCapability",
    "pre_commit_check",
]
