"""Structured candidate modifications and fail-closed SICA safety gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .evolution_controls import EvolutionController, SnapshotStore
from .immutable_core import ImmutablePathError, pre_commit_check


@dataclass(frozen=True)
class CandidateModification:
    candidate_id: str
    target: str
    content: dict[str, Any]
    recursion_depth: int = 0
    affected_paths: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    evidence_count: int = 0
    distinct_games: int = 0
    complexity_delta: float = 0.0
    expected_gain: float = 0.0

    def __post_init__(self) -> None:
        if not self.candidate_id or self.target not in {"world_model", "rule_set", "policy"}:
            raise ValueError("candidate_id must be non-empty and target must be world_model/rule_set/policy")
        if not isinstance(self.content, dict):
            raise TypeError("candidate content must be an object")
        if self.recursion_depth < 0:
            raise ValueError("recursion_depth must be non-negative")
        if self.complexity_delta < 0 and self.expected_gain < 0:
            raise ValueError("negative complexity and negative expected gain are invalid together")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)


class CandidateRegistry:
    """Small in-process merge boundary for structured SICA candidates."""

    def __init__(self, gate: SafetyGate | None = None, *, controls: EvolutionController | None = None,
                 snapshot_store: SnapshotStore | None = None) -> None:
        self.gate = gate or SafetyGate()
        self.controls = controls or EvolutionController(max_meta_depth=self.gate.max_recursion_depth)
        self.snapshot_store = snapshot_store
        self.rejected_count = 0
        self.rejection_reasons: list[str] = []
        self._accepted: dict[str, CandidateModification] = {}

    def submit(self, candidate: CandidateModification, *, tick: int = 0,
               is_exploration: bool = False) -> CandidateModification:
        try:
            self.controls.admit(tick=tick, recursion_depth=candidate.recursion_depth,
                                is_exploration=is_exploration)
            self.gate.require(candidate)
            if candidate.candidate_id in self._accepted:
                raise ValueError(f"candidate already submitted: {candidate.candidate_id}")
            if self.snapshot_store is not None:
                self.snapshot_store.create(
                    candidate.candidate_id,
                    {"accepted": [asdict(item) for item in self._accepted.values()]},
                    metadata={"event": "before_candidate_merge", "candidate_id": candidate.candidate_id},
                )
            self._accepted[candidate.candidate_id] = candidate
            self.controls.commit(tick=tick, is_exploration=is_exploration)
            return candidate
        except (PermissionError, ValueError, KeyError, ImmutablePathError) as exc:
            self.rejected_count += 1
            self.rejection_reasons.append(str(exc))
            raise

    def get(self, candidate_id: str) -> CandidateModification | None:
        return self._accepted.get(candidate_id)

    def all(self) -> tuple[CandidateModification, ...]:
        return tuple(self._accepted.values())


class SafetyGate:
    """R4/R5/R9/R3 pre-merge checks; R2 holdout belongs to a later gate."""

    def __init__(self, *, max_recursion_depth: int = 2, min_evidence: int = 3,
                 min_distinct_games: int = 2, complexity_lambda: float = 0.1,
                 protected_paths: Iterable[str] | None = None) -> None:
        if max_recursion_depth < 0 or min_evidence < 1 or min_distinct_games < 1:
            raise ValueError("safety thresholds must be positive, except recursion depth")
        if complexity_lambda < 0:
            raise ValueError("complexity_lambda must be non-negative")
        self.max_recursion_depth = max_recursion_depth
        self.min_evidence = min_evidence
        self.min_distinct_games = min_distinct_games
        self.complexity_lambda = complexity_lambda
        self.protected_paths = tuple(protected_paths) if protected_paths is not None else None

    def evaluate(self, candidate: CandidateModification) -> GateResult:
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        try:
            pre_commit_check(candidate.affected_paths, protected_paths=self.protected_paths)
            checks["immutable_core"] = True
        except ImmutablePathError as exc:
            checks["immutable_core"] = False
            reasons.append(str(exc))

        checks["recursion_depth"] = candidate.recursion_depth <= self.max_recursion_depth
        if not checks["recursion_depth"]:
            reasons.append(f"recursion depth {candidate.recursion_depth} exceeds {self.max_recursion_depth}")

        checks["evidence_threshold"] = (
            candidate.evidence_count >= self.min_evidence
            and len(candidate.evidence_refs) >= self.min_evidence
        )
        if not checks["evidence_threshold"]:
            reasons.append(f"evidence requires at least {self.min_evidence} independent references")

        checks["cross_game_evidence"] = candidate.distinct_games >= self.min_distinct_games
        if not checks["cross_game_evidence"]:
            reasons.append(f"evidence requires at least {self.min_distinct_games} distinct games")

        required_gain = self.complexity_lambda * candidate.complexity_delta
        checks["complexity_ratio"] = candidate.expected_gain >= required_gain
        if not checks["complexity_ratio"]:
            reasons.append(
                f"expected gain {candidate.expected_gain} is below complexity cost {required_gain}"
            )
        return GateResult(not reasons, tuple(reasons), checks)

    def require(self, candidate: CandidateModification) -> CandidateModification:
        result = self.evaluate(candidate)
        if not result.passed:
            raise PermissionError("SICA safety gate rejected candidate: " + "; ".join(result.reasons))
        return candidate
