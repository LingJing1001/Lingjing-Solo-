"""SICA phase-four R10 isolation, cross-family validation and monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


class RuleScope(StrEnum):
    GENERAL = "general"
    SPECIALIZED = "specialized"


@dataclass(frozen=True)
class CrossGameValidation:
    passed: bool
    rule_id: str
    scope: RuleScope
    families: tuple[str, ...]
    reason: str


class CrossGameValidator:
    """R10 boundary: general rules must transfer; specialized rules cannot leak."""

    def validate(self, *, rule_id: str, scope: RuleScope | str,
                 declared_family: str | None, observed_families: Iterable[str],
                 minimum_families: int = 2) -> CrossGameValidation:
        scope = RuleScope(scope)
        families = tuple(sorted({str(item) for item in observed_families if item}))
        if minimum_families < 1:
            raise ValueError("minimum_families must be positive")
        if scope is RuleScope.GENERAL:
            passed = declared_family is None and len(families) >= minimum_families
            reason = ("general rule validated across game families"
                      if passed else "general rule requires no family binding and cross-family evidence")
        else:
            passed = bool(declared_family) and all(item == declared_family for item in families)
            reason = ("specialized rule contained to declared family"
                      if passed else "specialized rule has missing or cross-family evidence")
        return CrossGameValidation(passed, rule_id, scope, families, reason)


@dataclass(frozen=True)
class MonitorSnapshot:
    tick: int
    potential: float
    performance: float
    active_rules: int
    hypothesis_rules: int
    archived_rules: int
    eliminated_rules: int
    complexity: float
    rejected_candidates: int
    rollback_count: int
    rule_categories: dict[str, int]
    rule_scopes: dict[str, int]


class SICAMonitor:
    """Append-only metric samples with a deterministic display snapshot."""

    def __init__(self) -> None:
        self._potential = 0.0
        self._performance = 0.0
        self._complexity = 0.0
        self._rejected_candidates = 0
        self._rollback_count = 0
        self._samples: list[MonitorSnapshot] = []
        self._last_source_rejected = 0
        self._last_source_rollbacks = 0

    def record(self, *, tick: int, potential: float | None = None,
               performance: float | None = None, complexity: float | None = None,
               rejected_candidates: int = 0, rollback_count: int = 0,
               rule_registry: Any = None) -> MonitorSnapshot:
        if tick < 0 or rejected_candidates < 0 or rollback_count < 0:
            raise ValueError("monitor counters and tick must be non-negative")
        if potential is not None:
            self._potential = float(potential)
        if performance is not None:
            self._performance = float(performance)
        if complexity is not None:
            self._complexity = float(complexity)
        self._rejected_candidates += rejected_candidates
        self._rollback_count += rollback_count
        records = tuple(rule_registry.all()) if rule_registry is not None else ()
        state_counts = {state: 0 for state in ("active", "hypothesis", "archived", "eliminated")}
        categories: dict[str, int] = {}
        scopes: dict[str, int] = {}
        for rule in records:
            state_counts[rule.state.value] = state_counts.get(rule.state.value, 0) + 1
            category = getattr(rule, "category", "uncategorized") or "uncategorized"
            scope = getattr(getattr(rule, "scope", None), "value", getattr(rule, "scope", "unknown"))
            categories[category] = categories.get(category, 0) + 1
            scopes[scope] = scopes.get(scope, 0) + 1
        snapshot = MonitorSnapshot(
            tick=tick, potential=self._potential, performance=self._performance,
            active_rules=state_counts.get("active", 0), hypothesis_rules=state_counts.get("hypothesis", 0),
            archived_rules=state_counts.get("archived", 0), eliminated_rules=state_counts.get("eliminated", 0),
            complexity=self._complexity, rejected_candidates=self._rejected_candidates,
            rollback_count=self._rollback_count, rule_categories=categories, rule_scopes=scopes,
        )
        self._samples.append(snapshot)
        return snapshot

    @property
    def latest(self) -> MonitorSnapshot | None:
        return self._samples[-1] if self._samples else None

    def samples(self) -> tuple[MonitorSnapshot, ...]:
        return tuple(self._samples)

    @staticmethod
    def potential(*, progress: float, novelty: float = 0.0, risk: float = 0.0) -> float:
        """Bounded potential Φ: progress/novelty help; risk penalizes."""
        return float(progress) + 0.25 * float(novelty) - float(risk)

    def record_system(self, *, tick: int, potential: float, performance: float,
                      complexity: float, rule_registry: Any = None,
                      candidate_registry: Any = None, rollback_manager: Any = None) -> MonitorSnapshot:
        rejected = getattr(candidate_registry, "rejected_count", 0)
        rollbacks = getattr(rollback_manager, "rollback_count", 0)
        rejected_delta = max(0, rejected - self._last_source_rejected)
        rollback_delta = max(0, rollbacks - self._last_source_rollbacks)
        self._last_source_rejected = rejected
        self._last_source_rollbacks = rollbacks
        return self.record(
            tick=tick, potential=potential, performance=performance, complexity=complexity,
            rejected_candidates=rejected_delta, rollback_count=rollback_delta,
            rule_registry=rule_registry,
        )

    def display(self) -> dict[str, Any]:
        latest = self.latest
        if latest is None:
            return {}
        return {
            "tick": latest.tick,
            "potential": latest.potential,
            "performance": latest.performance,
            "rules": {
                "active": latest.active_rules, "hypothesis": latest.hypothesis_rules,
                "archived": latest.archived_rules, "eliminated": latest.eliminated_rules,
            },
            "complexity": latest.complexity,
            "rejected_candidates": latest.rejected_candidates,
            "rollback_count": latest.rollback_count,
            "rule_categories": dict(latest.rule_categories),
            "rule_scopes": dict(latest.rule_scopes),
        }
