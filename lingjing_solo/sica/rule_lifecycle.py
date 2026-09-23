"""SICA phase-2 rule governance: R3, R9 and R6.

This module keeps hypotheses separate from verified rules. A rule is promoted
only after three distinct evidence identities are observed; repeated playback
of one evidence record cannot satisfy the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


class RuleState(StrEnum):
    HYPOTHESIS = "hypothesis"
    ACTIVE = "active"
    ARCHIVED = "archived"
    ELIMINATED = "eliminated"


@dataclass(frozen=True)
class ComplexityProfile:
    nodes: int
    depth: int
    dependencies: int
    action_count: int
    cost: float


def measure_complexity(value: Any) -> ComplexityProfile:
    """Measure structural complexity deterministically without model claims."""
    nodes = 0
    max_depth = 0
    dependencies = 0
    action_count = 0

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes, max_depth, dependencies, action_count
        nodes += 1
        max_depth = max(max_depth, depth)
        if isinstance(item, Mapping):
            dependencies += sum(1 for key in item if "depend" in str(key).lower())
            action_count += sum(1 for key in item if "action" in str(key).lower())
            for key, child in item.items():
                walk(key, depth + 1)
                walk(child, depth + 1)
        elif isinstance(item, (list, tuple, set, frozenset)):
            action_count += 1 if item and all(isinstance(x, str) for x in item) else 0
            for child in item:
                walk(child, depth + 1)

    walk(value, 0)
    cost = float(nodes + 2 * max_depth + 3 * dependencies + action_count)
    return ComplexityProfile(nodes, max_depth, dependencies, action_count, cost)


@dataclass(frozen=True)
class ComplexityBudget:
    max_cost: float
    max_depth: int | None = None

    def accept(self, profile: ComplexityProfile) -> bool:
        return profile.cost <= self.max_cost and (
            self.max_depth is None or profile.depth <= self.max_depth
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    rule_id: str
    source_id: str
    game_id: str
    episode_id: str
    success: bool
    tick: int

    @property
    def independence_key(self) -> tuple[str, str, str]:
        return self.source_id, self.game_id, self.episode_id


@dataclass
class RuleRecord:
    rule_id: str
    content: dict[str, Any]
    state: RuleState = RuleState.HYPOTHESIS
    evidence: dict[str, Evidence] = field(default_factory=dict)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.0
    last_used_tick: int = 0
    decayed_score: float = 0.0
    archived_tick: int | None = None
    eliminated_tick: int | None = None

    @property
    def independent_evidence_count(self) -> int:
        return len({item.independence_key for item in self.evidence.values()})

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    def recompute_confidence(self) -> None:
        total = self.success_count + self.failure_count
        self.confidence = self.success_count / total if total else 0.0


class RuleRegistry:
    """Hypothesis area plus lifecycle management for verified rules."""

    def __init__(self, *, evidence_threshold: int = 3, decay_factor: float = 0.9,
                 archive_after: int = 10, eliminate_after: int = 30,
                 min_success_rate: float = 0.25) -> None:
        if evidence_threshold < 1 or not 0 < decay_factor <= 1:
            raise ValueError("invalid evidence threshold or decay factor")
        if archive_after < 1 or eliminate_after < archive_after:
            raise ValueError("eliminate_after must be >= archive_after >= 1")
        if not 0 <= min_success_rate <= 1:
            raise ValueError("min_success_rate must be in 0..1")
        self.evidence_threshold = evidence_threshold
        self.decay_factor = decay_factor
        self.archive_after = archive_after
        self.eliminate_after = eliminate_after
        self.min_success_rate = min_success_rate
        self._rules: dict[str, RuleRecord] = {}

    @property
    def hypotheses(self) -> tuple[RuleRecord, ...]:
        return tuple(r for r in self._rules.values() if r.state == RuleState.HYPOTHESIS)

    @property
    def active_rules(self) -> tuple[RuleRecord, ...]:
        return tuple(r for r in self._rules.values() if r.state == RuleState.ACTIVE)

    def propose(self, rule_id: str, content: Mapping[str, Any], *, budget: ComplexityBudget | None = None) -> RuleRecord:
        if not rule_id or rule_id in self._rules:
            raise ValueError("rule_id must be unique and non-empty")
        profile = measure_complexity(dict(content))
        if budget is not None and not budget.accept(profile):
            raise PermissionError(f"complexity budget rejected {rule_id}: cost={profile.cost}")
        record = RuleRecord(rule_id, dict(content), decayed_score=profile.cost)
        self._rules[rule_id] = record
        return record

    def add_evidence(self, evidence: Evidence) -> RuleRecord:
        rule = self._rules.get(evidence.rule_id)
        if rule is None:
            raise KeyError(f"unknown rule: {evidence.rule_id}")
        if rule.state in {RuleState.ARCHIVED, RuleState.ELIMINATED}:
            raise PermissionError(f"cannot add evidence to {rule.state.value} rule")
        if evidence.evidence_id in rule.evidence:
            raise ValueError(f"duplicate evidence: {evidence.evidence_id}")
        rule.evidence[evidence.evidence_id] = evidence
        if evidence.success:
            rule.success_count += 1
        else:
            rule.failure_count += 1
        rule.recompute_confidence()
        if rule.independent_evidence_count >= self.evidence_threshold:
            rule.state = RuleState.ACTIVE
        return rule

    def use(self, rule_id: str, *, tick: int) -> RuleRecord:
        rule = self._get_active(rule_id)
        rule.usage_count += 1
        rule.last_used_tick = tick
        rule.decayed_score = max(rule.decayed_score, 1.0)
        return rule

    def report_outcome(self, rule_id: str, *, success: bool, tick: int) -> RuleRecord:
        rule = self._get_active(rule_id)
        if success:
            rule.success_count += 1
        else:
            rule.failure_count += 1
        rule.last_used_tick = tick
        rule.recompute_confidence()
        return rule

    def decay(self, *, current_tick: int) -> tuple[RuleRecord, ...]:
        changed: list[RuleRecord] = []
        for rule in self._rules.values():
            if rule.state not in {RuleState.ACTIVE, RuleState.ARCHIVED}:
                continue
            idle = max(0, current_tick - rule.last_used_tick)
            rule.decayed_score *= self.decay_factor ** idle
            if rule.state == RuleState.ACTIVE and idle >= self.archive_after and rule.confidence < self.min_success_rate:
                rule.state = RuleState.ARCHIVED
                rule.archived_tick = current_tick
                changed.append(rule)
            elif rule.state == RuleState.ARCHIVED and idle >= self.eliminate_after:
                rule.state = RuleState.ELIMINATED
                rule.eliminated_tick = current_tick
                changed.append(rule)
        return tuple(changed)

    def archive(self, rule_id: str, *, tick: int) -> RuleRecord:
        rule = self._rules[rule_id]
        if rule.state == RuleState.ELIMINATED:
            raise PermissionError("eliminated rule cannot be archived")
        rule.state = RuleState.ARCHIVED
        rule.archived_tick = tick
        return rule

    def eliminate(self, rule_id: str, *, tick: int) -> RuleRecord:
        rule = self._rules[rule_id]
        rule.state = RuleState.ELIMINATED
        rule.eliminated_tick = tick
        return rule

    def get(self, rule_id: str) -> RuleRecord:
        return self._rules[rule_id]

    def _get_active(self, rule_id: str) -> RuleRecord:
        rule = self._rules[rule_id]
        if rule.state != RuleState.ACTIVE:
            raise PermissionError(f"rule is not active: {rule.state.value}")
        return rule
