"""Conservative feedback attribution and tabu state for R4 probes."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .hotspot_candidates import HotspotCandidate
from .probe_planner import ProbeAction


@dataclass(frozen=True)
class ProbeFeedback:
    state_changed: bool = False
    levels_before: int = 0
    levels_after: int = 0
    score_before: float = 0.0
    score_after: float = 0.0
    terminal: bool = False
    dynamic_noise: bool = False
    frame_delta_ratio: float = 0.0
    before_frame_ref: str | None = None
    after_frame_ref: str | None = None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeedbackOutcome:
    outcome: str
    state_delta: float
    level_delta: int
    score_delta: float
    frame_delta_ratio: float
    terminal: bool
    explanation: str
    evidence_refs: tuple[str, ...]


def calculate_frame_delta(before: object, after: object) -> float:
    """Return changed-pixel ratio for two normalized 2-D frames."""
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    if before_array.shape != after_array.shape or before_array.size == 0:
        raise ValueError("before and after frames must have the same non-empty shape")
    if before_array.ndim != 2:
        raise ValueError("before and after frames must be 2-D")
    return float(np.mean(before_array != after_array))


@dataclass(frozen=True)
class TabuEntry:
    hotspot_id: str
    failures: int
    last_outcome: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class TabuContext:
    entries: tuple[TabuEntry, ...] = ()

    def ids(self) -> tuple[str, ...]:
        return tuple(entry.hotspot_id for entry in self.entries)

    def record(self, action: ProbeAction, outcome: FeedbackOutcome) -> "TabuContext":
        current = next((e for e in self.entries if e.hotspot_id == action.hotspot_id), None)
        if outcome.outcome in {"progress", "score_progress", "terminal"}:
            return self
        failures = 1 if current is None else current.failures + 1
        entry = TabuEntry(action.hotspot_id, failures, outcome.outcome, outcome.evidence_refs)
        retained = tuple(e for e in self.entries if e.hotspot_id != action.hotspot_id)
        return TabuContext(retained + (entry,))


def attribute_feedback(feedback: ProbeFeedback) -> FeedbackOutcome:
    """Never infer success from visual change alone."""
    level_delta = feedback.levels_after - feedback.levels_before
    score_delta = feedback.score_after - feedback.score_before
    evidence = tuple(feedback.evidence_refs)
    if feedback.terminal:
        outcome = "terminal"
        explanation = "environment reported terminal state"
    elif level_delta > 0:
        outcome = "progress"
        explanation = "levels completed increased"
    elif score_delta > 0:
        outcome = "score_progress"
        explanation = "score increased without level completion"
    elif feedback.dynamic_noise:
        outcome = "noisy"
        explanation = "change classified as dynamic noise"
    elif feedback.state_changed:
        outcome = "changed_no_progress"
        explanation = "state changed but no progress signal increased"
    else:
        outcome = "no_effect"
        explanation = "no observable state change"
    state_delta = float(level_delta) + float(score_delta != 0)
    return FeedbackOutcome(
        outcome,
        state_delta,
        level_delta,
        score_delta,
        feedback.frame_delta_ratio,
        feedback.terminal,
        explanation,
        evidence,
    )


def update_candidate_after_feedback(
    candidate: HotspotCandidate, outcome: FeedbackOutcome
) -> HotspotCandidate:
    """Return a new candidate; the original detector result remains immutable."""
    if outcome.outcome in {"progress", "score_progress", "terminal"}:
        return replace(
            candidate,
            status=outcome.outcome,
            confidence=min(1.0, candidate.confidence + 0.10),
        )
    if outcome.outcome in {"no_effect", "noisy", "changed_no_progress"}:
        penalty = 0.15 if outcome.outcome == "noisy" else 0.10
        features = replace(
            candidate.features,
            repeated_failure=min(1.0, candidate.features.repeated_failure + penalty),
        )
        return replace(candidate, status=outcome.outcome,
                       confidence=max(0.0, candidate.confidence - penalty), features=features)
    return replace(candidate, status=outcome.outcome)
