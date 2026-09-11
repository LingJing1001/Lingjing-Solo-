"""Bounded, fail-closed probe planning for R4 candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..perception.observation import NormalizedObservation
from .hotspot_candidates import HotspotDetectionResult


@dataclass(frozen=True)
class ProbeAction:
    hotspot_id: str
    action: str
    x: int
    y: int
    mode: str
    evidence_refs: tuple[str, ...]
    reason: str
    hypothesis_space_type: str | None = None
    evidence_role: str = "r4_hotspot"


@dataclass(frozen=True)
class ProbePlan:
    actions: tuple[ProbeAction, ...]
    skipped_hotspot_ids: tuple[str, ...]
    remaining_budget: int | None
    warnings: tuple[str, ...] = ()


class ProbePlanner:
    """Turn detector output into at most three auditable, legal probes."""

    def __init__(self, *, min_click_confidence: float = 0.50) -> None:
        if not 0 <= min_click_confidence <= 1:
            raise ValueError("min_click_confidence must be in [0, 1]")
        self.min_click_confidence = min_click_confidence

    def plan(
        self,
        observation: NormalizedObservation,
        detection: HotspotDetectionResult,
        *,
        max_probes: int = 3,
        tabu_hotspot_ids: Iterable[str] = (),
    ) -> ProbePlan:
        if (
            not isinstance(max_probes, int)
            or isinstance(max_probes, bool)
            or not 0 <= max_probes <= 3
        ):
            raise ValueError("max_probes must be in [0, 3]")
        budget = observation.remaining_budget
        if budget is not None and budget < 1:
            return ProbePlan(
                (),
                tuple(c.hotspot_id for c in detection.candidates),
                budget,
                ("remaining budget is exhausted",),
            )
        if observation.game_family == "keyboard":
            return ProbePlan(
                (),
                tuple(c.hotspot_id for c in detection.candidates),
                budget,
                ("keyboard game family has no click probes",),
            )
        if observation.legal_actions is not None and "click" not in observation.legal_actions:
            return ProbePlan(
                (),
                tuple(c.hotspot_id for c in detection.candidates),
                budget,
                ("click is not a legal action",),
            )
        tabu = set(tabu_hotspot_ids)
        actions: list[ProbeAction] = []
        skipped: list[str] = []
        effective_limit = min(max_probes, budget) if budget is not None else max_probes
        hypothesis = observation.current_hypothesis
        for candidate in detection.candidates:
            if len(actions) >= effective_limit:
                skipped.append(candidate.hotspot_id)
                continue
            if candidate.hotspot_id in tabu:
                skipped.append(candidate.hotspot_id)
                continue
            if not (0 <= candidate.x < observation.width and 0 <= candidate.y < observation.height):
                skipped.append(candidate.hotspot_id)
                continue
            mode = "click" if candidate.confidence >= self.min_click_confidence else "observe_only"
            actions.append(
                ProbeAction(
                    hotspot_id=candidate.hotspot_id,
                    action="click" if mode == "click" else "observe",
                    x=candidate.x,
                    y=candidate.y,
                    mode=mode,
                    evidence_refs=candidate.evidence_refs + detection.evidence_refs,
                    reason=(
                        "detector_ranked_candidate"
                        if mode == "click"
                        else "low_confidence_observe_only"
                    ),
                    hypothesis_space_type=(
                        hypothesis.hypothesis_space_type if hypothesis is not None else None
                    ),
                    evidence_role=(
                        "phi_interactive_hotspot"
                        if hypothesis is not None
                        else "r4_hotspot"
                    ),
                )
            )
        return ProbePlan(
            actions=tuple(actions),
            skipped_hotspot_ids=tuple(skipped),
            remaining_budget=None if budget is None else max(0, budget - len(actions)),
        )
