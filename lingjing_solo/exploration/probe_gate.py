"""Feature-flag boundary for R4 probe planning.

The gate is deliberately outside Field/Learner code. Disabled mode and keyboard
observations fail closed, while the returned plan remains immutable and can be
used by the dry-run adapter without executing an external action.
"""
from __future__ import annotations

from dataclasses import replace

from ..perception.observation import NormalizedObservation
from .probe_planner import ProbePlan


class R4ProbeGate:
    """Isolate optional R4 probe planning from the existing learner path."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def apply(self, observation: NormalizedObservation, plan: ProbePlan) -> ProbePlan:
        """Return a safe plan; never mutate the incoming plan or execute actions."""
        if not self.enabled:
            return replace(
                plan,
                actions=(),
                skipped_hotspot_ids=tuple(
                    dict.fromkeys(
                        plan.skipped_hotspot_ids
                        + tuple(action.hotspot_id for action in plan.actions)
                    )
                ),
                warnings=plan.warnings + ("R4 probe feature flag is disabled",),
            )
        if observation.game_family == "keyboard":
            return replace(
                plan,
                actions=(),
                skipped_hotspot_ids=tuple(
                    dict.fromkeys(
                        plan.skipped_hotspot_ids
                        + tuple(action.hotspot_id for action in plan.actions)
                    )
                ),
                warnings=plan.warnings + ("keyboard observations cannot consume click probes",),
            )
        return plan
