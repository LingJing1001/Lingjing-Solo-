"""Evidence record for offline and synthetic R4 probe loops."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..perception.observation import NormalizedObservation
from .hotspot_candidates import HotspotDetectionResult
from .probe_feedback import FeedbackOutcome, ProbeFeedback, TabuContext, attribute_feedback
from .probe_planner import ProbePlan, ProbePlanner


@dataclass(frozen=True)
class ProbeEvidence:
    run_id: str
    frame_id: str
    raw_candidate_count: int
    retained_count: int
    probe_plan: ProbePlan
    outcomes: tuple[FeedbackOutcome, ...]
    tabu: TabuContext
    feedbacks: tuple[ProbeFeedback, ...] = ()
    exit_code: int = 0


def run_synthetic_probe_loop(
    observation: NormalizedObservation,
    detection: HotspotDetectionResult,
    feedback_by_hotspot: Mapping[str, ProbeFeedback],
    *,
    run_id: str = "synthetic-r4-run",
    planner: ProbePlanner | None = None,
    tabu: TabuContext | None = None,
    max_probes: int = 3,
) -> ProbeEvidence:
    """Run planner→feedback attribution without calling an external environment."""
    planner = planner or ProbePlanner()
    tabu = tabu or TabuContext()
    plan = planner.plan(observation, detection, max_probes=max_probes, tabu_hotspot_ids=tabu.ids())
    outcomes: list[FeedbackOutcome] = []
    feedbacks: list[ProbeFeedback] = []
    for action in plan.actions:
        feedback = feedback_by_hotspot.get(action.hotspot_id, ProbeFeedback(False))
        feedbacks.append(feedback)
        outcome = attribute_feedback(feedback)
        outcomes.append(outcome)
        tabu = tabu.record(action, outcome)
    return ProbeEvidence(
        run_id=run_id,
        frame_id=observation.frame_id,
        raw_candidate_count=detection.raw_candidate_count,
        retained_count=detection.retained_count,
        probe_plan=plan,
        outcomes=tuple(outcomes),
        tabu=tabu,
        feedbacks=tuple(feedbacks),
    )


def write_probe_evidence_jsonl(path: str | Path, evidence: ProbeEvidence) -> int:
    """Append one run header and one auditable record per planned probe."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = [{
        "record_type": "probe_run",
        "run_id": evidence.run_id,
        "frame_id": evidence.frame_id,
        "raw_candidate_count": evidence.raw_candidate_count,
        "retained_count": evidence.retained_count,
        "exit_code": evidence.exit_code,
        "tabu_hotspot_ids": list(evidence.tabu.ids()),
    }]
    for index, (action, outcome) in enumerate(zip(evidence.probe_plan.actions, evidence.outcomes)):
        feedback = (
            evidence.feedbacks[index]
            if index < len(evidence.feedbacks)
            else ProbeFeedback(False)
        )
        records.append({
            "record_type": "probe",
            "run_id": evidence.run_id,
            "index": index,
            "hotspot_id": action.hotspot_id,
            "requested_action": action.action,
            "mode": action.mode,
            "x": action.x,
            "y": action.y,
            "before_frame_ref": feedback.before_frame_ref,
            "after_frame_ref": feedback.after_frame_ref,
            "outcome": outcome.outcome,
            "state_delta": outcome.state_delta,
            "level_delta": outcome.level_delta,
            "score_delta": outcome.score_delta,
            "frame_delta_ratio": outcome.frame_delta_ratio,
            "terminal": outcome.terminal,
            "evidence_refs": list(outcome.evidence_refs),
            "tabu_hotspot_ids": list(evidence.tabu.ids()),
        })
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)
