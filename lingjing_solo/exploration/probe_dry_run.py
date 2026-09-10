"""Dry-run adapter for auditable probe requests.

This module validates planned probe payloads and writes evidence before any real
harness is called. It deliberately never executes clicks or keyboard actions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..perception.observation import NormalizedObservation
from .probe_planner import ProbeAction, ProbePlan


@dataclass(frozen=True)
class DryRunProbe:
    run_id: str
    index: int
    hotspot_id: str
    requested_action: str
    mode: str
    x: int
    y: int
    evidence_refs: tuple[str, ...]
    executed: bool = False
    status: str = "dry_run"


def prepare_probe_dry_run(
    observation: NormalizedObservation,
    plan: ProbePlan,
    *,
    run_id: str = "dry-run-r4",
) -> tuple[DryRunProbe, ...]:
    """Validate a plan and return harness-neutral requests without execution."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    prepared: list[DryRunProbe] = []
    for index, action in enumerate(plan.actions):
        _validate_action(observation, action)
        prepared.append(
            DryRunProbe(
                run_id=run_id,
                index=index,
                hotspot_id=action.hotspot_id,
                requested_action=action.action,
                mode=action.mode,
                x=action.x,
                y=action.y,
                evidence_refs=action.evidence_refs,
            )
        )
    return tuple(prepared)


def _validate_action(observation: NormalizedObservation, action: ProbeAction) -> None:
    if action.action not in {"click", "observe"}:
        raise ValueError(f"unsupported probe action: {action.action}")
    if action.mode not in {"click", "observe_only"}:
        raise ValueError(f"unsupported probe mode: {action.mode}")
    if action.action == "click" and action.mode != "click":
        raise ValueError("click action requires click mode")
    if action.action == "observe" and action.mode != "observe_only":
        raise ValueError("observe action requires observe_only mode")
    if not (0 <= action.x < observation.width and 0 <= action.y < observation.height):
        raise ValueError("probe coordinates are outside the observation bounds")


def write_probe_dry_run_jsonl(
    path: str | Path,
    requests: Iterable[DryRunProbe],
    *,
    observation: NormalizedObservation,
) -> int:
    """Append a dry-run header and requests; no execution result is implied."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = [{
        "record_type": "probe_dry_run",
        "run_id": requests_tuple[0].run_id if (requests_tuple := tuple(requests)) else "",
        "frame_id": observation.frame_id,
        "executed": False,
        "status": "dry_run",
        "request_count": len(requests_tuple),
    }]
    records.extend({
        "record_type": "probe_request",
        "run_id": request.run_id,
        "index": request.index,
        "hotspot_id": request.hotspot_id,
        "requested_action": request.requested_action,
        "mode": request.mode,
        "x": request.x,
        "y": request.y,
        "evidence_refs": list(request.evidence_refs),
        "executed": request.executed,
        "status": request.status,
    } for request in requests_tuple)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)
