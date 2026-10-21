"""Fail-closed GateBench to shadow-registry admission boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import math

from lingjing_solo.evidence.shadow_registry import ShadowCandidate, ShadowCandidateRegistry


@dataclass(frozen=True)
class AdmissionResult:
    candidate_id: str
    authority_status: str
    holdout_delta: float
    environments: tuple[str, ...]
    rollback_snapshot: Mapping[str, Any]


def _validate_packet(packet: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    metrics = packet.get("metrics")
    manifest = packet.get("manifest")
    if not isinstance(metrics, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("GateBench packet requires metrics and manifest objects")
    environments = packet.get("environments")
    if not isinstance(environments, list) or len(set(environments)) < 2:
        raise ValueError("GateBench admission requires at least two environments")
    if int(packet.get("holdout_cases", 0)) <= 0:
        raise ValueError("GateBench admission requires a non-empty holdout")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping) or len(hashes) < 3:
        raise ValueError("GateBench admission requires at least three hashed artifacts")
    holdout_delta = float(metrics.get("holdout_delta", -1.0))
    if not math.isfinite(holdout_delta) or holdout_delta < 0:
        raise PermissionError("GateBench holdout delta is non-finite or negative")
    for metric in ("replay_determinism", "false_accept", "false_reject"):
        if metric not in metrics:
            raise ValueError(f"GateBench metric missing: {metric}")
        if not math.isfinite(float(metrics[metric])):
            raise PermissionError(f"GateBench metric is non-finite: {metric}")
    if float(metrics["replay_determinism"]) < 1.0:
        raise PermissionError("GateBench replay is not deterministic")
    if float(metrics["false_accept"]) > 0 or float(metrics["false_reject"]) > 0:
        raise PermissionError("GateBench contains false accept/reject failures")
    return metrics, manifest


def admit_gatebench_candidate(
    packet: Mapping[str, Any],
    *,
    candidate_id: str,
    candidate_type: str,
    cases: Iterable[Any],
    baseline: Callable[[Any], Mapping[str, float]],
    candidate_eval: Callable[[Any], Mapping[str, float]],
    safety_report_ref: str,
    replay_hashes: tuple[str, ...],
    registry: ShadowCandidateRegistry | None = None,
) -> AdmissionResult:
    """Admit a clean GateBench result to shadow only after SICA holdout checks."""
    metrics, manifest = _validate_packet(packet)
    if not safety_report_ref:
        raise ValueError("safety report reference must be non-empty")
    if not replay_hashes:
        raise ValueError("replay hashes must be non-empty")
    registry = registry or ShadowCandidateRegistry()
    hashes = manifest["artifact_hashes"]
    evidence_refs = tuple(f"gatebench:{name}:{value}" for name, value in sorted(hashes.items()))
    candidate = ShadowCandidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        authority_status="shadow",
        source_environments=tuple(sorted(set(packet["environments"]))),
        evidence_refs=evidence_refs,
        holdout_report_ref=f"gatebench:{packet.get('run_id', 'unknown')}:report.md",
        safety_report_ref=safety_report_ref,
        replay_hashes=replay_hashes,
        rollback_snapshot_ref=f"{candidate_id}.rollback.json",
        holdout_delta=float(metrics["holdout_delta"]),
    )
    admitted = registry.submit_checked(
        candidate,
        cases=cases,
        baseline=baseline,
        candidate_eval=candidate_eval,
    )
    snapshot = registry.rollback_snapshot(admitted.candidate_id)
    return AdmissionResult(
        candidate_id=admitted.candidate_id,
        authority_status=admitted.authority_status,
        holdout_delta=admitted.holdout_delta,
        environments=admitted.source_environments,
        rollback_snapshot=snapshot,
    )


def write_admission_result(result: AdmissionResult, path: Path) -> None:
    """Persist an auditable, shadow-only admission record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(asdict(result)) + "\n", encoding="utf-8")


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
