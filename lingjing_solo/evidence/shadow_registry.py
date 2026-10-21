"""Shadow-only candidate admission and rollback snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from lingjing_solo.sica.candidate import CandidateModification, SafetyGate
from lingjing_solo.sica.holdout import HoldoutGate


@dataclass(frozen=True)
class ShadowCandidate:
    candidate_id: str
    candidate_type: str
    authority_status: str
    source_environments: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    holdout_report_ref: str
    safety_report_ref: str
    replay_hashes: tuple[str, ...]
    rollback_snapshot_ref: str
    holdout_delta: float

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")


class ShadowCandidateRegistry:
    def __init__(self, snapshot_dir: Path | None = None) -> None:
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._candidates: dict[str, ShadowCandidate] = {}
        self.rejections: list[str] = []

    def submit(self, candidate: ShadowCandidate) -> ShadowCandidate:
        failures: list[str] = []
        if candidate.authority_status != "shadow": failures.append("authority must remain shadow")
        if len(candidate.evidence_refs) < 3: failures.append("evidence_count < 3")
        if len(set(candidate.source_environments)) < 2: failures.append("distinct_games < 2")
        if not candidate.holdout_report_ref: failures.append("holdout report missing")
        if not candidate.safety_report_ref: failures.append("safety report missing")
        if not candidate.replay_hashes: failures.append("replay hashes missing")
        if candidate.candidate_id in self._candidates: raise ValueError(f"candidate already submitted: {candidate.candidate_id}")
        if failures:
            self.rejections.extend(failures)
            raise PermissionError("shadow candidate rejected: " + "; ".join(failures))
        self._candidates[candidate.candidate_id] = candidate
        if self.snapshot_dir:
            self.rollback_snapshot(candidate.candidate_id)
        return candidate

    def submit_checked(self, candidate: ShadowCandidate, *, cases: Iterable[Any], baseline: Callable[[Any], Mapping[str, float]], candidate_eval: Callable[[Any], Mapping[str, float]], metric: str = "score") -> ShadowCandidate:
        """Run the existing SICA holdout and safety gates before shadow admission."""
        holdout = HoldoutGate(metric=metric, min_delta=0.0).evaluate(cases, baseline, candidate_eval)
        if not holdout.passed:
            raise PermissionError(f"SICA holdout rejected candidate: {holdout.reason}")
        safety_candidate = CandidateModification(
            candidate_id=candidate.candidate_id, target="rule_set", content={"candidate_type": candidate.candidate_type},
            evidence_refs=candidate.evidence_refs, evidence_count=len(candidate.evidence_refs),
            distinct_games=len(set(candidate.source_environments)), expected_gain=max(candidate.holdout_delta, 0.0),
        )
        SafetyGate().require(safety_candidate)
        return self.submit(candidate)

    def get(self, candidate_id: str) -> ShadowCandidate | None:
        return self._candidates.get(candidate_id)

    def rollback_snapshot(self, candidate_id: str) -> dict:
        candidate = self._candidates.get(candidate_id)
        if candidate is None: raise KeyError(candidate_id)
        snapshot = {"candidate_id": candidate_id, "authority_status": "shadow", "candidate": asdict(candidate), "active": False}
        if self.snapshot_dir:
            (self.snapshot_dir / f"{candidate_id}.rollback.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return snapshot

    def activate(self, candidate_id: str, *, approval: Mapping[str, Any] | None = None) -> bool:
        """Activate only with an authenticated, hash-bound MC verifier approval."""
        candidate = self._candidates.get(candidate_id)
        required = {"verdict", "verifier_task_id", "report_sha256", "candidate_sha256", "approved_at"}
        if candidate is None:
            self.rejections.append(f"activation denied for unknown candidate: {candidate_id}")
            return False
        if not approval or approval.get("verdict") != "passed" or not required.issubset(approval):
            self.rejections.append(f"activation denied without complete verifier approval: {candidate_id}")
            return False
        if approval.get("candidate_id", candidate_id) != candidate_id:
            self.rejections.append(f"activation denied for mismatched candidate approval: {candidate_id}")
            return False
        if candidate.authority_status != "shadow":
            self.rejections.append(f"activation denied for non-shadow candidate: {candidate_id}")
            return False
        activated = ShadowCandidate(**{**asdict(candidate), "authority_status": "production"})
        self._candidates[candidate_id] = activated
        if self.snapshot_dir:
            (self.snapshot_dir / f"{candidate_id}.production.json").write_text(
                json.dumps({"candidate_id": candidate_id, "authority_status": "production", "approval": dict(approval), "active": True}, indent=2, sort_keys=True) + "\n"
            )
        return True
