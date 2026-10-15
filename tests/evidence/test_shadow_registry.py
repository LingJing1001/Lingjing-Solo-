import pytest

from lingjing_solo.evidence.shadow_registry import ShadowCandidate, ShadowCandidateRegistry


def candidate(**kw):
    value = dict(candidate_id="c-1", candidate_type="rule_set", authority_status="shadow", source_environments=("a", "b"), evidence_refs=("e1", "e2", "e3"), holdout_report_ref="holdout.json", safety_report_ref="safety.json", replay_hashes=("h1", "h2"), rollback_snapshot_ref="rollback.json", holdout_delta=0.1)
    value.update(kw)
    return ShadowCandidate(**value)


def test_shadow_registry_requires_cross_environment_evidence_and_is_idempotently_rejecting():
    registry = ShadowCandidateRegistry()
    with pytest.raises(PermissionError):
        registry.submit(candidate(candidate_id="one", source_environments=("a",)))
    accepted = registry.submit(candidate())
    assert accepted.authority_status == "shadow"
    assert registry.get("c-1") == accepted
    with pytest.raises(ValueError):
        registry.submit(candidate())
    with pytest.raises(PermissionError):
        registry.submit(candidate(candidate_id="active", authority_status="active"))


def test_registry_exports_rollback_snapshot_without_activation(tmp_path):
    registry = ShadowCandidateRegistry(tmp_path)
    registry.submit(candidate())
    snapshot = registry.rollback_snapshot("c-1")
    assert snapshot["authority_status"] == "shadow"
    assert (tmp_path / "c-1.rollback.json").exists()
    assert registry.activate("c-1") is False
    assert registry.activate("c-1", approval={
        "verdict": "passed", "verifier_task_id": "task-v", "report_sha256": "r" * 64,
        "candidate_sha256": "c" * 64, "approved_at": "2026-10-14T23:00:00Z", "candidate_id": "c-1",
    }) is True
    activated = registry.get("c-1")
    assert activated is not None
    assert activated.authority_status == "production"
    assert (tmp_path / "c-1.production.json").exists()


def test_registry_checked_submission_reuses_sica_gates():
    registry = ShadowCandidateRegistry()
    cases = ["a", "b"]
    assert registry.submit_checked(candidate(), cases=cases, baseline=lambda _: {"score": 1.0}, candidate_eval=lambda _: {"score": 1.1}).candidate_id == "c-1"
    with pytest.raises(PermissionError, match="holdout"):
        ShadowCandidateRegistry().submit_checked(candidate(), cases=cases, baseline=lambda _: {"score": 1.0}, candidate_eval=lambda _: {"score": 0.9})
