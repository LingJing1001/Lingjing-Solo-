from pathlib import Path

from tools.evidence.production_gate import check


def _ready_report():
    return {
        "policy": {"authority_status": "production", "external_ugc": "disabled"},
        "canonical": {"source_commits": {"null": 0}},
        "counterfactual_verified_holdout": {"delta": 0.01, "passed": True, "rows": 10, "valid_prediction_rows": 10, "replay_verified_rows": 10},
        "gatebench": {"metrics": {"false_accept": 0, "false_reject": 0, "replay_determinism": 1}},
        "projection": {"status": "pass", "invalid_rows": 0},
    }


def test_production_gate_blocks_zero_holdout_delta():
    report = _ready_report()
    report["counterfactual_verified_holdout"]["delta"] = 0.0
    result = check(report, {"passed": True, "stable": True}, None, None)
    assert result["status"] == "blocked"
    assert any(item["check"] == "holdout_positive_delta" for item in result["failures"])


def test_production_gate_fails_closed_without_approval_and_rollback(tmp_path: Path):
    result = check(_ready_report(), {"passed": True, "stable": True}, None, None)
    assert result["status"] == "blocked"
    assert {item["check"] for item in result["failures"]} == {"rollback_artifact", "independent_approval", "audit_write_path"}


def test_production_gate_accepts_only_when_explicit_artifacts_exist(tmp_path: Path):
    rollback = tmp_path / "rollback"
    (rollback / "artifacts").mkdir(parents=True)
    (rollback / "manifest.json").write_text('{"schema":"appendix_a.rollback.v1","runtime_authority":false,"artifacts":[]}')
    result = check(_ready_report(), {"passed": True, "stable": True}, rollback, None)
    assert result["status"] == "blocked"
    assert {item["check"] for item in result["failures"]} == {"independent_approval", "audit_write_path"}