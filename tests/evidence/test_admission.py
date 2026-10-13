import json

import pytest

from lingjing_solo.evidence.admission import admit_gatebench_candidate, write_admission_result
from lingjing_solo.evidence.benchmark import build_gatebench_packet
from lingjing_solo.evidence.metrics import GateBenchCase
from lingjing_solo.evidence.shadow_registry import ShadowCandidateRegistry


def clean_packet() -> dict:
    cases = [
        GateBenchCase("a", "env-a", True, True, True, True),
        GateBenchCase("b", "env-b", True, True, True, True),
    ]
    packet = build_gatebench_packet(
        cases,
        holdout_delta=0.1,
        holdout_cases=2,
        run_id="clean-run",
        source_commit="abc",
    )
    packet["manifest"] = {
        "artifact_hashes": {
            "baseline.json": "sha256:base",
            "candidate.json": "sha256:candidate",
            "metrics.json": "sha256:metrics",
        }
    }
    return packet


def test_clean_gatebench_packet_enters_shadow_and_writes_rollback(tmp_path):
    registry = ShadowCandidateRegistry(tmp_path)
    result = admit_gatebench_candidate(
        clean_packet(),
        candidate_id="candidate-1",
        candidate_type="rule_set",
        cases=["a", "b"],
        baseline=lambda _: {"score": 1.0},
        candidate_eval=lambda _: {"score": 1.1},
        safety_report_ref="safety.json",
        replay_hashes=("hash-a", "hash-b"),
        registry=registry,
    )

    assert result.authority_status == "shadow"
    assert result.environments == ("env-a", "env-b")
    assert registry.get("candidate-1") is not None
    assert (tmp_path / "candidate-1.rollback.json").exists()

    output = tmp_path / "admission.json"
    write_admission_result(result, output)
    assert json.loads(output.read_text())["authority_status"] == "shadow"


def test_gatebench_failures_are_rejected_before_sica_submission():
    packet = clean_packet()
    packet["metrics"]["replay_determinism"] = 0.5

    with pytest.raises(PermissionError, match="not deterministic"):
        admit_gatebench_candidate(
            packet,
            candidate_id="candidate-bad",
            candidate_type="rule_set",
            cases=["a", "b"],
            baseline=lambda _: {"score": 1.0},
            candidate_eval=lambda _: {"score": 1.1},
            safety_report_ref="safety.json",
            replay_hashes=("hash-a",),
        )


def test_non_finite_gatebench_metrics_fail_closed():
    packet = clean_packet()
    packet["metrics"]["holdout_delta"] = float("nan")

    with pytest.raises(PermissionError, match="non-finite"):
        admit_gatebench_candidate(
            packet,
            candidate_id="candidate-nan",
            candidate_type="rule_set",
            cases=["a", "b"],
            baseline=lambda _: {"score": 1.0},
            candidate_eval=lambda _: {"score": 1.1},
            safety_report_ref="safety.json",
            replay_hashes=("hash-a",),
        )


def test_holdout_regression_is_rejected():
    with pytest.raises(PermissionError, match="holdout"):
        admit_gatebench_candidate(
            clean_packet(),
            candidate_id="candidate-regression",
            candidate_type="rule_set",
            cases=["a", "b"],
            baseline=lambda _: {"score": 1.0},
            candidate_eval=lambda _: {"score": 0.9},
            safety_report_ref="safety.json",
            replay_hashes=("hash-a",),
        )
