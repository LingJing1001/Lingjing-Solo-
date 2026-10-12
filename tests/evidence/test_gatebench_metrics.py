import json

import pytest

from lingjing_solo.evidence.benchmark import build_gatebench_packet, write_gatebench_packet
from lingjing_solo.evidence.metrics import GateBenchCase, compute_metrics


def test_gatebench_metrics_and_determinism():
    cases = [
        GateBenchCase("a", "env-a", True, True, True, True),
        GateBenchCase("b", "env-b", True, False, True, True),
        GateBenchCase("c", "env-a", False, True, False, False),
    ]
    metrics = compute_metrics(cases, holdout_delta=0.2, combination_failures=1, combination_total=4)
    assert metrics["single_environment_pass_rate"] == 1.0
    assert metrics["cross_environment_pass_rate"] == 2 / 3
    assert metrics["replay_determinism"] == 2 / 3
    assert metrics["false_accept"] == 1 / 3
    assert metrics["false_reject"] == 1 / 3
    assert metrics["combination_activation_failure_rate"] == 0.25


def test_empty_cases_and_invalid_denominators_fail_closed():
    with pytest.raises(ValueError):
        compute_metrics([], holdout_delta=0.0)
    with pytest.raises(ValueError):
        compute_metrics([GateBenchCase("a", "e", True, True, True, True)], holdout_delta=0.0, combination_failures=1, combination_total=0)


def test_gatebench_packet_writes_auditable_artifacts(tmp_path):
    cases = [
        GateBenchCase("a", "env-a", True, True, True, True),
        GateBenchCase("b", "env-b", True, False, False, True),
    ]
    packet = build_gatebench_packet(cases, holdout_delta=0.1, holdout_cases=2, run_id="run-1", source_commit="abc")
    write_gatebench_packet(packet, tmp_path)
    assert packet["metrics"]["environments"] == 2
    assert (tmp_path / "baseline.json").exists()
    assert (tmp_path / "candidate.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "failures.jsonl").read_text().count("replay_nondeterminism") == 1
    assert "artifact_hashes" in json.loads((tmp_path / "manifest.json").read_text())


def test_gatebench_packet_requires_two_environments_and_nonempty_holdout():
    one = [GateBenchCase("a", "env-a", True, True, True, True)]
    with pytest.raises(ValueError, match="two environments"):
        build_gatebench_packet(one, holdout_delta=0.0, holdout_cases=1)
    two = one + [GateBenchCase("b", "env-a", True, True, True, True)]
    with pytest.raises(ValueError, match="holdout"):
        build_gatebench_packet(two, holdout_delta=0.0, holdout_cases=0)
