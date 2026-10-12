import pytest

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
