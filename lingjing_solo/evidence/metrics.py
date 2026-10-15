"""Small, explicit GateBench metric calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GateBenchCase:
    case_id: str
    environment_id: str
    baseline_passed: bool
    candidate_passed: bool
    replay_deterministic: bool
    expected_pass: bool


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(cases: Iterable[GateBenchCase], *, holdout_delta: float, combination_failures: int = 0, combination_total: int = 0) -> dict[str, Any]:
    values = list(cases)
    if not values:
        raise ValueError("GateBench cases must not be empty")
    if combination_total < 0 or combination_failures < 0 or combination_failures > combination_total:
        raise ValueError("combination failure counts are invalid")
    n = len(values)
    accepted = sum(case.candidate_passed for case in values)
    expected = sum(case.expected_pass for case in values)
    false_accept = sum(case.candidate_passed and not case.expected_pass for case in values)
    false_reject = sum((not case.candidate_passed) and case.expected_pass for case in values)
    envs = {case.environment_id for case in values}
    reference_environment = sorted(envs)[0]
    by_environment = {}
    for environment_id in sorted(envs):
        env_cases = [case for case in values if case.environment_id == environment_id]
        env_n = len(env_cases)
        env_accepted = sum(case.candidate_passed for case in env_cases)
        env_baseline = sum(case.baseline_passed for case in env_cases)
        by_environment[environment_id] = {
            "cases": env_n,
            "candidate_pass_rate": _rate(env_accepted, env_n),
            "baseline_pass_rate": _rate(env_baseline, env_n),
            "gate_drift": _rate(env_accepted - env_baseline, env_n),
            "false_accept": _rate(sum(case.candidate_passed and not case.expected_pass for case in env_cases), env_n),
            "false_reject": _rate(sum((not case.candidate_passed) and case.expected_pass for case in env_cases), env_n),
        }
    return {
        "cases": n,
        "environments": len(envs),
        "single_environment_pass_rate": _rate(sum(case.candidate_passed for case in values if case.environment_id == reference_environment), sum(case.environment_id == reference_environment for case in values)),
        "cross_environment_pass_rate": _rate(accepted, n),
        "holdout_delta": float(holdout_delta),
        "replay_determinism": _rate(sum(case.replay_deterministic for case in values), n),
        "false_accept": _rate(false_accept, n),
        "false_reject": _rate(false_reject, n),
        "gate_drift": _rate(accepted - sum(case.baseline_passed for case in values), n),
        "gate_drift_by_environment": by_environment,
        "combination_activation_failure_rate": _rate(combination_failures, combination_total),
        "combination_failures": combination_failures,
        "combination_total": combination_total,
        "expected_passes": expected,
    }


def markdown_report(metrics: dict[str, float | int]) -> str:
    return "# GateBench report\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in sorted(metrics.items())) + "\n"
