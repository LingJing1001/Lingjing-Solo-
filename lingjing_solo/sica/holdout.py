"""R2 holdout A/B performance gate.

The evaluator is injected so this layer cannot silently invent an environment
score. A candidate passes only when both sides ran on the same non-empty cases
and the candidate does not regress the selected metric beyond tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class HoldoutResult:
    passed: bool
    baseline_score: float
    candidate_score: float
    delta: float
    cases: int
    metric: str
    reason: str = ""


class HoldoutGate:
    def __init__(self, *, metric: str = "score", tolerance: float = 0.0) -> None:
        if not metric:
            raise ValueError("metric must be non-empty")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.metric = metric
        self.tolerance = float(tolerance)

    def evaluate(
        self,
        cases: Iterable[Any],
        baseline: Callable[[Any], Mapping[str, float]],
        candidate: Callable[[Any], Mapping[str, float]],
    ) -> HoldoutResult:
        case_list = list(cases)
        if not case_list:
            return HoldoutResult(False, 0.0, 0.0, 0.0, 0, self.metric, "holdout is empty")
        baseline_values: list[float] = []
        candidate_values: list[float] = []
        for case in case_list:
            base_metrics = baseline(case)
            candidate_metrics = candidate(case)
            if self.metric not in base_metrics or self.metric not in candidate_metrics:
                return HoldoutResult(
                    False, 0.0, 0.0, 0.0, len(case_list), self.metric,
                    f"metric {self.metric!r} missing from evaluator output",
                )
            baseline_values.append(float(base_metrics[self.metric]))
            candidate_values.append(float(candidate_metrics[self.metric]))
        base_score = sum(baseline_values) / len(baseline_values)
        candidate_score = sum(candidate_values) / len(candidate_values)
        delta = candidate_score - base_score
        passed = candidate_score + self.tolerance >= base_score
        reason = "candidate is non-regressing" if passed else "candidate regresses holdout metric"
        return HoldoutResult(passed, base_score, candidate_score, delta, len(case_list), self.metric, reason)

    def require(self, result: HoldoutResult) -> HoldoutResult:
        if not result.passed:
            raise PermissionError(f"R2 holdout rejected candidate: {result.reason}")
        return result
