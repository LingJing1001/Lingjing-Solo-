"""Auditable offline GateBench packet construction."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import GateBenchCase, compute_metrics, markdown_report


def build_gatebench_packet(
    cases: Iterable[GateBenchCase],
    *,
    holdout_delta: float,
    holdout_cases: int,
    combination_failures: int = 0,
    combination_total: int = 0,
    run_id: str = "offline-gatebench",
    source_commit: str = "unknown",
) -> dict[str, Any]:
    values = list(cases)
    if not values:
        raise ValueError("GateBench cases must not be empty")
    if holdout_cases <= 0:
        raise ValueError("holdout must not be empty")
    environments = {case.environment_id for case in values}
    if len(environments) < 2:
        raise ValueError("GateBench requires at least two environments")
    metrics = compute_metrics(values, holdout_delta=holdout_delta, combination_failures=combination_failures, combination_total=combination_total)
    failures = [
        {
            "case_id": case.case_id,
            "environment_id": case.environment_id,
            "failure_types": [
                *( ["false_accept"] if case.candidate_passed and not case.expected_pass else []),
                *( ["false_reject"] if not case.candidate_passed and case.expected_pass else []),
                *( ["replay_nondeterminism"] if not case.replay_deterministic else []),
            ],
        }
        for case in values
        if (case.candidate_passed and not case.expected_pass)
        or (not case.candidate_passed and case.expected_pass)
        or not case.replay_deterministic
    ]
    baseline = {
        "cases": len(values),
        "passed": sum(case.baseline_passed for case in values),
        "pass_rate": sum(case.baseline_passed for case in values) / len(values),
    }
    candidate = {
        "cases": len(values),
        "passed": sum(case.candidate_passed for case in values),
        "pass_rate": sum(case.candidate_passed for case in values) / len(values),
    }
    return {
        "run_id": run_id,
        "source_commit": source_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_cases": holdout_cases,
        "environments": sorted(environments),
        "cases": [asdict(case) for case in values],
        "baseline": baseline,
        "candidate": candidate,
        "metrics": metrics,
        "failures": failures,
        "report": markdown_report(metrics),
    }


def write_gatebench_packet(packet: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline.json").write_text(json.dumps(packet["baseline"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "candidate.json").write_text(json.dumps(packet["candidate"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(packet["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = packet["failures"]
    if not isinstance(failures, list):
        raise ValueError("packet failures must be a list")
    with (output / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
    (output / "report.md").write_text(str(packet["report"]), encoding="utf-8")
    manifest = {
        key: packet[key]
        for key in ("run_id", "source_commit", "created_at", "holdout_cases", "environments")
    }
    manifest["artifact_hashes"] = {
        name: "sha256:" + hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("baseline.json", "candidate.json", "metrics.json", "failures.jsonl", "report.md")
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
