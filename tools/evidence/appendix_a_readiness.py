#!/usr/bin/env python3
"""Produce an auditable Appendix A readiness report from canonical/evidence artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--gatebench", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--authority-status", choices=("shadow", "production"), default="shadow")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    canonical_files = sorted(args.canonical_root.glob("*/counterfactual_labels.jsonl"))
    canonical_rows = [row for path in canonical_files for row in read_jsonl(path)]
    required = ("schema_version", "provenance", "replay_state_hash", "state_delta")
    canonical_missing = [
        (row.get("episode_id"), key)
        for row in canonical_rows
        for key in required
        if not row.get(key)
    ]
    semantic_keys = [
        (row.get("episode_id"), row.get("step_id"), row.get("candidate_action_id"), row.get("thread"))
        for row in canonical_rows
    ]
    duplicates = sum(count - 1 for count in Counter(semantic_keys).values() if count > 1)
    schemas = Counter(row.get("schema_version") for row in canonical_rows)
    source_commits = Counter((row.get("provenance") or {}).get("source_commit") for row in canonical_rows)

    projection_rows = read_jsonl(args.projection)
    projection_bad = []
    for row in projection_rows:
        provenance = row.get("provenance") or {}
        transition = row.get("transition") or {}
        if row.get("schema_version") != "episode_event.v1":
            projection_bad.append("schema_version")
        if row.get("authority_status") != "shadow":
            projection_bad.append("authority_status")
        if not provenance.get("canonical_shard") or not provenance.get("input_hash"):
            projection_bad.append("canonical_provenance")
        if not row.get("replay_hash") or not transition.get("before_hash") or not transition.get("after_hash"):
            projection_bad.append("replay_or_delta")

    gatebench = json.loads((args.gatebench / "metrics.json").read_text(encoding="utf-8"))
    gatebench_manifest = json.loads((args.gatebench / "manifest.json").read_text(encoding="utf-8"))
    failures = read_jsonl(args.gatebench / "failures.jsonl")
    holdout = json.loads(args.holdout_report.read_text(encoding="utf-8")) if args.holdout_report else None
    approval = json.loads(args.approval.read_text(encoding="utf-8")) if args.approval else None
    if holdout is not None:
        holdout = dict(holdout)
        holdout.setdefault("status", "pass" if holdout.get("passed") else "blocked")
    report = {
        "status": "production_ready" if args.authority_status == "production" else "shadow_only",
        "canonical": {
            "files": len(canonical_files),
            "rows": len(canonical_rows),
            "schema_versions": dict(schemas),
            "source_commits": dict(source_commits),
            "missing_required_fields": len(canonical_missing),
            "duplicate_semantic_keys": duplicates,
            "cross_group_compatibility": {
                "status": "pass" if not canonical_missing and not duplicates and len(schemas) == 1 else "fail",
                "basis": "schema/provenance/replay/action-delta scan across all canonical model shards",
            },
        },
        "projection": {
            "rows": len(projection_rows),
            "invalid_rows": len(projection_bad),
            "authority_status": sorted(str(row.get("authority_status")) for row in projection_rows),
            "status": "pass" if projection_rows and not projection_bad else "fail",
        },
        "gatebench": {
            "run_id": gatebench_manifest.get("run_id"),
            "metrics": gatebench,
            "failure_records": len(failures),
            "drift_by_environment": gatebench.get("gate_drift_by_environment", {}),
            "combination_activation": {
                "failures": gatebench.get("combination_failures", 0),
                "total": gatebench.get("combination_total", 0),
                "rate": gatebench.get("combination_activation_failure_rate", 0.0),
            },
        },
        "counterfactual_verified_holdout": holdout or {
            "status": "blocked",
            "reason": "当前 GateBench packet 只有 recording-adapter cases，没有 candidate predictions keyed to counterfactual verified holdout rows；不得把 recording pass-rate 当作独立 holdout effect。",
        },
        "policy": {
            "authority_status": args.authority_status,
            "external_ugc": "disabled",
            "training_performed": False,
        },
    }
    if approval is not None:
        report["independent_approval"] = approval
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.out.write_text(payload, encoding="utf-8")
    print(f"APPENDIX_A status={report['status']} canonical_rows={len(canonical_rows)} projection_rows={len(projection_rows)} output={args.out}")
    print(f"CROSS_GROUP status={report['canonical']['cross_group_compatibility']['status']} duplicates={duplicates}")
    print(f"HOLDOUT status={report['counterfactual_verified_holdout']['status']}")
    return 0 if report["canonical"]["cross_group_compatibility"]["status"] == "pass" and report["projection"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
