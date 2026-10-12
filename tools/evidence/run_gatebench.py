#!/usr/bin/env python3
"""Run the deterministic offline GateBench calculation from a JSON case list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lingjing_solo.evidence.metrics import GateBenchCase, compute_metrics, markdown_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="JSON list of GateBench case objects")
    parser.add_argument("--holdout-delta", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--combination-failures", type=int, default=0)
    parser.add_argument("--combination-total", type=int, default=0)
    args = parser.parse_args()
    try:
        raw = json.loads(args.cases.read_text(encoding="utf-8"))
        cases = [GateBenchCase(**item) for item in raw]
        metrics = compute_metrics(cases, holdout_delta=args.holdout_delta, combination_failures=args.combination_failures, combination_total=args.combination_total)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"REJECTED: {exc}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown_report(metrics), encoding="utf-8")
    print(f"GATEBENCH cases={metrics['cases']} environments={metrics['environments']} output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
