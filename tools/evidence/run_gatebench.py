#!/usr/bin/env python3
"""Run the deterministic offline GateBench calculation from a JSON case list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lingjing_solo.evidence.benchmark import build_gatebench_packet, write_gatebench_packet
from lingjing_solo.evidence.metrics import GateBenchCase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="JSON list of GateBench case objects")
    parser.add_argument("--holdout-delta", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--combination-failures", type=int, default=0)
    parser.add_argument("--combination-total", type=int, default=0)
    parser.add_argument("--holdout-cases", type=int, required=True)
    parser.add_argument("--run-id", default="offline-gatebench")
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()
    try:
        raw = json.loads(args.cases.read_text(encoding="utf-8"))
        cases = [GateBenchCase(**item) for item in raw]
        packet = build_gatebench_packet(cases, holdout_delta=args.holdout_delta, holdout_cases=args.holdout_cases, combination_failures=args.combination_failures, combination_total=args.combination_total, run_id=args.run_id, source_commit=args.source_commit)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"REJECTED: {exc}")
        return 1
    write_gatebench_packet(packet, args.out)
    print(f"GATEBENCH cases={packet['metrics']['cases']} environments={len(packet['environments'])} output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
