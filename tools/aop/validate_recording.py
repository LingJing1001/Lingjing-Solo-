#!/usr/bin/env python3
"""Validate an ARC/CEAX recording before converting it to AOP labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aop.recording_gate import RecordingValidationError, validate_jsonl  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("recording", type=Path)
parser.add_argument("--min-episodes", type=int, default=1)
args = parser.parse_args()
try:
    report = validate_jsonl(str(args.recording))
    if report.episodes < args.min_episodes:
        raise RecordingValidationError(
            f"need at least {args.min_episodes} episodes, got {report.episodes}"
        )
except (OSError, RecordingValidationError) as exc:
    print(f"REJECTED: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
print(f"ACCEPTED rows={report.rows} episodes={report.episodes} resets={report.resets} actions={report.actions}")
