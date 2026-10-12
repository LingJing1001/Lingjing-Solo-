#!/usr/bin/env python3
"""Build an auditable EpisodeEvent v1 evidence bundle from a local recording."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lingjing_solo.evidence.adapter import RecordingAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--environment-version", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = RecordingAdapter(args.recording, environment_id=args.environment_id, environment_version=args.environment_version, seed=args.seed, source_commit=args.source_commit).build(args.out)
    except (OSError, ValueError) as exc:
        print(f"REJECTED: {exc}")
        return 1
    print(f"BUILT events={result.event_count} recording_rows={result.recording_rows} replay_hash={result.replay_hash} output={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
