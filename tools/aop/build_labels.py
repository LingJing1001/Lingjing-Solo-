#!/usr/bin/env python3
"""Build deterministic AOP JSONL labels from an ARC/CEAX recording JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aop.labels import LabelError, row_to_label  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                label = row_to_label(json.loads(line))
            except (json.JSONDecodeError, LabelError) as exc:
                raise SystemExit(f"line {line_no}: rejected recording row: {exc}") from exc
            dst.write(json.dumps(label.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    print(f"labels_written={written} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
