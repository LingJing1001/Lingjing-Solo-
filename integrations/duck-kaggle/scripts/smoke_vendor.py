#!/usr/bin/env python3
"""Validate vendored duck-harness tree (no GPU / no vLLM required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "ARC3-Inference" / "Makefile",
    ROOT / "ARC3-Inference" / "configs" / "inference.json",
    ROOT / "ARC3-Inference" / "inference" / "framework" / "kaggle.py",
    ROOT / "tufa-arc-agi-framework",
    ROOT / "taaf-duck-harness-kaggle-share.ipynb",
]


def main() -> int:
    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        print("FAIL missing:", *missing, sep="\n  ")
        return 1

    kaggle_py = (ROOT / "ARC3-Inference" / "inference" / "framework" / "kaggle.py").read_text(
        encoding="utf-8"
    )
    for needle in (
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        "vrfai/Qwen3.6-27B-FP8",
    ):
        if needle not in kaggle_py:
            print(f"FAIL kaggle.py missing ref: {needle}")
            return 1

    cfg = json.loads(
        (ROOT / "ARC3-Inference" / "configs" / "inference.json").read_text(encoding="utf-8")
    )
    model = cfg.get("shared", {}).get("model_name") or cfg.get("model_name")
    print("OK duck-harness vendor")
    print("  root:", ROOT)
    print("  inference.json model:", model)
    print("  wheelhouse: driessmit1/arc3-vllm-h100-wheelhouse-v3")
    print("  model ds:   driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
