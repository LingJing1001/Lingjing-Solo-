"""Evaluate AOP action predictions with and without legal-action masking."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from lingjing_solo.neural.action_mask import choose_masked_action
from lingjing_solo.neural.aop_encoder import encode_label
from lingjing_solo.neural.aop_model import load_checkpoint
from tools.aop.train import read_labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(checkpoint: str | Path, labels_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    labels_file = Path(labels_path)
    model, config = load_checkpoint(checkpoint_path)
    names = list(config.get("action_names", []))
    action_to_id = {name: i for i, name in enumerate(names)}
    labels = read_labels(labels_file)
    if not names:
        raise ValueError("checkpoint has no action_names")
    counts = Counter()
    masked_counts = Counter()
    unknown_legal_rows = 0
    with torch.no_grad():
        for row in labels:
            actual = row["requested_action"]["name"]
            if actual not in action_to_id:
                raise ValueError(f"label action not present in checkpoint: {actual}")
            feature = encode_label(row, action_id=action_to_id[actual], action_count=len(names)).unsqueeze(0)
            logits = model(feature)["action"][0]
            raw = names[int(logits.argmax().item())]
            counts["rows"] += 1
            counts["raw_correct"] += int(raw == actual)
            legal = row.get("legal_actions") or []
            try:
                masked = choose_masked_action(logits, legal, names)
            except ValueError:
                unknown_legal_rows += 1
                continue
            counts["masked_legal_rows"] += int(masked in legal)
            counts["masked_correct"] += int(masked == actual)
            masked_counts[f"actual_{actual}"] += 1
            masked_counts[f"predicted_{masked}"] += 1
    rows = counts["rows"]
    evaluated = counts["masked_legal_rows"]
    return {
        "schema": "aop.evaluation.v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "labels": str(labels_file),
        "labels_sha256": _sha256(labels_file),
        "git_commit": _git_commit(),
        "model_version": config.get("version"),
        "rows": rows,
        "masked_rows": evaluated,
        "unknown_legal_rows": unknown_legal_rows,
        "raw_action_accuracy": counts["raw_correct"] / rows if rows else 0.0,
        "masked_action_accuracy": counts["masked_correct"] / evaluated if evaluated else 0.0,
        "masked_legal_prediction_rate": counts["masked_legal_rows"] / rows if rows else 0.0,
        "counts": dict(sorted(masked_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"EVALUATED rows={report['rows']} raw_acc={report['raw_action_accuracy']:.4f} masked_acc={report['masked_action_accuracy']:.4f} unknown_legal={report['unknown_legal_rows']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
