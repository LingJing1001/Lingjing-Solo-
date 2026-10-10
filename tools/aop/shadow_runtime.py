#!/usr/bin/env python3
"""Non-authoritative AOP shadow runtime.

Consumes already-recorded labels, emits model predictions, and never changes
or approves the authoritative action. This is deliberately offline and
fail-closed: inference errors abort rather than silently enabling control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lingjing_solo.neural.aop_encoder import encode_label  # noqa: E402
from lingjing_solo.neural.aop_model import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    model, config = load_checkpoint(args.checkpoint)
    action_names = list(config["action_names"])
    action_to_id = {name: i for i, name in enumerate(action_names)}
    action_count = max(2, len(action_names))
    rows = []
    with torch.no_grad():
        for index, label in enumerate(labels):
            requested = label.get("requested_action", {}).get("name")
            if requested not in action_to_id:
                raise ValueError(f"row {index}: action not in checkpoint vocabulary: {requested}")
            features = encode_label(label, action_id=action_to_id[requested], action_count=action_count).unsqueeze(0)
            outputs = model(features)
            pred = {
                "action_effective": int(outputs["action_effective"].argmax(1).item()),
                "progressed": int(outputs["progressed"].argmax(1).item()),
                "action": action_names[int(outputs["action"].argmax(1).item())]
                if int(outputs["action"].argmax(1).item()) < len(action_names)
                else None,
            }
            rows.append({
                "episode_id": label["episode_id"],
                "step_id": label["step_id"],
                "authoritative_action": requested,
                "prediction": pred,
                "used_for_control": False,
                "model_version": config.get("version", "unknown"),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"SHADOW_OK rows={len(rows)} used_for_control=False output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
