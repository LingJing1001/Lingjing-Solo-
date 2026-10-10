#!/usr/bin/env python3
"""Train the offline AOP multitask model from converted labels JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lingjing_solo.neural.aop_encoder import encode_label  # noqa: E402
from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint  # noqa: E402
from tools.aop.dataset import split_labels  # noqa: E402


def read_labels(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_no}: label must be an object")
                rows.append(row)
    return rows


def targets(labels: list[dict], action_to_id: dict[str, int]):
    action_count = max(1, len(action_to_id))
    features = torch.stack([
        encode_label(x, action_id=action_to_id[x["requested_action"]["name"]], action_count=action_count)
        for x in labels
    ])
    effective = torch.tensor([int(bool(x["state_delta"]["action_effective"])) for x in labels])
    progressed = torch.tensor([int(bool(x["state_delta"]["progressed"])) for x in labels])
    actions = torch.tensor([action_to_id[x["requested_action"]["name"]] for x in labels])
    return features, effective, progressed, actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--synthetic", action="store_true", help="mark output as synthetic smoke evidence")
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    labels = read_labels(args.labels)
    train, valid = split_labels(labels, args.validation_fraction)
    names = sorted({x["requested_action"]["name"] for x in labels})
    action_to_id = {name: i for i, name in enumerate(names)}
    model = AOPModel(16, max(2, len(names)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        features, y_effective, y_progressed, y_action = targets(train, action_to_id)
        outputs = model(features)
        loss = (criterion(outputs["action_effective"], y_effective) + criterion(outputs["progressed"], y_progressed) + criterion(outputs["action"], y_action))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        features, y_effective, y_progressed, y_action = targets(valid, action_to_id)
        outputs = model(features)
        val_loss = float((criterion(outputs["action_effective"], y_effective) + criterion(outputs["progressed"], y_progressed) + criterion(outputs["action"], y_action)).item())
    save_checkpoint(args.output, model, {"action_names": names, "train_rows": len(train), "validation_rows": len(valid), "evidence": "synthetic_smoke" if args.synthetic else "converted_recording"})
    print(f"TRAINED epochs={args.epochs} train_rows={len(train)} validation_rows={len(valid)} val_loss={val_loss:.6f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
