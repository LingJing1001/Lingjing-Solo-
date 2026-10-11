#!/usr/bin/env python3
"""Train the offline AOP multitask model from converted labels JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lingjing_solo.neural.aop_encoder import encode_label  # noqa: E402
from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint  # noqa: E402
from tools.aop.action_schema import action_names_from_labels, normalize_label  # noqa: E402
from tools.aop.dataset import split_labels  # noqa: E402


def read_labels(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_no}: label must be an object")
                rows.append(normalize_label(row))
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
    names = action_names_from_labels(labels)
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
    save_checkpoint(args.output, model, {
        "action_names": names,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_episodes": sorted({x["episode_id"] for x in train}),
        "validation_episodes": sorted({x["episode_id"] for x in valid}),
        "labels_sha256": _sha256(args.labels),
        "git_commit": _git_commit(),
        "training_command": " ".join(sys.argv),
        "evidence": "synthetic_smoke" if args.synthetic else "converted_recording",
    })
    print(f"TRAINED epochs={args.epochs} train_rows={len(train)} validation_rows={len(valid)} val_loss={val_loss:.6f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
