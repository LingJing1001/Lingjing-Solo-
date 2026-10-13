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
        encode_label(
            x,
            action_id=action_to_id[x["requested_action"]["name"]],
            action_count=action_count,
        )
        for x in labels
    ])
    effective = torch.tensor([int(bool(x["state_delta"]["action_effective"])) for x in labels])
    progressed = torch.tensor([int(bool(x["state_delta"]["progressed"])) for x in labels])
    actions = torch.tensor([action_to_id[x["requested_action"]["name"]] for x in labels])
    legal = torch.tensor([
        [name in (x.get("legal_actions") or []) for name in action_to_id]
        for x in labels
    ], dtype=torch.bool)
    return features, effective, progressed, actions, legal


def action_loss(logits, targets, legal, class_weights):
    masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    return nn.functional.cross_entropy(masked, targets, weight=class_weights)


def visual_delta(before, after):
    left = (before.get("frame") or {}).get("frame")
    right = (after.get("frame") or {}).get("frame")
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    return sum(a != b for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def counterfactual_rank_loss(logits, action_targets, labels):
    """Rank candidate actions only when a group has informative outcomes."""
    groups = {}
    for index, label in enumerate(labels):
        group_id = label.get("counterfactual_group_id")
        if group_id:
            groups.setdefault(group_id, []).append(index)
    losses = []
    for indexes in groups.values():
        rewards = torch.tensor([
            100.0 * float(bool(labels[i].get("state_delta", {}).get("progressed")))
            + float(bool(labels[i].get("state_delta", {}).get("action_effective")))
            + 0.01 * float(labels[i].get("state_delta", {}).get("visual_delta") or 0.0)
            for i in indexes
        ], dtype=logits.dtype, device=logits.device)
        if rewards.unique().numel() < 2:
            continue
        scores = logits[indexes, action_targets[indexes]]
        target = int(rewards.argmax().item())
        losses.append(nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=logits.device)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


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


def _split_key(label: dict) -> str:
    return str(label.get("counterfactual_group_id") or label["episode_id"])


def assert_disjoint(train: list[dict], holdout: list[dict]) -> None:
    overlap = {_split_key(row) for row in train} & {_split_key(row) for row in holdout}
    if overlap:
        raise ValueError(f"train/holdout leakage detected: {sorted(overlap)[:3]}")


def evaluate(model, labels, action_to_id, class_weights) -> dict:
    if not labels:
        raise ValueError("evaluation labels must not be empty")
    criterion = nn.CrossEntropyLoss()
    features, y_effective, y_progressed, y_action, legal = targets(labels, action_to_id)
    with torch.no_grad():
        outputs = model(features)
        raw_action = outputs["action"].argmax(dim=1)
        masked_logits = outputs["action"].masked_fill(~legal, torch.finfo(outputs["action"].dtype).min)
        masked_action = masked_logits.argmax(dim=1)
        loss = (
            criterion(outputs["action_effective"], y_effective)
            + criterion(outputs["progressed"], y_progressed)
            + action_loss(outputs["action"], y_action, legal, class_weights)
        )
    names = list(action_to_id)
    per_action = {}
    for name, action_id in action_to_id.items():
        selected = y_action == action_id
        support = int(selected.sum().item())
        per_action[name] = {
            "support": support,
            "raw_accuracy": float((raw_action[selected] == y_action[selected]).float().mean().item()) if support else None,
            "masked_accuracy": float((masked_action[selected] == y_action[selected]).float().mean().item()) if support else None,
        }
    return {
        "rows": len(labels),
        "val_loss": float(loss.item()),
        "action_raw_accuracy": float((raw_action == y_action).float().mean().item()),
        "action_masked_accuracy": float((masked_action == y_action).float().mean().item()),
        "action_effective_accuracy": float((outputs["action_effective"].argmax(dim=1) == y_effective).float().mean().item()),
        "progressed_accuracy": float((outputs["progressed"].argmax(dim=1) == y_progressed).float().mean().item()),
        "per_action": {name: per_action[name] for name in names},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path, help="training labels JSONL")
    parser.add_argument("--holdout-labels", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--synthetic", action="store_true", help="mark output as synthetic smoke evidence")
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    labels = read_labels(args.labels)
    if args.holdout_labels:
        train = labels
        valid = read_labels(args.holdout_labels)
    else:
        train, valid = split_labels(labels, args.validation_fraction)
    if not train or not valid:
        raise ValueError("train and holdout must both be non-empty")
    assert_disjoint(train, valid)
    names = action_names_from_labels(train + valid)
    if len(names) == 1:
        names.append("__UNUSED__")
    action_to_id = {name: i for i, name in enumerate(names)}
    model = AOPModel(16, len(names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    class_counts = torch.bincount(
        torch.tensor([action_to_id[x["requested_action"]["name"]] for x in train]),
        minlength=len(names),
    ).float()
    class_weights = (class_counts.clamp_min(1.0).sqrt().reciprocal())
    class_weights = class_weights / class_weights.mean()
    model.eval()
    baseline = evaluate(model, valid, action_to_id, class_weights)
    for _ in range(args.epochs):
        model.train()
        features, y_effective, y_progressed, y_action, legal = targets(train, action_to_id)
        outputs = model(features)
        loss = (
            criterion(outputs["action_effective"], y_effective)
            + criterion(outputs["progressed"], y_progressed)
            + action_loss(outputs["action"], y_action, legal, class_weights)
            + counterfactual_rank_loss(outputs["action"], y_action, train)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    trained = evaluate(model, valid, action_to_id, class_weights)
    train_hash = _sha256(args.labels)
    holdout_hash = _sha256(args.holdout_labels) if args.holdout_labels else train_hash
    manifest_hash = _sha256(args.dataset_manifest) if args.dataset_manifest else None
    deltas = {
        "val_loss_improvement": baseline["val_loss"] - trained["val_loss"],
        "action_raw_accuracy": trained["action_raw_accuracy"] - baseline["action_raw_accuracy"],
        "action_masked_accuracy": trained["action_masked_accuracy"] - baseline["action_masked_accuracy"],
        "action_effective_accuracy": trained["action_effective_accuracy"] - baseline["action_effective_accuracy"],
        "progressed_accuracy": trained["progressed_accuracy"] - baseline["progressed_accuracy"],
    }
    evidence = "synthetic_smoke" if args.synthetic else "real_stage3_packet" if args.holdout_labels else "converted_recording"
    checkpoint_config = {
        "action_names": names,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_split_keys": sorted({_split_key(x) for x in train}),
        "validation_split_keys": sorted({_split_key(x) for x in valid}),
        "train_sha256": train_hash,
        "holdout_sha256": holdout_hash,
        "dataset_manifest_sha256": manifest_hash,
        "git_commit": _git_commit(),
        "training_command": " ".join(sys.argv),
        "seed": args.seed,
        "epochs": args.epochs,
        "evidence": evidence,
        "runtime_authority": False,
        "holdout_metrics": trained,
    }
    save_checkpoint(args.output, model, checkpoint_config)
    metrics = {
        "schema_version": "aop.training.metrics.v1",
        "evidence": evidence,
        "runtime_authority": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_rows": len(train),
        "holdout_rows": len(valid),
        "train_sha256": train_hash,
        "holdout_sha256": holdout_hash,
        "dataset_manifest_sha256": manifest_hash,
        "checkpoint_sha256": _sha256(args.output),
        "git_commit": _git_commit(),
        "baseline": baseline,
        "trained": trained,
        "delta": deltas,
    }
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
