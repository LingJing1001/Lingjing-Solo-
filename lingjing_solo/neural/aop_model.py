"""Small multitask AOP model with strict versioned checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch import nn


class AOPModel(nn.Module):
    def __init__(self, input_width: int, action_classes: int, dropout: float = 0.1):
        super().__init__()
        if input_width <= 0 or action_classes <= 1:
            raise ValueError("invalid model dimensions")
        self.config = {"input_width": input_width, "action_classes": action_classes, "dropout": dropout, "version": "aop.model.v1"}
        self.body = nn.Sequential(
            nn.Linear(input_width, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(),
        )
        self.action_effective = nn.Linear(128, 2)
        self.progressed = nn.Linear(128, 2)
        self.action = nn.Linear(128, action_classes)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.config["input_width"] or not torch.isfinite(features).all():
            raise ValueError("features have invalid shape or non-finite values")
        hidden = self.body(features)
        return {"action_effective": self.action_effective(hidden), "progressed": self.progressed(hidden), "action": self.action(hidden)}


def save_checkpoint(path: str | Path, model: AOPModel, config: dict[str, Any]) -> None:
    payload = {"format": "aop.model.v1", "config": {**model.config, **config}, "state_dict": model.state_dict()}
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> tuple[AOPModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "aop.model.v1":
        raise ValueError("unsupported AOP checkpoint")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint missing config")
    model = AOPModel(int(config["input_width"]), int(config["action_classes"]), float(config.get("dropout", 0.1)))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, config
