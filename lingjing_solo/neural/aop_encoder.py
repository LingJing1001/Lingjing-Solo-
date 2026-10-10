"""Versioned finite-width encoder for AOP labels."""
from __future__ import annotations

import math
from typing import Any
import torch

INPUT_WIDTH = 16


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return default


def encode_label(label: dict[str, Any], *, action_id: int = 0, action_count: int = 1) -> torch.Tensor:
    """Encode only information available before executing the candidate action.

    Targets such as action_effective/progressed, after-state, deltas, and reward
    are deliberately excluded. ``action_id`` represents the candidate action
    being scored by the model.
    """
    before = label.get("observation_before")
    action = label.get("requested_action")
    if not isinstance(before, dict) or not isinstance(action, dict):
        raise ValueError("label is missing before observation or action")
    legal = label.get("legal_actions") or before.get("legal_actions") or []
    if not isinstance(legal, list):
        raise ValueError("legal_actions must be a list")
    delta = label.get("state_delta")
    if isinstance(delta, dict) and isinstance(delta.get("score_delta"), float) and not math.isfinite(delta["score_delta"]):
        raise ValueError("label contains non-finite score_delta")
    state = str(before.get("state", "")).upper()
    state_features = [
        float(state == "NOT_FINISHED"), float(state == "WIN"),
        float(state == "GAME_OVER"), float(state == "RESET"),
    ]
    payload = action.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("action payload must be an object")
    denom = max(1, int(action_count) - 1)
    features = [
        _num(before.get("levels_completed")), _num(before.get("score")),
        _num(before.get("tick")), _num(label.get("step_id")),
        _num(len(legal)), float(action_id) / denom,
        _num(payload.get("x")), _num(payload.get("y")),
        float(action.get("name") in legal), *state_features,
        _num(action.get("id")),
        _num(before.get("frame_id")), _num(before.get("steps_left")),
    ]
    tensor = torch.tensor(features, dtype=torch.float32)
    if tensor.shape != (INPUT_WIDTH,) or not torch.isfinite(tensor).all():
        raise ValueError("encoded feature vector is invalid")
    return tensor
