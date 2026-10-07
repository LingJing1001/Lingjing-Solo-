import json

import pytest
import torch

from lingjing_solo.neural.aop_encoder import encode_label
from lingjing_solo.neural.aop_model import AOPModel, load_checkpoint, save_checkpoint
from tools.aop.dataset import split_labels


def label(ep, step, effective=1, progressed=0):
    return {
        "episode_id": ep, "step_id": step,
        "requested_action": {"name": "CLICK", "id": 6},
        "observation_before": {"tick": step, "state": "NOT_FINISHED", "levels_completed": 0, "legal_actions": ["CLICK"]},
        "observation_after": {"tick": step + 1, "state": "WIN" if progressed else "NOT_FINISHED", "levels_completed": progressed, "legal_actions": ["CLICK"]},
        "state_delta": {"action_effective": bool(effective), "progressed": bool(progressed), "levels_delta": progressed, "score_delta": 0.0, "visual_changed": bool(effective)},
        "reward": float(progressed), "level_completed": progressed, "legal_actions": ["CLICK"],
        "temporal": {"tick_before": step, "tick_after": step + 1, "dt": 1},
    }


def test_encoder_has_fixed_finite_width():
    x = encode_label(label("e1", 0))
    assert x.shape == (16,)
    assert torch.isfinite(x).all()


def test_encoder_rejects_nan_and_model_heads_have_expected_shapes():
    bad = label("e1", 0)
    bad["state_delta"]["score_delta"] = float("nan")
    with pytest.raises(ValueError):
        encode_label(bad)
    model = AOPModel(16, 4)
    outputs = model(torch.zeros(2, 16))
    assert outputs["action_effective"].shape == (2, 2)
    assert outputs["progressed"].shape == (2, 2)


def test_split_is_episode_disjoint():
    train, valid = split_labels([label("a", 0), label("a", 1), label("b", 0), label("c", 0)], validation_fraction=0.34)
    assert {x["episode_id"] for x in train}.isdisjoint({x["episode_id"] for x in valid})
    assert train and valid


def test_checkpoint_round_trip(tmp_path):
    model = AOPModel(16, 4)
    path = tmp_path / "aop.pt"
    save_checkpoint(path, model, {"input_width": 16, "action_classes": 4})
    restored, config = load_checkpoint(path)
    model.eval()
    assert config["input_width"] == 16
    assert torch.allclose(model(torch.zeros(1, 16))["progressed"], restored(torch.zeros(1, 16))["progressed"])
