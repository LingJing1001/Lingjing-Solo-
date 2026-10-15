import json

import pytest
import torch

from lingjing_solo.neural.aop_encoder import encode_label
from lingjing_solo.neural.aop_model import AOPModel, load_checkpoint, save_checkpoint
from tools.aop.dataset import split_labels
from tools.aop.train import assert_disjoint, evaluate, better_holdout, macro_action_accuracy


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


def test_split_keeps_counterfactual_branches_together():
    rows = [
        {**label("branch-a", 0), "counterfactual_group_id": "group-1"},
        {**label("branch-b", 0), "counterfactual_group_id": "group-1"},
        {**label("episode-c", 0)},
        {**label("episode-d", 0)},
    ]
    train, valid = split_labels(rows, validation_fraction=0.5)
    train_groups = {x.get("counterfactual_group_id", x["episode_id"]) for x in train}
    valid_groups = {x.get("counterfactual_group_id", x["episode_id"]) for x in valid}
    assert train_groups.isdisjoint(valid_groups)
    assert {x["episode_id"] for x in train} != {"branch-a", "branch-b"}
    assert {x["episode_id"] for x in valid} != {"branch-a", "branch-b"}


def test_explicit_holdout_leakage_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint([label("same", 0)], [label("same", 1)])


def test_evaluation_reports_raw_masked_and_per_action_metrics():
    model = AOPModel(16, 2, dropout=0.0)
    rows = [label("holdout", 0), label("holdout", 1, effective=0)]
    metrics = evaluate(model, rows, {"CLICK": 0, "__UNUSED__": 1}, torch.ones(2))
    assert metrics["rows"] == 2
    assert set(metrics["per_action"]) == {"CLICK", "__UNUSED__"}
    assert metrics["per_action"]["CLICK"]["support"] == 2
    assert metrics["per_action"]["__UNUSED__"]["support"] == 0
    assert 0.0 <= metrics["action_raw_accuracy"] <= 1.0
    assert 0.0 <= metrics["action_masked_accuracy"] <= 1.0


def test_runtime_candidate_scoring_does_not_count_candidate_echo_as_correct():
    class EchoCandidate(torch.nn.Module):
        def forward(self, features):
            candidate = (features[:, 12] > 0.5).long()
            action = torch.zeros((len(features), 2))
            action[torch.arange(len(features)), candidate] = 10.0
            binary = torch.zeros((len(features), 2))
            return {"action": action, "action_effective": binary, "progressed": binary}

    first = label("holdout-a", 0)
    second = label("holdout-b", 0)
    for row in (first, second):
        row["legal_actions"] = ["CLICK", "OTHER"]
        row["observation_before"]["legal_actions"] = ["CLICK", "OTHER"]
    second["requested_action"] = {"name": "OTHER", "id": 2}
    metrics = evaluate(EchoCandidate(), [first, second], {"CLICK": 0, "OTHER": 1}, torch.ones(2))
    assert metrics["action_raw_accuracy"] == 0.5
    assert metrics["action_masked_accuracy"] == 0.5


def test_checkpoint_round_trip(tmp_path):
    model = AOPModel(16, 4)
    path = tmp_path / "aop.pt"
    save_checkpoint(path, model, {"input_width": 16, "action_classes": 4})
    restored, config = load_checkpoint(path)
    model.eval()
    assert config["input_width"] == 16
    assert torch.allclose(model(torch.zeros(1, 16))["progressed"], restored(torch.zeros(1, 16))["progressed"])


def test_better_holdout_prefers_masked_action_without_progress_regression():
    baseline = {"action_masked_accuracy": 0.40, "progressed_accuracy": 0.90}
    candidate = {"action_masked_accuracy": 0.41, "progressed_accuracy": 0.89}
    assert better_holdout(candidate, baseline)
    rejected = {"action_masked_accuracy": 0.42, "progressed_accuracy": 0.79}
    assert not better_holdout(rejected, baseline)


def test_macro_action_accuracy_ignores_zero_support():
    metrics = {"per_action": {"A": {"support": 2, "masked_accuracy": 0.5}, "B": {"support": 0, "masked_accuracy": None}, "C": {"support": 1, "masked_accuracy": 1.0}}}
    assert macro_action_accuracy(metrics) == 0.75
