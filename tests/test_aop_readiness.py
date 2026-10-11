import json

import pytest
import torch

from lingjing_solo.neural.action_mask import choose_masked_action, mask_action_logits
from tools.aop.action_schema import normalize_label
from tools.aop.qa import build_report


def test_masked_logits_cannot_select_illegal_action():
    logits = torch.tensor([[0.1, 9.0, 0.2]])
    masked = mask_action_logits(logits, ["ACTION1", "ACTION3"], ["ACTION1", "ACTION2", "ACTION3"])
    assert int(masked.argmax(dim=-1).item()) == 2
    assert torch.isneginf(masked[0, 1])


def test_unknown_legal_action_fails_closed():
    with pytest.raises(ValueError, match="unknown legal action"):
        choose_masked_action(torch.tensor([1.0, 2.0]), ["ACTION9"], ["ACTION1", "ACTION2"])


def test_numeric_legal_actions_are_canonicalized():
    row = normalize_label({
        "episode_id": "g1_rep0",
        "requested_action": {"id": "ACTION2"},
        "legal_actions": [1, 2, 7],
    })
    assert row["requested_action"]["name"] == "ACTION2"
    assert row["legal_actions"] == ["ACTION1", "ACTION2", "ACTION7"]


def test_qa_report_counts_real_rows_and_episodes(tmp_path):
    rows = [
        {"episode_id": "g1_rep0", "requested_action": {"name": "ACTION1"}, "legal_actions": ["ACTION1"]},
        {"episode_id": "g1_rep0", "requested_action": {"name": "ACTION2"}, "legal_actions": ["ACTION1", "ACTION2"]},
        {"episode_id": "g2_rep0", "requested_action": {"name": "ACTION1"}, "legal_actions": ["ACTION1"]},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = build_report(path)
    assert report["rows"] == 3
    assert report["episodes"] == 2
    assert report["action_counts"] == {"ACTION1": 2, "ACTION2": 1}
    assert report["requested_action_legal_rate"] == 1.0
    assert report["duplicate_transition_rows"] == 0
