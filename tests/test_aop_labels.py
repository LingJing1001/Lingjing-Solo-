import json
import subprocess
import sys

import pytest

from models.aop.labels import LabelError, row_to_label


def row(**overrides):
    base = {
        "episode_id": "ep-1",
        "tick": 3,
        "requested_action": {"name": "CLICK", "id": 6},
        "observation": {
            "tick": 2, "state": "NOT_FINISHED", "state_hash": "before",
            "frame": [[0]], "levels_completed": 0, "legal_actions": ["CLICK"],
        },
        "data": {
            "tick": 3, "state": "NOT_FINISHED", "state_hash": "after",
            "frame": [[1]], "levels_completed": 0, "legal_actions": ["CLICK"],
        },
    }
    base.update(overrides)
    return base


def test_label_separates_visual_change_from_progress():
    label = row_to_label(row())
    assert label.state_delta["visual_changed"] is True
    assert label.state_delta["action_effective"] is True
    assert label.state_delta["progressed"] is False
    assert label.reward == 0.0


def test_level_advance_is_progress():
    item = row()
    item["data"]["levels_completed"] = 1
    item["data"]["state"] = "WIN"
    label = row_to_label(item)
    assert label.state_delta["progressed"] is True
    assert label.state_delta["levels_delta"] == 1
    assert label.reward == 1.0


def test_reset_and_missing_observation_are_rejected():
    with pytest.raises(LabelError):
        row_to_label(row(requested_action={"name": "RESET", "id": 0}))
    with pytest.raises(LabelError):
        row_to_label(row(data=None))


def test_visual_delta_is_not_required_for_state_progress():
    item = row()
    item["observation"]["state_hash"] = "same"
    item["data"]["state_hash"] = "same"
    item["observation"]["frame"] = [[0]]
    item["data"]["frame"] = [[0]]
    item["data"]["state"] = "WIN"
    label = row_to_label(item)
    assert label.state_delta["visual_changed"] is False
    assert label.state_delta["progressed"] is True


def test_cli_is_deterministic(tmp_path):
    source = tmp_path / "recording.jsonl"
    output = tmp_path / "labels.jsonl"
    source.write_text(json.dumps(row(), ensure_ascii=False) + "\n", encoding="utf-8")
    cmd = [sys.executable, "tools/aop/build_labels.py", str(source), str(output)]
    first = subprocess.run(cmd, capture_output=True, text=True, check=True)
    content1 = output.read_text(encoding="utf-8")
    second = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "labels_written=1" in first.stdout
    assert "labels_written=1" in second.stdout
    assert content1 == output.read_text(encoding="utf-8")
