import json

import pytest

from tools.aop.normalize_arc_recording import normalize
from models.aop.recording_gate import validate_jsonl


def test_normalize_arc_recorder_pairs_settled_action(tmp_path):
    raw = tmp_path / "raw.recording.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    frames = [
        {"game_id": "ls20", "guid": "ep-1", "frame": [[0]], "state": "NOT_FINISHED", "levels_completed": 0,
         "available_actions": ["ACTION1"]},
        {"game_id": "ls20", "guid": "ep-1", "frame": [[1]], "state": "NOT_FINISHED", "levels_completed": 0,
         "available_actions": ["ACTION1"], "requested_action": {"name": "ACTION1", "id": 1}},
        {"game_id": "ls20", "guid": "ep-1", "frame": [[2]], "state": "WIN", "levels_completed": 1,
         "available_actions": [], "requested_action": {"name": "ACTION1", "id": 1}},
    ]
    raw.write_text("\n".join(json.dumps({"data": x}) for x in frames) + "\n", encoding="utf-8")

    assert normalize(raw, normalized) == 2
    report = validate_jsonl(str(normalized))
    assert report.rows == 2
    assert report.episodes == 1
    assert report.actions == 2


def test_normalize_tick_trail_uses_outer_observation(tmp_path):
    raw = tmp_path / "evidence.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    row = {
        "observation": {"frame": [[0]], "state": "NOT_FINISHED", "levels_completed": 0,
                        "legal_actions": ["ACTION1"]},
        "data": {"episode_id": "ep-tick", "tick": 1, "frame": [[1]], "state": "NOT_FINISHED",
                  "levels_completed": 0, "legal_actions": ["ACTION1"],
                  "requested_action": {"name": "ACTION1", "id": 1}},
    }
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert normalize(raw, normalized) == 1
    output = json.loads(normalized.read_text(encoding="utf-8"))
    assert output["episode_id"] == "ep-tick"
    assert output["observation"]["frame"] == [[0]]
    assert output["data"]["frame"] == [[1]]


def test_normalize_rejects_missing_action(tmp_path):
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [{"data": {"guid": "ep", "frame": [[0]]}}, {"data": {"guid": "ep", "frame": [[1]]}}]
    raw.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no action-bearing"):
        normalize(raw, out)
