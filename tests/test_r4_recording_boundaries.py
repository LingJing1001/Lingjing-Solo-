import json

import numpy as np

from lingjing_solo.exploration.action_diff import analyze_recording


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def frame(x):
    grid = np.zeros((4, 4), dtype=int)
    grid[1, x] = 1
    return grid.tolist()


def test_recording_reset_starts_a_new_baseline(tmp_path):
    path = tmp_path / "reset.jsonl"
    write_jsonl(
        path,
        [
            {"data": {"frame": frame(0), "state": "NOT_FINISHED", "levels_completed": 0}},
            {"data": {"frame": frame(1), "requested_action": "ACTION1", "state": "NOT_FINISHED", "levels_completed": 0}},
            {"data": {"frame": frame(3), "requested_action": "RESET", "state": "RESET", "levels_completed": 0}},
            {"data": {"frame": frame(2), "requested_action": {"name": "ACTION2"}, "state": "NOT_FINISHED", "levels_completed": 1}},
        ],
    )

    deltas = analyze_recording(path)

    assert [delta.action for delta in deltas] == ["ACTION1", "ACTION2"]
    assert deltas[1].player_before == (3.0, 1.0)
    assert deltas[1].levels_completed_after == 1


def test_recording_rejects_missing_action_after_baseline(tmp_path):
    path = tmp_path / "missing-action.jsonl"
    write_jsonl(path, [{"frame": frame(0)}, {"frame": frame(1)}])

    try:
        analyze_recording(path)
    except ValueError as exc:
        assert "requested_action missing" in str(exc)
    else:
        raise AssertionError("missing action must fail closed")
