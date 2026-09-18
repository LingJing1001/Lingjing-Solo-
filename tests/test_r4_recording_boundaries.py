import json

import numpy as np

from lingjing_solo.core import Frame, SoloConfig
from lingjing_solo.exploration.action_diff import analyze_recording
from lingjing_solo.world_model.field import WorldModelField


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


def test_ls20_frame_data_schema_replays_into_r4_field():
    grid0 = np.zeros((64, 64), dtype=np.int8)
    grid1 = grid0.copy(); grid1[10, 10] = 1
    grid2 = grid1.copy(); grid2[10, 11] = 1
    frames = [
        {"frame": grid0.tolist(), "state": "NOT_FINISHED", "levels_completed": 0},
        {"frame": grid1.tolist(), "state": "NOT_FINISHED", "levels_completed": 0,
         "requested_action": {"name": "ACTION3", "id": 3}},
        {"frame": grid2.tolist(), "state": "WIN", "levels_completed": 1,
         "requested_action": {"name": "ACTION4", "id": 4}},
    ]
    field = WorldModelField(SoloConfig())
    previous = None
    for tick, data in enumerate(frames):
        current = Frame(np.asarray(data["frame"], dtype=np.int8), t=tick,
                        state=data["state"], levels_completed=data["levels_completed"])
        action = data.get("requested_action", {}).get("name") if data.get("requested_action") else None
        field.update(current, previous, action)
        previous = current

    assert len(field.transition_table) == 2
    assert field.levels == 1
    assert field.env_state == "WIN"
    assert field.win_hashes


def test_ar25_action_schema_replays_into_r4_field():
    grid0 = np.zeros((8, 8), dtype=np.int8)
    grid1 = grid0.copy(); grid1[2, 2] = 9
    grid2 = grid1.copy(); grid2[2, 3] = 9
    grid3 = grid2.copy(); grid3[3, 3] = 9
    frames = [
        (grid0, "NOT_FINISHED", 0, None),
        (grid1, "NOT_FINISHED", 0, {"name": "ACTION3", "id": 3}),
        (grid2, "NOT_FINISHED", 0, {"name": "ACTION5", "id": 5}),
        (grid3, "WIN", 1, {"name": "ACTION7", "id": 7}),
    ]
    field = WorldModelField(SoloConfig())
    previous = None
    for tick, (grid, state, level, requested_action) in enumerate(frames):
        current = Frame(grid, t=tick, state=state, levels_completed=level)
        action = requested_action["name"] if requested_action else None
        field.update(current, previous, action)
        previous = current

    assert len(field.transition_table) == 3
    assert [t.action for t in field.transition_table] == ["ACTION3", "ACTION5", "ACTION7"]
    assert field.levels == 1
    assert field.env_state == "WIN"
    assert field.win_hashes

def test_recording_rejects_missing_action_after_baseline(tmp_path):
    path = tmp_path / "missing-action.jsonl"
    write_jsonl(path, [{"frame": frame(0)}, {"frame": frame(1)}])

    try:
        analyze_recording(path)
    except ValueError as exc:
        assert "requested_action missing" in str(exc)
    else:
        raise AssertionError("missing action must fail closed")
