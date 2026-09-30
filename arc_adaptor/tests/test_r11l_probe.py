from types import SimpleNamespace

import numpy as np

from tools.r11l_single_action_probe import evidence, frame_hash, plane


def observation(frame, state="GameState.NOT_FINISHED", levels_completed=0, actions=(6,)):
    return SimpleNamespace(
        frame=frame,
        state=state,
        levels_completed=levels_completed,
        available_actions=list(actions),
    )


def test_plane_accepts_single_channel_arc_frame():
    assert plane(observation([[[1, 2], [3, 4]]])).tolist() == [[1, 2], [3, 4]]


def test_frame_hash_is_deterministic():
    grid = np.array([[1, 2], [3, 4]], dtype=np.int16)
    assert frame_hash(grid) == frame_hash(grid.copy())


def test_evidence_records_click_and_changed_bbox():
    before = observation([[[0, 0], [0, 0]]])
    after = observation([[[0, 1], [0, 0]]])
    result = evidence(before, after, 12, 34)
    assert result["action"] == {"name": "ACTION6", "id": 6, "x": 12, "y": 34}
    assert result["changed_cells"] == 1
    assert result["changed_bbox"] == {"row_min": 0, "row_max": 0, "col_min": 1, "col_max": 1}


def test_evidence_preserves_terminal_and_level_state():
    before = observation([[[0]]], levels_completed=0)
    after = observation([[[0]]], state="GameState.WIN", levels_completed=6)
    result = evidence(before, after, 0, 0)
    assert result["after"]["state"] == "GameState.WIN"
    assert result["after"]["levels_completed"] == 6
    assert result["changed_cells"] == 0
    assert result["changed_bbox"] is None
