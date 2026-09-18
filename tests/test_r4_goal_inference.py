import numpy as np

from lingjing_solo.core import Frame, SoloConfig
from lingjing_solo.exploration.explorer import ExplorationEngine
from lingjing_solo.world_model.field import WorldModelField


def make_engine():
    field = WorldModelField(SoloConfig())
    engine = ExplorationEngine(SoloConfig(), field)
    return engine, field


def test_infer_goal_records_callback_feedback_with_confidence_and_kind():
    engine, field = make_engine()
    field.grid_state = np.zeros((64, 64), dtype=np.int8)

    result = engine.infer_goal(
        lambda _field: {
            "state_hash": "win-state",
            "description": "reach the completed level",
            "confidence": 0.9,
            "kind": "win",
        }
    )

    assert result == "reach the completed level"
    assert field.goals[0].state_hash == "win-state"
    assert field.goals[0].confidence == 0.9
    assert field.goals[0].kind == "win"


def test_infer_goal_uses_authoritative_win_state_when_callback_is_absent():
    engine, field = make_engine()
    grid = np.zeros((64, 64), dtype=np.int8)
    field.update(Frame(grid=grid, state="WIN", levels_completed=1, t=1))

    assert engine.infer_goal(None) == f"win:{next(iter(field.win_hashes))}"
    assert any(goal.kind == "win" for goal in field.goals)
