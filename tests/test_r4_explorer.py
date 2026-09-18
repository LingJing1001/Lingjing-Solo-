import numpy as np

from lingjing_solo.core import GoalHypothesis, SoloConfig, Transition
from lingjing_solo.exploration.explorer import ExplorationEngine
from lingjing_solo.world_model.field import WorldModelField


def make_engine(**config_overrides):
    cfg = SoloConfig(**config_overrides)
    field = WorldModelField(cfg)
    return ExplorationEngine(cfg, field), field


def test_probe_budget_allows_exactly_configured_number_of_steps():
    engine, _ = make_engine(probe_max_steps=2)

    engine.start_probe()

    assert engine.step_probe() is True
    assert engine.step_probe() is True
    assert engine.step_probe() is False
    assert engine._probe_budget == 0


def test_empty_actions_score_to_empty_list():
    engine, _ = make_engine()

    assert engine.score_actions([]) == []
    assert engine.last_score_details == {}


def test_score_tie_break_is_stable_and_preserves_input_order():
    engine, _ = make_engine()

    assert [action for action, _ in engine.score_actions(["DOWN", "UP"])] == ["DOWN", "UP"]


def test_goal_successor_gets_optional_score_bonus_and_reason():
    engine, field = make_engine(goal_score_bonus=1.0)
    field.grid_state = np.zeros((4, 4), dtype=np.uint8)
    current = field.current_hash()
    field.predict_graph[(current, "ACTION1")] = "goal-state"
    field.goals = [GoalHypothesis("reach goal", confidence=0.8, state_hash="goal-state")]

    scored = engine.score_actions(["ACTION2", "ACTION1"])

    assert scored[0][0] == "ACTION1"
    assert "goal_successor" in engine.last_score_details["ACTION1"]["reason"]


def test_info_gain_rewards_uncertain_successors():
    deterministic, field = make_engine()
    field.grid_state = np.zeros((8, 8), dtype=np.uint8)
    state = field.current_hash()
    field.transition_index[(state, "ACTION1")] = [
        Transition(state, "ACTION1", "next", 1, 1, False, 1),
        Transition(state, "ACTION1", "next", 1, 2, False, 2),
    ]
    uncertain, uncertain_field = make_engine()
    uncertain_field.grid_state = np.zeros((8, 8), dtype=np.uint8)
    uncertain_state = uncertain_field.current_hash()
    uncertain_field.transition_index[(uncertain_state, "ACTION1")] = [
        Transition(uncertain_state, "ACTION1", "next-a", 1, 1, False, 1),
        Transition(uncertain_state, "ACTION1", "next-b", 1, 2, False, 2),
    ]

    assert uncertain.info_gain("ACTION1") > deterministic.info_gain("ACTION1")
