import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingjing_solo import LingjingSoloAgent, SoloConfig
from lingjing_solo.core import FieldSnapshot
from lingjing_solo.core.types import Frame
from lingjing_solo.planning import LLMPlanner
from lingjing_solo.reflection import ReflectionTrigger
from lingjing_solo.world_model import WorldModelField


def grid_with_marker(x: int | None = None) -> np.ndarray:
    grid = np.zeros((64, 64), dtype=np.int8)
    if x is not None:
        grid[0, x] = 1
    return grid


def test_reflection_reports_rule_conflict_from_world_model():
    cfg = SoloConfig()
    field = WorldModelField(cfg)
    initial = grid_with_marker()
    first_successor = grid_with_marker(1)
    second_successor = grid_with_marker(2)

    field.update(Frame(grid=initial, t=0))
    field.propose_rule("same state and action", "same successor")
    field.update(
        Frame(grid=first_successor, t=1),
        prev_grid=Frame(grid=initial, t=0),
        action="UP",
    )

    # Re-run the same state/action with a different successor.
    field.grid_state = initial.copy()
    field.update(
        Frame(grid=second_successor, t=2),
        prev_grid=Frame(grid=initial, t=0),
        action="UP",
    )

    signal = ReflectionTrigger(cfg, field).evaluate()

    assert signal.rule_conflict is True
    assert signal.should_reflect is True


def test_reflection_context_contains_valid_actions():
    cfg = SoloConfig()
    field = WorldModelField(cfg)
    field.update(Frame(grid=grid_with_marker(), t=7))

    snapshot = ReflectionTrigger(cfg, field).pack_context(["UP", "RIGHT"], recent_n=3)

    assert snapshot.valid_actions == ["UP", "RIGHT"]
    assert snapshot.step == 7


def test_reflection_throttles_repeated_trigger():
    cfg = SoloConfig(loop_detect_window=4, reflection_min_interval=3)
    field = WorldModelField(cfg)
    grid = grid_with_marker()
    for step in range(4):
        field.update(
            Frame(grid=grid, t=step),
            prev_grid=Frame(grid=grid, t=step - 1) if step else None,
            action="UP" if step else None,
        )

    reflector = ReflectionTrigger(cfg, field)

    assert reflector.should_reflect_now() is True
    assert reflector.should_reflect_now() is False

    field.step += cfg.reflection_min_interval
    assert reflector.should_reflect_now() is True


def test_llm_planner_rejects_invalid_action_and_callback_errors():
    cfg = SoloConfig(llm_calls_per_game=2)
    planner = LLMPlanner(cfg)
    snapshot = FieldSnapshot(
        grid_summary="test",
        rules=[],
        goals=[],
        recent_transitions=[],
        visited_count=0,
        step=1,
    )

    planner.inject_llm(lambda _snapshot, _actions: "NOT_ALLOWED")
    assert planner.plan(snapshot, ["UP"]) is None

    def failing_llm(_snapshot, _actions):
        raise RuntimeError("test failure")

    planner.inject_llm(failing_llm)
    assert planner.plan(snapshot, ["UP"]) is None
    assert planner.calls_used == 2


def test_agent_falls_back_when_llm_returns_invalid_action():
    agent = LingjingSoloAgent.with_llm(
        lambda _snapshot, _actions: "NOT_ALLOWED",
        llm_calls_per_game=1,
        loop_detect_window=4,
        reflection_min_interval=1,
    )
    agent.reset()
    grid = grid_with_marker()
    frames = []

    actions = []
    for _ in range(4):
        action = agent.choose_action(frames, grid)
        actions.append(action)
        frames.append(grid)

    assert all(action in agent.cfg.allowed_actions for action in actions)
    assert agent.llm.calls_used == 1
