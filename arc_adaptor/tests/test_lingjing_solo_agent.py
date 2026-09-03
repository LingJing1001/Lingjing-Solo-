from types import SimpleNamespace

from arcengine import FrameData, GameAction, GameState
from lingjing_solo.planning import LS20Solver

from agents.templates.lingjing_solo_agent import (
    LingjingSolo,
    _available_actions,
    _frame_grid,
    _ls20_level_plan,
)


def make_frame(state=GameState.NOT_FINISHED, actions=None, game_id="test"):
    return FrameData(
        game_id=game_id,
        frame=[[[0, 1], [2, 3]]],
        state=state,
        levels_completed=0,
        win_levels=7,
        available_actions=(
            actions
            if actions is not None
            else [GameAction.ACTION1, GameAction.ACTION6]
        ),
    )


def test_frame_conversion_and_legal_action_filtering():
    frame = make_frame()
    assert _frame_grid(frame).tolist() == [[0, 1], [2, 3]]
    assert _available_actions(frame) == [GameAction.ACTION1, GameAction.ACTION6]
    assert _available_actions(make_frame(actions=[GameAction.RESET])) == []
    assert _available_actions(SimpleNamespace(available_actions=[999, "bad"])) == []


def test_empty_legal_actions_fail_closed_to_reset():
    agent = object.__new__(LingjingSolo)
    agent.solo = SimpleNamespace(choose_action=lambda *args, **kwargs: "ACTION1")
    assert agent.choose_action([], make_frame(actions=[])) is GameAction.RESET


def test_is_done_uses_official_terminal_states():
    agent = object.__new__(LingjingSolo)
    assert agent.is_done([], make_frame(GameState.WIN))
    assert agent.is_done([], make_frame(GameState.GAME_OVER))
    assert not agent.is_done([], make_frame(GameState.NOT_FINISHED))


def test_choose_action_returns_reset_before_game_start():
    agent = object.__new__(LingjingSolo)
    calls = []
    agent.solo = SimpleNamespace(reset=lambda: calls.append("reset"))
    assert agent.choose_action([], make_frame(GameState.NOT_PLAYED)) is GameAction.RESET
    assert calls == ["reset"]


def test_choose_action_prefers_valid_ls20_solver_plan():
    agent = object.__new__(LingjingSolo)
    agent._ls20_plan = []
    agent._seeded_level = 0
    agent._experiment_actions = []
    agent._experiment_index = 0
    agent.solo = SimpleNamespace(
        choose_action=lambda frames, latest, valid_actions: "ACTION1"
    )
    agent.ls20_solver = SimpleNamespace(
        next_action=lambda grid, valid_actions: "ACTION6"
    )
    frame = make_frame(actions=[GameAction.ACTION1, GameAction.ACTION6], game_id="ls20-9607627b")
    assert agent.choose_action([], frame) is GameAction.ACTION6


def test_ls20_uses_verified_default_plan_without_environment_override(monkeypatch):
    monkeypatch.delenv("LINGJING_LS20_PLAN", raising=False)
    agent = LingjingSolo("card", "ls20-9607627b", "test", "", False, None)
    assert agent._ls20_plan == []
    frame = make_frame(GameState.NOT_PLAYED, game_id="ls20-9607627b")
    assert agent.choose_action([], frame) is GameAction.RESET
    assert agent.ls20_solver._plan == _ls20_level_plan(0)


def test_explicit_ls20_plan_overrides_default(monkeypatch):
    monkeypatch.setenv("LINGJING_LS20_PLAN", "ACTION4,ACTION2")
    agent = LingjingSolo("card", "ls20-9607627b", "test", "", False, None)
    frame = make_frame(GameState.NOT_PLAYED, game_id="ls20-9607627b")
    assert agent.choose_action([], frame) is GameAction.RESET
    assert agent.ls20_solver._plan == ["ACTION4", "ACTION2"]


def test_choose_action_converts_lingjing_name_and_falls_back_to_legal_action():
    agent = object.__new__(LingjingSolo)
    agent.solo = SimpleNamespace(
        choose_action=lambda frames, latest, valid_actions: "ACTION6"
    )
    frame = make_frame(actions=[GameAction.ACTION1, GameAction.ACTION6])
    chosen = agent.choose_action([], frame)
    assert chosen is GameAction.ACTION6
    assert chosen.action_data.x == 0
    assert chosen.action_data.y == 0

    agent.solo = SimpleNamespace(
        choose_action=lambda frames, latest, valid_actions: "UNKNOWN"
    )
    assert agent.choose_action([], frame) is GameAction.ACTION1
