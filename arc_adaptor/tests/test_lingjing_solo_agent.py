from types import SimpleNamespace

from arcengine import FrameData, GameAction, GameState

from agents.templates.lingjing_solo_agent import (
    LingjingSolo,
    _available_actions,
    _frame_grid,
    _ls20_level_plan,
)
from agents.strategies import GenericStrategy, LS20Strategy


def make_frame(state=GameState.NOT_FINISHED, actions=None, game_id="test"):
    return FrameData(
        game_id=game_id,
        frame=[[[0, 1], [2, 3]]],
        state=state,
        levels_completed=0,
        win_levels=7,
        available_actions=actions if actions is not None else [GameAction.ACTION1, GameAction.ACTION6],
    )


def test_frame_conversion_and_legal_action_filtering():
    assert _frame_grid(make_frame()).tolist() == [[0, 1], [2, 3]]
    assert _available_actions(make_frame()) == [GameAction.ACTION1, GameAction.ACTION6]
    assert _available_actions(make_frame(actions=[GameAction.RESET])) == []
    assert _available_actions(SimpleNamespace(available_actions=[999, "bad"])) == []


def test_registry_resolves_specialized_and_generic_strategies():
    agent = LingjingSolo("card", "test", "test", "", False, None)
    assert isinstance(agent.strategies.resolve("ls20-9607627b"), LS20Strategy)
    assert isinstance(agent.strategies.resolve("r11l-495a7899"), GenericStrategy)


def test_empty_legal_actions_fail_closed_to_reset():
    agent = LingjingSolo("card", "test", "test", "", False, None)
    assert agent.choose_action([], make_frame(actions=[])) is GameAction.RESET


def test_is_done_uses_official_terminal_states():
    agent = LingjingSolo("card", "test", "test", "", False, None)
    assert agent.is_done([], make_frame(GameState.WIN))
    assert agent.is_done([], make_frame(GameState.GAME_OVER))
    assert not agent.is_done([], make_frame(GameState.NOT_FINISHED))


def test_ls20_strategy_loads_verified_default_plan(monkeypatch):
    monkeypatch.delenv("LINGJING_LS20_PLAN", raising=False)
    agent = LingjingSolo("card", "ls20-9607627b", "test", "", False, None)
    frame = make_frame(GameState.NOT_PLAYED, game_id="ls20-9607627b")
    assert agent.choose_action([], frame) is GameAction.RESET
    strategy = agent.strategies.resolve("ls20-9607627b")
    assert strategy.solver._plan == _ls20_level_plan(0)


def test_explicit_ls20_plan_overrides_default(monkeypatch):
    monkeypatch.setenv("LINGJING_LS20_PLAN", "ACTION4,ACTION2")
    agent = LingjingSolo("card", "ls20-9607627b", "test", "", False, None)
    frame = make_frame(GameState.NOT_PLAYED, game_id="ls20-9607627b")
    assert agent.choose_action([], frame) is GameAction.RESET
    strategy = agent.strategies.resolve("ls20-9607627b")
    assert strategy.solver._plan == ["ACTION4", "ACTION2"]


def test_strategy_returns_valid_ls20_action():
    agent = LingjingSolo("card", "ls20-9607627b", "test", "", False, None)
    frame = make_frame(GameState.NOT_PLAYED, game_id="ls20-9607627b")
    agent.choose_action([], frame)
    action = agent.choose_action([], make_frame(game_id="ls20-9607627b"))
    assert action in {GameAction.ACTION1, GameAction.ACTION6}


def test_generic_strategy_uses_core_and_falls_back_to_legal_action():
    agent = LingjingSolo("card", "r11l-495a7899", "test", "", False, None)
    agent.solo.choose_action = lambda *args, **kwargs: "UNKNOWN"
    action = agent.choose_action([], make_frame(game_id="r11l-495a7899"))
    assert action is GameAction.ACTION1
