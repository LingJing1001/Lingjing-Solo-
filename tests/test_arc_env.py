"""官方 ARC-AGI-3 harness 桥接层：动作映射、帧→网格、Env 适配、Agent 子类。

需要 ``arcengine``（starter venv 里有）。``agents``（官方 vendor 包）不在
sys.path 上时，``harness_agent_class`` 应当报错而不是让整个模块 import 失败 ——
这条也被测了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lingjing_solo.harness import arc_env
from lingjing_solo.harness.actions import ACTION_NAMES, make_action

arcengine = pytest.importorskip("arcengine")
from arcengine import GameAction, GameState  # noqa: E402

VENDOR = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "ARC-AGI-3-Kaggle-Starter"
    / "vendor"
    / "ARC-AGI-3-Agents"
)


# --------------------------------------------------------------------------- #
# 1. 动作映射
# --------------------------------------------------------------------------- #


def test_every_harness_action_has_an_official_mapping():
    for name in ACTION_NAMES:
        assert name in arc_env.CANONICAL_TO_GAME_ACTION, name


def test_mapping_is_exactly_the_documented_official_convention():
    assert arc_env.CANONICAL_TO_GAME_ACTION == {
        "UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3",
        "RIGHT": "ACTION4", "SPACE": "ACTION5", "CLICK": "ACTION6",
    }


def test_reverse_mapping_roundtrips():
    for canon, official in arc_env.CANONICAL_TO_GAME_ACTION.items():
        assert arc_env.GAME_ACTION_TO_CANONICAL[official] == canon


def test_no_undo_mapping_is_exposed():
    """harness 不给 Undo / RESET —— 桥接层也不许偷偷补一个。"""
    assert "UNDO" not in arc_env.CANONICAL_TO_GAME_ACTION
    assert "ACTION7" not in arc_env.CANONICAL_TO_GAME_ACTION.values()
    assert "RESET" not in arc_env.CANONICAL_TO_GAME_ACTION


def test_load_game_action_returns_enum():
    assert arc_env.load_game_action() is GameAction


# --------------------------------------------------------------------------- #
# 2. available_actions
# --------------------------------------------------------------------------- #


def test_canonical_action_names_from_int_ids_drops_reset():
    names = arc_env.canonical_action_names([0, 1, 2, 3, 4, 5, 6, 7])
    assert names == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
    assert "RESET" not in names


def test_canonical_action_names_accepts_strings_and_dedups():
    assert arc_env.canonical_action_names(["action1", "ACTION1", "RESET"]) == ["ACTION1"]


def test_canonical_action_names_tolerates_garbage():
    assert arc_env.canonical_action_names([None, "??", 99, 3]) == ["ACTION3"]
    assert arc_env.canonical_action_names(None) == []
    assert arc_env.canonical_action_names([]) == []


# --------------------------------------------------------------------------- #
# 3. resolve_game_action
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,official",
    [("UP", "ACTION1"), ("DOWN", "ACTION2"), ("LEFT", "ACTION3"),
     ("RIGHT", "ACTION4"), ("SPACE", "ACTION5")],
)
def test_resolve_simple_actions(name, official):
    assert arc_env.resolve_game_action(make_action(name)) is GameAction[official]


def test_resolve_click_converts_row_col_to_x_y():
    ga = arc_env.resolve_game_action(make_action("CLICK", 5, 9), available=[6])
    assert ga is GameAction.ACTION6
    assert ga.action_data.model_dump() == {"game_id": "", "x": 9, "y": 5}


def test_resolve_downgrades_when_action_not_available():
    """引擎这一帧关掉了某个键时降级，而不是让整步作废。"""
    ga = arc_env.resolve_game_action(make_action("CLICK", 1, 1), available=[1, 2])
    assert ga is GameAction.ACTION1


def test_resolve_prefers_direction_over_click_when_downgrading():
    ga = arc_env.resolve_game_action(make_action("SPACE"), available=[6, 4])
    assert ga is GameAction.ACTION4


def test_resolve_without_available_does_not_consult_the_engine():
    assert arc_env.resolve_game_action(make_action("CLICK", 0, 0)) is GameAction.ACTION6


def test_resolve_rejects_unknown_kind():
    class Fake:
        kind = "TELEPORT"

    with pytest.raises(ValueError):
        arc_env.resolve_game_action(Fake())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 4. frame_to_grid
# --------------------------------------------------------------------------- #


def test_frame_to_grid_from_nested_list():
    assert arc_env.frame_to_grid([[0, 1], [2, 3]]) == [[0, 1], [2, 3]]


def test_frame_to_grid_from_numpy():
    assert arc_env.frame_to_grid(np.array([[4, 5], [6, 7]])) == [[4, 5], [6, 7]]


def test_frame_to_grid_takes_last_of_frame_list():
    """官方真实结构：frame 是"一帧一层"的列表，取最后一层。"""
    frames = [[[0, 1], [2, 3]], [[9, 8], [7, 6]]]
    assert arc_env.frame_to_grid(frames) == [[9, 8], [7, 6]]


def test_frame_to_grid_from_frame_data_like_object():
    class FD:
        frame = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    assert arc_env.frame_to_grid(FD()) == [[5, 6], [7, 8]]


def test_frame_to_grid_from_real_frame_data():
    from arcengine import FrameData

    fd = FrameData(levels_completed=0, frame=[np.zeros((4, 4), dtype=int)])
    g = arc_env.frame_to_grid(fd)
    assert g is not None and len(g) == 4 and len(g[0]) == 4


def test_frame_to_grid_returns_none_for_empty_or_none():
    assert arc_env.frame_to_grid(None) is None
    assert arc_env.frame_to_grid([]) is None
    assert arc_env.frame_to_grid(7) is None


def test_frame_to_grid_casts_floats_to_int():
    assert arc_env.frame_to_grid(np.array([[1.0, 2.0]])) == [[1, 2]]


# --------------------------------------------------------------------------- #
# 5. OfficialEnv（用鸭子类型的 wrapper，不需要真引擎）
# --------------------------------------------------------------------------- #


class _Raw:
    def __init__(self, grid, state, levels, win, avail, game_id="toy"):
        self.frame = [np.array(grid)]
        self.state = state
        self.levels_completed = levels
        self.win_levels = win
        self.available_actions = avail
        self.game_id = game_id


class _Wrapper:
    """最小可用的 EnvironmentWrapper 替身：每 3 步过一关。"""

    def __init__(self, avail=(1, 2, 3, 4, 5, 6), win=2):
        self.avail = list(avail)
        self.win = win
        self.calls = []
        self.levels = 0
        self.n = 0

    def reset(self):
        self.n = 0
        return _Raw([[0, 1], [2, 3]], GameState.NOT_FINISHED, 0, self.win, self.avail)

    def step(self, action, data=None, reasoning=None):
        self.calls.append({"action": action.name, "data": data, "reasoning": reasoning})
        self.n += 1
        if self.n % 3 == 0:
            self.levels += 1
        state = GameState.WIN if self.levels >= self.win else GameState.NOT_FINISHED
        return _Raw([[1, 0], [3, 2]], state, self.levels, self.win, self.avail)


def test_official_env_reset_returns_grid_and_shape():
    env = arc_env.OfficialEnv(_Wrapper(), game_id="toy")
    grid = env.reset()
    assert grid == [[0, 1], [2, 3]]
    assert env.shape == (2, 2)


def test_official_env_available_actions_is_canonical_without_reset():
    env = arc_env.OfficialEnv(_Wrapper(avail=(0, 1, 2, 5, 6)))
    env.reset()
    assert env.available_actions == ["ACTION1", "ACTION2", "ACTION5", "ACTION6"]


def test_official_env_step_reports_reward_only_on_level_up():
    w = _Wrapper()
    env = arc_env.OfficialEnv(w)
    env.reset()
    r1 = env.step(make_action("UP"))
    r2 = env.step(make_action("UP"))
    r3 = env.step(make_action("UP"))
    assert (r1.reward, r2.reward, r3.reward) == (0.0, 0.0, 1.0)
    assert r3.info["levels_completed"] == 1


def test_official_env_step_done_on_win():
    env = arc_env.OfficialEnv(_Wrapper(win=1))
    env.reset()
    out = [env.step(make_action("UP")) for _ in range(3)][-1]
    assert out.done is True and out.info["state"] == "WIN"


def test_official_env_passes_action_data_and_reasoning_through():
    w = _Wrapper(avail=(6,))
    env = arc_env.OfficialEnv(w, reasoning_tag="tag")
    env.reset()
    env.step(make_action("CLICK", 3, 7))
    call = w.calls[-1]
    assert call["action"] == "ACTION6"
    assert call["data"]["x"] == 7 and call["data"]["y"] == 3
    assert call["reasoning"] == {"text": "tag:CLICK(3,7)"}


def test_official_env_downgrades_unavailable_action():
    w = _Wrapper(avail=(1,))
    env = arc_env.OfficialEnv(w)
    env.reset()
    env.step(make_action("SPACE"))
    assert w.calls[-1]["action"] == "ACTION1"


def test_official_env_raises_on_none_from_engine():
    class Bad(_Wrapper):
        def reset(self):
            return None

    with pytest.raises(RuntimeError):
        arc_env.OfficialEnv(Bad()).reset()


def test_official_env_drivable_by_harness():
    """OfficialEnv 能直接喂给 Harness.run_episode（真跑一局 2 步的小局）。"""
    from lingjing_solo.harness import Harness
    from lingjing_solo.harness.loop import FakeLLM

    env = arc_env.OfficialEnv(_Wrapper(win=1), game_id="toy")
    res = Harness(FakeLLM()).run_episode(env, max_turns=3)
    assert len(res.steps) == 3
    assert all(s.ok for s in res.steps)
    assert res.solved is True


# --------------------------------------------------------------------------- #
# 6. HarnessAgent 的按需构造
# --------------------------------------------------------------------------- #


def test_module_getattr_rejects_unknown_names():
    with pytest.raises(AttributeError):
        arc_env.__getattr__("NoSuchThing")


def test_harness_agent_class_is_lazy_and_reports_missing_vendor():
    """vendor 不在 sys.path 上时报 ImportError（而不是让模块 import 失败）。"""
    if (VENDOR / "agents").is_dir() and str(VENDOR) not in sys.path:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            arc_env.harness_agent_class()
    else:  # vendor 已可用，或环境里根本没有 vendor → 只验证可调用
        assert callable(arc_env.harness_agent_class)


@pytest.mark.skipif(not (VENDOR / "agents").is_dir(), reason="official vendor agents not present")
def test_harness_agent_implements_the_official_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(VENDOR))
    arc_env._HARNESS_AGENT_CLS = None  # 清缓存，确保这次真的走 vendor
    try:
        cls = arc_env.harness_agent_class()
    finally:
        arc_env._HARNESS_AGENT_CLS = None

    from lingjing_solo.harness.loop import FakeLLM

    wrapper = _Wrapper(win=9)
    agent = cls(
        card_id="t", game_id="toy", agent_name="t.toy",
        ROOT_URL="http://localhost", record=False, arc_env=wrapper,
        tags=[], llm=FakeLLM(agent_color="5", goal_color="3"),
    )
    assert agent.MAX_ACTIONS > 0

    frame = _Raw([[0, 5], [3, 0]], GameState.NOT_FINISHED, 0, 9, [1, 2, 3, 4, 5])
    assert agent.is_done([frame], frame) is False
    won = _Raw([[0, 5], [3, 0]], GameState.WIN, 1, 9, [1, 2, 3, 4, 5])
    assert agent.is_done([won], won) is True

    action = agent.choose_action([frame], frame)
    assert action in tuple(GameAction)
    # 官方路径也要留证据，且用的是同一套 EpisodeState 语义
    assert len(agent.records) == 1
    assert agent.records[0].exit_code == 0
    assert len(agent.ep_state.history) == 1
    assert agent.ep_state.history[0]["action"] is not None


@pytest.mark.skipif(not (VENDOR / "agents").is_dir(), reason="official vendor agents not present")
def test_harness_agent_falls_back_when_model_emits_no_code(monkeypatch):
    monkeypatch.syspath_prepend(str(VENDOR))
    arc_env._HARNESS_AGENT_CLS = None
    try:
        cls = arc_env.harness_agent_class()
    finally:
        arc_env._HARNESS_AGENT_CLS = None

    from lingjing_solo.harness.loop import FakeLLM

    agent = cls(
        card_id="t", game_id="toy", agent_name="t.toy",
        ROOT_URL="http://localhost", record=False, arc_env=_Wrapper(),
        tags=[], llm=FakeLLM(script=["I cannot do that."]),
    )
    frame = _Raw([[0, 1], [2, 3]], GameState.NOT_FINISHED, 0, 9, [3, 4])
    action = agent.choose_action([frame], frame)
    assert action is GameAction.ACTION3          # 降级到第一个可用动作
    assert agent.records[0].termination_reason == "no_action"
    assert agent.records[0].exit_code == 126
