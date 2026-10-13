"""learned graph 边键的坐标盲区测试（#9 / P1 证据线）。

**缺陷（2026-10-12 由 dc22 旁路产物坐实）**：边键的动作分量只用枚举名
（`r3_shadow.py:156` 写入、`r3_kernel.py:239` 读取），而 `ACTION6/ACTION7` 是带
`(x,y)` payload 的复杂动作。同一状态下不同坐标的点击因此塌成一条边：

- 写入端 `LearnedGraphBackend._graph[(s, "ACTION6")] = s'` 后写覆盖先写；
- 读取端 P1 noop 用同一个名字查边，判据与该点哪个坐标无关。

实测证据（`state/r3_shadow/dc22_20261011_122838.jsonl`，399 tick）：108 次 ACTION6
回显只留下 19 个边键，其中 6 个键存过 >1 个不同后继 ⇒ 28 条点击转移里 9 条被静默覆盖；
整轮 `P1_NOOP` 命中 1751 次、落在 `ACTION6` 上的次数是 **0**（点击通道恒失明）。

本文件覆盖：
    1. `action_edge_key()` 口径：键盘动作不变名、点击动作带坐标
    2. 写入端：从官方回显 `ActionInput(id, data)` 取坐标，不同坐标留两条边
    3. 读取端：P1 noop 只废掉那个坐标的候选，不误伤同状态其他坐标
    4. plan 出口的动作名不变（§3.2.1-3：坐标只进 payload_hint，不塞进枚举名）

只读被测代码，不碰打分入口、不写任何产物。
"""
from __future__ import annotations

from typing import Any, Hashable

from arcengine import ActionInput, GameAction

from lingjing_solo.planning.backends import LearnedGraphBackend
from lingjing_solo.planning.r3_kernel import PruningContract, R3Kernel, SearchBudget
from lingjing_solo.planning.plan_contract import validate_plan
from lingjing_solo.planning.r3_shadow import _action_edge_key, _action_name
from lingjing_solo.planning.world_view import (
    ActionSpec, ConfigSpace, WorldView, action_edge_key,
)


# ── 测试用 WorldView：两个坐标的点击候选 + 一个键盘动作 ──────────

class _Frame:
    def __init__(self, grid=None, state="NOT_FINISHED", levels=0):
        self.grid = grid or [[0, 0], [0, 0]]
        self.state = state
        self.levels = levels


class _ClickWorldView(WorldView):
    """点击游戏最小 profile：proposals 带 payload_hint 坐标。"""

    def grids(self, frame: Any) -> list[Any]:
        return [frame.grid]

    def key(self, grids: list[Any]) -> Hashable:
        return tuple(tuple(row) for row in grids[0])

    def counters(self, grids: list[Any]) -> dict[str, int]:
        return {"steps": 0}

    def proposals(self, grids: list[Any], key: Hashable) -> list[ActionSpec]:
        return [
            ActionSpec("ACTION3"),
            ActionSpec("ACTION6", payload_hint={"x": 1, "y": 2}),
            ActionSpec("ACTION6", payload_hint={"x": 3, "y": 4}),
        ]

    def config_space(self, grids: list[Any]) -> ConfigSpace | None:
        return None

    def score(self, grids: list[Any]) -> float:
        return 0.0

    def progress(self, frame: Any) -> int:
        return frame.levels

    def done(self, frame: Any) -> bool:
        return frame.state == "WIN"

    def dead(self, frame: Any) -> bool:
        return frame.state == "GAME_OVER"


LEGAL = ["ACTION3", "ACTION6"]


# ── 1. 边键口径 ───────────────────────────────────────────────

class TestActionEdgeKey:
    def test_keyboard_action_stays_name_only(self):
        """键盘/UNDO 两端都不带坐标（写入=官方回显 data，读取=payload_hint）。"""
        assert action_edge_key("ACTION3") == "ACTION3"
        assert action_edge_key("ACTION3", {}) == "ACTION3"
        assert action_edge_key("ACTION3", {"x": None, "y": None}) == "ACTION3"

    def test_click_action_carries_coordinates(self):
        assert action_edge_key("ACTION6", {"x": 12, "y": 34}) == "ACTION6@12,34"

    def test_coordinate_bearing_name_is_always_keyed_by_coord(self):
        """两端都过同一个函数 ⇒ 键口径由构造一致，不靠"哪个名字算点击"的名单。"""
        assert action_edge_key("ACTION7", {"x": 5, "y": 6}) == "ACTION7@5,6"

    def test_non_int_payload_falls_back_to_name(self):
        """缺坐标 / 类型不对时退回名字，不能让边键变成不可比的东西。"""
        assert action_edge_key("ACTION6", {}) == "ACTION6"
        assert action_edge_key("ACTION6", {"x": 1}) == "ACTION6"
        assert action_edge_key("ACTION6", {"x": "a", "y": "b"}) == "ACTION6"

    def test_empty_name_is_empty_key(self):
        assert action_edge_key("") == ""
        assert action_edge_key("", {"x": 1, "y": 2}) == ""


# ── 2. 写入端：官方回显里的坐标必须进边键 ───────────────────────

class TestWriteSideFromEcho:
    def test_action_name_drops_coordinates(self):
        """旧口径（保留）：动作名仍只是枚举名，给日志与 violation 用。"""
        echoed = ActionInput(id=GameAction.ACTION6, data={"x": 12, "y": 34})
        assert _action_name(echoed) == "ACTION6"

    def test_edge_key_reads_echo_coordinates(self):
        echoed = ActionInput(id=GameAction.ACTION6, data={"x": 12, "y": 34})
        assert _action_edge_key(echoed) == "ACTION6@12,34"

    def test_keyboard_echo_has_no_coordinate_suffix(self):
        echoed = ActionInput(id=GameAction.ACTION3, data={})
        assert _action_edge_key(echoed) == "ACTION3"

    def test_empty_echo_gives_empty_key(self):
        assert _action_edge_key(None) == ""

    def test_two_clicks_same_state_keep_two_edges(self):
        """#9 原故障：同一状态下两个坐标的点击，旧键只剩一条、后写覆盖先写。"""
        graph = LearnedGraphBackend()
        state = "s"
        graph.update(state, _action_edge_key(ActionInput(id=GameAction.ACTION6,
                                                          data={"x": 1, "y": 2})), "a")
        graph.update(state, _action_edge_key(ActionInput(id=GameAction.ACTION6,
                                                          data={"x": 3, "y": 4})), "b")
        assert graph.edge_count == 2
        assert graph.predict(state, "ACTION6@1,2") == "a"
        assert graph.predict(state, "ACTION6@3,4") == "b"


# ── 3. 读取端：P1 noop 不能再一视同仁 ──────────────────────────

class _NoPayloadClickWorldView(_ClickWorldView):
    """旧口径等价物：点击候选只有动作名、没有 payload。"""

    def proposals(self, grids: list[Any], key: Hashable) -> list[ActionSpec]:
        return [
            ActionSpec("ACTION3"),
            ActionSpec("ACTION6"),
            ActionSpec("ACTION6"),
        ]


class TestP1NoopIsCoordinateAware:
    def test_old_name_only_key_overwrites_the_other_coordinate(self):
        """原故障（图层面）：两条点击转移用同一个名字当键 ⇒ 只剩一条、后写覆盖。"""
        graph = LearnedGraphBackend()
        graph.update("s", "ACTION6", "a")
        graph.update("s", "ACTION6", "b")
        assert graph.edge_count == 1
        assert graph.predict("s", "ACTION6") == "b"

    def test_old_key_prunes_every_click_candidate(self):
        """原故障（闸门层面）：键里没坐标时，一条 noop 自环废掉全部点击候选。"""
        graph = LearnedGraphBackend()
        graph.update("s", "ACTION6", "s")
        kernel = R3Kernel(_NoPayloadClickWorldView(), graph,
                          budget=SearchBudget(time_ms=5000))
        kernel.plan(_Frame(), LEGAL, "s")
        p1 = [v for v in kernel.violations if v.contract == PruningContract.P1_NOOP]
        assert len(p1) == 2

    def test_legacy_name_only_edge_does_not_prune_coord_candidates(self):
        """新口径：带坐标的候选查不到纯名边 ⇒ 不误判 noop（fail-open）。"""
        graph = LearnedGraphBackend()
        graph.update("s", "ACTION6", "s")
        kernel = R3Kernel(_ClickWorldView(), graph, budget=SearchBudget(time_ms=5000))
        plan = kernel.plan(_Frame(), LEGAL, "s")
        p1 = [v for v in kernel.violations if v.contract == PruningContract.P1_NOOP]
        assert not p1
        assert plan["candidate_actions"].count("ACTION6") == 2

    def test_noop_prunes_only_that_coordinate(self):
        """(s, ACTION6@1,2) 学到自环 ⇒ 只废 (1,2) 这个坐标，(3,4) 与键盘动作保留。"""
        graph = LearnedGraphBackend()
        graph.update("s", "ACTION6@1,2", "s")
        kernel = R3Kernel(_ClickWorldView(), graph, budget=SearchBudget(time_ms=5000))
        plan = kernel.plan(_Frame(), LEGAL, "s")

        assert "ACTION3" in plan["candidate_actions"]
        p1 = [v for v in kernel.violations if v.contract == PruningContract.P1_NOOP]
        assert len(p1) == 1, f"应只有 (1,2) 那一条 noop，实际 {len(p1)}"
        assert "ACTION6@1,2" in p1[0].reason
        # 候选出口里仍要能看见 ACTION6（(3,4) 那条活着）
        assert plan["candidate_actions"].count("ACTION6") == 1

    def test_plan_still_emits_abstract_names(self):
        """§3.2.1-3：坐标只进边键与 payload，plan 出口的动作名不许被改坏。"""
        graph = LearnedGraphBackend()
        graph.update("s", "ACTION6@1,2", "s")
        kernel = R3Kernel(_ClickWorldView(), graph, budget=SearchBudget(time_ms=5000))
        plan = kernel.plan(_Frame(), LEGAL, "s")
        assert all("@" not in a for a in plan["candidate_actions"])
        validate_plan(plan)
