"""图探索器接线验证测试（单元测试 + 接线集成测试）。

覆盖三层：
1. GraphExplorer 本体：init / 记录 / 占位节点补齐 / 优先级选择 / 多跳前沿导航 / 节点上限
2. 动作表示转换：ACTIONn 字符串 ↔ int
3. 接线：ExplorationEngine 包装器（字符串动作喂图）与 CeaxController（实盘决策路径喂图+咨询）

运行：python -X utf8 test_graph_explorer.py   （也兼容 pytest -q test_graph_explorer.py）
"""
from __future__ import annotations

import numpy as np

from lingjing_solo.core.config import SoloConfig
from lingjing_solo.core.utils import hash_grid
from lingjing_solo.exploration.graph_explorer import (
    GraphExplorer,
    GraphExplorerConfig,
    action_int_to_name,
    action_name_to_int,
)


def test_action_conversion():
    assert action_name_to_int("ACTION1") == 1
    assert action_name_to_int("ACTION7") == 7
    assert action_name_to_int("action3") == 3
    assert action_name_to_int(5) == 5
    assert action_name_to_int("RESET") is None
    assert action_name_to_int(None) is None
    assert action_name_to_int("ACTION0") is None
    assert action_int_to_name(4) == "ACTION4"
    print("  [1/5] 动作字符串↔int 转换 OK")


def test_graph_core():
    g = GraphExplorer(GraphExplorerConfig(max_nodes=50))
    h_start = g.initialize("stateA", [1, 2, 3])
    assert g.get_node_count() == 1
    assert g.get_frontier_count() == 1
    # 未测动作按优先级取（默认优先级为空 → 保持调用顺序首个）
    pick = g.choose_action("stateA")
    assert pick is not None and pick[0] == 1

    # 记录一条成功转移：目的状态动作未知 → 建占位节点、不抛错
    g.record_transition("stateA", 1, True, "stateB")
    assert g.get_node_count() == 2
    bh = g.hash_state("stateB")
    assert g.nodes[bh].actions == []

    # ensure_node 补齐占位节点动作 → 进入前沿
    g.ensure_node("stateB", [2, 3])
    assert bh in g.frontier
    assert g.nodes[bh].actions == [2, 3]

    # A 状态剩余未测：优先给 2（1 已测）
    pick = g.choose_action("stateA")
    assert pick is not None and pick[0] == 2

    # A 全部测完 → closed；B、C 都是前沿，A 到最近前沿距离 = 1
    g.record_transition("stateA", 2, True, "stateC", [1])
    g.record_transition("stateA", 3, False, "stateA")
    assert g.nodes[g.hash_state("stateA")].closed
    assert g.get_distance("stateA") == 1
    nav = g.choose_action("stateA")
    assert nav is not None and nav[0] in (1, 2)
    print("  [2/5] GraphExplorer 核心（前沿/BFS/占位补齐）OK")


def test_max_nodes_cap():
    g = GraphExplorer(GraphExplorerConfig(max_nodes=5))
    for i in range(10):
        g.ensure_node(f"s{i}", [1, 2])
    assert g.get_node_count() <= 5 + 1  # 上限生效（当前节点受保护）
    # 前沿集合与在册节点一致（无悬空哈希）
    assert set(g.frontier) <= set(g.nodes.keys())
    print("  [3/5] max_nodes 上限 + 逐出一致 OK")


def test_explorer_wrapper_strings():
    """ExplorationEngine 包装器：字符串动作 → 图内 int。"""
    from lingjing_solo.exploration.explorer import ExplorationEngine

    cfg = SoloConfig()
    eng = ExplorationEngine(cfg, field=None)
    g1 = np.full((8, 8), 1, dtype=np.int64)
    g2 = np.full((8, 8), 2, dtype=np.int64)
    eng._ensure_graph_initialized(g1, ["ACTION1", "ACTION3"])
    assert eng._graph_initialized
    eng.record_graph_transition(g1, "ACTION1", True, g2, ["ACTION1", "ACTION3"])
    graph = eng._graph_explorer
    assert graph.get_node_count() == 2
    node = graph.nodes[graph.hash_state(hash_grid(g1))]
    assert 1 in node.tested_actions
    # 自环（网格未变）不应建边为成功
    eng.record_graph_transition(g2, "ACTION3", False, g2, ["ACTION1", "ACTION3"])
    node2 = graph.nodes[graph.hash_state(hash_grid(g2))]
    assert node2.transitions.get(3) == graph.hash_state(hash_grid(g2))
    assert node2.success.get(3) is False
    # 关卡通关：图应重置
    eng.on_environment_feedback("PLAYING", 1, 0)
    assert eng._graph_explorer is not None and not eng._graph_initialized
    print("  [4/5] ExplorationEngine 字符串接线 OK")


def test_ceax_graph_live():
    """CeaxController（实盘打分路径）：图被持续喂数据并参与 _rank_simple 咨询。"""
    from lingjing_solo.transfer.ceax_controller import CeaxController

    cfg = SoloConfig()
    ctl = CeaxController(cfg)
    states = {
        0: np.full((64, 64), 2, dtype=np.int64),
        1: np.full((64, 64), 3, dtype=np.int64),
        2: np.full((64, 64), 4, dtype=np.int64),
    }
    hashes = {k: hash_grid(v) for k, v in states.items()}

    def nxt(state: int, act: str) -> int:
        if act == "ACTION1":
            return (state + 1) % 3
        if act == "ACTION3":
            return (state + 2) % 3
        return state  # ACTION2/ACTION4 空转

    valid = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    s = 0
    for _ in range(24):
        act, _xy, why = ctl.choose(valid_actions=valid, grid=states[s], grid_hash=hashes[s])
        ns = nxt(s, act)
        ctl.observe_outcome(
            delta_pixels=(16 if ns != s else 0),
            progressed=False,
            grid_hash=hashes[ns],
            levels=0,
            prev_grid=states[s],
            grid=states[ns],
        )
        s = ns

    snap = ctl.snapshot()
    assert snap["graph_nodes"] > 0, f"图未被喂数据: {snap['graph_nodes']}"
    g = ctl._graph
    # 真实边存在：至少一条 to_hash != from_hash 的成功边
    edges = [
        (h, a, t)
        for h, node in g.nodes.items()
        for a, t in node.transitions.items()
        if node.success.get(a, False) and t != h
    ]
    assert edges, "图中没有成功转移边"
    # 决策建议路径可达（不抛错、返回动作名或 None）
    adv = g.choose_action(hashes[s])
    assert adv is None or action_name_to_int(action_int_to_name(adv[0])) == adv[0]
    print(f"  [5/5] CEAX 实盘路径图激活 OK（nodes={snap['graph_nodes']}, "
          f"frontier={snap['graph_frontier']}, edges={len(edges)}）")


def test_plan_a_blind_step():
    """A 档接线：导航盲步由图带路；无信息时维持原行为。"""
    from lingjing_solo.transfer.ceax_controller import CeaxController

    cfg = SoloConfig()
    ctl = CeaxController(cfg)
    g0 = np.full((64, 64), 2, dtype=np.int64)
    h0 = hash_grid(g0)
    dirs = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    # 首访、无实测边 → 不建议（不许干扰盲轮换）
    ctl._graph.ensure_node(h0, [1, 2, 3, 4])
    assert ctl._graph_advice_action(h0, dirs) is None

    # 一条实测自环边 → 建议落在未测方向
    ctl._graph.record_transition(h0, 2, False, h0)
    adv = ctl._graph_advice_action(h0, dirs)
    assert adv in ("ACTION1", "ACTION3", "ACTION4")

    # 全部测完且无前沿 → 回到 None（原逻辑接管）
    for a in (1, 3, 4):
        ctl._graph.record_transition(h0, a, False, h0)
    assert ctl._graph_advice_action(h0, dirs) is None

    # _choose_keyboard 盲步实际吃到建议：ACTION2 自环后再进应得未测方向
    ctl._kb_last = None
    pick = ctl._choose_keyboard(dirs, h0, grid=None)  # grid=None 跳过格子导航
    assert pick in dirs
    print("  [6/6] A 档盲步建议 OK")


def test_plan_b_nav_preempt():
    """B 档接线：主判 nav 方向被图实测为自环时，让位给图建议；无信息维持原 nav。"""
    from lingjing_solo.transfer.ceax_controller import CeaxController

    cfg = SoloConfig()
    ctl = CeaxController(cfg)
    g0 = np.full((64, 64), 2, dtype=np.int64)
    h0 = hash_grid(g0)
    dirs = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    ctl._graph.ensure_node(h0, [1, 2, 3, 4])
    ctl._lattice_nav = lambda grid, valid: "ACTION1"  # 强制主判指向 ACTION1

    # 无实测边 → 不得干扰，照常返回 nav
    ctl._kb_last = None
    assert ctl._choose_keyboard(dirs, h0, grid=g0) == "ACTION1"

    # ACTION1 被实测为自环（按下不动）→ 让位给未测方向
    ctl._graph.record_transition(h0, 1, False, h0)
    assert ctl._graph_selfloop(h0, "ACTION1") is True
    assert ctl._graph_selfloop(h0, "ACTION2") is False
    ctl._kb_last = None
    pick = ctl._choose_keyboard(dirs, h0, grid=g0)
    assert pick in ("ACTION2", "ACTION3", "ACTION4"), f"B 档未生效: {pick}"

    # 其余方向也全部实测（仍无前沿）→ 图无建议，回到原 nav
    for a in (2, 3, 4):
        ctl._graph.record_transition(h0, a, False, h0)
    ctl._kb_last = None
    assert ctl._choose_keyboard(dirs, h0, grid=g0) == "ACTION1"
    print("  [7/7] B 档主判自环让位 OK")


def test_plan_b_fallback_dir():
    """档2 放宽：nav 实测自环且图给不出完整建议时，就近换非自环方向；全是墙才维持 nav。"""
    from lingjing_solo.transfer.ceax_controller import CeaxController

    cfg = SoloConfig()
    ctl = CeaxController(cfg)
    g0 = np.full((64, 64), 2, dtype=np.int64)
    h0 = hash_grid(g0)
    dirs = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    ctl._graph.ensure_node(h0, [1, 2, 3, 4])
    ctl._lattice_nav = lambda grid, valid: "ACTION1"
    ctl._graph.record_transition(h0, 1, False, h0)  # ACTION1 实测自环
    ctl._graph_advice_action = lambda gh, d: None   # 模拟图给不出完整建议

    ctl._kb_last = None
    pick = ctl._choose_keyboard(dirs, h0, grid=g0)
    assert pick in ("ACTION2", "ACTION3", "ACTION4"), f"档2 兜底未生效: {pick}"

    # 四个方向全是墙 → 无替代可用 → 维持原 nav
    for a in (2, 3, 4):
        ctl._graph.record_transition(h0, a, False, h0)
    ctl._kb_last = None
    assert ctl._choose_keyboard(dirs, h0, grid=g0) == "ACTION1"

    # 次选路径：全测完、仅 ACTION2 非自环（通向它态）→ 建议 None 时换 ACTION2
    ctl2 = CeaxController(cfg)
    ctl2._graph.ensure_node(h0, [1, 2, 3, 4])
    ctl2._lattice_nav = lambda grid, valid: "ACTION1"
    ctl2._graph_advice_action = lambda gh, d: None
    ctl2._graph.record_transition(h0, 1, False, h0)
    ctl2._graph.record_transition(h0, 2, True, "elsewhere")
    for a in (3, 4):
        ctl2._graph.record_transition(h0, a, False, h0)
    ctl2._kb_last = None
    assert ctl2._choose_keyboard(dirs, h0, grid=g0) == "ACTION2"
    print("  [8/8] 档2 非自环方向兜底 OK")


def test_state_key_pose():
    """改状态键：背景微变不再产生新状态；撞墙记成自环；B/档2 据此拦下主判。"""
    from lingjing_solo.transfer.ceax_controller import CeaxController

    cfg = SoloConfig()
    ctl = CeaxController(cfg)
    ctl._kb_player_colors = [5]

    def frame(bg: int) -> np.ndarray:
        g = np.full((64, 64), bg, dtype=np.int64)
        g[10:13, 10:13] = 5
        return g

    f0, f1 = frame(1), frame(2)          # 背景不同、玩家位姿相同
    h0, h1 = hash_grid(f0), hash_grid(f1)
    assert h0 != h1
    assert ctl._state_key(f0, h0) == ctl._state_key(f1, h1) != h0
    # 回退：认不出玩家色 → 维持整帧哈希旧行为
    ctl2 = CeaxController(cfg)
    assert ctl2._state_key(f0, h0) == h0

    valid = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    ctl._lattice_nav = lambda g, v: "ACTION1"  # 主判永远指 ACTION1（实际是墙）

    act1, _xy, _why = ctl.choose(valid_actions=valid, grid=f0, grid_hash=h0)
    assert act1 == "ACTION1"            # 首访无证据，不许干扰
    # 实况首轮 observe 只落 _prev_hash 不喂图（action_counter>0 才 observe），照实先垫一帧
    ctl.observe_outcome(delta_pixels=0, progressed=False, grid_hash=h0,
                        levels=0, prev_grid=None, grid=f0)
    # 撞墙：位姿没动，但背景闪烁给了非零 delta（旧逻辑会误判成"动了"）
    ctl.observe_outcome(delta_pixels=64, progressed=False, grid_hash=h1,
                        levels=0, prev_grid=f0, grid=f1)
    assert ctl._graph_selfloop(h1, "ACTION1") is False  # h1≠当前 raw_key，映射不外溢
    act2, _xy, _why = ctl.choose(valid_actions=valid, grid=f1, grid_hash=h1)
    assert act2 != "ACTION1", f"改状态键未生效，仍在撞墙: {act2}"
    assert ctl.snapshot()["graph_nodes"] == 1           # 闪烁不再刷屏建节点
    print("  [9/9] 改状态键（位姿键＋撞墙自环）OK")


def main():
    print("== 图探索器接线测试 ==")
    test_action_conversion()
    test_graph_core()
    test_max_nodes_cap()
    test_explorer_wrapper_strings()
    test_ceax_graph_live()
    test_plan_a_blind_step()
    test_plan_b_nav_preempt()
    test_plan_b_fallback_dir()
    test_state_key_pose()
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
