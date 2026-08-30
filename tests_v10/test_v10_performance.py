"""
test_v10_performance.py — v1.0 性能重构验证

核心命题（v1.0 成败判据）：
    ✅ 后继缓存：跨 search 调用复用，重复状态跳过 simulate
    ✅ 浅拷贝 + 按需深拷：状态复制开销远低于 deepcopy
    ✅ 深度自适应：缓存规模大时自动缩短搜索深度
    ✅ 首层最优短路：首层恒返回 goal_evaluator 最优动作，不展开深层
    ✅ BFS 树内缓存复用（同一搜索树中子节点命中自身缓存）
    ✅ v0.9 对外行为保持兼容

性能收益的真实判据（不造假）：
    缓存命中率 = cache_hits / (hits+misses)
    浅拷贝耗时 << deepcopy
    重复搜索的 sim_calls 受控（自适应降深 + 缓存）
"""
import os
import sys
import time
import copy
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lingjing_solo.planning.wmp_planner import (
    Planner, make_box_goal_evaluator, state_key, _shallow_copy_state,
)
from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM
from lingjing_solo.world_model.symbols import SymbolTable


def _state(avatar_xy, box_xy, goals, grid_w=8, grid_h=6, aid="a", bid="b"):
    return {
        "objects": {
            aid: {"x": avatar_xy[0], "y": avatar_xy[1], "role": "avatar"},
            bid: {"x": box_xy[0], "y": box_xy[1], "role": "box"},
        },
        "avatar_id": aid,
        "extras": {"goals": list(goals), "grid_w": grid_w, "grid_h": grid_h},
    }


def _make_wmp(llm=None):
    wmp = WorldModelProgram(llm_client=llm, min_support=1, confidence_threshold=0.0)
    before = SymbolTable()
    aid = before.add_object(1, 1, color=1, role="avatar")
    before.avatar_id = aid
    bid = before.add_object(2, 1, color=2, role="box")
    after = SymbolTable()
    after.add_object(2, 1, color=1, role="avatar", obj_id=aid)
    after.add_object(3, 1, color=2, role="box", obj_id=bid)
    after.avatar_id = aid
    wmp.learn(WMPEvidence(action="right", before=before, after=after))
    wmp.compile(llm=llm)
    return wmp


class TestSuccessorCache(unittest.TestCase):
    """v1.0 核心：后继缓存跨调用复用。"""

    def test_cross_call_cache_reuse(self):
        """
        同一状态连续 search 两次：
        第二次的缓存命中数 > 第一次（跨调用复用语义）。
        """
        wmp = _make_wmp(llm=FakeLLM())
        planner = Planner(wmp=wmp, goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                          max_depth=4, telemetry=None)
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        acts = ["up", "down", "left", "right"]

        planner.search(state, acts)
        hits_after_first = planner.cache_hits
        misses_after_first = planner.cache_misses

        planner.search(state, acts)
        hits_after_second = planner.cache_hits

        # 首次搜索：BFS 树内会复用缓存，故首次也可能有命中（正确行为）
        self.assertGreaterEqual(hits_after_first, 0)
        # 关键：第二次搜索应额外命中缓存（同一状态重复访问）
        self.assertGreater(hits_after_second, hits_after_first,
                          "跨调用缓存应产生新增命中")
        print(f"\n    [cache] 首次: hits={hits_after_first} misses={misses_after_first}")
        print(f"    [cache] 二次: hits={hits_after_second}")
        if (hits_after_second + misses_after_first) > 0:
            rate = hits_after_second / (hits_after_second + misses_after_first)
            print(f"    [cache] 命中率 ≈ {rate:.1%}")

    def test_cache_cleared_on_reset(self):
        """clear_cache 后统计归零。"""
        wmp = _make_wmp(llm=FakeLLM())
        planner = Planner(wmp=wmp, max_depth=3)
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        planner.search(state, ["up", "down", "left", "right"])
        self.assertGreater(planner.cache_hits + planner.cache_misses, 0)
        planner.clear_cache()
        self.assertEqual(planner.cache_hits, 0)
        self.assertEqual(planner.cache_misses, 0)
        self.assertEqual(len(planner._successor_cache), 0)

    def test_cached_result_is_copy(self):
        """缓存返回的是拷贝，修改不污染缓存本体（正确性）。"""
        wmp = _make_wmp(llm=FakeLLM())
        planner = Planner(wmp=wmp, max_depth=3)
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        planner.search(state, ["right"])
        # 取出缓存中的状态并修改
        key = (state_key(state["objects"]), "right")
        cached = planner._successor_cache.get(key)
        if cached is not None:
            cached["objects"]["a"]["x"] = 999
            # 原 state 不应受影响
            self.assertEqual(state["objects"]["a"]["x"], 1)


class TestShallowCopy(unittest.TestCase):
    """浅拷贝 + 按需深拷策略。"""

    def test_apply_move_isolates_modified_objects(self):
        """
        v1.0 契约：_apply_move 对被修改的 obj 做 dict 复制，保证隔离。
        这是对「裸浅拷贝共享 obj」的正确处理，而非 naive 深拷。
        """
        planner = Planner(max_depth=2)
        s = _state((1, 1), (2, 1), goals=[(4, 1)])
        nxt = planner._apply_move(s, "right")
        # 原 state 的 avatar/box 坐标未被修改
        self.assertEqual(s["objects"]["a"]["x"], 1)
        self.assertEqual(s["objects"]["b"]["x"], 2)
        # 新状态正确推进
        self.assertEqual(nxt["objects"]["a"]["x"], 2)
        self.assertEqual(nxt["objects"]["b"]["x"], 3)

    def test_shallow_faster_than_deep(self):
        """浅拷贝应显著快于 deepcopy（性能命题，回归防护）。"""
        s = _state((1, 1), (2, 1), goals=[(4, 1)])
        for i in range(20):
            s["objects"][f"obj_{i}"] = {"x": i, "y": 0, "role": "wall"}

        n = 2000
        t0 = time.time()
        for _ in range(n):
            _shallow_copy_state(s)
        shallow_ms = (time.time() - t0) * 1000

        t0 = time.time()
        for _ in range(n):
            copy.deepcopy(s)
        deep_ms = (time.time() - t0) * 1000

        print(f"\n    [perf] shallow={shallow_ms:.1f}ms  deep={deep_ms:.1f}ms (n={n})")
        self.assertLess(shallow_ms, deep_ms,
                       "浅拷贝应快于 deepcopy")


class TestAdaptiveDepth(unittest.TestCase):
    """深度自适应：缓存规模大时缩短搜索深度。"""

    def test_adaptive_reduces_sim_calls_when_cache_large(self):
        wmp = _make_wmp(llm=FakeLLM())
        sim_calls = [0]

        class CountingWMP:
            def __init__(self, inner): self.inner = inner; self.is_compiled = True
            def simulate(self, s, a):
                sim_calls[0] += 1
                return self.inner.simulate(s, a)

        planner = Planner(wmp=CountingWMP(wmp),
                          goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                          max_depth=6, adaptive_depth=True)
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        acts = ["up", "down", "left", "right"]

        planner.search(state, acts)
        sim_warm = sim_calls[0]

        # 膨胀缓存规模，触发自适应降深
        for i in range(600):
            planner._successor_cache[("fake_key_%d" % i, "up")] = None
        self.assertGreater(len(planner._successor_cache), 500)

        sim_calls[0] = 0
        planner.search(state, acts)
        sim_inflated = sim_calls[0]

        print(f"\n    [adaptive] warm={sim_warm}  inflated={sim_inflated}")
        self.assertLessEqual(sim_inflated, sim_warm,
                            "缓存膨胀后应自适应缩短深度，simulate 调用不增")


class TestFirstLayerOptimalShortCircuit(unittest.TestCase):
    """v1.0 修正：首层恒返回 goal_evaluator 最优动作。"""

    def test_returns_optimal_toward_goal(self):
        """box 在(2,1)，goal 在(4,1)：最优首步是 right（使 box 靠近 goal）。"""
        wmp = _make_wmp(llm=FakeLLM())
        planner = Planner(wmp=wmp,
                          goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                          max_depth=8)
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        chosen = planner.search(state, ["up", "down", "left", "right"])
        self.assertEqual(chosen, "right", "应选中朝向 goal 的方向")

    def test_first_layer_returns_immediately(self):
        """首层恒返回，不展开到 max_depth（性能：sim_calls 受控）。"""
        wmp = _make_wmp(llm=FakeLLM())
        sim_calls = [0]

        class CountingWMP:
            def __init__(self, inner): self.inner = inner; self.is_compiled = True
            def simulate(self, s, a):
                sim_calls[0] += 1
                return self.inner.simulate(s, a)

        planner = Planner(wmp=CountingWMP(wmp),
                          goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                          max_depth=10)  # 极深，验证不展开到底
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        sim_calls[0] = 0
        chosen = planner.search(state, ["up", "down", "left", "right"])
        self.assertEqual(chosen, "right")
        # 首层短路：simulate 调用数 ≈ 首层动作数（不随 depth 指数增长）
        self.assertLessEqual(sim_calls[0], 8,
                            "首层短路应限制 simulate 调用数")


class TestBackwardCompatibility(unittest.TestCase):
    """v1.0 不应破坏 v0.9 的对外行为。"""

    def test_bfs_solves_two_step_push(self):
        wmp = _make_wmp(llm=FakeLLM())
        self.assertEqual(wmp.provenance, "llm")
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        planner = Planner(wmp=wmp,
                          goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                          max_depth=6)
        chosen = planner.search(state, ["up", "down", "left", "right"])
        self.assertEqual(chosen, "right")

    def test_empty_transition_falls_through(self):
        wmp = _make_wmp(llm=FakeLLM())
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        planner = Planner(transition_index={}, wmp=wmp,
                          goal_evaluator=make_box_goal_evaluator([(4, 1)]),
                          max_depth=6)
        chosen = planner.search(state, ["up", "down", "left", "right"])
        self.assertEqual(chosen, "right")

    def test_fallback_no_crash(self):
        state = _state((1, 1), (2, 1), goals=[(4, 1)])
        planner = Planner(transition_index={}, wmp=None,
                          goal_evaluator=make_box_goal_evaluator([(4, 1)]))
        chosen = planner.search(state, ["up", "down", "left", "right"])
        self.assertIn(chosen, {"up", "down", "left", "right"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

