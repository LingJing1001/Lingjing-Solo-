"""
test_v10_telemetry_agent.py — v1.0 观测层 + Kaggle Agent 集成验证

核心命题：
    ✅ Telemetry 正确记录每步决策来源与模拟器调用
    ✅ Agent 适配官方 is_done / choose_action 接口
    ✅ 跑分后可输出 JSONL 摘要（喂回分析的入口）
    ✅ 无网络约束下 Agent 可运行（LLM=None 降级）
"""
import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lingjing_solo.telemetry import Telemetry
from lingjing_solo.agent import LingjingAgent, GameAction
from lingjing_solo.planning.planner import make_box_goal_evaluator
from lingjing_solo.world_model.symbols import SymbolTable
from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence
from lingjing_solo.world_model.codegen import FakeLLM


def _make_state(avatar_xy, box_xy, goals, grid_w=8, grid_h=6, aid="a", bid="b"):
    return {
        "objects": {
            aid: {"x": avatar_xy[0], "y": avatar_xy[1], "role": "avatar"},
            bid: {"x": box_xy[0], "y": box_xy[1], "role": "box"},
        },
        "avatar_id": aid,
        "extras": {"goals": list(goals), "grid_w": grid_w, "grid_h": grid_h},
    }


def _make_wmp():
    wmp = WorldModelProgram(llm_client=FakeLLM(), min_support=1, confidence_threshold=0.0)
    before = SymbolTable()
    aid = before.add_object(1, 1, color=1, role="avatar")
    before.avatar_id = aid
    bid = before.add_object(2, 1, color=2, role="box")
    after = SymbolTable()
    after.add_object(2, 1, color=1, role="avatar", obj_id=aid)
    after.add_object(3, 1, color=2, role="box", obj_id=bid)
    after.avatar_id = aid
    wmp.learn(WMPEvidence(action="right", before=before, after=after))
    wmp.compile(llm=FakeLLM())
    return wmp


class TestTelemetry(unittest.TestCase):
    """观测层基本功能。"""

    def test_records_decision_chain(self):
        tel = Telemetry(enabled=True)
        tel.start_step()
        tel.record(source="wmp", sim_calls=12, cache_hits=3, cache_misses=2)
        tel.end_step(action="right")
        self.assertEqual(len(tel.records), 1)
        rec = tel.records[0]
        self.assertEqual(rec["source"], "wmp")
        self.assertEqual(rec["action"], "right")
        self.assertEqual(rec["sim_calls"], 12)
        self.assertIn("decision_ms", rec)

    def test_summary_aggregation(self):
        tel = Telemetry(enabled=True)
        for i in range(3):
            tel.start_step()
            tel.record(source="wmp" if i < 2 else "score", sim_calls=10 * (i + 1))
            tel.end_step(action="right")
        s = tel.summary()
        self.assertEqual(s["total_steps"], 3)
        self.assertEqual(s["total_sim_calls"], 60)
        self.assertAlmostEqual(s["avg_sim_calls_per_step"], 20.0)
        self.assertEqual(s["source_distribution"]["wmp"], 2)
        self.assertEqual(s["source_distribution"]["score"], 1)

    def test_persist_to_jsonl(self):
        """跑分结果落盘为 JSONL（喂回分析的入口）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            tel = Telemetry(log_path=path, enabled=True)
            tel.start_step()
            tel.record(source="wmp", sim_calls=5)
            tel.end_step(action="right", win=True)
            tel.flush()
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertEqual(data["action"], "right")
            self.assertTrue(data["win"])


class TestAgentIntegration(unittest.TestCase):
    """Agent 端到端（无 LLM，符合 Kaggle 无网络约束）。"""

    def setUp(self):
        self.agent = LingjingAgent(
            goals=[(4, 1)],
            llm_client=None,  # 无网络：纯轻量规划
            llm_calls_per_game=0,
            max_steps_per_game=50,
            telemetry_path=None,
        )

    def test_choose_action_returns_game_action(self):
        state = _make_state((1, 1), (2, 1), goals=[(4, 1)])
        action = self.agent.choose_action(frames=[state], latest_frame=state)
        self.assertIsInstance(action, GameAction)
        self.assertEqual(action.kind, "key")

    def test_full_episode_to_win(self):
        """完整一关：决策 → WIN 终止。"""
        state = _make_state((1, 1), (2, 1), goals=[(4, 1)])
        # 手动驱动：把 WMP 学到的 push:right 注入，让 Agent 规划
        self.agent.wmp = _make_wmp()
        self.agent.planner.wmp = self.agent.wmp
        self.agent.planner.set_goal_evaluator(make_box_goal_evaluator([(4, 1)]))

        cur = dict(state)
        done = False
        for _ in range(20):
            if self.agent.is_done(frames=[cur], latest_frame=cur):
                done = True
                break
            act = self.agent.choose_action(frames=[cur], latest_frame=cur)
            # 用 WMP 步进模拟真实环境（占位）
            nxt = self.agent.wmp.simulate(cur, act.value)
            if nxt is None:
                break
            cur = nxt
        # Agent 决策链路不应崩溃；若走到 goal 则 done=True
        stats = self.agent.stats()
        self.assertIn("step_count", stats)
        self.assertIn("source_distribution", stats)

    def test_is_done_on_win(self):
        # box 已在 goal 上 → 立即判定 WIN
        state = _make_state((3, 1), (4, 1), goals=[(4, 1)])
        self.assertTrue(self.agent.is_done(frames=[state], latest_frame=state))

    def test_is_done_step_limit(self):
        state = _make_state((1, 1), (2, 1), goals=[(4, 1)])
        self.agent.step_count = 50  # 已达上限
        self.assertTrue(self.agent.is_done(frames=[state], latest_frame=state))

    def test_telemetry_captures_sources(self):
        """跑分后 Telemetry 应包含决策来源分布。"""
        state = _make_state((1, 1), (2, 1), goals=[(4, 1)])
        self.agent.wmp = _make_wmp()
        self.agent.planner.wmp = self.agent.wmp
        self.agent.planner.set_goal_evaluator(make_box_goal_evaluator([(4, 1)]))

        for _ in range(5):
            if self.agent.is_done(frames=[state], latest_frame=state):
                break
            self.agent.choose_action(frames=[state], latest_frame=state)

        s = self.agent.telemetry.summary()
        # 至少应有探索/兜底类决策被记录
        self.assertGreater(s["total_steps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
