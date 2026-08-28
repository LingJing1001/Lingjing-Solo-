"""Lingjing-Solo 框架验证脚本

跑通五层闭环的最小测试：
1. 导入无错误
2. 感知编码（Layer 0）：差分 + 分割 + 特征
3. 世界模型场（Layer 1）：转移记录 + 规则归纳 + 循环检测
4. 探索评分（Layer 2）
5. 规划（Layer 3）：LLM 预算节制
6. Agent 决策闭环：模拟 30 步随机环境
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from lingjing_solo import SoloConfig, LingjingSoloAgent
from lingjing_solo.core import Logger, hash_grid, GameObject, RuleHypothesis
from lingjing_solo.core.types import Frame
from lingjing_solo.perception import PerceptionEncoder
from lingjing_solo.world_model import WorldModelField
from lingjing_solo.exploration import ExplorationEngine
from lingjing_solo.planning import LightweightPlanner, LLMPlanner
from lingjing_solo.reflection import ReflectionTrigger


def make_grid(t=0, seed=0):
    """构造一个 64x64 网格：背景 0 + 几个彩色块。"""
    rng = np.random.RandomState(seed + t)
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:14, 10:14] = 3   # 一块颜色 3
    g[20:23, 20:25] = 7   # 一块颜色 7
    if t % 3 == 0:
        g[30:32, 30:32] = 5
    return g


def fake_llm(snapshot, valid_actions):
    """模拟 LLM：总是选第一个合法动作（仅用于验证调用链）。"""
    return valid_actions[0] if valid_actions else None


def test_imports():
    print("=" * 50)
    print("TEST 1 · 导入与配置")
    cfg = SoloConfig()
    assert cfg.grid_size == 64
    assert "UP" in cfg.allowed_actions
    print(f"  OK · allowed_actions={cfg.allowed_actions}")


def test_perception():
    print("=" * 50)
    print("TEST 2 · Layer 0 感知编码")
    cfg = SoloConfig()
    enc = PerceptionEncoder(cfg)
    prev = make_grid(0)
    curr = make_grid(1)
    out = enc(prev, curr)
    assert "feature" in out and "delta_pixels" in out and "objects" in out
    assert out["feature"].shape[0] == cfg.cnn_feature_dim
    assert len(out["delta_pixels"]) > 0, "差分应检测到变化"
    print(f"  OK · feature_dim={out['feature'].shape[0]}, delta_px={len(out['delta_pixels'])}, objs={len(out['objects'])}")


def test_field():
    print("=" * 50)
    print("TEST 3 · Layer 1 世界模型场 Φ")
    cfg = SoloConfig()
    field = WorldModelField(cfg)
    for t in range(8):
        field.update(Frame(grid=make_grid(t), t=t),
                     prev_grid=Frame(grid=make_grid(t - 1), t=t - 1) if t > 0 else None,
                     action="RIGHT")
    assert len(field.transition_table) == 7
    field.propose_rule("at state X do RIGHT", "-> Y")
    field.propose_rule("at state X do RIGHT", "-> Z")  # 重复前提不同结论 → 冲突
    # 手动触发冲突降级
    field._reconcile_rules(field.transition_table[-1])
    print(f"  OK · transitions={len(field.transition_table)}, rules={len(field.rules)}, visited={len(field.visited)}")
    snap = field.snapshot()
    assert snap.step == 7
    print(f"  OK · snapshot step={snap.step}")


def test_exploration():
    print("=" * 50)
    print("TEST 4 · Layer 2 探索评分")
    cfg = SoloConfig()
    field = WorldModelField(cfg)
    exp = ExplorationEngine(cfg, field)
    scored = exp.score_actions(["UP", "DOWN", "LEFT", "RIGHT", "SPACE"])
    assert len(scored) == 5
    assert all(0.0 <= s <= 1.0 for _, s in scored)
    exp.induce_rules()  # 转移不足时应安全跳过
    print(f"  OK · scored_actions={scored}")


def test_planning_and_reflection():
    print("=" * 50)
    print("TEST 5 · Layer 3/4 规划 + 反思 + LLM 预算")
    cfg = SoloConfig(llm_calls_per_game=3)
    llm = LLMPlanner(cfg)
    llm.inject_llm(fake_llm)
    assert llm.can_call()
    snap = None
    field = WorldModelField(cfg)
    from lingjing_solo.core import FieldSnapshot
    snap = FieldSnapshot(grid_summary="test", rules=[], goals=[], recent_transitions=[], visited_count=0, step=50)
    a1 = llm.plan(snap, ["UP", "DOWN"])
    a2 = llm.plan(snap, ["UP", "DOWN"])
    a3 = llm.plan(snap, ["UP", "DOWN"])
    a4 = llm.plan(snap, ["UP", "DOWN"])  # 应因预算耗尽返回 None
    assert a1 == "UP" and a4 is None, f"预算节制失效: {a1},{a4}"
    print(f"  OK · LLM 预算: 3 次后第 4 次返回 None (calls_used={llm.calls_used})")

    # 反思触发
    ref = ReflectionTrigger(cfg, field)
    sig = ref.evaluate()
    assert hasattr(sig, "should_reflect")
    print(f"  OK · reflection signal: {sig}")


def test_agent_loop():
    print("=" * 50)
    print("TEST 6 · Agent 决策闭环 (30 步)")
    cfg = SoloConfig(llm_calls_per_game=2, human_baseline_estimate=10)
    agent = LingjingSoloAgent.with_llm(fake_llm, llm_calls_per_game=2, human_baseline_estimate=10)
    agent.reset()
    frames = []
    for t in range(30):
        latest = make_grid(t)
        action = agent.choose_action(frames, latest)
        frames.append(latest)
        assert isinstance(action, str), f"动作应为字符串, 得到 {action}"
        if agent.is_done(frames, latest):
            break
    print(f"  OK · 运行 {agent.step} 步, 结束于 is_done={agent.is_done(frames, frames[-1])}")
    print(f"       rules={len(agent.field.rules)}, visited={len(agent.field.visited)}, llm_calls={agent.llm.calls_used}")
    assert agent.step > 0


def test_kaggle_adapter():
    print("=" * 50)
    print("TEST 7 · Kaggle 适配层")
    from lingjing_solo.harness import MyAgent, make_agent
    a = make_agent(llm_fn=fake_llm)
    assert hasattr(a, "is_done") and hasattr(a, "choose_action")
    action = a.choose_action([make_grid(0)], make_grid(1))
    assert isinstance(action, str)
    print(f"  OK · MyAgent.choose_action -> {action}")


if __name__ == "__main__":
    test_imports()
    test_perception()
    test_field()
    test_exploration()
    test_planning_and_reflection()
    test_agent_loop()
    test_kaggle_adapter()
    print()
    print("=" * 50)
    print("ALL TESTS PASSED ✅")
