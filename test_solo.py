"""Lingjing-Solo 框架验证脚本

跑通五层闭环的最小测试：
1. 导入无错误
2. 感知编码（Layer 0）：差分 + 分割 + 特征
3. 世界模型场（Layer 1）：转移记录 + 规则归纳 + 循环检测
4. 探索评分（Layer 2）
5. 规划（Layer 3）：LLM 预算节制
6. Agent 决策闭环：模拟 30 步随机环境
"""
import json
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from lingjing_solo import SoloConfig, LingjingSoloAgent
from lingjing_solo.core import Logger, hash_grid, GameObject, RuleHypothesis
from lingjing_solo.core.types import Frame
from lingjing_solo.perception import PerceptionEncoder
from lingjing_solo.world_model import WorldModelField
from lingjing_solo.exploration import ExplorationEngine
from lingjing_solo.exploration import (
    ActionObservation,
    analyze_observation,
    analyze_recording,
    summarize_actions,
)
from lingjing_solo.planning import (
    GridObject,
    MotionObject,
    LightweightPlanner,
    LLMPlanner,
    LS20Solver,
    extract_objects,
    moved_objects,
    observe_motion,
)
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


def test_ls20_level1_verified_route():
    route = LS20Solver.level1_verified_route()
    assert route == (
        ["ACTION3"] * 3
        + ["ACTION1"] * 6
        + ["ACTION4"] * 3
        + ["ACTION1"] * 3
    )


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


def test_agent_honors_authoritative_terminal_state():
    agent = LingjingSoloAgent()
    agent.reset()
    grid = make_grid(0)
    agent.observe(grid, state="WIN", levels_completed=1)
    assert agent.is_done([], grid)


def test_planner_returns_unvisited_known_transition():
    cfg = SoloConfig()
    field = WorldModelField(cfg)
    planner = LightweightPlanner(cfg, field)
    first = make_grid(0)
    second = make_grid(1)
    third = make_grid(2)
    field.update(Frame(grid=first, t=0))
    field.update(Frame(grid=second, t=1), prev_grid=Frame(grid=first, t=0), action="ACTION2")
    field.update(Frame(grid=third, t=2), prev_grid=Frame(grid=second, t=1), action="ACTION2")
    field.grid_state = second.copy()
    field.visited_set.discard(hash_grid(third))
    assert planner.search(valid_actions=["ACTION2"]) == "ACTION2"


def test_exploration_penalizes_repeated_action_for_current_state():
    cfg = SoloConfig()
    field = WorldModelField(cfg)
    explorer = ExplorationEngine(cfg, field)
    grid = make_grid(0)
    field.update(Frame(grid=grid, t=0))
    next_grid = make_grid(1)
    field.update(Frame(grid=next_grid, t=1), prev_grid=Frame(grid=grid, t=0), action="ACTION0")
    field.update(Frame(grid=make_grid(2), t=2), prev_grid=Frame(grid=next_grid, t=1), action="ACTION1")
    field.update(Frame(grid=make_grid(3), t=3), prev_grid=Frame(grid=next_grid, t=2), action="ACTION1")
    field.grid_state = next_grid
    scores = dict(explorer.score_actions(["ACTION1", "ACTION2"]))
    assert scores["ACTION2"] > scores["ACTION1"]


def test_action_diff_records_single_action_evidence():
    before = np.zeros((8, 8), dtype=np.int8)
    after = before.copy()
    before[2:4, 2:4] = 1
    after[2:4, 3:5] = 1
    delta = analyze_observation(ActionObservation("ACTION3", before, after))
    assert delta.changed_pixels == 4
    assert delta.player_before == (2.5, 2.5)
    assert delta.player_after == (3.5, 2.5)
    assert delta.displacement == (1.0, 0.0)
    assert not delta.triggered_level_change


def test_action_diff_summary_is_fail_closed_on_inconsistent_motion():
    base = np.zeros((4, 4), dtype=np.int8)
    one = base.copy()
    one[1, 1] = 1
    two = base.copy()
    two[1, 2] = 1
    three = base.copy()
    three[2, 2] = 1
    observations = [
        ActionObservation("ACTION3", one, two),
        ActionObservation("ACTION3", two, three),
    ]
    summary = summarize_actions(observations)["ACTION3"]
    assert summary.samples == 2
    assert summary.moved_samples == 2
    assert summary.consistent_displacement is None
    assert summary.confidence == 0.5


def test_action_diff_rejects_ambiguous_multichannel_frame():
    frame = np.zeros((6, 4, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="expected HxW or 1xHxW"):
        analyze_observation(ActionObservation("ACTION1", frame, frame))


def test_ls20_solver_discards_stale_route_on_observation_change():
    solver = LS20Solver()
    first = make_grid(0)
    changed = make_grid(1)
    solver.set_plan(["ACTION1", "ACTION2"])
    assert solver.next_action(first, ["ACTION1", "ACTION2"]) == "ACTION1"
    solver.observe(changed)
    assert solver.next_action(changed, ["ACTION1", "ACTION2"]) is None
    assert solver.replan_required


def test_ls20_solver_plans_aligned_waypoints():
    solver = LS20Solver()
    route = solver.plan_waypoints((0, 0), [(0, 10), (10, 10)])
    assert route == ["ACTION4", "ACTION4", "ACTION2", "ACTION2"]
    assert solver.next_action(np.zeros((2, 2), dtype=np.int8), route) == "ACTION4"


def test_ls20_solver_records_dynamic_transition_and_checks_next_step_collision():
    previous = np.zeros((8, 8), dtype=np.int8)
    current = previous.copy()
    previous[2:4, 1:3] = 9
    current[2:4, 4:6] = 9
    solver = LS20Solver()
    solver.set_plan(["ACTION1", "ACTION2"])
    assert solver.observe_transition(previous, current, player=(0, 0)) == [(0.0, 3.0)]
    assert solver.state.dynamic_obstacles == [(2, 4, 3, 5)]
    assert not solver.replan_required
    assert solver.next_action(current, ["ACTION1", "ACTION2"]) == "ACTION1"

    blocked_previous = np.zeros((8, 8), dtype=np.int8)
    blocked_current = blocked_previous.copy()
    blocked_previous[5:7, 3:5] = 9
    blocked_current[5:7, 0:2] = 9
    solver = LS20Solver()
    solver.set_plan(["ACTION2", "ACTION3"])
    solver.observe_transition(blocked_previous, blocked_current, player=(0, 0))
    assert solver.replan_required
    assert solver.next_action(blocked_current, ["ACTION1", "ACTION2"]) is None

    solver.set_plan(["ACTION1", "ACTION2"])
    solver.observe_transition(blocked_current, current, player=(0, 0))
    assert not solver.replan_required
    assert solver.next_action(current, ["ACTION1", "ACTION2"]) == "ACTION1"


def test_ls20_perception_extracts_objects_and_tracks_motion():
    previous = np.zeros((8, 8), dtype=np.int8)
    current = previous.copy()
    previous[2:4, 1:3] = 9
    current[2:4, 4:6] = 9
    objects = extract_objects(previous)
    assert objects == [GridObject(9, 4, (2, 1, 3, 2))]
    assert moved_objects(previous, current, max_distance=2.0) == [
        GridObject(9, 4, (2, 4, 3, 5)),
    ]
    assert observe_motion(previous, current, max_distance=2.0) == [
        MotionObject(GridObject(9, 4, (2, 4, 3, 5)), (0.0, 3.0)),
    ]


def test_field_win_detector_callback():
    cfg = SoloConfig()
    field = WorldModelField(cfg, win_detector=lambda grid: int(grid.sum()) == 0)
    assert field.detect_win(np.zeros((2, 2), dtype=np.int8))
    assert not field.detect_win(np.ones((2, 2), dtype=np.int8))


def test_kaggle_adapter():
    print("=" * 50)
    print("TEST 7 · Kaggle 适配层")
    from lingjing_solo.harness import MyAgent, make_agent
    a = make_agent(llm_fn=fake_llm)
    assert hasattr(a, "is_done") and hasattr(a, "choose_action")
    action = a.choose_action([make_grid(0)], make_grid(1))
    assert isinstance(action, str)
    print(f"  OK · MyAgent.choose_action -> {action}")


def test_analyze_recording_preserves_multi_action_sequence(tmp_path):
    first = np.zeros((1, 4, 4), dtype=np.int8)
    first[0, 1, 0] = 1
    second = first.copy()
    second[0, 1, 0] = 0
    second[0, 1, 1] = 1
    third = second.copy()
    third[0, 1, 1] = 0
    third[0, 1, 2] = 1
    recording = tmp_path / "probe.recording.jsonl"
    recording.write_text(
        "\n".join(
            [
                json.dumps({"data": {"frame": first.tolist(), "requested_action": {"name": "ACTION1"}}}),
                json.dumps({"data": {"frame": second.tolist(), "requested_action": {"name": "ACTION2"}}}),
                json.dumps({"data": {"frame": third.tolist(), "requested_action": {"name": "ACTION3"}}}),
            ]
        )
    )
    deltas = analyze_recording(recording)
    assert [delta.action for delta in deltas] == ["ACTION2", "ACTION3"]
    assert [delta.displacement for delta in deltas] == [(1.0, 0.0), (1.0, 0.0)]


def test_analyze_recording_rejects_missing_requested_action(tmp_path):
    recording = tmp_path / "invalid.recording.jsonl"
    frame = np.zeros((1, 2, 2), dtype=np.int8).tolist()
    recording.write_text(
        "\n".join(
            [
                json.dumps({"data": {"frame": frame, "requested_action": {"name": "ACTION1"}}}),
                json.dumps({"data": {"frame": frame}}),
            ]
        )
    )
    with pytest.raises(ValueError, match="requested_action"):
        analyze_recording(recording)


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
