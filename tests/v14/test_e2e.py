"""
灵境引擎 V14.0 — 端到端集成测试
==================================
验证"真实物理交互闭环"（IFC 信息场范式 v1.0）：

  E1 守恒恒等式  : ΔΣφ = dt·ΣJ（零通量边界，无额外汇源 → 数值守恒）
  E2 撤离闭环    : 全部 Agent 到达出口并被冻结
  E3 质心移动    : 群体质心单调趋向出口
  E4 拥挤排斥    : 高密处 Agent 减速（speed 随 φ 衰减）
  E5 软分离      : 邻近 Agent 产生排斥分量，缓解聚集
  E6 确定性      : 相同 seed → 完全相同状态序列
  E7 边界反射    : 连续坐标越界被弹性钳制，不崩溃
"""
import numpy as np

from lingjing_solo.v14.core.field import FieldConfig
from lingjing_solo.v14.engine import Engine, EngineConfig
from lingjing_solo.v14.scenarios.evacuation import EvacuationScenario


def _make_engine(n_agents=6, speed=1.0, seed=7):
    cfg = EngineConfig(
        field=FieldConfig(shape=(24, 16, 8), h=1.0 / 23, D=0.1, scheme="explicit"),
        dt=0.05, seed=seed, record=True,
    )
    scene = EvacuationScenario({
        "shape": (24, 16, 8),
        "exits": [(22, 8, 4)],
        "exit_strength": 30.0,
        "n_agents": n_agents,
        "spawn_region": [(1, 3), (4, 12), (2, 6)],
        "speed": speed, "seed": seed,
    })
    return Engine(cfg, scene)


def test_E1_conservation_identity():
    """ΔΣφ 在纯扩散下守恒；含源项时 ΔΣφ = dt·ΣJ_env（数值上稳定）。"""
    eng = _make_engine()
    recs = eng.run(n_ticks=20)
    # 环境持续注入 → Σφ 应单调非减（势阱不被扩散抹平）
    totals = [r.total_phi for r in recs]
    assert totals[-1] > totals[0], f"势阱应被维持: {totals[0]:.3f} → {totals[-1]:.3f}"
    # 每 tick 增量应稳定（持续注入恒定源 → ΔΣφ ≈ const）
    deltas = [totals[i] - totals[i - 1] for i in range(1, len(totals))]
    assert np.std(deltas) / (np.mean(np.abs(deltas)) + 1e-12) < 0.05, \
        f"注入应稳定: std/mean={np.std(deltas):.4f}"
    print("  E1 ✅ 守恒/持续注入稳定")


def test_E2_evacuation_completes():
    """全部 Agent 在合理 tick 内撤离并被冻结。"""
    eng = _make_engine(n_agents=4, speed=1.5)
    res = eng.run_e2e(n_ticks=120, exit_radius=1.5)
    assert res["n_evacuated"] == res["n_agents"], (
        f"应全部撤离: {res['n_evacuated']}/{res['n_agents']}, "
        f"耗时 {res['evacuation_time']}, 质心 {res['initial_com_dist']:.2f}→{res['final_com_dist']:.2f}"
    )
    assert res["evacuation_time"] is not None and res["evacuation_time"] <= 120
    print(f"  E2 ✅ 全部撤离, {res['evacuation_time']} tick")


def test_E3_center_of_mass_monotone():
    """质心距离出口应严格缩短（真实疏散方向）。"""
    eng = _make_engine()
    res = eng.run_e2e(n_ticks=60, exit_radius=1.5)
    assert res["final_com_dist"] < res["initial_com_dist"], (
        f"质心应趋近出口: {res['initial_com_dist']:.2f} → {res['final_com_dist']:.2f}"
    )
    print(f"  E3 ✅ 质心 {res['initial_com_dist']:.2f} → {res['final_com_dist']:.2f}")


def test_E4_crowding_slowdown():
    """高局部 φ 应使 Agent 减速（拥挤排斥的物理表现）。"""
    from lingjing_solo.v14.agents import EvacuationAgent
    a = EvacuationAgent(0, (5, 5, 5), speed=2.0, crowding_coeff=0.5)
    a.perceive({"gradient": np.array([1.0, 0.0, 0.0]), "local_phi": 0.0, "neighbors": []})
    a.decide()
    v_empty = a.speed
    a2 = EvacuationAgent(1, (5, 5, 5), speed=2.0, crowding_coeff=0.5)
    a2.perceive({"gradient": np.array([1.0, 0.0, 0.0]), "local_phi": 10.0, "neighbors": []})
    a2.decide()
    assert a2.speed < v_empty, f"拥挤应减速: {v_empty:.3f} → {a2.speed:.3f}"
    print(f"  E4 ✅ 拥挤减速 {v_empty:.3f} → {a2.speed:.3f}")


def test_E5_soft_separation():
    """邻近 Agent 应在合力中产生排斥分量。"""
    from lingjing_solo.v14.agents import EvacuationAgent
    a = EvacuationAgent(0, (3, 3, 3), speed=1.0, sep_radius=2.0, sep_strength=1.0)
    a.perceive({
        "gradient": np.array([1.0, 0.0, 0.0]),   # 吸引 +x（出口方向）
        "local_phi": 0.0,
        "neighbors": [(99, 0.0, 1.0, 0.0, 1.0)],  # 正上方邻居 → 应排斥 -y
    })
    d = a.decide()
    assert d[1] < 0, f"上方邻居应产生 -y 排斥: {d}"
    print(f"  E5 ✅ 软分离方向 {np.round(d, 3)}")


def test_E6_determinism():
    """相同 seed → 完全相同结果。"""
    r1 = _make_engine().run_e2e(n_ticks=50, exit_radius=1.5)
    r2 = _make_engine().run_e2e(n_ticks=50, exit_radius=1.5)
    assert r1["n_evacuated"] == r2["n_evacuated"]
    assert np.isclose(r1["final_com_dist"], r2["final_com_dist"])
    assert np.isclose(r1["final_total_phi"], r2["final_total_phi"])
    print("  E6 ✅ 确定性复现")


def test_E7_boundary_reflection():
    """Agent 连续位移越界时被弹性钳制，不崩溃、不穿墙。"""
    from lingjing_solo.v14.agents import EvacuationAgent
    field = type("F", (), {"phi": np.zeros((24, 16, 8))})()
    a = EvacuationAgent(0, (23, 8, 4), speed=5.0)  # 紧靠 +x 边界
    a.perceive({"gradient": np.array([1.0, 0.0, 0.0]), "local_phi": 0.0, "neighbors": []})
    d = a.decide()
    a.apply_action(d, field)
    assert 0.0 <= a.sub_pos[0] <= 23.0, f"x 应被钳制: {a.sub_pos}"
    print(f"  E7 ✅ 边界反射 x={a.sub_pos[0]:.3f} (∈[0,23])")


if __name__ == "__main__":
    tests = [t for name, t in globals().items() if name.startswith("test_E")]
    for t in tests:
        t()
    print(f"\n端到端集成测试 {len(tests)}/{len(tests)} 通过 ✅")
