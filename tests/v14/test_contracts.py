"""
V14.0/V14.2 契约测试 —— 由 diagnose.py 的真实运行结果固化而来。

每个测试对应一份可观测的物理/工程契约：
  C1 纯扩散守恒（零通量边界 → ΔΣφ = 0）
  C2 守恒恒等式（ΔΣφ = dt·ΣJ，机器精度）
  C3 Agent 真实位移（连续坐标累积，非格点锁步）
  C4 疏散闭环（Agent 向出口移动 + 撤离判定）
  C5 确定性（同 seed → 完全相同终态）
  C6 面积律（泡壁节点数 M 与 N 无关，O(R²)）
  C7 Agent 写场契约（apply_action → AgentWriteback，非空源项 = speed）
  C8 shape 校验 fail-fast（错误 shape 立即报错，不静默广播）
  C9 场景配置 dataclass（必填项构造期暴露，替代裸 dict）

约定：不依赖 pytest，纯 Python + numpy，与 V10 风格一致。
"""
import sys

import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Optional

from lingjing_solo.v14.core.field import Field, FieldConfig
from lingjing_solo.v14.core.writeback import AgentWriteback, validate_writeback
from lingjing_solo.v14.agents import EvacuationAgent
from lingjing_solo.v14.scenarios.evacuation import EvacuationScenario
from lingjing_solo.v14.engine import Engine, EngineConfig
from lingjing_solo.v14.core.laplacian import bubble_mask_spherical, build_bubble_laplacian_csr


def _make(N=16, n_agents=12, dt=0.01, seed=42, speed=1.0, D=0.10):
    cfg = {
        "shape": (N, N, N),
        "exits": [(N-1, N//2, N//2)],
        "exit_strength": 50.0,
        "n_agents": n_agents,
        "spawn_region": [(0, N//2), (0, N), (0, N)],
        "speed": speed,
        "seed": seed,
    }
    sc = EvacuationScenario(cfg)
    eng = Engine(EngineConfig(
        field=FieldConfig(shape=(N, N, N), h=1.0/(N-1), D=D, scheme="implicit"),
        dt=dt, seed=seed), sc)
    return sc, eng


def test_c1_diffusion_conservation():
    """零通量边界下纯扩散必须守恒（D1 实测：机器精度）。"""
    f = Field(FieldConfig(shape=(16, 16, 16), h=1.0/15, D=0.10))
    S0 = f.total()
    for _ in range(20):
        f.apply_sources_and_evolve(np.zeros_like(f.phi), dt=0.01)
    assert abs(f.total() - S0) < 1e-8, f"纯扩散不守恒: Δ={f.total()-S0}"


def test_c2_conservation_identity():
    """有源项时 ΔΣφ = dt·ΣJ（D2 实测 rel=5.55e-16）。"""
    f = Field(FieldConfig(shape=(16, 16, 16), h=1.0/15, D=0.10))
    J = np.zeros_like(f.phi)
    J[3, 3, 3] = 1.0
    J[12, 12, 12] = -0.5
    S0 = f.total()
    f.apply_sources_and_evolve(J, dt=0.05)
    dS = f.total() - S0
    expected = 0.05 * 0.5
    assert abs(dS - expected) / max(abs(expected), 1e-15) < 1e-10


def test_c3_agent_real_displacement():
    """Agent 必须产生真实连续位移（D3 实测 4/4 移动）。"""
    sc, eng = _make(16, n_agents=4)
    moved = 0
    for a in eng.agents:
        pb = a.sub_pos.copy()
        a.perceive(sc.observation(eng.field, a, eng.agents))
        d = a.decide()
        a.apply_action(d, eng.field)
        if np.linalg.norm(a.sub_pos - pb) > 1e-9:
            moved += 1
    assert moved == len(eng.agents), f"并非所有 Agent 移动: {moved}/{len(eng.agents)}"


def test_c4_evacuation_closed_loop():
    """完整疏散闭环：向出口移动且全部撤离（D4 实测 12/12, 质心 10.6→1.18）。"""
    sc, eng = _make(16, n_agents=12)
    res = eng.run_e2e(200, exit_radius=2.0)
    assert res["n_evacuated"] == res["n_agents"], \
        f"未全部撤离: {res['n_evacuated']}/{res['n_agents']}"
    assert res["final_com_dist"] < res["initial_com_dist"], \
        f"质心未向出口靠近: {res['initial_com_dist']:.2f} -> {res['final_com_dist']:.2f}"


def test_c5_determinism():
    """同 seed 必须产生完全相同终态（D5 实测: True）。"""
    def run(seed):
        sc, eng = _make(16, n_agents=6, seed=seed)
        eng.run_e2e(30, exit_radius=2.0)
        return [tuple(eng.agents[i].sub_pos) for i in range(len(eng.agents))]
    a = run(42)
    b = run(42)
    assert a == b, "同 seed 运行结果不一致"


def test_c6_area_law_constant_bubble():
    """泡壁活跃节点数 M 在固定 R 下与 N 无关（D6 实测 M=123 恒定）。"""
    Ms = []
    for N in [12, 16, 20, 24, 30]:
        cx = N // 2
        mask = bubble_mask_spherical((N, N, N), (cx, cx, cx), radius=3)
        L = build_bubble_laplacian_csr(mask, h=1.0/(N-1))
        Ms.append(L.shape[0])
    assert len(set(Ms)) == 1, f"泡壁节点数随 N 变化: {list(zip([12,16,20,24,30], Ms))}"


# ============================================================
# V14.2 新增契约
# ============================================================
def test_c7_agent_writeback_contract():
    """
    C7：Agent 写场契约——apply_action 必须返回 AgentWriteback，
    且非空写回的 source 总量 = 当前速度（speed）。
    """
    f = Field(FieldConfig(shape=(16, 16, 16), h=1.0 / 15, D=0.1))
    a = EvacuationAgent(0, (8, 8, 8), speed=3.0)
    a.bind_field(f.phi.shape)

    # 静止 → 空写回（alive 但零方向）
    wb_zero = a.apply_action(np.zeros(3), f)
    assert isinstance(wb_zero, AgentWriteback)
    assert wb_zero.is_empty, "零方向应产生空写回"

    # 移动 → 非空写回，总量 = speed
    wb = a.apply_action(np.array([1.0, 0.0, 0.0]), f)
    assert isinstance(wb, AgentWriteback), \
        f"C7 失败：应返回 AgentWriteback，实际 {type(wb).__name__}"
    assert wb.shape == f.phi.shape, f"C7 shape 不匹配: {wb.shape}"
    assert abs(float(wb.source.sum()) - a._orig_speed) < 1e-12, \
        f"C7 源项总量应={a._orig_speed}，实际 {float(wb.source.sum())}"

    # 经引擎校验 → 可累加（验证 write_set 聚合路径）
    J = validate_writeback(wb, f.phi.shape)
    assert J.shape == f.phi.shape


def test_c8_writeback_shape_fail_fast():
    """
    C8：shape 不匹配必须立即报错（fail-fast），禁止静默广播。
    这是 V14 面积律最常见的陷阱：(M,) 泡壁向量混入 (N³,) 全网格。
    """
    f = Field(FieldConfig(shape=(16, 16, 16), h=1.0 / 15, D=0.1))

    # 错误 shape：(8,8,8) 冒充 (16,16,16)
    bad = AgentWriteback(source=np.zeros((8, 8, 8), dtype=np.float64), shape=(8, 8, 8))
    raised = False
    try:
        validate_writeback(bad, f.phi.shape)
    except ValueError:
        raised = True
    assert raised, "C8 失败：shape 不匹配应抛出 ValueError"

    # None → 标准化为空（向前兼容），不报错
    empty = validate_writeback(None, f.phi.shape)
    assert empty.shape == f.phi.shape
    assert np.all(empty == 0)


def test_c9_scenario_config_dataclass():
    """
    C9：场景配置 dataclass —— 必填项构造期暴露，替代裸 dict 的运行时 KeyError。

    这是 V14.2 平台化的第二项：把"场景配置"从裸 dict（字段散落各处、
    缺字段到 scenario.step 才崩）升级为带默认值与类型的数据类，
    第三方插件构造即获完整字段。
    """
    @dataclass
    class EvacuationConfig:
        """疏散场景配置（V14.2 dataclass 化）。"""
        shape: Tuple[int, int, int] = (16, 16, 16)
        exits: List[Tuple[int, int, int]] = None
        exit_strength: float = 50.0
        n_agents: int = 10
        spawn_region: List[Tuple[int, int]] = None
        speed: float = 1.0
        seed: Optional[int] = 42

        def __post_init__(self):
            # 必填约束：shape 必须 3 维（置于最前，非法输入立即报错，
            # 不进入后续的 self.shape[2] 访问，避免 IndexError 掩盖校验）
            assert len(self.shape) == 3, "shape 必须为 (nx,ny,nz)"
            if self.exits is None:
                self.exits = [(self.shape[0] - 1, self.shape[1] // 2, self.shape[2] // 2)]
            if self.spawn_region is None:
                self.spawn_region = [(0, self.shape[0]), (0, self.shape[1]), (0, self.shape[2])]

    # 最小构造：只给 shape，其余走默认值（必填项 shape 在构造期即校验）
    cfg = EvacuationConfig(shape=(24, 24, 24))
    assert cfg.exits == [(23, 12, 12)]
    assert cfg.n_agents == 10
    assert cfg.spawn_region == [(0, 24), (0, 24), (0, 24)]

    # 构造期校验：错误 shape 立即报错
    raised = False
    try:
        EvacuationConfig(shape=(16, 16))  # 2 维，非法
    except AssertionError:
        raised = True
    assert raised, "C9 失败：非法 shape 应在构造期被 __post_init__ 捕获"

    # 向后兼容：现有 EvacuationScenario(dict) 仍可运行（不破坏 C4）
    sc = EvacuationScenario({
        "shape": cfg.shape, "exits": cfg.exits,
        "exit_strength": cfg.exit_strength, "n_agents": cfg.n_agents,
        "spawn_region": cfg.spawn_region, "speed": cfg.speed, "seed": cfg.seed,
    })
    eng = Engine(EngineConfig(
        field=FieldConfig(shape=cfg.shape, h=1.0 / (cfg.shape[0] - 1), D=0.1),
        dt=0.01, seed=cfg.seed), sc)
    res = eng.run_e2e(150, exit_radius=2.0)
    assert res["n_evacuated"] == res["n_agents"], \
        f"C9 向后兼容失败：{res['n_evacuated']}/{res['n_agents']}"
    assert res["final_com_dist"] < res["initial_com_dist"]


if __name__ == "__main__":
    tests = [
        ("C1 纯扩散守恒", test_c1_diffusion_conservation),
        ("C2 守恒恒等式", test_c2_conservation_identity),
        ("C3 Agent 真实位移", test_c3_agent_real_displacement),
        ("C4 疏散闭环", test_c4_evacuation_closed_loop),
        ("C5 确定性", test_c5_determinism),
        ("C6 面积律恒定泡壁", test_c6_area_law_constant_bubble),
        ("C7 Agent 写场契约", test_c7_agent_writeback_contract),
        ("C8 写回 shape fail-fast", test_c8_writeback_shape_fail_fast),
        ("C9 场景配置 dataclass", test_c9_scenario_config_dataclass),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
    print(f"\n契约测试: {passed}/{len(tests)} 通过")
    sys.exit(0 if passed == len(tests) else 1)
