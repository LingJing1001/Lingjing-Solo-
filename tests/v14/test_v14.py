"""
灵境引擎 V14.0/14.2 — 验证套件（pytest 可选版）
==============================================
性质测试（可证伪）：
1. 守恒恒等式：ΔΣφ = dt·ΣJ（机器精度）
2. 因果闭环：Agent 写场 → 下一帧感知变化
3. 确定性：相同 seed → 完全相同状态序列
4. 面积律：泡壁活动节点 = O(R²)，与 N 脱钩；CSR == Python 内部点一致
5. 疏散闭环：Agent 质心向出口移动

V14.2 改动：
- pytest 改为**可选依赖**：import 失败则降级为纯 unittest/裸跑，
  消除"unittest discover 因缺 pytest 而 ERROR"的硬依赖问题。
- 采用模块级函数（pytest 可收集，unittest 可收集，__main__ 可裸跑）。
- 因果闭环改用新的 AgentWriteback 契约验证。

运行:
    python -m pytest lingjing_solo/v14/tests/test_v14.py -v  # 有 pytest
    python -m lingjing_solo.v14.tests.test_v14  # 无 pytest（自运行）
    python -m unittest discover -s tests     # 无 pytest 也不再报错
"""
from __future__ import annotations
import sys

import numpy as np

try:
    import pytest  # type: ignore
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from lingjing_solo.v14.core import (Field, FieldConfig, laplacian_7point,
                  build_bubble_laplacian_csr, bubble_mask_spherical)
from lingjing_solo.v14.core.writeback import AgentWriteback, validate_writeback  # V14.2
from lingjing_solo.v14.core.scenario import Scenario
from lingjing_solo.v14.agents import Agent, EvacuationAgent
from lingjing_solo.v14.engine import Engine, EngineConfig
from lingjing_solo.v14.scenarios import EvacuationScenario


# ============================================================
# T1: 守恒恒等式
# ============================================================
def test_pure_diffusion_conservation():
    """纯扩散（无源项，零通量边界）：Σφ 严格守恒。"""
    cfg = FieldConfig(shape=(16, 16, 16), h=1.0, D=1.0)
    f = Field(cfg)
    x = np.arange(16)[:, None, None]
    y = np.arange(16)[None, :, None]
    z = np.arange(16)[None, None, :]
    f.inject_sources(np.exp(-((x - 8) ** 2 + (y - 8) ** 2 + (z - 8) ** 2) / 8.0))
    total0 = f.total()
    for _ in range(100):
        f.apply_sources_and_evolve(np.zeros_like(f.phi), dt=0.01)
    total1 = f.total()
    assert np.isclose(total0, total1, atol=1e-8), \
        f"纯扩散不守恒（零通量边界）: {total0} -> {total1}"


def test_source_identity():
    """源项恒等式：ΔΣφ = dt·ΣJ。"""
    cfg = FieldConfig(shape=(16, 16, 16), h=1.0, D=1.0)
    f = Field(cfg)
    f.inject_sources(np.ones((16, 16, 16)) * 0.1)
    total0 = f.total()
    J = np.ones_like(f.phi) * 0.05
    dt = 0.02
    f.apply_sources_and_evolve(J, dt)
    total1 = f.total()
    expected = total0 + dt * np.sum(J)
    assert np.isclose(total1, expected, atol=1e-10), \
        f"守恒恒等式失败: Δ={total1-total0}, 期望 dt·ΣJ={dt*np.sum(J)}"


# ============================================================
# T2: 因果闭环（感知→写回→下一帧感知变化）— V14.2 写场契约
# ============================================================
def test_agent_writeback_contract():
    """V14.2：Agent apply_action 返回 AgentWriteback，不再返回 None/ndarray。"""
    f = Field(FieldConfig(shape=(16, 16, 16), h=1.0, D=0.1))
    agent = EvacuationAgent(0, (8, 8, 8), speed=1.0)
    agent.bind_field(f.phi.shape)
    wb = agent.apply_action(np.array([1.0, 0.0, 0.0]), f)
    assert isinstance(wb, AgentWriteback), \
        f"应返回 AgentWriteback，实际 {type(wb).__name__}"
    assert wb.shape == f.phi.shape, f"写回 shape 不匹配: {wb.shape}"
    # 有速度 → 源项非零（speed=1.0 累加到 (8,8,8)）
    assert not wb.is_empty, "EvacuationAgent 移动后应留下非空源项"
    assert float(wb.source.sum()) == float(agent._orig_speed), \
        f"源项总量应 = speed，实际 {float(wb.source.sum())}"


def test_agent_writes_field():
    """
    V14.2 修正：Agent 写场经 Engine 聚合后，守恒恒等式 ΔΣφ = dt·ΣJ 成立。

    关键：Agent 只在"有方向"（梯度≠0）时才写场。全零场无梯度 → Agent 不动。
    因此这里**开启出口势阱**（exit_strength>0），使 Agent 有真实方向可决策，
    从而在本 tick 留下源项；再单独验证"Agent 源项 + 环境源项"合计的恒等式。

    恒等式：ΔΣφ = dt · (ΣJ_env + ΣJ_agent)
    其中 J_env = exit_strength（setup 注入的势阱作为持续源，此处取 step 返回值）。
    """
    speed = 2.0
    exit_strength = 5.0
    dt = 0.01
    cfg = EngineConfig(
        field=FieldConfig(shape=(16, 16, 16), h=1.0 / 15, D=0.1),
        dt=dt, seed=0, record=True)
    sc = EvacuationScenario({
        "shape": (16, 16, 16), "exits": [(15, 8, 8)],
        "exit_strength": exit_strength,   # 开启势阱 → Agent 有方向
        "n_agents": 1, "spawn_region": [(8, 9), (8, 9), (8, 9)], "speed": speed})
    eng = Engine(cfg, sc)
    a = eng.agents[0]
    phi0 = eng.field.total()

    # 手动复现 Engine.step 的聚合，便于精确校验恒等式
    # （不用 eng.step() 是为了单独分离"Agent 源项"贡献，避免隐式混合）
    f = eng.field
    # 计算 Agent 本 tick 源项总量
    a.perceive(sc.observation(f, a, eng.agents))
    direction = a.decide()
    wb = a.apply_action(direction, f)
    J_agent_sum = float(wb.source.sum())
    # 环境源项（与 scenario.step 一致）
    J_env = sc.step(f, eng.agents, dt)
    J_env_sum = float(J_env.sum()) if J_env is not None else 0.0

    # 预测：ΔΣφ = dt · (J_env_sum + J_agent_sum)
    predicted = dt * (J_env_sum + J_agent_sum)
    assert J_agent_sum > 0, \
        f"有势阱时 Agent 应有方向并写场，实际 J_agent_sum={J_agent_sum}"

    # 实际演化
    f.apply_sources_and_evolve(wb.source + (J_env if J_env is not None else 0), dt)
    phi1 = f.total()
    actual = phi1 - phi0
    assert np.isclose(actual, predicted, atol=1e-9), \
        f"守恒恒等式失败: ΔΣφ={actual}, 期望 dt·(J_env+J_agent)={predicted} " \
        f"(J_env={J_env_sum}, J_agent={J_agent_sum})"


def test_perception_changes_after_write():
    """
    V14.2 修正：Agent 写场是"返回 AgentWriteback"，不直接修改 phi。
    正确因果链 = Agent 源项经 apply_sources_and_evolve 注入 → 演化 → 下一帧感知变化。

    验证：把 Agent 写回的 source 随演化带入后，局部 phi 确实改变且扩散传播。
    """
    f = Field(FieldConfig(shape=(16, 16, 16), h=1.0, D=0.1))
    agent = EvacuationAgent(0, (8, 8, 8), speed=1.0)
    agent.bind_field(f.phi.shape)
    agent.alive = True

    # Agent 决策+位移 → 拿到写回（此时 phi 尚未变）
    wb = agent.apply_action(np.array([1.0, 0.0, 0.0]), f)
    deposited = float(wb.source.sum())
    assert deposited > 0, "Agent 移动后应留下正源项"

    # V14.2 因果链：源项经演化注入（dt·J 通道）
    phi_before_evolve = float(f.phi[8, 8, 8])
    f.apply_sources_and_evolve(wb.source, dt=0.01)
    phi_after_inject = float(f.phi[8, 8, 8])
    assert not np.isclose(phi_before_evolve, phi_after_inject, atol=1e-12), \
        "Agent 源项注入后场应立即变化（守恒恒等式 ΔΣφ=dt·ΣJ）"

    # 再纯扩散一帧：局部值因扩散传播而继续变化（因果传播）
    phi_after_diffuse = float(f.phi[8, 8, 8])
    f.apply_sources_and_evolve(np.zeros_like(f.phi), dt=0.01)
    phi_after_propagate = float(f.phi[8, 8, 8])
    assert not np.isclose(phi_after_diffuse, phi_after_propagate, atol=1e-12), \
        "演化后局部 phi 应因扩散传播而变化"


# ============================================================
# T3: 确定性（可重放）
# ============================================================
def _make_engine(seed: int) -> Engine:
    cfg = EngineConfig(
        field=FieldConfig(shape=(16, 16, 16), h=1.0 / (16 - 1), D=0.5),
        dt=0.05, seed=seed, record=True)
    scenario = EvacuationScenario({
        "shape": (16, 16, 16), "exits": [(14, 14, 14)],
        "exit_strength": 10.0, "n_agents": 5,
        "spawn_region": [(0, 5), (0, 5), (0, 5)], "speed": 0.5, "seed": seed,
    })
    return Engine(cfg, scenario)


def test_deterministic_replay():
    """相同 seed → 完全相同状态序列。"""
    eng1 = _make_engine(42)
    eng1.run(n_ticks=10)
    eng2 = _make_engine(42)
    eng2.run(n_ticks=10)
    assert eng1.total_phi() == eng2.total_phi(), "总 phi 应一致"
    for r1, r2 in zip(eng1.records, eng2.records):
        assert r1.n_moved == r2.n_moved
        assert r1.agent_positions == r2.agent_positions, \
            f"tick {r1.tick} Agent 位置不一致"


def test_different_seed_diverges():
    """不同 seed → 初始 Agent 位置不同（seed 确实控制随机性）。"""
    eng1 = _make_engine(42)
    eng2 = _make_engine(99)
    pos1 = sorted([a.pos for a in eng1.agents])
    pos2 = sorted([a.pos for a in eng2.agents])
    assert pos1 != pos2, f"不同 seed 应产生不同初始位置: {pos1} vs {pos2}"


# ============================================================
# T4: 面积律（泡壁 O(R²) 与 N 脱钩）
# ============================================================
def test_bubble_nodes_constant():
    """泡壁活动节点数恒定（R 固定），与 N 无关。"""
    R = 4
    center = (8, 8, 8)
    nodes_by_N = {}
    for N in [16, 24, 32]:
        mask = bubble_mask_spherical((N, N, N), center, R)
        nodes_by_N[N] = int(mask.sum())
    values = set(nodes_by_N.values())
    assert len(values) == 1, f"泡壁节点数应恒定: {nodes_by_N}"


def test_csr_equals_python_internal():
    """泡壁 CSR Laplacian == 体积 7 点 Laplacian（内部点逐点一致）。"""
    N = 16
    mask = bubble_mask_spherical((N, N, N), (8, 8, 8), 4)
    h = 1.0 / (N - 1)
    f = Field(FieldConfig(shape=(N, N, N), h=h))
    f.set_bubble(mask)
    phi = np.random.default_rng(0).random((N, N, N))
    lap_vol = laplacian_7point(phi, h=h)
    lap_csr = np.zeros_like(phi)
    active = np.flatnonzero(mask)
    lap_csr_flat = f.laplacian_bubble(phi)
    rows = active // (N * N)
    cols = (active % (N * N)) // N
    deps = active % N
    lap_csr[rows, cols, deps] = lap_csr_flat
    # 内部点 = 活动节点且 6 邻域全在活动集内
    internal = mask.copy()
    for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        internal &= np.roll(mask, -di, 0) & np.roll(mask, -dj, 1) & np.roll(mask, -dk, 2)
    internal &= mask
    max_err = np.max(np.abs(lap_vol[internal] - lap_csr[internal]))
    assert max_err < 1e-10, \
        f"CSR vs Python 内部点不一致: max_err={max_err} (内部点数={internal.sum()})"


# ============================================================
# T5: 疏散真实闭环
# ============================================================
def test_agent_moves_toward_exit():
    """Agent 质心向出口移动。"""
    cfg = EngineConfig(
        field=FieldConfig(shape=(24, 24, 24), h=1.0 / (24 - 1), D=0.3),
        dt=0.05, seed=42, record=True)
    scenario = EvacuationScenario({
        "shape": (24, 24, 24), "exits": [(20, 12, 12)],
        "exit_strength": 20.0, "n_agents": 8,
        "spawn_region": [(0, 6), (8, 16), (8, 16)], "speed": 0.3,
    })
    eng = Engine(cfg, scenario)
    eng.run(n_ticks=40)
    com_final = eng.agent_center_of_mass()
    assert com_final[0] > 5.0, f"Agent 质心应向出口(+x)移动，实际 x={com_final[0]}"


# ============================================================
# 自运行入口（无 pytest 时）
# ============================================================
_TESTS = [
    test_pure_diffusion_conservation,
    test_source_identity,
    test_agent_writeback_contract,       # V14.2 新增
    test_agent_writes_field,
    test_perception_changes_after_write,
    test_deterministic_replay,
    test_different_seed_diverges,
    test_bubble_nodes_constant,
    test_csr_equals_python_internal,
    test_agent_moves_toward_exit,
]


if __name__ == "__main__":
    passed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\ntest_v14.py: {passed}/{len(_TESTS)} 通过")
    sys.exit(0 if passed == len(_TESTS) else 1)
