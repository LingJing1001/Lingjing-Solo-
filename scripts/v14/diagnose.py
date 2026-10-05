"""
V14.0 真实状态摸底 v2 —— 严格对齐当前 API，裸跑诊断。
目标：搞清楚现在到底什么能用、什么不能用，用真实数据说话。
"""
import numpy as np

from lingjing_solo.v14.core.field import Field, FieldConfig
from lingjing_solo.v14.agents import EvacuationAgent
from lingjing_solo.v14.scenarios.evacuation import EvacuationScenario
from lingjing_solo.v14.engine import Engine, EngineConfig


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# 公共构造
def make_field(N=16, D=0.10, scheme="implicit"):
    return Field(FieldConfig(shape=(N, N, N), h=1.0/(N-1), D=D, scheme=scheme))


def make_scenario_engine(N=16, n_agents=12, dt=0.01, seed=42, speed=1.0, exit_radius=2.0,
                          D=0.10, scheme="implicit"):
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
    eng = Engine(EngineConfig(field=FieldConfig(shape=(N, N, N), h=1.0/(N-1), D=D, scheme=scheme),
                            dt=dt, seed=seed), sc)
    # Engine 构造即完成 setup + create_agents；sc.field / eng.agents 均已就绪。
    return sc, eng


# ----------------------------------------------------------------------
banner("D1. Field 纯扩散守恒（零通量边界，期望 ΔΣφ = 0）")
# ----------------------------------------------------------------------
f = make_field(16)
S0 = f.total()
for _ in range(20):
    f.apply_sources_and_evolve(np.zeros_like(f.phi), dt=0.01)
S1 = f.total()
print(f"  20 步纯扩散: Σφ {S0:.10f} -> {S1:.10f}, Δ = {S1-S0:.3e}")
print(f"  {'✅ 守恒（机器精度）' if abs(S1-S0) < 1e-8 else '❌ 不守恒'}")

# ----------------------------------------------------------------------
banner("D2. 守恒恒等式 ΔΣφ = dt·ΣJ（有源项）")
# ----------------------------------------------------------------------
f = make_field(16)
J = np.zeros_like(f.phi)
J[3, 3, 3] = 1.0
J[12, 12, 12] = -0.5
S0 = f.total()
f.apply_sources_and_evolve(J, dt=0.05)
dS = f.total() - S0
expected = 0.05 * (1.0 + (-0.5))
print(f"  ΣJ=0.5, dt=0.05, 期望 dt·ΣJ = {expected:.4f}")
print(f"  实测 ΔΣφ = {dS:.6f}, 相对差 = {abs(dS-expected)/max(abs(expected),1e-15):.3e}")


# ----------------------------------------------------------------------
banner("D3. Agent 单次 step 是否真实位移")
# ----------------------------------------------------------------------
sc, eng = make_scenario_engine(16, n_agents=4)
agent = eng.agents[0]
print(f"  出口: {sc.exits}")
print(f"  Agent 0 初始: sub_pos={agent.sub_pos}, pos={agent.pos}")
obs = sc.observation(eng.field, agent, eng.agents)
agent.perceive(obs)
print(f"  感知: local_phi={agent._local_phi:.4f}, grad={agent._gradient}, neighbors={len(agent._neighbors)}")
direction = agent.decide()
print(f"  决策 direction = {direction}")
J_ret = agent.apply_action(direction, eng.field)
print(f"  apply_action 返回 J is None: {J_ret is None}（V14契约：只位移不写场）")
print(f"  Agent 0 位移后: sub_pos={agent.sub_pos}, pos={agent.pos}")
moved = 0
for a in eng.agents:
    pb = a.sub_pos.copy()
    o = sc.observation(eng.field, a, eng.agents)
    a.perceive(o)
    d = a.decide()
    a.apply_action(d, eng.field)
    if np.linalg.norm(a.sub_pos - pb) > 1e-9:
        moved += 1
print(f"  本轮真实移动: {moved}/{len(eng.agents)}")


# ----------------------------------------------------------------------
banner("D4. 完整 Engine.run 疏散闭环 + 守恒审计")
# ----------------------------------------------------------------------
sc, eng = make_scenario_engine(16, n_agents=12, exit_radius=2.0, D=0.10)
print(f"  出口: {sc.exits}, agents={len(eng.agents)}")
S0 = eng.field.total()
result = eng.run_e2e(200, exit_radius=2.0)
S1 = eng.field.total()
print(f"  撤离: {result['n_evacuated']}/{result['n_agents']}")
print(f"  质心距离: {result['initial_com_dist']:.3f} -> {result['final_com_dist']:.3f}")
print(f"  Σφ: {S0:.4f} -> {S1:.4f}  (场景持续注入源项，故总量不守恒属正常)")
# 审计：纯扩散部分（不含环境源项）的守恒需另测；此处记录源项总量
print(f"  conservation 序列前5项: {result['conservation'][:5]}")


# ----------------------------------------------------------------------
banner("D5. 确定性（同 seed 是否完全一致）")
# ----------------------------------------------------------------------
def run_com(seed):
    sc, eng = make_scenario_engine(16, n_agents=6, seed=seed)
    eng.run_e2e(30, exit_radius=2.0)
    return [tuple(a.sub_pos) for a in eng.agents]

c42a = run_com(42)
c42b = run_com(42)
c99 = run_com(99)
same = all(a == b for a, b in zip(c42a, c42b))
print(f"  seed=42 两次一致: {same}")
print(f"  seed=42 终态: {c42a[:2]}")
print(f"  seed=99 终态: {c99[:2]}")


# ----------------------------------------------------------------------
banner("D6. 面积律：泡壁活跃节点数 vs N（O(R²) 应为常数）")
# ----------------------------------------------------------------------
from lingjing_solo.v14.core.laplacian import bubble_mask_spherical, build_bubble_laplacian_csr
R = 3  # 泡壁半径（与场景 bubble_radius 一致）
for N in [12, 16, 20, 24, 30]:
    cx = N // 2
    mask = bubble_mask_spherical((N, N, N), (cx, cx, cx), radius=R)
    M = int(mask.sum())
    L = build_bubble_laplacian_csr(mask, h=1.0/(N-1))
    print(f"  N={N:3d}  泡壁活跃节点 M = {M:4d}  L.shape={L.shape}")
print(f"  （R={R} 固定时 M 应恒定 = O(R²)，与 N 无关）")

print("\n诊断完成。以上全部为真实运行输出。")
