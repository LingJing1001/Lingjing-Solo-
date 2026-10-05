"""
灵境引擎 V14.0 — 端到端演示：疏散场景
========================================
真实物理交互闭环：
  Agent 感知信息场梯度 → 决策（+∇φ）→ 位移 + 写回源项
  → 场演化（扩散）→ 下一帧感知变化

输出：
- 每 tick 的 Agent 位置 + 场总量
- 质心轨迹（验证向出口移动）
- 守恒监测
"""
from __future__ import annotations
import json
import os

from lingjing_solo.core import Field, FieldConfig
from lingjing_solo.engine import Engine, EngineConfig
from lingjing_solo.scenarios import EvacuationScenario


def main():
    print("=" * 70)
    print("灵境引擎 V14.0 — 疏散场景端到端演示")
    print("IFC 信息场范式工程 v1.0")
    print("=" * 70)

    cfg = EngineConfig(
        field=FieldConfig(shape=(48, 24, 24), h=1.0/(48-1), D=0.2),
        dt=0.1, seed=42, record=True,
    )
    scenario = EvacuationScenario({
        "shape": (48, 24, 24),
        "exits": [(44, 12, 12)],   # 出口在 +x 远端
        "exit_strength": 20.0,      # 中等势阱
        "n_agents": 16,
        "spawn_region": [(0, 4), (4, 20), (4, 20)],  # 生成区在 -x 近端
        "speed": 0.3,               # 较慢，疏散过程可见
        "seed": 42,
    })

    eng = Engine(cfg, scenario)
    print(f"\n初始: {eng.total_phi():.2f}")
    print(f"Agent 数: {len(eng.agents)}")
    print(f"出口: {scenario.exits}")
    print(f"\n{'tick':>4} | {'total_phi':>12} | {'com_x':>8} | {'com_y':>8} | {'com_z':>8} | {'n_moved':>8}")
    print("-" * 60)

    n_ticks = 60
    eng.run(n_ticks=n_ticks)

    # 打印轨迹
    for rec in eng.records[::5]:  # 每 5 tick
        com = eng.agent_center_of_mass()
        print(f"{rec.tick:>4} | {rec.total_phi:>12.4f} | {com[0]:>8.2f} | {com[1]:>8.2f} | {com[2]:>8.2f} | {rec.n_moved:>8}")

    # 最终统计
    com_final = eng.agent_center_of_mass()
    # 初始质心（从第一帧记录）
    init_positions = np.array([list(a.pos) for a in eng.agents], dtype=float)
    com_init = init_positions.mean(axis=0)
    dist_init = np.linalg.norm(com_init - np.array(scenario.exits[0], dtype=float))
    dist_final = np.linalg.norm(com_final - np.array(scenario.exits[0], dtype=float))
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"初始质心: {com_init}, 距离出口: {dist_init:.2f}")
    print(f"最终质心: {com_final}, 距离出口: {dist_final:.2f}")
    print(f"净移动: {dist_init - dist_final:.2f} (向出口方向)")
    print(f"守恒监测: total_phi 终值 = {eng.total_phi():.4f}")

    # 保存轨迹（可重放/可视化）
    out = os.path.join(os.path.dirname(__file__), "out", "evacuation_trajectory.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trajectory = {
        "config": {
            "shape": list(cfg.field.shape),
            "dt": cfg.dt,
            "seed": cfg.seed,
            "n_agents": scenario.n_agents,
            "exits": scenario.exits,
        },
        "records": [
            {
                "tick": r.tick,
                "t": r.t,
                "total_phi": r.total_phi,
                "n_moved": r.n_moved,
                "agent_positions": r.agent_positions,
            }
            for r in eng.records
        ],
    }
    with open(out, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"轨迹已保存: {out}")

    # 验收
    assert dist_final < dist_init, "质心应向出口移动"
    print("\n✅ 疏散闭环验证通过：Agent 群体向出口方向移动")


if __name__ == "__main__":
    import numpy as np
    main()
