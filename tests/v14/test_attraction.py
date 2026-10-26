"""
V14.4.1 端到端测试 —— 第二场景 AttractionScenario（平台化验收）。

  E8 AttractionScenario 端到端：Agent 被吸引到中心捕获区
"""
import numpy as np

from lingjing_solo.core.field import Field, FieldConfig
from lingjing_solo.engine import Engine, EngineConfig
from lingjing_solo.scenarios.attraction import AttractionScenario


def test_e8_attraction_end_to_end():
    """E8：用同一 Engine 跑 AttractionScenario，Agent 被吸引到中心。

    本测试的存在意义：证明 V14.2 的 Scenario 接口是真平台化的——
    换一个完全不同的场景（从疏散到吸引），Engine 代码零修改。
    """
    N = 16
    cfg = {
        "shape": (N, N, N),
        "attractor": (N // 2, N // 2, N // 2),
        "attractor_strength": 30.0,
        "n_agents": 6,
        "spawn_region": [(0, 4), (0, 4), (0, 4)],  # 从左上角外围生成
        "speed": 1.0,
        "capture_radius": 3.0,
        "seed": 42,
    }
    sc = AttractionScenario(cfg)
    eng = Engine(EngineConfig(
        field=FieldConfig(shape=(N, N, N), h=1.0 / (N - 1), D=0.10,
                          scheme="implicit"),
        dt=0.01, seed=42), sc)

    # 初始距离：Agent 在 (0-3, 0-3, 0-3)，中心在 (8,8,8)
    agents = eng.agents
    init_dist = [
        float(np.sqrt(sum((a.pos[i] - cfg["attractor"][i]) ** 2 for i in range(3))))
        for a in agents
    ]
    # 跑 200 tick
    for _ in range(200):
        eng.step()

    # 最终距离
    final_dist = [
        float(np.sqrt(sum((a.pos[i] - cfg["attractor"][i]) ** 2 for i in range(3))))
        for a in eng.agents
    ]

    # 至少一半 Agent 被吸引到更靠近中心
    closer = sum(1 for d0, d1 in zip(init_dist, final_dist) if d1 < d0)
    assert closer >= len(agents) // 2, \
        f"E8 吸引失败: {closer}/{len(agents)} Agent 靠近中心"

    # 至少 1 个 Agent 进入捕获区
    captured = sum(1 for a in eng.agents if sc.captured(a))
    assert captured >= 1, \
        f"E8 无 Agent 被捕获: {captured}/{len(agents)} 在捕获区内"
