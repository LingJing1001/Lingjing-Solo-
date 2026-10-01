"""
灵境引擎 V14.0 — scenarios.evacuation
======================================
疏散场景（L5 应用插件示例）。

物理设定：
- φ = 信息密度场（人群密度 + 出口势阱）
- 出口处持续注入源项（维持吸引子，对抗扩散抹平）
- Agent 沿 +∇φ 移动（向出口）
- 到达出口（势阱中心）即"撤离成功"

场景契约（Scenario 接口）：
    setup(field)          : 建立出口势阱 + 环境源项
    create_agents(field) : 在远离出口处生成 Agent
    step(field, agents)  : 持续注入出口源项（维持势阱）
    observation(field, a) : 返回 Agent 位置的局部梯度
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

from core.scenario import Scenario
from agents import Agent, EvacuationAgent


class EvacuationScenario(Scenario):
    """
    疏散场景：单出口势阱。

    配置项：
        shape      : 网格尺寸
        exits      : 出口列表 [(x,y,z), ...]
        exit_strength : 出口源项强度（持续注入）
        n_agents   : Agent 数量
        spawn_region : Agent 生成区域 [(x0,x1),(y0,y1),(z0,z1)]
        speed      : Agent 移动速度
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.shape = tuple(config["shape"])
        self.exits: List[Tuple[int, int, int]] = config.get("exits", [])
        self.exit_strength = float(config.get("exit_strength", 50.0))
        self.n_agents = int(config.get("n_agents", 10))
        self.spawn_region = config.get(
            "spawn_region",
            [(0, self.shape[0]), (0, self.shape[1]), (0, self.shape[2])],
        )
        self.speed = float(config.get("speed", 1.0))
        self._env_sources: np.ndarray = None  # 环境源项（持续注入）
        self._seed: Optional[int] = config.get("seed", None)  # 可配置随机种子

    def setup(self, field) -> None:
        """建立出口势阱（高斯型源项，持续注入维持）。"""
        self._env_sources = np.zeros(self.shape, dtype=np.float64)
        nx, ny, nz = self.shape
        i = np.arange(nx)[:, None, None]
        j = np.arange(ny)[None, :, None]
        k = np.arange(nz)[None, None, :]
        for ex in self.exits:
            exi, exj, exk = ex
            dist2 = (i - exi) ** 2 + (j - exj) ** 2 + (k - exk) ** 2
            # 高斯势阱：中心高，向外衰减
            sigma = max(nx, ny, nz) / 8.0
            self._env_sources += self.exit_strength * np.exp(-dist2 / (2 * sigma ** 2))
        # 初始场：注入势阱
        field.inject_sources(self._env_sources)

    def create_agents(self, field) -> List[EvacuationAgent]:
        """在生成区域随机放置 Agent。"""
        agents = []
        rng = np.random.default_rng(self._seed)  # 可配置种子（None=随机）
        for aid in range(self.n_agents):
            pos = tuple(
                int(rng.integers(lo, hi)) for (lo, hi) in self.spawn_region
            )
            agents.append(EvacuationAgent(aid, pos, speed=self.speed))
        return agents

    def step(self, field, agents: List[Agent], dt: float) -> np.ndarray:
        """
        每 tick：持续注入出口源项（维持势阱，对抗扩散抹平）。

        V14 严格源项语义：不再直接 inject_sources，而是返回 J_env，
        由 Engine 在 apply_sources_and_evolve 中统一注入（单一写入点）。

        返回：
            J_env : ndarray，环境源项密度（场单位/时间）；None 表示无环境源。
        """
        if self._env_sources is None:
            return None
        # 按时间累积：单位时间注入量 = exit_strength，整步积分 = strength * dt
        return self._env_sources * dt

    def observation(self, field, agent: Agent, all_agents: list = None) -> Dict[str, Any]:
        """
        感知：局部梯度（零通量边界）+ 局部密度 + 邻近 Agent（软分离用）。

        V14 真实物理交互：
        - 梯度：零通量（Neumann）边界，中心差分；边界处外侧=自身 → 梯度只指向场内
        - 邻居：以 agent 为中心、sep_radius 内的其他 Agent（空间哈希桶，O(n)）
          供 Agent 施加短程软排斥，避免格点聚集（真实疏散的"挤不过去"）。
        """
        phi = field.phi
        gx, gy, gz = agent.pos
        nx, ny, nz = phi.shape
        h = field.cfg.h
        exit_pos = np.array(self.exits[0], dtype=np.float64) if self.exits else None

        # ---- 梯度（零通量边界，中心差分）----
        x_fwd = gx + 1 if gx + 1 < nx else gx
        x_bwd = gx - 1 if gx - 1 >= 0 else gx
        y_fwd = gy + 1 if gy + 1 < ny else gy
        y_bwd = gy - 1 if gy - 1 >= 0 else gy
        z_fwd = gz + 1 if gz + 1 < nz else gz
        z_bwd = gz - 1 if gz - 1 >= 0 else gz
        grad = np.array([
            (phi[x_fwd, gy, gz] - phi[x_bwd, gy, gz]) / (2 * h),
            (phi[gx, y_fwd, gz] - phi[gx, y_bwd, gz]) / (2 * h),
            (phi[gx, gy, z_fwd] - phi[gx, gy, z_bwd]) / (2 * h),
        ], dtype=np.float64)

        # ---- 邻居查询（真实 Agent-Agent 物理交互）----
        neighbors = []
        if all_agents is not None:
            r = self._sep_radius
            r2 = r * r
            for other in all_agents:
                if other.id == agent.id or not other.alive:
                    continue
                dx = other.pos[0] - gx
                dy = other.pos[1] - gy
                dz = other.pos[2] - gz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < r2:
                    neighbors.append((other.id, float(dx), float(dy), float(dz), np.sqrt(d2)))

        # 出口邻近区：以出口中心为吸引子做微弱收束，
        # 让 Agent 能完成"最后一段"（避免卡在势阱边缘振荡）。
        # 这是真实疏散的物理：出口对人群有持续吸引，直至穿过出口平面。
        if self.exits and self._near_exit(gx, gy, gz, grad):
            # 已在出口邻域（dist<=2 格）：给一个指向出口中心的线性吸引
            center = np.array([float(v) for v in self.exits[0]], dtype=np.float64)
            to_center = center - np.array([gx, gy, gz], dtype=np.float64)
            n = np.linalg.norm(to_center)
            if n > 1e-6:
                grad = grad + 1.0 * to_center / n   # 收束吸引（足以克服墙角钉滞）

        return {"gradient": grad, "local_phi": float(phi[gx, gy, gz]),
                "neighbors": neighbors}

    def _near_exit(self, gx, gy, gz, grad) -> bool:
        """
        是否已进入出口邻近收束区（3D 欧氏距离 ≤ 3.5 格）。

        一旦进入该区，observation 叠加一个指向出口中心的吸引，
        使 Agent 能完成最后收敛（避免卡在势阱边缘/墙角振荡）。

        3.5 格的选取：覆盖出口格点 + 相邻两层体素，
        足以让"已靠近出口但被 x 梯度钉在墙边"的 Agent 被拉向中心。
        """
        if not self.exits:
            return False
        ex = self.exits[0]
        d2 = (gx - ex[0]) ** 2 + (gy - ex[1]) ** 2 + (gz - ex[2]) ** 2
        return d2 <= 3.5 ** 2

    # 软分离半径（observation 查询用）
    @property
    def _sep_radius(self) -> float:
        # 与 Agent 默认 sep_radius 保持一致（格点单位）
        return 1.5
