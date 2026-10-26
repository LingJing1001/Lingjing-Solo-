"""
灵境引擎 V14.4.1 — scenarios.attraction
==========================================
吸引场景（L5 第二插件，验证平台化）。

物理设定（与疏散对偶）：
- φ = 信息密度场（中心吸引子）
- 中心持续注入源项（维持吸引势阱）
- Agent 沿 +∇φ 移动（向中心聚集）
- 到达中心（势阱核心）即"聚集成功"

与 Evacuation 的对偶：
    Evacuation: 向外疏散 → 出口在边界
    Attraction: 向心聚集 → 目标在中心

本场景的存在意义：证明"换场景不改 Engine 内核"（V14.1 接口自检的真正验收）。
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from lingjing_solo.core.scenario import Scenario
from lingjing_solo.agents import Agent, EvacuationAgent


class AttractionScenario(Scenario):
    """
    吸引场景：中心吸引子。

    配置项：
        shape      : 网格尺寸 (Nx,Ny,Nz)
        attractor  : 吸引子位置 (x,y,z)，默认中心
        attractor_strength : 吸引子源项强度
        n_agents   : Agent 数量
        spawn_region : Agent 生成区域（外围壳层）
        speed      : Agent 移动速度
        capture_radius : 捕获半径（距吸引子 ≤ r 即成功）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.shape = tuple(config["shape"])
        nx, ny, nz = self.shape
        self.attractor: Tuple[int, int, int] = tuple(
            config.get("attractor", (nx // 2, ny // 2, nz // 2))
        )
        self.attractor_strength = float(config.get("attractor_strength", 30.0))
        self.n_agents = int(config.get("n_agents", 8))
        self.spawn_region = config.get(
            "spawn_region",
            [(0, nx), (0, ny), (0, nz)],
        )
        self.speed = float(config.get("speed", 1.0))
        self.capture_radius = float(config.get("capture_radius", 2.5))
        # exits 属性是 Scenario.validate() 要求的；吸引场景用吸引子替代出口
        self.exits: List[Tuple[int, int, int]] = [self.attractor]
        self._env_sources: np.ndarray = None
        self._seed: Optional[int] = config.get("seed", None)

    def setup(self, field) -> None:
        """建立中心吸引子（高斯势阱，持续注入维持）。"""
        self._env_sources = np.zeros(self.shape, dtype=np.float64)
        nx, ny, nz = self.shape
        i = np.arange(nx)[:, None, None]
        j = np.arange(ny)[None, :, None]
        k = np.arange(nz)[None, None, :]
        ai, aj, ak = self.attractor
        dist2 = (i - ai) ** 2 + (j - aj) ** 2 + (k - ak) ** 2
        sigma = max(nx, ny, nz) / 6.0
        self._env_sources += self.attractor_strength * np.exp(-dist2 / (2 * sigma ** 2))
        field.inject_sources(self._env_sources)

    def create_agents(self, field) -> List[EvacuationAgent]:
        """在生成区域随机放置 Agent。"""
        agents = []
        rng = np.random.default_rng(self._seed)
        for aid in range(self.n_agents):
            pos = tuple(
                int(rng.integers(lo, hi)) for (lo, hi) in self.spawn_region
            )
            agents.append(EvacuationAgent(aid, pos, speed=self.speed))
        return agents

    def step(self, field, agents: List[Agent], dt: float) -> np.ndarray:
        """每 tick：持续注入吸引子源项（维持势阱，对抗扩散抹平）。"""
        if self._env_sources is None:
            return None
        return self._env_sources * dt

    def observation(self, field, agent: Agent, all_agents: list = None) -> Dict[str, Any]:
        """感知：局部梯度 + 局部密度 + 邻近 Agent（软分离）。"""
        phi = field.phi
        gx, gy, gz = agent.pos
        nx, ny, nz = phi.shape
        h = field.cfg.h

        # 梯度（Neumann 零通量边界，中心差分）
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

        # 邻居（软分离）
        neighbors = []
        if all_agents is not None:
            r = 1.5
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

        # 靠近吸引子时给线性收束（完成最后一段）
        ai, aj, ak = self.attractor
        d2 = (gx - ai) ** 2 + (gy - aj) ** 2 + (gz - ak) ** 2
        if d2 <= self.capture_radius ** 2 * 2.0:
            to_center = np.array([ai - gx, aj - gy, ak - gz], dtype=np.float64)
            n = np.linalg.norm(to_center)
            if n > 1e-6:
                grad = grad + 1.0 * to_center / n

        return {"gradient": grad, "local_phi": float(phi[gx, gy, gz]),
                "neighbors": neighbors}

    def captured(self, agent: Agent) -> bool:
        """Agent 是否已到达吸引子捕获区。"""
        ai, aj, ak = self.attractor
        d2 = (agent.pos[0] - ai) ** 2 + (agent.pos[1] - aj) ** 2 + (agent.pos[2] - ak) ** 2
        return d2 <= self.capture_radius ** 2
