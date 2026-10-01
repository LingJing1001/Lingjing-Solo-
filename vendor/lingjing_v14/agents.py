"""
灵境引擎 V14.2 — agents
=========================
Agent 接口与疏散 Agent 实现（真实物理交互版）。

设计原则（V14 严格源项语义）：
- _sub_pos : 连续坐标（唯一真实位置，累积位移）
- pos      : 整数格点（仅投影，单向不回写）
- apply_action : 只做位移，**返回 AgentWriteback**（携带源项 source）。
  写场统一由 Engine 在 apply_sources_and_evolve 中完成，单一写入点。
- ΔΣφ = dt·ΣJ 守恒恒等式可直接验证。

V14.2 写场契约升级：
- 不再返回 Optional[ndarray]（靠注释约束），改为返回 AgentWriteback。
- 基类提供 _writeback_shape + _emit_source() 钩子；
  子类在 _emit_source(wb) 中调用 wb.add((x,y,z), value) 追加源项。
- 忘记实现 _emit_source → 返回空写回（显式，不静默）；
  shape 不匹配 → 引擎校验时 fail-fast（core.writeback.validate_writeback）。

真实物理交互（疏散）：
1. 出口吸引：direction = +∇φ（信息场梯度指向出口）
2. 拥挤排斥：局部 φ 过高时减速（越挤越慢）
3. 软分离：与相邻 Agent 过近时短程排斥，缓解格点聚集
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from core.writeback import AgentWriteback, validate_writeback


@dataclass
class AgentState:
    """Agent 状态（可序列化，用于重放）。"""
    id: int
    pos: Tuple[int, int, int]
    sub_pos: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    alive: bool


class Agent(ABC):
    """Agent 抽象基类。"""

    def __init__(self, agent_id: int, pos: Tuple[int, int, int], speed: float = 1.0):
        self.id = agent_id
        self.pos = tuple(int(x) for x in pos)
        self._sub_pos = [float(x) for x in pos]   # 连续坐标（真实位置）
        self._orig_speed = float(speed)            # 基准速度（不随拥挤衰减）
        self.speed = float(speed)                  # 当前速度（可被 decide 动态修改）
        self.alive = True
        # V14.2：写场 shape 由 Engine 在首次 step 前注入（避免 Agent 自己算 shape）
        self._field_shape: Optional[Tuple[int, int, int]] = None

    def bind_field(self, shape: Tuple[int, int, int]) -> None:
        """由 Engine 调用：告知场 shape，供写回校验。"""
        self._field_shape = tuple(int(x) for x in shape)

    @property
    def sub_pos(self) -> np.ndarray:
        return np.array(self._sub_pos, dtype=np.float64)

    @abstractmethod
    def perceive(self, obs: Dict[str, Any]) -> None:
        """接收场感知（梯度、密度、邻居），更新内部状态。"""

    @abstractmethod
    def decide(self) -> np.ndarray:
        """决策：返回运动方向（单位向量）或零向量。"""

    def apply_action(self, direction: np.ndarray, field: "Field") -> AgentWriteback:
        """
        执行动作：连续位移，返回 AgentWriteback。

        V14.2 契约流程：
            1. 位移（连续坐标 + 边界弹性反射）
            2. 构造空 AgentWriteback（shape 取自 field）
            3. 调用子类钩子 _emit_source(wb)，由子类追加源项
            4. 返回 wb（引擎侧统一校验 shape + 累加）

        边界处理：连续坐标越界采用弹性反射回弹，避免索引崩溃。
        """
        if self._field_shape is None:
            self.bind_field(field.phi.shape)

        # 1. 位移（仅在有效方向且存活时）
        moved = False
        if self.alive and np.linalg.norm(direction) >= 1e-15:
            moved = True
            direction = direction / (np.linalg.norm(direction) + 1e-15)
            shape = field.phi.shape
            for i in range(3):
                nx_i = shape[i]
                self._sub_pos[i] += direction[i] * self.speed
                if self._sub_pos[i] < 0.0:
                    self._sub_pos[i] = 0.0
                elif self._sub_pos[i] > nx_i - 1:
                    self._sub_pos[i] = nx_i - 1
            self.pos = tuple(int(round(x)) for x in self._sub_pos)

        # 2&3. 构造写回；仅当本 tick 真正行动（位移）时才追加源项。
        # 语义：零方向 = Agent 未行动 → 不留痕迹（C7）。
        # 注意：冻结(alive=False)或零方向的 Agent 返回显式空写回，
        # 保证"每 tick 都有明确写场声明"（Engine 侧可直接忽略空写回）。
        wb = AgentWriteback.empty(self._field_shape)
        if moved:
            self._emit_source(wb)
        return wb

    def _emit_source(self, wb: AgentWriteback) -> None:
        """
        子类钩子：向 wb 追加本 tick 的源项。

        默认实现：空写回（纯观测 Agent）。需要写场的子类（如 EvacuationAgent）
        重写此方法，调用 wb.add((x,y,z), value)。
        """
        return  # 基类：不写场（显式空，非静默 None）

    def snapshot(self) -> AgentState:
        return AgentState(
            id=self.id, pos=self.pos, sub_pos=tuple(self._sub_pos),
            velocity=tuple(self.decide()), alive=self.alive,
        )


class EvacuationAgent(Agent):
    """
    疏散 Agent：信息场驱动 + 拥挤排斥 + 软分离。

    感知（由 Scenario.observation 提供）：
        gradient   : 局部 φ 梯度（+∇φ 指向出口）
        local_phi  : 局部信息密度（人群密度代理）
        neighbors  : 邻近 Agent 列表 [(id, dx, dy, dz, dist), ...]

    决策：
        force = 出口吸引(+∇φ) + 邻居软排斥
        direction = force / |force|
        speed = v0 / (1 + crowding_coeff · φ_local)   # 越挤越慢

    写场（V14.2）：
        在当前格点留下正源项（强度=当前速度），通过 wb.add() 追加，
        由 Engine 校验 shape 后统一注入。
    """

    def __init__(self, agent_id: int, pos: Tuple[int, int, int], speed: float = 1.0,
                 crowding_coeff: float = 0.15, sep_radius: float = 1.5,
                 sep_strength: float = 0.4):
        super().__init__(agent_id, pos, speed)
        self._gradient = np.zeros(3, dtype=np.float64)
        self._local_phi = 0.0
        self._neighbors: List[Tuple[int, float, float, float, float]] = []
        self.crowding_coeff = crowding_coeff   # 拥挤减速系数
        self.sep_radius = sep_radius            # 软分离半径（格点单位）
        self.sep_strength = sep_strength       # 分离力强度

    def perceive(self, obs: Dict[str, Any]) -> None:
        grad = obs.get("gradient", np.zeros(3))
        self._gradient = np.array(grad, dtype=np.float64)
        self._local_phi = float(obs.get("local_phi", 0.0))
        self._neighbors = obs.get("neighbors", []) or []

    def decide(self) -> np.ndarray:
        """合力 = 出口吸引 + 邻居排斥；速度随密度衰减。"""
        g = self._gradient
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-12:
            return np.zeros(3)

        attract = g / gnorm

        # 邻居软排斥（指向远离邻居）
        repel = np.zeros(3)
        for nid, dx, dy, dz, dist in self._neighbors:
            if dist < self.sep_radius and dist > 1e-6:
                w = (self.sep_radius - dist) / self.sep_radius
                repel[0] -= (dx / dist) * w
                repel[1] -= (dy / dist) * w
                repel[2] -= (dz / dist) * w
        repel *= self.sep_strength

        force = attract + repel
        fnorm = np.linalg.norm(force)
        if fnorm < 1e-12:
            return np.zeros(3)

        # 拥挤减速：越挤越慢
        self.speed = self._orig_speed / (1.0 + self.crowding_coeff * self._local_phi)
        return force / fnorm

    def _emit_source(self, wb: AgentWriteback) -> None:
        """
        V14.2：在当前格点留下正源项 = 当前速度。

        通过 wb.add() 追加——自动校验索引在 shape 内，无需手动判边界。
        """
        gx, gy, gz = self.pos
        wb.add((gx, gy, gz), self.speed)


__all__ = ["Agent", "EvacuationAgent", "AgentState", "AgentWriteback"]
