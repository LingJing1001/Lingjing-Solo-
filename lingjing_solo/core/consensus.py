"""
灵境引擎 V15.0 — core.consensus
==================================
多 Agent 无通讯共识层（v13.2 协议 6，R1-R5）。

核心契约：
    R1 所有 Agent 读同一个 V_N 场快照
    R2 Agent 决策只依赖 V_N（不偷看其他 Agent 的实时写入）
    R3 Action 入队（submit_action），不立即写场
    R4 统一求解：ΣΔJ_i 一次性注入场
    R5 版本号 V_{N+1} = V_N + 1（所有 Agent 同步推进）

写入顺序无关：
    J_total = J_env + ΣJ_i，加法交换律 → 与 Agent 提交顺序无关。
    这是"无通讯共识"的物理基础：Agent 不需要互发消息，
    只需要读同一个 V_N，各自提交 ΔJ_i，由场统一求解。
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import List, Optional
import numpy as np

from lingjing_solo.core.field import Field
from lingjing_solo.core.writeback import AgentWriteback, validate_writeback


@dataclass
class ConsensusRecord:
    """单 tick 共识记录。"""
    version: int
    n_agents: int
    n_submitted: int
    total_source: float  # ΣJ（注入场的总源项）
    c_value: float = 0.0  # C 值（只读监视器，不写场）


class ConsensusLayer:
    """
    统一提交队列 + 版本号管理。

    用法（由 Engine.step 内部调用）：
        consensus.begin_tick()           # R1/R2：冻结当前场快照
        for agent in agents:
            obs = scenario.observation(field, agent, agents)
            wb = agent.apply_action(direction, field)
            consensus.submit(agent.id, wb)  # R3：入队
        J_total = consensus.aggregate(J_env)  # R4：统一聚合
        field.apply_sources_and_evolve(J_total, dt)
        consensus.commit()               # R5：版本号 +1
    """

    def __init__(self, field: Field):
        self.field = field
        self.version: int = 0  # V_0
        self._queue: List[AgentWriteback] = []
        self._snapshot_phi: Optional[np.ndarray] = None
        self.records: List[ConsensusRecord] = []

    def begin_tick(self) -> None:
        """R1/R2：冻结当前场快照（本 tick 所有 Agent 读同一个版本）。"""
        self._snapshot_phi = self.field.phi.copy()
        self._queue.clear()

    def read_snapshot(self, agent_id: str) -> np.ndarray:
        """R1：Agent 读当前版本场快照（只读，不偷看其他 Agent 的实时写入）。"""
        if self._snapshot_phi is None:
            return self.field.phi
        return self._snapshot_phi

    def submit(self, agent_id: str, wb: AgentWriteback) -> None:
        """R3：Agent Action 入队（不立即写场）。"""
        self._queue.append(wb)

    def aggregate(self, J_env: Optional[np.ndarray]) -> np.ndarray:
        """R4：统一聚合 J_total = J_env + ΣJ_i。

        写入顺序无关：加法交换律。
        """
        J_total = np.zeros_like(self.field.phi)
        if J_env is not None:
            J_total += J_env
        for wb in self._queue:
            J_total += validate_writeback(wb, self.field.phi.shape)
        return J_total

    def commit(self, c_value: float = 0.0) -> ConsensusRecord:
        """R5：版本号 +1，记录本 tick 共识。"""
        self.version += 1
        rec = ConsensusRecord(
            version=self.version,
            n_agents=len(self._queue),
            n_submitted=len(self._queue),
            total_source=float(np.sum(self.field.phi)),  # 提交后再算
            c_value=c_value,
        )
        self.records.append(rec)
        return rec
