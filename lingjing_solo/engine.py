"""
灵境引擎 V14.0 — engine
=========================
主循环：因果闭合的核心。

每 tick 严格顺序（V14 契约）：
    1. 场景感知：scenario.observation(field, agent) -> obs
    2. Agent 感知：agent.perceive(obs)
    3. Agent 决策：direction = agent.decide()
    4. Agent 写回：agent.apply_action(direction, field)  # 注入源项（绝对通道）
    5. 场景 step：scenario.step(field, agents, dt)        # 环境源项
    6. 聚合源项 + 演化：field.apply_sources_and_evolve(J_total, dt)  # 唯一演化入口
    7. 时钟推进（单一推进点）
    8. 记录状态（可重放）

守恒保证：
    - Agent 写场：通过 field.inject_sources（绝对，不参与 dt）
    - 演化写场：通过 apply_sources_and_evolve（含 dt·J）
    - 两者严格分离（V13.2 A2 修复）

确定性：
    - 所有随机性通过 np.random.default_rng(seed) 控制
    - 相同 seed + 相同输入 = 完全相同状态序列
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Dict, Any
import numpy as np

from lingjing_solo.core.field import Field, FieldConfig
from lingjing_solo.core.scenario import Scenario
from lingjing_solo.core.writeback import AgentWriteback, validate_writeback
from lingjing_solo.agents import Agent


@dataclass
class EngineConfig:
    """引擎配置。"""
    field: FieldConfig
    dt: float = 0.01
    seed: Optional[int] = 42
    record: bool = True  # 是否记录状态序列（重放用）


@dataclass
class TickRecord:
    """单 tick 记录（用于重放/审计）。"""
    tick: int
    t: float
    total_phi: float
    n_moved: int
    agent_positions: list  # [(id, x, y, z), ...]


class Engine:
    """
    灵境引擎主循环。

    用法：
        engine = Engine(config, scenario)
        engine.run(n_ticks=100)
        # 或
        for _ in range(100):
            engine.step()
    """

    def __init__(self, config: EngineConfig, scenario: Scenario):
        self.config = config
        self.scenario = scenario
        # V14.1：构造时立即校验场景接口契约，错误场景 fail-fast
        scenario.validate()
        self.field = Field(config.field)
        self.dt = config.dt
        self.rng = np.random.default_rng(config.seed)
        # 状态
        self.tick_count = 0
        self.agents: List[Agent] = []
        self.records: List[TickRecord] = []
        # 初始化场景
        self._initialize()

    @property
    def _exit_pos(self) -> Optional[np.ndarray]:
        """主出口位置（连续坐标），供撤离判定使用。"""
        exits = getattr(self.scenario, "exits", None) or []
        if not exits:
            return None
        return np.array(exits[0], dtype=np.float64)

    def _initialize(self) -> None:
        """setup 场 + 创建 Agent + 绑定场 shape（V14.2 写场契约）。"""
        self.scenario.setup(self.field)
        self.agents = self.scenario.create_agents(self.field)
        # V14.2：让每个 Agent 知晓场 shape，写回时可本地校验
        for agent in self.agents:
            agent.bind_field(self.field.phi.shape)

    # ---- 主循环 ----
    def run(self, n_ticks: int) -> List[TickRecord]:
        """
        运行 n_ticks。

        返回 records；若调用方需要撤离/守恒统计，使用 run_e2e()。
        """
        for _ in range(n_ticks):
            self.step()
        return self.records

    def run_e2e(self, n_ticks: int, exit_radius: float = 1.5) -> dict:
        """
        端到端运行 + 物理闭环审计。

        参数:
            exit_radius: 以**连续格点**为单位的出口捕获半径（默认 1.5，覆盖邻格）

        返回 dict:
            n_evacuated, evacuation_time, initial/final_com_dist,
            final_total_phi, conservation(前若干 tick 审计)
        """
        scene = self.scenario
        exit_pos = self._exit_pos
        evacuated_ids = set()
        evacuation_time = None
        com_dist_init = None
        conservation = []

        for _ in range(n_ticks):
            # step 内部：位移 → 即时冻结（原地更新 evacuated_ids）→ 源项聚合 → 演化
            phi_before = float(self.field.total())
            self.step(evacuated_ids=evacuated_ids, exit_radius=exit_radius)
            phi_after = float(self.field.total())
            conservation.append({
                "tick": self.tick_count - 1,
                "delta_phi": phi_after - phi_before,
                "total_phi": phi_after,
            })

            # 全撤离时刻（evacuated_ids 已被 step 原地更新）
            if evacuation_time is None and exit_pos is not None and \
               len(evacuated_ids) == len(self.agents):
                evacuation_time = self.tick_count

            if com_dist_init is None and exit_pos is not None:
                com = self.agent_center_of_mass()
                com_dist_init = float(np.linalg.norm(com - exit_pos))

        final_com = self.agent_center_of_mass()
        com_dist_final = float(np.linalg.norm(final_com - exit_pos)) if exit_pos is not None else None

        return {
            "n_agents": len(self.agents),
            "n_evacuated": len(evacuated_ids),
            "evacuation_time": evacuation_time,
            "initial_com_dist": com_dist_init,
            "final_com_dist": com_dist_final,
            "final_total_phi": float(self.field.total()),
            "conservation": conservation,
        }

    def step(self, evacuated_ids: set = None, exit_radius: float = 1.5) -> TickRecord:
        """
        单 tick：因果闭合（V14 严格源项语义）。

        顺序：
            1-4. 感知 → 决策 → apply_action（位移，返回源项 J_agent）
                已撤离 Agent 冻结：不感知、不决策、不写场。
                位移后**立即**判定出口距离：进入 exit_radius 者当 tick 冻结，
                避免"到达出口被边界反射弹回"的非物理行为。
            5. 场景 step：构建环境源项 J_env（出口势阱等）
            6. 聚合 J = J_env + ΣJ_agent，唯一演化入口
               apply_sources_and_evolve(J, dt)
            7. 记录 + 时钟推进

        参数:
            evacuated_ids: 累计已撤离集合（**同一对象会被本方法原地更新**）。
                           传入者可在 step 后读到最新冻结结果。
        返回:
            TickRecord（evacuated_ids 通过参数原地修改，无需返回值）。
        """
        evacuated_ids = evacuated_ids if evacuated_ids is not None else set()
        dt = self.dt
        field = self.field
        exit_pos = self._exit_pos

        # 1-4. 感知 → 决策 → 位移 + 即时撤离判定
        write_set = []  # [AgentWriteback, ...]
        n_moved = 0
        for agent in self.agents:
            if not agent.alive or agent.id in evacuated_ids:
                continue  # 已撤离：冻结（不移动、不写场）
            obs = self.scenario.observation(field, agent, self.agents)
            agent.perceive(obs)
            direction = agent.decide()
            if np.linalg.norm(direction) > 1e-12:
                pos_before = agent.pos
                wb = agent.apply_action(direction, field)  # V14.2：返回 AgentWriteback
                if agent.pos != pos_before:
                    n_moved += 1
                write_set.append(wb)
            else:
                # 零方向也算一次写回（空），保证"写场契约"每 tick 都有明确声明
                wb = agent.apply_action(direction, field)
                write_set.append(wb)
            # ---- 即时撤离判定（位移后，同 tick 冻结，原地更新集合）----
            if exit_pos is not None and agent.id not in evacuated_ids:
                d = float(np.linalg.norm(agent.sub_pos - exit_pos))
                if d <= exit_radius:
                    evacuated_ids.add(agent.id)

        # 5. 场景 step：环境源项 J_env（出口势阱持续注入，对抗扩散）
        J_env = self.scenario.step(field, self.agents, dt)  # ndarray 或 None

        # 6. 聚合源项 + 演化（唯一写入点，V13.2 A2 严格版）
        # V14.2：所有 AgentWriteback 经校验后累加，shape 不匹配立即报错
        J_total = np.zeros_like(field.phi)
        if J_env is not None:
            J_total += J_env
        for wb in write_set:
            J_total += validate_writeback(wb, field.phi.shape)
        field.apply_sources_and_evolve(J_total, dt)

        # 7. 记录
        record = TickRecord(
            tick=self.tick_count,
            t=field.t,
            total_phi=float(field.total()),
            n_moved=n_moved,
            agent_positions=[(a.id, *a.pos) for a in self.agents],
        )
        if self.config.record:
            self.records.append(record)

        self.tick_count += 1
        return record

    # ---- 诊断 ----
    def total_phi(self) -> float:
        return float(self.field.total())

    def agent_center_of_mass(self) -> np.ndarray:
        """Agent 质心（用于验证"是否向出口移动"）。"""
        if not self.agents:
            return np.zeros(3)
        pts = np.array([a.pos for a in self.agents], dtype=np.float64)
        return pts.mean(axis=0)

    def snapshot(self) -> dict:
        """完整状态快照（用于重放比对）。"""
        return {
            "tick": self.tick_count,
            "t": self.field.t,
            "phi": self.field.phi.copy(),
            "agents": [a.snapshot() for a in self.agents],
        }
