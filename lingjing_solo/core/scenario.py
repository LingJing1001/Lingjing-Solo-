"""
灵境引擎 V14.0 — core.scenario
================================
Scenario 抽象基类（L0-L3 契约）。

设计原则（V14 平台化核心）：
- Engine 不知道具体场景，只认 Scenario 接口
- 场景插件（疏散/仓储/人群）继承此类，实现 setup/create_agents/step
- 同一 Engine 应能跑任意 Scenario，零代码修改

接口定义：
    setup(field)        : 初始化场（势阱、障碍物、环境源项）
    create_agents()     : 返回 Agent 列表
    step(field, agents) : 每 tick 的场景逻辑（注入环境源项等）
    observation(field, agent, all_agents=None) -> dict : 感知接口

V14.1 接口自检：
    场景插件常犯的错误是 observation 签名与引擎调用不一致
    （抽象方法 2 参数，实际实现多加 all_agents）。
    validate() 在 Engine 构造时自动调用，签名错误立即报错，
    而不是跑到一半才崩——这是"底座"应有的自省能力。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING, List, Dict, Any
import numpy as np

if TYPE_CHECKING:
    from lingjing_solo.agents import Agent
    from lingjing_solo.core.field import Field


class ScenarioInterfaceError(Exception):
    """场景插件未满足 Scenario 接口契约时抛出。"""


class Scenario(ABC):
    """场景插件接口。"""

    # 引擎调用 observation 时实际传入的参数名（不含 self）
    _OBSERVATION_ENGINE_ARGS = ("field", "agent", "all_agents")

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def setup(self, field: "Field") -> None:
        """初始化场：势阱、障碍物、环境源项。"""

    @abstractmethod
    def create_agents(self, field: "Field") -> List["Agent"]:
        """创建 Agent 列表。"""

    @abstractmethod
    def step(self, field: "Field", agents: List["Agent"], dt: float) -> None:
        """每 tick 场景逻辑（如：持续注入势阱源项，维持目标不被扩散抹平）。"""

    @abstractmethod
    def observation(self, field: "Field", agent: "Agent",
                    all_agents: List["Agent"] = None) -> Dict[str, Any]:
        """
        Agent 感知接口：返回该 Agent 位置的局部场信息（梯度、密度、邻居等）。

        参数:
            all_agents: 可选，供场景做 Agent-Agent 交互（软分离、拥挤等）。
                       引擎调用时**总会**传入，子类应将其声明为可选参数
                       以保持向前兼容（签名必须能接受 3 个位置参数）。
        """

    # ---- V14.1 接口自检 ----
    def validate(self) -> None:
        """
        校验本场景是否满足引擎契约。由 Engine.__init__ 自动调用。

        检查项：
          1. observation 签名能接受 (field, agent, all_agents)
          2. 必要属性 exits / shape 存在
        """
        # 1) 签名检查：observation 必须能接受 (field, agent, all_agents)
        #    —— 即含 self 在内至少 4 个参数位置，或第 3 个参数为 *args/**kwargs
        obs = getattr(type(self), "observation", None)
        if obs is None:
            raise ScenarioInterfaceError("缺少 observation 方法")
        sig = inspect.signature(obs)
        params = list(sig.parameters.values())
        n = len(params)
        # 接受可变参数也算合规（如 def observation(self, field, agent, *args)）
        has_var = any(p.kind in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD) for p in params)
        # 业务参数 = 除 self 外；需要 field, agent, all_agents 三个固定位置
        if n < 4 and not has_var:
            names = [p.name for p in params]
            raise ScenarioInterfaceError(
                f"observation 签名不满足契约: 需要能接收 (field, agent, all_agents), "
                f"即至少 4 个参数(含 self)。实际参数: {names}。"
                f"请改为 def observation(self, field, agent, all_agents=None): ..."
            )
        # 2) 必要属性
        for attr in ("exits", "shape"):
            if not hasattr(self, attr):
                raise ScenarioInterfaceError(f"场景缺少必要属性: {attr}")


__all__ = ["Scenario", "ScenarioInterfaceError"]
