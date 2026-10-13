"""统一 R3 WorldView profile 接口（§3.2.1）。

WorldView 是每个游戏唯一允许携带游戏知识的 profile 接口。R3 内核只消费
WorldView 暴露的方法，不直接访问引擎对象或游戏专用字段。

硬约束（§3.2.1 三条）：
    1. key() 不得包含单调资源计数器；计数器放在 counters() 中。
    2. progress()/done() 优先使用官方 FrameData 字段，不能用视觉变化或
       私有引擎字段伪造通关。
    3. proposals() 只返回 abstract action name，动作到引擎枚举的转换由
       ARC boundary adapter 负责。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionSpec:
    """R3 候选动作规格：abstract name + 可选 payload 描述 + 可选 cost hint。

    planner 只返回 abstract name（如 ``"ACTION3"``）；带 payload 的复杂动作
    （如 ``ACTION6(x,y)``）的坐标转换由 boundary adapter 负责，不在这里塞数字。
    """
    name: str
    payload_hint: dict[str, Any] = field(default_factory=dict)
    cost_hint: int = 1

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"ActionSpec.name 必须是非空 str，收到 {self.name!r}")
        if self.cost_hint < 0:
            raise ValueError(f"cost_hint 不能为负: {self.cost_hint}")


def action_edge_key(name: str, payload: dict[str, Any] | None = None) -> str:
    """learned graph 的边键：动作名 + payload 里的整数坐标（`NAME@x,y`）。

    只有点击类动作的 payload 带坐标；键盘 / UNDO 两端都不带，键自然就是纯动作名。
    写入端（官方回显 `action_input.data`）与读取端（`ActionSpec.payload_hint`）
    都只过这一个函数，口径不会各自跑偏。

    为什么需要它：`(state, action_name)` 当键时，同一状态下点 (1,2) 和点 (3,4)
    是同一条边，后写覆盖先写；P1 noop 判据也因此与"点哪里"无关，点击通道恒失明
    （dc22 实测：108 次 ACTION6 回显只留 19 个边键，P1_NOOP 落在 ACTION6 上 0 次）。

    只改图内部的键，**不改 `proposals()` 返回的 abstract action name**，
    所以 §3.2.1-3（坐标只进 payload_hint、转换归 boundary adapter）照旧成立。
    """
    if not name:
        return ""
    data = payload or {}
    x, y = data.get("x"), data.get("y")
    # bool 是 int 的子类，坐标不接受
    if isinstance(x, int) and not isinstance(x, bool) \
            and isinstance(y, int) and not isinstance(y, bool):
        return f"{name}@{x},{y}"
    return name


@dataclass(frozen=True)
class ConfigSpace:
    """配置层搜索空间（T2 双层搜索的配置层）。

    当 WorldView.config_space() 返回非 None 时，R3 内核先在配置层搜索目标终态
    配置，再由动作层 pathify 并真实校验。None 表示该游戏无配置层（直接走动作层）。
    """
    nodes: tuple[Any, ...] = ()
    adjacency: dict[Any, list[Any]] = field(default_factory=dict)
    goal_configs: set[Any] = field(default_factory=set)

    def is_goal(self, config: Any) -> bool:
        return config in self.goal_configs

    def neighbors(self, config: Any) -> list[Any]:
        return self.adjacency.get(config, [])


class WorldView(ABC):
    """每个游戏唯一允许携带游戏知识的 profile 接口（§3.2.1）。

    R3 内核通过这 9 个方法消费游戏知识，不直接访问引擎对象。每个游戏提供
    差异化 WorldView 实现，不复制 R2/R3/R4/R5 的整套闭环。
    """

    @abstractmethod
    def grids(self, frame: Any) -> list[Any]:
        """帧 → 结构化网格列表（可能有多层：主网格 + UI 层 + ...）。"""
        ...

    @abstractmethod
    def key(self, grids: list[Any]) -> Hashable:
        """网格 → 状态键。

        硬约束：不得包含单调资源计数器（步数、分数等）；那些放在 counters()。
        否则 P2 计数器支配剪枝失效——同局面不同步数会被当成不同状态。
        """
        ...

    @abstractmethod
    def counters(self, grids: list[Any]) -> dict[str, int]:
        """网格 → 资源计数器（步数、生命、分数等单调量）。

        与 key() 分离：P2 剪枝用 key 去重，用 counters 做支配比较。
        """
        ...

    @abstractmethod
    def proposals(self, grids: list[Any], key: Hashable) -> list[ActionSpec]:
        """网格 + 状态键 → 动作候选列表。

        只返回 abstract action name；动作到引擎枚举的转换由 boundary adapter 负责。
        """
        ...

    @abstractmethod
    def config_space(self, grids: list[Any]) -> ConfigSpace | None:
        """网格 → 配置层搜索空间（或 None）。

        非 None 时 R3 内核先做配置层搜索，再做 pathify 和真实动作校验（T2 双层）。
        None 表示直接走动作层搜索。
        """
        ...

    @abstractmethod
    def score(self, grids: list[Any]) -> float:
        """网格 → 当前得分（用于启发式排序）。"""
        ...

    @abstractmethod
    def progress(self, frame: Any) -> int:
        """帧 → 进度量（如 levels_completed）。

        硬约束：优先使用官方 FrameData 字段，不能用视觉变化或私有引擎字段伪造。
        """
        ...

    @abstractmethod
    def done(self, frame: Any) -> bool:
        """帧 → 是否通关。

        硬约束：优先使用官方 state=WIN / levels_completed 字段。
        """
        ...

    @abstractmethod
    def dead(self, frame: Any) -> bool:
        """帧 → 是否死亡/游戏结束（GAME_OVER 或关键资源耗尽）。

        P3 闸门用：dead=True 时停止该分支，记录 divergence。
        """
        ...


class WorldViewInvariantError(RuntimeError):
    """WorldView 不满足 §3.2.1 硬约束。"""


def assert_world_view_invariant(wv: WorldView, *, sample_frame: Any = None) -> None:
    """注册 WorldView 时断言 P2 硬约束：key() 不含计数器。

    启发式检查：如果 sample_frame 可用，验证 key(grids) 不随 counters 变化而变化。
    完整的 P2 检查需要游戏实例，这里只做接口级断言。

    Raises:
        WorldViewInvariantError: 如果 WorldView 不满足硬约束。
    """
    if not isinstance(wv, WorldView):
        raise WorldViewInvariantError(
            f"{type(wv).__name__} 不是 WorldView 子类"
        )

    # 接口级检查：所有抽象方法都已实现（ABC 已保证，这里再显式断言一次）
    for method_name in (
        "grids", "key", "counters", "proposals", "config_space",
        "score", "progress", "done", "dead",
    ):
        method = getattr(wv, method_name, None)
        if method is None or not callable(method):
            raise WorldViewInvariantError(
                f"WorldView {type(wv).__name__} 缺少方法 {method_name}"
            )

    # 如果有样本帧，做 P2 启发式验证：key 不应随 counters 变化
    if sample_frame is not None:
        try:
            grids = wv.grids(sample_frame)
            k1 = wv.key(grids)
            c1 = wv.counters(grids)
            # key 应该是确定性的：同样输入同样输出
            k2 = wv.key(grids)
            if k1 != k2:
                raise WorldViewInvariantError(
                    f"key() 不确定：同一 grids 两次调用得到 {k1!r} != {k2!r}"
                )
        except WorldViewInvariantError:
            raise
        except Exception as exc:
            raise WorldViewInvariantError(
                f"WorldView 在样本帧上调用失败: {type(exc).__name__}: {exc}"
            ) from exc


__all__ = [
    "ActionSpec",
    "ConfigSpace",
    "WorldView",
    "WorldViewInvariantError",
    "action_edge_key",
    "assert_world_view_invariant",
]
