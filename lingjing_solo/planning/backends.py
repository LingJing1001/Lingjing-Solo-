"""R3 后端边界（§3.2.1 后端边界）。

四种后端各有不同的能力和证据层级。R3 内核通过后端接口隔离线上/离线，
不直接接触引擎对象（R3-A 线上边界）。

    EngineBackend        离线快照/回滚，只生成和回归罐头解
    SimulatorBackend     离线纯 Python 模拟器，只能证明模拟器语义
    LearnedGraphBackend  线上默认，以 learned (state, action) -> state 图预测
    UndoBackend          仅在 action → undo → key() 实测等价时启用
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable


class SearchBackend(ABC):
    """R3 搜索后端抽象基类。

    后端只暴露 predict / validate / 快照能力，不暴露引擎对象。
    R3 内核通过这个接口隔离线上/离线（R3-A）。
    """

    @property
    @abstractmethod
    def is_online(self) -> bool:
        """是否为线上后端（禁止快照/回滚）。"""
        ...

    @property
    @abstractmethod
    def can_snapshot(self) -> bool:
        """是否支持快照/回滚（仅离线后端）。"""
        ...

    @property
    @abstractmethod
    def evidence_tier(self) -> str:
        """该后端产出的证据层级。

        返回值必须属于 {"simulator", "offline_engine", "live_scorecard"}。
        """
        ...

    @abstractmethod
    def predict(self, state_hash: Hashable, action: str) -> Hashable | None:
        """预测 (state, action) -> next_state。

        返回 None 表示预测不可用（未知转移）。线上后端用 learned graph 预测；
        离线后端用引擎快照执行一步再取 hash。
        """
        ...

    def validate(self, state_hash: Hashable, action: str, observed_next: Hashable) -> bool:
        """校验预测是否与观测一致。

        线上后端每步后必须校验：预测 != 观测 时需要重规划。
        默认实现：predict 与 observed_next 一致则 True。
        """
        predicted = self.predict(state_hash, action)
        return predicted is not None and predicted == observed_next

    def snapshot(self) -> Any:
        """保存当前状态快照（仅 can_snapshot=True 的后端）。

        默认抛出：不支持快照的后端调用此方法说明调用方逻辑错误。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持快照")

    def restore(self, snap: Any) -> None:
        """从快照恢复（仅 can_snapshot=True 的后端）。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持恢复")


class EngineBackend(SearchBackend):
    """离线引擎后端：快照/回滚，只生成和回归罐头解。

    evidence_tier = "offline_engine"
    """

    @property
    def is_online(self) -> bool:
        return False

    @property
    def can_snapshot(self) -> bool:
        return True

    @property
    def evidence_tier(self) -> str:
        return "offline_engine"

    def __init__(self, *, predict_fn=None, snapshot_fn=None, restore_fn=None):
        self._predict_fn = predict_fn
        self._snapshot_fn = snapshot_fn
        self._restore_fn = restore_fn

    def predict(self, state_hash: Hashable, action: str) -> Hashable | None:
        if self._predict_fn is None:
            return None
        return self._predict_fn(state_hash, action)

    def snapshot(self) -> Any:
        if self._snapshot_fn is None:
            raise NotImplementedError("EngineBackend 未提供 snapshot_fn")
        return self._snapshot_fn()

    def restore(self, snap: Any) -> None:
        if self._restore_fn is None:
            raise NotImplementedError("EngineBackend 未提供 restore_fn")
        self._restore_fn(snap)


class SimulatorBackend(SearchBackend):
    """离线纯 Python 模拟器后端：只能证明模拟器语义。

    evidence_tier = "simulator"
    """

    @property
    def is_online(self) -> bool:
        return False

    @property
    def can_snapshot(self) -> bool:
        return True

    @property
    def evidence_tier(self) -> str:
        return "simulator"

    def __init__(self, *, predict_fn=None, snapshot_fn=None, restore_fn=None):
        self._predict_fn = predict_fn
        self._snapshot_fn = snapshot_fn
        self._restore_fn = restore_fn

    def predict(self, state_hash: Hashable, action: str) -> Hashable | None:
        if self._predict_fn is None:
            return None
        return self._predict_fn(state_hash, action)

    def snapshot(self) -> Any:
        if self._snapshot_fn is None:
            raise NotImplementedError("SimulatorBackend 未提供 snapshot_fn")
        return self._snapshot_fn()

    def restore(self, snap: Any) -> None:
        if self._restore_fn is None:
            raise NotImplementedError("SimulatorBackend 未提供 restore_fn")
        self._restore_fn(snap)


class LearnedGraphBackend(SearchBackend):
    """线上默认后端：以 learned (state, action) -> state 图预测并在每步后校验。

    evidence_tier = "live_scorecard"

    硬约束（R3-A）：is_online=True，不支持快照/回滚。线上路径不读取 _game、
    不依赖快照回滚。
    """

    @property
    def is_online(self) -> bool:
        return True

    @property
    def can_snapshot(self) -> bool:
        return False

    @property
    def evidence_tier(self) -> str:
        return "live_scorecard"

    def __init__(self, predict_graph: dict | None = None):
        """predict_graph: {(state_hash, action) -> next_state_hash}"""
        self._graph = predict_graph or {}

    def predict(self, state_hash: Hashable, action: str) -> Hashable | None:
        return self._graph.get((state_hash, action))

    def update(self, state_hash: Hashable, action: str, next_hash: Hashable) -> None:
        """每步后用观测更新 learned graph。"""
        self._graph[(state_hash, action)] = next_hash

    @property
    def edge_count(self) -> int:
        """已学到的 (state, action) 转移条数（只读诊断口径，不参与搜索）。"""
        return len(self._graph)


class UndoBackend(SearchBackend):
    """Undo 后端：仅在 action → undo → key() 实测等价时启用。

    evidence_tier = "live_scorecard"

    启用条件：profile 经过实测证明 UNDO 恢复的状态维度覆盖 key()。
    未验证时退回 LearnedGraphBackend。
    """

    @property
    def is_online(self) -> bool:
        return True

    @property
    def can_snapshot(self) -> bool:
        return False

    @property
    def evidence_tier(self) -> str:
        return "live_scorecard"

    def __init__(self, *, undo_verified: bool = False,
                 predict_fn=None, undo_fn=None):
        if not undo_verified:
            raise ValueError(
                "UndoBackend 仅在 action → undo → key() 实测等价时启用；"
                "未验证时请用 LearnedGraphBackend"
            )
        self._predict_fn = predict_fn
        self._undo_fn = undo_fn

    def predict(self, state_hash: Hashable, action: str) -> Hashable | None:
        if self._predict_fn is None:
            return None
        return self._predict_fn(state_hash, action)

    def undo(self, state_hash: Hashable, action: str) -> Hashable | None:
        """撤销动作，返回前一个状态 hash。"""
        if self._undo_fn is None:
            return None
        return self._undo_fn(state_hash, action)


__all__ = [
    "SearchBackend",
    "EngineBackend",
    "SimulatorBackend",
    "LearnedGraphBackend",
    "UndoBackend",
]
