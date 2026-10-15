"""状态空间图探索器 — 基于 BFS 的全局状态探索。

设计原则：
- 有向图：节点 = 状态哈希，边 = 动作转移
- BFS 最短路径到前沿节点
- 优先级动作选择（高优先级动作先测试）
- fail-closed：未知状态回退到简单策略

防线 1：CEAX 异常计数 + 日志
- 记录所有异常和错误
- 统计异常次数，超过阈值则告警
- 提供详细的错误日志

注意：此文件使用 spec_from_file_location 导入时，需要手动注册模块到 sys.modules。
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================================
# 修复：手动注册模块到 sys.modules（防止 spec_from_file_location 导入失败）
# ============================================================================

# 在 dataclass 装饰器之前注册模块
_module_name = __name__
if _module_name not in sys.modules:
    # 创建一个虚拟模块
    import types
    _dummy_module = types.ModuleType(_module_name)
    sys.modules[_module_name] = _dummy_module


# ============================================================================
# 防线 1：CEAX 异常计数 + 日志
# ============================================================================

# 配置日志
logger = logging.getLogger("graph_explorer")

# 异常计数
_exception_counts: Dict[str, int] = {}
_exception_threshold = 10  # 异常阈值


def log_exception(exc: Exception, context: str = "") -> None:
    """记录异常。

    Args:
        exc: 异常对象
        context: 上下文信息
    """
    exc_type = type(exc).__name__
    _exception_counts[exc_type] = _exception_counts.get(exc_type, 0) + 1

    # 记录日志
    logger.error(f"[CEAX] {context}: {exc_type}: {str(exc)}")

    # 检查阈值
    if _exception_counts[exc_type] >= _exception_threshold:
        logger.warning(f"[CEAX] 异常 {exc_type} 超过阈值 {_exception_threshold}！")


def get_exception_counts() -> Dict[str, int]:
    """获取异常计数。"""
    return _exception_counts.copy()


def reset_exception_counts() -> None:
    """重置异常计数。"""
    _exception_counts.clear()


# ============================================================================
# 动作表示转换（实盘决策用 "ACTIONn" 字符串，图内用 int）
# ============================================================================

def action_name_to_int(action: Any) -> Optional[int]:
    """ACTIONn 字符串（或 int）→ 动作编号 int；无法识别返回 None。"""
    if isinstance(action, bool):
        return None
    if isinstance(action, int):
        return action if 1 <= action <= 9 else None
    s = str(action or "").strip().upper()
    if s.startswith("ACTION") and s[6:].isdigit():
        n = int(s[6:])
        return n if 1 <= n <= 9 else None
    return None


def action_int_to_name(n: int) -> str:
    """动作编号 int → ACTIONn 字符串。"""
    return f"ACTION{int(n)}"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StateNode:
    """状态节点。"""
    state_hash: str
    actions: List[int]                    # 可用动作列表
    tested_actions: Set[int] = field(default_factory=set)  # 已测试动作
    transitions: Dict[int, str] = field(default_factory=dict)  # action -> next_state_hash
    success: Dict[int, bool] = field(default_factory=dict)     # action -> success
    distance: float = float("inf")        # 到前沿节点的距离
    closed: bool = False                  # 是否已关闭（所有动作测试完毕）
    action_priorities: Dict[int, int] = field(default_factory=dict)  # action -> priority (0=highest)

    def has_untested_actions(self) -> bool:
        """是否有未测试动作。"""
        return any(a not in self.tested_actions for a in self.actions)

    def get_untested_actions(self) -> List[int]:
        """获取未测试动作列表。"""
        return [a for a in self.actions if a not in self.tested_actions]

    def get_untested_actions_by_priority(self) -> List[int]:
        """获取按优先级排序的未测试动作列表（优先级高的先选）。"""
        untested = self.get_untested_actions()
        return sorted(untested, key=lambda a: self.action_priorities.get(a, 999))

    def is_frontier(self) -> bool:
        """是否是前沿节点（有未测试动作）。"""
        return not self.closed and self.has_untested_actions()


@dataclass
class GraphExplorerConfig:
    """探索器配置。"""
    max_nodes: int = 1000                 # 最大节点数
    max_depth: int = 50                   # 最大搜索深度
    priority_levels: int = 5              # 优先级层级数
    verbose: bool = False                 # 是否输出调试信息
    default_action_priorities: Dict[int, int] = field(default_factory=dict)


# ============================================================================
# 图探索器
# ============================================================================

class GraphExplorer:
    """状态空间图探索器。

    核心逻辑：
    1. 维护状态图（节点 = 状态，边 = 动作转移）
    2. BFS 计算到前沿节点的距离
    3. 选择动作：优先测试当前状态的未测试动作
    4. 如果当前状态无未测试动作，移动到最近的前沿节点
    """

    def __init__(self, config: Optional[GraphExplorerConfig] = None):
        self.config = config or GraphExplorerConfig()
        self.nodes: Dict[str, StateNode] = {}
        self.frontier: Set[str] = set()
        self._dist: Dict[str, float] = {}
        self._next: Dict[str, Tuple[int, str]] = {}
        self._current_state: Optional[str] = None

    # ----------------------------------------------------------------------
    # 状态管理
    # ----------------------------------------------------------------------

    def hash_state(self, state: Any) -> str:
        """计算状态哈希。"""
        if isinstance(state, np.ndarray):
            state = state.tolist()
        blob = json.dumps(state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def initialize(self, state: Any, actions: List[int],
                 action_priorities: Optional[Dict[int, int]] = None) -> str:
        """初始化探索器，添加起始状态。"""
        state_hash = self.hash_state(state)
        self._current_state = state_hash
        self._add_node(state_hash, actions, action_priorities)
        self._rebuild_distances()
        return state_hash

    def _add_node(self, state_hash: str, actions: List[int],
                action_priorities: Optional[Dict[int, int]] = None) -> StateNode:
        """添加状态节点（含 max_nodes 上限与空动作补齐）。"""
        if action_priorities is None:
            action_priorities = self.config.default_action_priorities

        if state_hash in self.nodes:
            node = self.nodes[state_hash]
            # 占位节点（record 时未知动作）后续拿到动作列表 → 补齐并重新入前沿
            if not node.actions and actions:
                node.actions = list(actions)
                if not node.action_priorities:
                    node.action_priorities = action_priorities
                if node.is_frontier():
                    self.frontier.add(state_hash)
            return node

        # 上限：逐出最早的、非前沿且非当前状态的节点
        if len(self.nodes) >= max(1, self.config.max_nodes):
            self._evict_older(state_hash)
            if len(self.nodes) >= max(1, self.config.max_nodes):
                # 无可逐出节点：返回未注册的临时节点，决策链路不受影响
                return StateNode(
                    state_hash=state_hash,
                    actions=list(actions),
                    action_priorities=action_priorities,
                )

        node = StateNode(
            state_hash=state_hash,
            actions=list(actions),
            action_priorities=action_priorities,
        )
        self.nodes[state_hash] = node

        if node.is_frontier():
            self.frontier.add(state_hash)

        return node

    def _evict_older(self, exclude: str) -> bool:
        """逐出一个可牺牲节点（非当前、非前沿）。返回是否成功腾位。"""
        for h in list(self.nodes.keys()):
            if h == exclude or h == self._current_state or h in self.frontier:
                continue
            self.nodes.pop(h, None)
            self.frontier.discard(h)
            self._dist.pop(h, None)
            self._next.pop(h, None)
            return True
        return False

    def ensure_node(self, state: Any, actions: List[int],
                    action_priorities: Optional[Dict[int, int]] = None) -> str:
        """登记/补齐当前状态节点（决策入口调用，供图被持续喂数据）。"""
        state_hash = self.hash_state(state)
        self._current_state = state_hash
        before = len(self.frontier)
        self._add_node(state_hash, list(actions or []), action_priorities)
        if len(self.frontier) != before:
            self._rebuild_distances()
        return state_hash

    # ----------------------------------------------------------------------
    # 记录转移
    # ----------------------------------------------------------------------

    def record_transition(
        self,
        from_state: Any,
        action: int,
        success: bool,
        to_state: Any,
        to_actions: Optional[List[int]] = None,
        to_action_priorities: Optional[Dict[int, int]] = None,
    ) -> None:
        """记录状态转移。

        防线 1：CEAX 异常计数 + 日志
        """
        try:
            from_hash = self.hash_state(from_state)
            to_hash = self.hash_state(to_state)

            # fail-closed：源状态未登记 → 先建空动作节点（后续 ensure_node 补齐）
            if from_hash not in self.nodes:
                self._add_node(from_hash, [])
            from_node = self.nodes[from_hash]

            from_node.tested_actions.add(action)
            from_node.transitions[action] = to_hash
            from_node.success[action] = success

            if success:
                # 目的状态动作列表未知时建占位节点（不抛错）
                to_node = self._add_node(
                    to_hash, list(to_actions or []), to_action_priorities
                )

                if to_hash in self.nodes and to_node.is_frontier():
                    self.frontier.add(to_hash)
                else:
                    self.frontier.discard(to_hash)

            if not from_node.has_untested_actions():
                from_node.closed = True
                self.frontier.discard(from_hash)

            self._rebuild_distances()

        except Exception as exc:
            log_exception(exc, f"record_transition(from={from_state}, action={action})")
            raise

    # ----------------------------------------------------------------------
    # 动作选择
    # ----------------------------------------------------------------------

    def choose_action(self, state: Any) -> Optional[Tuple[int, str]]:
        """选择下一个动作。

        防线 1：CEAX 异常计数 + 日志
        """
        try:
            state_hash = self.hash_state(state)

            if state_hash not in self.nodes:
                return None

            node = self.nodes[state_hash]

            untested_by_priority = node.get_untested_actions_by_priority()
            if untested_by_priority:
                action = untested_by_priority[0]
                return (action, state_hash)

            nearest = self._find_nearest_frontier(state_hash)
            if nearest:
                nearest_hash, path = nearest
                if path:
                    first_action = path[0]
                    return (first_action, state_hash)

            return None

        except Exception as exc:
            log_exception(exc, f"choose_action(state={state})")
            return None

    def _find_nearest_frontier(self, from_hash: str) -> Optional[Tuple[str, List[int]]]:
        """找到最近的前沿节点及其路径。"""
        if from_hash not in self._dist or self._dist[from_hash] == float("inf"):
            return None

        path: List[int] = []
        current = from_hash

        while current in self._next:
            action, next_hash = self._next[current]
            path.append(action)
            current = next_hash
            if current in self.frontier:
                return (current, path)

        return None

    # ----------------------------------------------------------------------
    # 距离计算（BFS）
    # ----------------------------------------------------------------------

    def _rebuild_distances(self) -> None:
        """重建到前沿节点的距离（多源 BFS，反向邻接表 → O(V+E)）。"""
        self._dist.clear()
        self._next.clear()

        # 前沿集合可能含被逐出/未注册哈希，先剪掉
        self.frontier = {h for h in self.frontier if h in self.nodes}

        for hash_val, node in self.nodes.items():
            self._dist[hash_val] = float("inf")
            node.distance = float("inf")

        # 反向邻接：target -> [(source_hash, action), ...]，仅统计成功边且端点在册
        rev: Dict[str, List[Tuple[str, int]]] = {h: [] for h in self.nodes}
        for h, node in self.nodes.items():
            for action, target_hash in node.transitions.items():
                if node.success.get(action, False) and target_hash in rev and target_hash != h:
                    rev[target_hash].append((h, action))

        queue = collections.deque()
        for frontier_hash in self.frontier:
            self._dist[frontier_hash] = 0
            self.nodes[frontier_hash].distance = 0
            queue.append(frontier_hash)

        while queue:
            current = queue.popleft()
            current_dist = self._dist[current]
            for prev_hash, action in rev.get(current, ()):
                new_dist = current_dist + 1
                if new_dist < self._dist.get(prev_hash, float("inf")):
                    self._dist[prev_hash] = new_dist
                    self.nodes[prev_hash].distance = new_dist
                    self._next[prev_hash] = (action, current)
                    queue.append(prev_hash)

    # ----------------------------------------------------------------------
    # 查询接口
    # ----------------------------------------------------------------------

    def get_distance(self, state: Any) -> float:
        """获取到最近前沿节点的距离。"""
        state_hash = self.hash_state(state)
        return self._dist.get(state_hash, float("inf"))

    def is_finished(self) -> bool:
        """是否探索完成（无前沿节点）。"""
        return len(self.frontier) == 0

    def get_frontier_count(self) -> int:
        """获取前沿节点数。"""
        return len(self.frontier)

    def get_node_count(self) -> int:
        """获取节点数。"""
        return len(self.nodes)

    def dump(self) -> str:
        """导出探索器状态。"""
        lines = [
            "=== Graph Explorer State ===",
            f"Nodes: {len(self.nodes)}",
            f"Frontier: {len(self.frontier)}",
            f"Finished: {self.is_finished()}",
        ]
        return "\n".join(lines)
