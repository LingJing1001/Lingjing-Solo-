"""DFS（深度优先搜索）算法。

移植自 C:\\newtask-pi\\arc_agi_solver\\solver\\dfs_solver.py。

适用于深度大、分支少的问题（配合剪枝使用）。
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class DFSResult:
    """DFS 搜索结果。"""
    actions: list | None
    nodes: int
    elapsed: float
    won: bool
    depth: int


class DFSSearch(Generic[T]):
    """通用 DFS 搜索器（迭代深化版本）。"""

    def __init__(
        self,
        generate_fn: Callable[[T], list[tuple]],
        is_goal_fn: Callable[[T], bool],
        state_key_fn: Callable[[T], tuple] | None = None,
    ):
        self.generate = generate_fn
        self.is_goal = is_goal_fn
        self.state_key = state_key_fn or (lambda s: s)

    def _dfs_recursive(
        self,
        state: T,
        path: list,
        visited: set,
        depth: int,
        max_depth: int,
    ) -> tuple[list | None, int]:
        """递归 DFS（返回 (解路径, 访问节点数)。"""
        nodes = 1

        if self.is_goal(state):
            return path, nodes

        if depth >= max_depth:
            return None, nodes

        key = self.state_key(state)
        if key in visited:
            return None, nodes

        visited.add(key)

        for action, next_state in self.generate(state):
            result, n = self._dfs_recursive(
                next_state, path + [action], visited, depth + 1, max_depth
            )
            nodes += n
            if result is not None:
                return result, nodes

        visited.discard(key)
        return None, nodes

    def search(
        self,
        start: T,
        max_depth: int = 30,
        *,
        verbose: bool = True,
    ) -> DFSResult:
        t0 = time.time()
        nodes = 0

        # 迭代深化：逐步增加深度上限
        for depth_limit in range(1, max_depth + 1):
            visited: set = set()
            result, n = self._dfs_recursive(start, [], visited, 0, depth_limit)
            nodes += n
            if result is not None:
                if verbose:
                    print(f"  DFS 成功: depth={len(result)}, nodes={nodes}, elapsed={time.time()-t0:.2f}s")
                return DFSResult(result, nodes, time.time() - t0, True, len(result))

        if verbose:
            print(f"  DFS 无解: nodes={nodes}, elapsed={time.time()-t0:.2f}s")
        return DFSResult(None, nodes, time.time() - t0, False, 0)


def dfs_search(
    start,
    generate_fn,
    is_goal_fn,
    state_key_fn=None,
    *,
    max_depth: int = 30,
    verbose: bool = True,
) -> DFSResult:
    """便捷封装: 单次调用 DFS 搜索。"""
    searcher = DFSSearch(generate_fn, is_goal_fn, state_key_fn)
    return searcher.search(start, max_depth=max_depth, verbose=verbose)


__all__ = ["DFSSearch", "DFSResult", "dfs_search"]
