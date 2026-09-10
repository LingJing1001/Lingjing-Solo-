"""BFS（广度优先搜索）算法。

移植自 C:\\newtask-pi\\arc_agi_solver\\solver\\bfs_solver.py。

适用于状态空间较小、深度适中的问题（如 ARC 网格变换）。
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class BFSResult:
    """BFS 搜索结果。"""
    actions: list | None
    nodes: int
    elapsed: float
    won: bool
    depth: int


class BFSSearch(Generic[T]):
    """通用 BFS 搜索器。"""

    def __init__(
        self,
        generate_fn: Callable[[T], list[tuple]],
        is_goal_fn: Callable[[T], bool],
        state_key_fn: Callable[[T], tuple] | None = None,
    ):
        self.generate = generate_fn
        self.is_goal = is_goal_fn
        self.state_key = state_key_fn or (lambda s: s)

    def search(
        self,
        start: T,
        max_depth: int = 30,
        *,
        verbose: bool = True,
    ) -> BFSResult:
        t0 = time.time()
        nodes = 0
        frontier: deque[tuple[T, list]] = deque([(start, [])])
        visited: set = {self.state_key(start)}

        while frontier:
            state, path = frontier.popleft()

            if len(path) > max_depth:
                continue

            if self.is_goal(state):
                if verbose:
                    print(f"  BFS 成功: depth={len(path)}, nodes={nodes}, elapsed={time.time()-t0:.2f}s")
                return BFSResult(path, nodes, time.time() - t0, True, len(path))

            for action, next_state in self.generate(state):
                key = self.state_key(next_state)
                if key in visited:
                    continue
                visited.add(key)
                nodes += 1
                frontier.append((next_state, path + [action]))

        if verbose:
            print(f"  BFS 无解: nodes={nodes}, elapsed={time.time()-t0:.2f}s")
        return BFSResult(None, nodes, time.time() - t0, False, 0)


def bfs_search(
    start,
    generate_fn,
    is_goal_fn,
    state_key_fn=None,
    *,
    max_depth: int = 30,
    verbose: bool = True,
) -> BFSResult:
    """便捷封装: 单次调用 BFS 搜索。"""
    searcher = BFSSearch(generate_fn, is_goal_fn, state_key_fn)
    return searcher.search(start, max_depth=max_depth, verbose=verbose)


__all__ = ["BFSSearch", "BFSResult", "bfs_search"]
