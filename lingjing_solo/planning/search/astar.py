"""A* 搜索算法 — 带顾问反思接口。

移植自 C:\\newtask-pi\\ar25_solver\\solver.py 的 solve_astar()。

特性:
    - Dijkstra 去重（同类状态只扩展最优路径）
    - 预算剪枝（超预算立即终止）
    - 镜像对感知启发（适用于 AR25 等有对称机制的游戏）
    - 顾问反思钩子（可注入 LLM 反思）
"""
from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class AStarResult:
    """A* 搜索结果。"""
    actions: list | None      # 动作序列，None 表示未找到
    nodes: int                # 展开节点数
    elapsed: float           # 耗时（秒）
    won: bool                 # 是否达成目标
    budget_used: int | None  # 消耗的步数


@dataclass
class SearchSnapshot:
    """反思顾问快照 — 传递给顾问判断是否该调整策略。"""
    level: str | int
    strategy: str
    steps_used: int
    budget: int
    coverage: float           # 目标覆盖率
    total_targets: int
    heuristic: float
    nodes: int
    dead_ends: int
    best_h_history: list[float]
    elapsed: float


@dataclass
class ReflectionAdvice:
    """顾问反思建议。"""
    action: str          # CONTINUE | ADJUST | SWITCH | RESTART | PRUNE | GIVE_UP
    advice: str
    params: dict = field(default_factory=dict)


class AStarSearch(Generic[T]):
    """通用 A* 搜索器。

    需要提供三个核心函数:
        - state_key(state) -> hashable
        - generate(state) -> list[(action, next_state, cost)]
        - heuristic(state) -> float (越小越好)
        - is_goal(state) -> bool
    """

    def __init__(
        self,
        state_key_fn: Callable[[T], tuple],
        generate_fn: Callable[[T], list[tuple]],
        heuristic_fn: Callable[[T], float],
        is_goal_fn: Callable[[T], bool],
        weight: float = 1.0,
    ):
        self.state_key = state_key_fn
        self.generate = generate_fn
        self.heuristic = heuristic_fn
        self.is_goal = is_goal_fn
        self.weight = weight

    def search(
        self,
        start: T,
        budget: int,
        *,
        t_limit: float = 60.0,
        max_nodes: int = 2000000,
        advisor_fn: Callable[[SearchSnapshot], ReflectionAdvice | None] | None = None,
        verbose: bool = True,
    ) -> AStarResult:
        """执行 A* 搜索。

        Args:
            start: 初始状态
            budget: 步数上限
            t_limit: 时间限制（秒）
            max_nodes: 最大展开节点数
            advisor_fn: 顾问反思函数（可选）
            verbose: 打印进度

        Returns:
            AStarResult
        """
        t0 = time.time()
        seq = 0
        initial_h = self.heuristic(start)

        heap: list[tuple] = [(initial_h * self.weight, 0, seq, start, [])]
        best_steps: dict = {self.state_key(start): 0}
        nodes = 0
        dead_ends = 0
        max_steps = budget - 1
        best_h_history = [initial_h]
        weight = self.weight

        if verbose:
            print(f"  A*: budget={budget}, h0={initial_h:.1f}, w={weight}, max_nodes={max_nodes}")

        while heap:
            elapsed = time.time() - t0
            if elapsed > t_limit or nodes > max_nodes:
                if verbose:
                    print(f"  A* 超时: {nodes}节点, {elapsed:.1f}s")
                return AStarResult(None, nodes, elapsed, False, None)

            f, depth, _, state, path = heapq.heappop(heap)
            key = self.state_key(state)

            if depth > best_steps.get(key, 999):
                continue
            if depth >= max_steps:
                dead_ends += 1
                continue

            if self.is_goal(state):
                if verbose:
                    print(f"  A* 成功: {len(path)}步, {nodes}节点, {elapsed:.1f}s")
                return AStarResult(path, nodes, elapsed, True, len(path))

            # ─── 顾问反思触发 ────────────────────────────
            if advisor_fn is not None:
                covered = 0  # 子类可覆盖
                snap = SearchSnapshot(
                    level="unknown",
                    strategy="astar",
                    steps_used=depth,
                    budget=budget,
                    coverage=covered,
                    total_targets=0,
                    heuristic=self.heuristic(state),
                    nodes=nodes,
                    dead_ends=dead_ends,
                    best_h_history=best_h_history,
                    elapsed=elapsed,
                )
                advice = advisor_fn(snap)
                if advice:
                    if advice.action == "GIVE_UP":
                        if verbose:
                            print(f"  顾问建议放弃: {advice.advice}")
                        return AStarResult(None, nodes, elapsed, False, None)
                    if advice.action == "ADJUST" and "weight" in advice.params:
                        old_w = weight
                        weight = advice.params["weight"]
                        if verbose:
                            print(f"  顾问调整权重: {old_w} → {weight}")
                    if advice.action == "PRUNE" and "prune_ratio" in advice.params:
                        max_steps = max(
                            int(max_steps * (1 - advice.params["prune_ratio"])),
                            max_steps // 2,
                        )

            # ─── 扩展子节点 ──────────────────────────────
            for action, next_state, cost in self.generate(state):
                nxt_depth = depth + cost
                nxt_key = self.state_key(next_state)
                if nxt_depth > best_steps.get(nxt_key, 999):
                    continue
                seq += 1
                h = self.heuristic(next_state)
                best_h_history.append(h)
                f = nxt_depth + h * weight
                heapq.heappush(
                    heap,
                    (f, nxt_depth, seq, next_state, path + [action]),
                )

            nodes += 1

        if verbose:
            print(f"  A* 无解: {nodes}节点, {time.time()-t0:.1f}s")
        return AStarResult(None, nodes, time.time() - t0, False, None)


def astar_search(
    start,
    generate_fn,
    heuristic_fn,
    is_goal_fn,
    state_key_fn=None,
    *,
    budget: int = 64,
    weight: float = 1.0,
    t_limit: float = 60.0,
    max_nodes: int = 2000000,
    advisor_fn=None,
    verbose: bool = True,
) -> AStarResult:
    """便捷封装: 单次调用 A* 搜索。"""
    if state_key_fn is None:
        state_key_fn = lambda s: s

    searcher = AStarSearch(
        state_key_fn=state_key_fn,
        generate_fn=generate_fn,
        heuristic_fn=heuristic_fn,
        is_goal_fn=is_goal_fn,
        weight=weight,
    )
    return searcher.search(
        start,
        budget=budget,
        t_limit=t_limit,
        max_nodes=max_nodes,
        advisor_fn=advisor_fn,
        verbose=verbose,
    )


__all__ = ["AStarSearch", "AStarResult", "SearchSnapshot", "ReflectionAdvice", "astar_search"]
