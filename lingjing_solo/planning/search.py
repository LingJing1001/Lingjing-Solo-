"""Layer 3a · 搜索算法 search.py —— BFS + A* 启发式（在 field 仿真图上推演）

R3 角色：接收 explorer 候选规则，在世界模型内仿真，输出下一步动作。
"""
from __future__ import annotations

import heapq
from collections import deque
from ..core import SoloConfig, Logger, clamp, canonicalize


class SearchEngine:
    """无 LLM 短程规划：BFS 目标搜索 + A* 信息增益启发。"""

    def __init__(self, cfg: SoloConfig, field, explorer=None, logger: Logger = None):
        self.cfg = cfg
        self.field = field
        self.explorer = explorer
        self.log = logger or Logger()
        self._plan: list[str] = []
        self._plan_age = 0
        self._search_failures = 0

    def clear_plan(self):
        self._plan = []
        self._plan_age = 0

    @property
    def search_budget_exhausted(self) -> bool:
        cap = int(self.cfg.human_baseline_estimate * self.cfg.hard_step_multiplier)
        return self._search_failures >= 3 or self.field.step >= cap * 0.85

    def search(self, goal_hint=None, depth=None, valid_actions=None) -> str | None:
        depth = depth or self.cfg.lightweight_search_depth
        if len(self.field.predict_graph) == 0:
            return None

        start = self.field.current_hash()
        if not start:
            return None

        actions = [canonicalize(a) for a in (valid_actions or self.cfg.allowed_actions)]

        if self._plan and self._plan_age < self.cfg.plan_replan_every:
            nxt = canonicalize(self._plan[0])
            if nxt in actions and self.field.predict(start, nxt) is not None:
                self._plan = self._plan[1:]
                self._plan_age += 1
                return nxt
            self.clear_plan()

        goals = set(goal_hint or [])
        goals |= self.field.goal_hashes()

        path = None
        if goals:
            if self.cfg.enable_astar:
                path = self._astar(start, actions, depth, goal_set=goals)
            if path is None:
                path = self._bfs(start, actions, depth, goal_set=goals)
            if path:
                self._plan = path[1:]
                self._plan_age = 0
                self._search_failures = 0
                mode = "astar" if self.cfg.enable_astar else "bfs"
                self.log.log("Search", f"{mode}_goal path_len={len(path)} next={path[0]}")
                return path[0]

        if any(v >= 1.0 for v in self.field.state_value.values()):
            path = self._bfs_best_value(start, actions, depth)
            if path:
                self._plan = path[1:]
                self._plan_age = 0
                self._search_failures = 0
                self.log.log("Search", f"bfs_value path_len={len(path)} next={path[0]}")
                return path[0]

        if len(self.field.predict_graph) >= 3:
            path = self._bfs_frontier(start, actions, depth)
            if path:
                self._plan = path[1:]
                self._plan_age = 0
                self._search_failures = 0
                self.log.log("Search", f"bfs_frontier path_len={len(path)} next={path[0]}")
                return path[0]

        self._search_failures += 1
        return None

    def _heuristic(self, state: str, goal_set: set | None = None) -> float:
        """A* 启发：价值距离 + 信息增益（越小越好）。"""
        if goal_set and state in goal_set:
            return 0.0
        val = self.field.state_value.get(state, 0.0)
        h_val = max(0.0, 5.0 - val) / 5.0
        h_gain = 0.0
        if self.explorer is not None:
            gains = [self.explorer.info_gain(a) for a in self.cfg.allowed_actions[:5]]
            if gains:
                h_gain = 1.0 / (1.0 + max(gains))
        return h_val + self.cfg.astar_info_gain_weight * h_gain

    def _astar(self, start, actions, depth, goal_set=None) -> list[str] | None:
        if not goal_set or start in goal_set:
            return None
        # (f, g, state, path)
        open_heap: list[tuple[float, int, str, list[str]]] = []
        heapq.heappush(open_heap, (self._heuristic(start, goal_set), 0, start, []))
        best_g: dict[str, int] = {start: 0}

        while open_heap:
            f, g, state, path = heapq.heappop(open_heap)
            if len(path) >= depth:
                continue
            if state in goal_set:
                return path
            for a in actions:
                nxt = self.field.predict(state, a)
                if nxt is None:
                    continue
                new_g = g + 1
                if nxt in best_g and best_g[nxt] <= new_g:
                    continue
                best_g[nxt] = new_g
                new_path = path + [a]
                if nxt in goal_set:
                    return new_path
                h = self._heuristic(nxt, goal_set)
                heapq.heappush(open_heap, (new_g + h, new_g, nxt, new_path))
        return None

    def _bfs(self, start, actions, depth, goal_set=None) -> list[str] | None:
        if not goal_set or start in goal_set:
            return None
        q = deque([(start, [])])
        visited = {start}
        while q:
            state, path = q.popleft()
            if len(path) >= depth:
                continue
            for a in actions:
                nxt = self.field.predict(state, a)
                if nxt is None or nxt in visited:
                    continue
                new_path = path + [a]
                if nxt in goal_set:
                    return new_path
                visited.add(nxt)
                q.append((nxt, new_path))
        return None

    def _bfs_best_value(self, start, actions, depth) -> list[str] | None:
        best_path = None
        best_val = self.field.state_value.get(start, 0.0)
        q = deque([(start, [])])
        visited = {start}
        while q:
            state, path = q.popleft()
            if len(path) >= depth:
                continue
            for a in actions:
                nxt = self.field.predict(state, a)
                if nxt is None or nxt in visited:
                    continue
                new_path = path + [a]
                val = self.field.state_value.get(nxt, 0.0)
                if val > best_val + 0.4:
                    best_val = val
                    best_path = new_path
                visited.add(nxt)
                q.append((nxt, new_path))
        return best_path

    def _bfs_frontier(self, start, actions, depth) -> list[str] | None:
        q = deque([(start, [])])
        visited = {start}
        while q:
            state, path = q.popleft()
            if len(path) >= depth:
                continue
            known = 0
            unknowns = []
            for a in actions:
                if self.field.predict(state, a) is None:
                    if self.field.pair_counts.get((state, a), 0) == 0:
                        unknowns.append(a)
                else:
                    known += 1
            if unknowns and not path:
                unknowns.sort(
                    key=lambda a: (
                        self.field.action_counts.get(a, 0),
                        self.field.pair_counts.get((state, a), 0),
                        a,
                    )
                )
                return [unknowns[0]]
            if path and known < max(2, len(actions) // 2):
                return path
            for a in actions:
                nxt = self.field.predict(state, a)
                if nxt is None or nxt in visited:
                    continue
                visited.add(nxt)
                q.append((nxt, path + [a]))
        return None

    def greedy_next(self, scored_actions: list) -> str | None:
        if not scored_actions:
            return None
        return scored_actions[0][0]


# 向后兼容别名
LightweightPlanner = SearchEngine
