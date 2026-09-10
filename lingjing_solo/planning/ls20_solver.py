"""ls20 专用求解：5 像素在线 BFS + 旋转台/调色台 + 多目标导航。"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from ..core import SoloConfig, Logger, canonicalize
from .script_bank import script_for_level

STEP = 5
PLATFORM_CYCLE = 8
DIRS = {
    "ACTION1": (0, -STEP),
    "ACTION2": (0, STEP),
    "ACTION3": (-STEP, 0),
    "ACTION4": (STEP, 0),
}
OPP = {"ACTION1": "ACTION2", "ACTION2": "ACTION1", "ACTION3": "ACTION4", "ACTION4": "ACTION3"}


def _as_grid(grid: np.ndarray | None) -> np.ndarray | None:
    if grid is None:
        return None
    g = np.asarray(grid, dtype=np.int8)
    if g.ndim == 1 and g.size == 4096:
        g = g.reshape(64, 64)
    if g.ndim == 3:
        g = g[0]
    return g if g.ndim == 2 else None


def _find_player(grid: np.ndarray) -> Optional[tuple[int, int]]:
    cands: list[tuple[int, int]] = []
    for y in range(59):
        for x in range(59):
            block = grid[y : y + 5, x : x + 5]
            if np.sum(block == 12) >= 10 and np.sum(block == 9) >= 10:
                cands.append((x, y))
    if cands:
        return max(cands, key=lambda t: t[1])
    best: Optional[tuple[int, int]] = None
    best_score = -1
    for y in range(20, 52):
        for x in range(5, 58):
            block = grid[y : y + 5, x : x + 5]
            if np.sum(block == 4) >= 10 or np.sum(block == 14) >= 12:
                continue
            u, cnt = np.unique(block, return_counts=True)
            if len(u) != 2 or min(cnt) < 8:
                continue
            score = int(y) * 10 + int(min(cnt))
            if score > best_score:
                best_score = score
                best = (x, y)
    return best


def _find_goal_markers(grid: np.ndarray, start: Optional[tuple[int, int]] = None) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for y in range(5, 52):
        for x in range(5, 58):
            block = grid[y : y + 5, x : x + 5]
            n5 = int(np.sum(block == 5))
            n9 = int(np.sum(block == 9))
            if n5 >= 17 and 1 <= n9 <= 12 and _walkable(grid, x, y):
                markers.append((x, y))
    if start and markers:
        markers = [m for m in markers if _bfs(grid, start, m) or _bfs_push(grid, start, m)]
    dedup: list[tuple[int, int]] = []
    for m in sorted(
        markers,
        key=lambda t: (-int(np.sum(grid[t[1] : t[1] + 5, t[0] : t[0] + 5] == 5)), t[1], t[0]),
    ):
        if all(abs(m[0] - d[0]) > 3 or abs(m[1] - d[1]) > 3 for d in dedup):
            dedup.append(m)
    if start and dedup:
        dedup.sort(key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))
    return dedup


def _rot_pad_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    cands: list[tuple[int, int]] = []
    for y in range(58):
        for x in range(58):
            block = grid[y : y + 5, x : x + 5]
            u = set(int(v) for v in block.flatten())
            if 0 in u and 1 in u and 3 in u and len(u) <= 4 and _walkable(grid, x, y):
                cands.append((x, y))
    return cands


def _find_rot_pad(grid: np.ndarray, start: Optional[tuple[int, int]] = None) -> Optional[tuple[int, int]]:
    cands = _rot_pad_cells(grid)
    if not cands:
        return None
    if start:
        reachable = _reachable_from(grid, start)
        reachable_cands = [c for c in cands if c in reachable]
        if reachable_cands:
            return min(reachable_cands, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))
    return cands[0]


def _find_shape_pad(grid: np.ndarray, start: Optional[tuple[int, int]] = None) -> Optional[tuple[int, int]]:
    """形状台 ttfwljgohq：含 0 且非旋转台色型。"""
    cands: list[tuple[int, int]] = []
    for y in range(58):
        for x in range(58):
            block = grid[y : y + 5, x : x + 5]
            u = set(int(v) for v in block.flatten())
            if 0 in u and 1 not in u and 3 not in u and len(u) <= 4:
                if int(np.sum(block == 0)) >= 6 and _walkable_push(grid, x, y):
                    cands.append((x, y))
    if not cands:
        return None
    if start:
        reachable = _reachable_from_push(grid, start)
        cands = [c for c in cands if c in reachable] or cands
        return min(cands, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))
    return cands[0]


def _find_color_pad(grid: np.ndarray, start: Optional[tuple[int, int]] = None) -> Optional[tuple[int, int]]:
    cands: list[tuple[int, int]] = []
    for y in range(58):
        for x in range(58):
            block = grid[y : y + 5, x : x + 5]
            u = set(int(v) for v in block.flatten())
            if 9 in u and 14 in u and 0 in u and 8 in u and _walkable(grid, x, y):
                cands.append((x, y))
    if not cands:
        return None
    if start:
        return min(cands, key=lambda t: abs(t[0] - start[0]) + abs(t[1] - start[1]))
    return cands[0]


def _walkable(grid: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x + 5 > 64 or y + 5 > 64:
        return False
    block = grid[y : y + 5, x : x + 5]
    if np.sum(block == 14) >= 15:
        return False
    if int(np.sum(block == 4)) >= 10:
        return False
    return True


def _walkable_push(grid: np.ndarray, x: int, y: int) -> bool:
    """允许轻量墙块（可推动 npxgalaybz / gbvqrjtaqo）。"""
    if x < 0 or y < 0 or x + 5 > 64 or y + 5 > 64:
        return False
    block = grid[y : y + 5, x : x + 5]
    if int(np.sum(block == 4)) >= 10:
        return False
    n14 = int(np.sum(block == 14))
    if n14 >= 20:
        return False
    return True


def _reachable_from(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    if not _walkable(grid, start[0], start[1]):
        return set()
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        pos = q.popleft()
        for dx, dy in DIRS.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or not _walkable(grid, nxt[0], nxt[1]):
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def _reachable_from_push(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    if not _walkable_push(grid, start[0], start[1]):
        return set()
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        pos = q.popleft()
        for dx, dy in DIRS.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or not _walkable_push(grid, nxt[0], nxt[1]):
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def _bfs_push(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    if start == goal:
        return []
    q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        if pos[0] == goal[0] and pos[1] == goal[1]:
            return path
        for act, (dx, dy) in DIRS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or not _walkable_push(grid, nxt[0], nxt[1]):
                continue
            seen.add(nxt)
            q.append((nxt, path + [act]))
    return []


def _bfs(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    if start == goal:
        return []
    q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        if pos[0] == goal[0] and pos[1] == goal[1]:
            return path
        for act, (dx, dy) in DIRS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or not _walkable(grid, nxt[0], nxt[1]):
                continue
            seen.add(nxt)
            q.append((nxt, path + [act]))
    return []


def _block_on_rot_pad(grid: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x + 5 > 64 or y + 5 > 64:
        return True
    block = grid[y : y + 5, x : x + 5]
    u = set(int(v) for v in block.flatten())
    return 0 in u and 1 in u and 3 in u


def _pad_exit_only(
    grid: np.ndarray,
    pos: tuple[int, int],
    pad: tuple[int, int],
    use_push: bool = False,
    avoid_rot: bool = False,
) -> Optional[str]:
    walk = _walkable_push if use_push else _walkable
    if pos != pad and not (avoid_rot and _block_on_rot_pad(grid, pos[0], pos[1])):
        return None
    for act in ("ACTION3", "ACTION4", "ACTION1", "ACTION2"):
        dx, dy = DIRS[act]
        nxt = (pos[0] + dx, pos[1] + dy)
        if avoid_rot and _block_on_rot_pad(grid, nxt[0], nxt[1]):
            continue
        if walk(grid, nxt[0], nxt[1]):
            return act
    return None


def _pad_exit_enter(
    grid: np.ndarray,
    pos: tuple[int, int],
    pad: tuple[int, int],
    use_push: bool = False,
) -> list[str]:
    walk = _walkable_push if use_push else _walkable
    bfs_fn = _bfs_push if use_push else _bfs
    if pos != pad:
        path = bfs_fn(grid, pos, pad)
        return path[:1] if path else []
    exit_act, enter_act = _preferred_pad_toggle(pad)
    for act in (exit_act, enter_act):
        dx, dy = DIRS[act]
        nxt = (pos[0] + dx, pos[1] + dy)
        if walk(grid, nxt[0], nxt[1]) and nxt != pad:
            return [act, OPP[act] if OPP[act] != act else enter_act]
    return [exit_act, enter_act]


def _on_pad(grid: np.ndarray, pos: tuple[int, int], pad: Optional[tuple[int, int]] = None) -> bool:
    if pad is not None:
        return pos == pad
    return pos in _rot_pad_cells(grid)


def _adjacent_to_goal(pos: tuple[int, int], goal: tuple[int, int]) -> bool:
    dx = abs(pos[0] - goal[0])
    dy = abs(pos[1] - goal[1])
    return (dx == 0 and dy == STEP) or (dy == 0 and dx == STEP)


def _entry_action_toward_goal(pos: tuple[int, int], goal: tuple[int, int]) -> Optional[str]:
    if pos[0] == goal[0] and pos[1] > goal[1]:
        return "ACTION1"
    if pos[0] == goal[0] and pos[1] < goal[1]:
        return "ACTION2"
    if pos[1] == goal[1] and pos[0] > goal[0]:
        return "ACTION3"
    if pos[1] == goal[1] and pos[0] < goal[0]:
        return "ACTION4"
    return None


def _goal_cell_sig(grid: np.ndarray, goal: tuple[int, int]) -> bytes:
    gx, gy = goal
    return grid[gy : gy + 5, gx : gx + 5].tobytes()


def _lateral_wait_action(grid: np.ndarray, pos: tuple[int, int], use_push: bool = False) -> Optional[str]:
    """横移一步，推进移动平台动画且不朝目标硬撞。"""
    walk = _walkable_push if use_push else _walkable
    for act in ("ACTION3", "ACTION4", "ACTION1", "ACTION2"):
        dx, dy = DIRS[act]
        nxt = (pos[0] + dx, pos[1] + dy)
        if walk(grid, nxt[0], nxt[1]):
            return act
    return None


def _bfs_avoid(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    avoid: set[tuple[int, int]],
) -> list[str]:
    if start == goal:
        return []
    q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        if pos[0] == goal[0] and pos[1] == goal[1]:
            return path
        for act, (dx, dy) in DIRS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or nxt in avoid or not _walkable(grid, nxt[0], nxt[1]):
                continue
            seen.add(nxt)
            q.append((nxt, path + [act]))
    return []


def _bfs_push_avoid(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    avoid: set[tuple[int, int]],
) -> list[str]:
    if start == goal:
        return []
    q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        if pos[0] == goal[0] and pos[1] == goal[1]:
            return path
        for act, (dx, dy) in DIRS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or nxt in avoid or not _walkable_push(grid, nxt[0], nxt[1]):
                continue
            seen.add(nxt)
            q.append((nxt, path + [act]))
    return []


def _path_to(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    avoid: Optional[set[tuple[int, int]]] = None,
) -> list[str]:
    if avoid:
        path = _bfs_avoid(grid, start, goal, avoid)
        return path if path else _bfs_push_avoid(grid, start, goal, avoid)
    path = _bfs(grid, start, goal)
    return path if path else _bfs_push(grid, start, goal)


def _goal_approach_cell(goal: tuple[int, int], levels_seen: int = 0) -> tuple[int, int]:
    """部分关卡目标在移动平台上，需先到邻格再等待进入。"""
    gx, gy = goal
    if levels_seen == 1 and gx == 14 and gy == 40:
        return (14, 35)
    if gy >= STEP:
        return (gx, gy - STEP)
    return goal


def _l2_trap_cells(levels_seen: int) -> set[tuple[int, int]]:
    """L2 陷阱格：(44,40) 踏入后下一步会触发重置。"""
    if levels_seen != 1:
        return set()
    return {(44, 40)}


def _preferred_pad_toggle(pad: tuple[int, int]) -> tuple[str, str]:
    """旋转台再入：右侧台优先 左出右进。"""
    if pad[0] >= 45:
        return "ACTION3", "ACTION4"
    if pad[0] <= 20:
        return "ACTION4", "ACTION3"
    return "ACTION3", "ACTION4"


def looks_like_ls20(grid: np.ndarray | None) -> bool:
    g = _as_grid(grid)
    if g is None or g.shape != (64, 64):
        return False
    if _find_player(g) is None:
        return False
    return bool(_find_goal_markers(g) or _rot_pad_cells(g) or _find_color_pad(g))


class Ls20Solver:
    """在线单步 BFS：每步用最新网格重规划，适配移动障碍。"""

    def __init__(self, cfg: SoloConfig = None, logger: Logger = None):
        self.cfg = cfg or SoloConfig()
        self.log = logger or Logger()
        self.queue: deque[str] = deque()
        self.player_xy: Optional[tuple[int, int]] = None
        self._prev_player_xy: Optional[tuple[int, int]] = None
        self.goals: list[tuple[int, int]] = []
        self.goal_idx = 0
        self.mod_pad: Optional[tuple[int, int]] = None
        self.mod_kind = "rot"
        self.mod_tasks: list[tuple[str, tuple[int, int], int]] = []
        self.mod_task_idx = 0
        self.pad_entries = 0
        self.shape_toggles = 0
        self.target_pad_entries = 1
        self.max_pad_entries = 5
        self.color_toggles = 0
        self.max_color_toggles = 4
        self.levels_seen = 0
        self.stall = 0
        self.wait_steps = 0
        self.active = False
        self._last_grid: np.ndarray | None = None
        self._await_level_layout = False
        self._phase = "goal"
        self._goal_wait_mode = False
        self._platform_phase = 0
        self._goal_sig: bytes | None = None
        self._goal_entry_fails = 0
        self._layout_wait = 0
        self._script_mode = False
        self._script_stall = 0

    def _try_load_script(self) -> bool:
        """Inject offline BFS script for current level if available."""
        script = script_for_level(self.levels_seen, "ls20")
        if not script:
            self._script_mode = False
            return False
        self.queue.clear()
        self.queue.extend(script)
        self._script_mode = True
        self._script_stall = 0
        self._phase = "script"
        self.active = True
        self.log.log("ls20", f"script L{self.levels_seen + 1}: {len(script)} acts")
        return True

    def reset_level(self, grid: np.ndarray):
        g = _as_grid(grid)
        if g is None:
            return
        self.queue.clear()
        self.stall = 0
        self.wait_steps = 0
        self.goal_idx = 0
        self.pad_entries = 0
        self.shape_toggles = 0
        self.color_toggles = 0
        self.mod_task_idx = 0
        self.mod_tasks = []
        self._goal_wait_mode = False
        self._platform_phase = 0
        self._goal_sig = None
        self._goal_entry_fails = 0
        self._layout_wait = 0
        self._prev_player_xy = None
        self._last_grid = g
        self._script_mode = False
        self._script_stall = 0

        self.player_xy = _find_player(g)
        self.goals = _find_goal_markers(g, self.player_xy)
        shape = _find_shape_pad(g, self.player_xy)
        rot = _find_rot_pad(g, self.player_xy)
        color = _find_color_pad(g, self.player_xy)
        if shape:
            self.mod_tasks.append(("shape", shape, 1))
        if rot:
            self.mod_tasks.append(("rot", rot, self._estimate_rot_entries(g)))
        if color:
            self.mod_tasks.append(("color", color, self._estimate_color_entries(g)))
        self._sync_current_mod()
        self.active = bool(self.player_xy and self.goals)

        if self._try_load_script():
            return

        if self.active:
            self._phase = "pad" if self.mod_tasks else "goal"
            self._queue_next(g)

    def _sync_current_mod(self):
        if self.mod_task_idx < len(self.mod_tasks):
            kind, pad, count = self.mod_tasks[self.mod_task_idx]
            self.mod_kind = kind
            self.mod_pad = pad
            self.target_pad_entries = count
        else:
            self.mod_kind = "none"
            self.mod_pad = None

    def _advance_mod_task(self):
        self.mod_task_idx += 1
        self.pad_entries = 0
        self.shape_toggles = 0
        self.color_toggles = 0
        self._sync_current_mod()

    def _estimate_rot_entries(self, grid: np.ndarray) -> int:
        # L2: arrival often lands at 180° after pickup route → one toggle to 270°.
        if self.levels_seen == 1:
            return 1
        if self.levels_seen == 2:
            return 2
        if self.levels_seen >= 5:
            return 2
        goal = self._current_goal()
        pad = None
        for t in self.mod_tasks:
            if t[0] == "rot":
                pad = t[1]
                break
        if pad is None:
            pad = _find_rot_pad(grid, self.player_xy)
        if not goal or not pad or not self.player_xy:
            return 1
        gy = goal[1]
        if gy <= 15:
            return 1
        if gy >= 35 and pad[0] >= 45:
            return 2
        if gy >= 30:
            return 2
        return 1

    def _estimate_color_entries(self, grid: np.ndarray) -> int:
        return 1

    def _current_mod_target(self) -> int:
        if self.mod_task_idx >= len(self.mod_tasks):
            return 0
        return self.mod_tasks[self.mod_task_idx][2]

    def _estimate_pad_entries(self, grid: np.ndarray) -> int:
        return self._current_mod_target()

    def _current_goal(self) -> Optional[tuple[int, int]]:
        if not self.goals:
            return None
        return self.goals[min(self.goal_idx, len(self.goals) - 1)]

    def _modifier_satisfied(self, grid: np.ndarray) -> bool:
        if self.mod_task_idx >= len(self.mod_tasks):
            return True
        kind, pad, target = self.mod_tasks[self.mod_task_idx]
        toggles = self.pad_entries if kind == "rot" else self.shape_toggles if kind == "shape" else self.color_toggles
        if toggles < target:
            return False
        if not self.player_xy:
            return False
        if kind == "rot" and _block_on_rot_pad(grid, self.player_xy[0], self.player_xy[1]):
            return False
        if self.player_xy == pad:
            return False
        return True

    def _avoid_pads(self) -> set[tuple[int, int]]:
        avoid: set[tuple[int, int]] = set(_l2_trap_cells(self.levels_seen))
        if self.mod_task_idx < len(self.mod_tasks):
            return avoid
        grid = self._last_grid
        for kind, pad, _ in self.mod_tasks:
            if kind == "rot" and grid is not None:
                avoid.update(_rot_pad_cells(grid))
            else:
                avoid.add(pad)
        return avoid

    def _nav_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[str]:
        avoid = self._avoid_pads()
        return _path_to(grid, start, goal, avoid if avoid else None)

    def _use_push_nav(self, grid: np.ndarray) -> bool:
        if not self.player_xy:
            return False
        goal = self._current_goal()
        avoid = self._avoid_pads()
        if goal:
            if avoid:
                if not _bfs_avoid(grid, self.player_xy, goal, avoid) and _bfs_push_avoid(
                    grid, self.player_xy, goal, avoid
                ):
                    return True
            elif not _bfs(grid, self.player_xy, goal) and _bfs_push(grid, self.player_xy, goal):
                return True
        pad = self.mod_pad
        if pad and self.mod_task_idx < len(self.mod_tasks):
            if not _bfs(grid, self.player_xy, pad) and _bfs_push(grid, self.player_xy, pad):
                return True
        return False

    def _queue_modifier(self, grid: np.ndarray):
        while self.mod_task_idx < len(self.mod_tasks) and self._modifier_satisfied(grid):
            self._advance_mod_task()
        if self.mod_task_idx >= len(self.mod_tasks):
            return False

        self._phase = "pad"
        kind, pad, target = self.mod_tasks[self.mod_task_idx]
        use_push = self._use_push_nav(grid)
        toggles = self.pad_entries if kind == "rot" else self.shape_toggles if kind == "shape" else self.color_toggles

        if kind == "color":
            if self.player_xy == pad:
                if toggles < target:
                    pair = _pad_exit_enter(grid, self.player_xy, pad, use_push)
                    self.queue.append(pair[0] if pair else "ACTION1")
                else:
                    exit_act = _pad_exit_only(grid, self.player_xy, pad, use_push)
                    if exit_act:
                        self.queue.append(exit_act)
                    else:
                        self._advance_mod_task()
                        return True
            else:
                path = _path_to(grid, self.player_xy, pad)
                if path:
                    self.queue.append(path[0])
                else:
                    self._queue_greedy(grid, pad, use_push)
            return False

        if self.player_xy == pad:
            if toggles < target:
                pair = _pad_exit_enter(grid, self.player_xy, pad, use_push)
                if pair:
                    self.queue.append(pair[0])
            elif kind == "rot":
                exit_act = _pad_exit_only(grid, self.player_xy, pad, use_push, avoid_rot=True)
                if exit_act:
                    self.queue.append(exit_act)
                else:
                    self._advance_mod_task()
                    return True
            else:
                exit_act = _pad_exit_only(grid, self.player_xy, pad, use_push)
                if exit_act:
                    self.queue.append(exit_act)
                else:
                    self._advance_mod_task()
                    return True
        elif _on_pad(grid, self.player_xy, pad):
            if kind == "rot":
                self.pad_entries = max(self.pad_entries, 1)
            elif kind == "shape":
                self.shape_toggles = max(self.shape_toggles, 1)
            pair = _pad_exit_enter(grid, self.player_xy, pad, use_push)
            if pair:
                self.queue.append(pair[0])
        else:
            path = _path_to(grid, self.player_xy, pad)
            if path:
                self.queue.append(path[0])
            else:
                self._queue_greedy(grid, pad, use_push)
        return False

    def _queue_next(self, grid: np.ndarray):
        self.queue.clear()
        if not self.player_xy:
            return
        goal = self._current_goal()
        if not goal:
            return

        if self.mod_task_idx < len(self.mod_tasks):
            advanced = self._queue_modifier(grid)
            if advanced:
                self._queue_next(grid)
            if self.queue:
                return

        if not self._modifier_satisfied(grid):
            return

        self._phase = "goal"
        use_push = self._use_push_nav(grid)
        if self.player_xy and goal:
            self._goal_sig = _goal_cell_sig(grid, goal)
        approach = _goal_approach_cell(goal, self.levels_seen)
        if self.player_xy == goal:
            return
        if self.player_xy == approach and approach != goal:
            self._goal_wait_mode = True

        if self._goal_wait_mode and self.player_xy and goal:
            if self.player_xy == approach or _adjacent_to_goal(self.player_xy, goal):
                self._platform_phase = (self._platform_phase + 1) % PLATFORM_CYCLE
                sig = _goal_cell_sig(grid, goal)
                sig_changed = self._goal_sig is not None and sig != self._goal_sig
                self._goal_sig = sig
                entry = _entry_action_toward_goal(self.player_xy, goal)
                if entry and (self._platform_phase == 0 or sig_changed):
                    self.queue.append(entry)
                elif self.player_xy == approach and approach[0] == 14:
                    self.queue.append("ACTION1")
                else:
                    lateral = _lateral_wait_action(grid, self.player_xy, use_push)
                    if lateral:
                        self.queue.append(lateral)
                    elif entry:
                        self.queue.append(entry)
                return

        nav_target = approach if self.levels_seen == 1 and goal == (14, 40) else goal
        path = self._nav_path(grid, self.player_xy, nav_target)
        if path:
            self.queue.append(path[0])
        else:
            avoid = self._avoid_pads()
            self._queue_greedy(grid, nav_target, use_push, avoid)

    def _queue_greedy(
        self,
        grid: np.ndarray,
        target: tuple[int, int],
        use_push: bool = False,
        avoid: Optional[set[tuple[int, int]]] = None,
    ):
        if not self.player_xy:
            return
        walk = _walkable_push if use_push else _walkable
        avoid = avoid or set()
        px, py = self.player_xy
        tx, ty = target
        order: list[str] = []
        if abs(tx - px) >= STEP:
            order.append("ACTION4" if tx > px else "ACTION3")
        if abs(ty - py) >= STEP:
            order.append("ACTION1" if ty < py else "ACTION2")
        for act in order:
            dx, dy = DIRS[act]
            nxt = (px + dx, py + dy)
            if nxt in avoid or not walk(grid, nxt[0], nxt[1]):
                continue
            self.queue.append(act)
            return
        for act in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            dx, dy = DIRS[act]
            nxt = (px + dx, py + dy)
            if nxt in avoid or not walk(grid, nxt[0], nxt[1]):
                continue
            self.queue.append(act)
            return

    def observe(
        self,
        prev_grid: np.ndarray | None,
        curr_grid: np.ndarray | None,
        action: str | None,
        levels: int,
    ):
        curr = _as_grid(curr_grid)
        if curr is None:
            return
        self._last_grid = curr

        if levels > self.levels_seen:
            self.levels_seen = levels
            self.queue.clear()
            self.active = False
            self._await_level_layout = False
            # Offline scripts assume immediate post-clear state; skip long layout wait.
            if script_for_level(self.levels_seen, "ls20"):
                self._layout_wait = 0
                if looks_like_ls20(curr):
                    self.reset_level(curr)
            else:
                self._layout_wait = 4
            return

        if self._layout_wait > 0:
            self._layout_wait -= 1
            if self._layout_wait == 0 and looks_like_ls20(curr):
                self.reset_level(curr)
            return

        if self._await_level_layout:
            self._await_level_layout = False
            self.reset_level(curr)
            return

        if not self.active:
            if looks_like_ls20(curr):
                self.reset_level(curr)
            return

        prev_player = self.player_xy
        detected = _find_player(curr)
        if detected:
            self.player_xy = detected

        if self.player_xy:
            new_goals = _find_goal_markers(curr, self.player_xy)
            if new_goals != self.goals:
                old_len = len(self.goals)
                self.goals = new_goals
                if len(new_goals) < old_len:
                    self.goal_idx = 0
                    self._goal_wait_mode = False
                    self._goal_entry_fails = 0
                elif self.goal_idx >= len(new_goals):
                    self.goal_idx = max(0, len(new_goals) - 1)

        goal = self._current_goal()
        if goal and self.player_xy and _adjacent_to_goal(self.player_xy, goal):
            self._goal_sig = _goal_cell_sig(curr, goal)

        moved = (
            self.player_xy
            and prev_player
            and self.player_xy != prev_player
        )

        if action and self.mod_task_idx < len(self.mod_tasks):
            kind, pad, _ = self.mod_tasks[self.mod_task_idx]
            if kind == "rot":
                if prev_player and self.player_xy == pad and prev_player != pad:
                    self.pad_entries += 1
                elif (
                    prev_player == pad
                    and self.player_xy
                    and self.player_xy != pad
                ):
                    pass
                elif self.player_xy == pad and prev_player == pad:
                    if action in DIRS:
                        dx, dy = DIRS[action]
                        expected = (prev_player[0] + dx, prev_player[1] + dy)
                        if self.player_xy != expected:
                            self.pad_entries += 1
            elif kind == "shape":
                if prev_player and self.player_xy == pad and prev_player != pad:
                    self.shape_toggles += 1
                elif self.player_xy == pad and prev_player == pad and action in DIRS:
                    dx, dy = DIRS[action]
                    expected = (prev_player[0] + dx, prev_player[1] + dy)
                    if self.player_xy != expected:
                        self.shape_toggles += 1
            elif kind == "color" and self.player_xy == pad and action in DIRS:
                if prev_player != pad or action in DIRS:
                    self.color_toggles += 1

        if action and not moved and self.player_xy and prev_player == self.player_xy:
            self.stall += 1
            self.wait_steps += 1
            if self._script_mode:
                self._script_stall += 1
                if self._script_stall >= 8:
                    # Script stuck — fall back to online BFS planner.
                    self._script_mode = False
                    self.queue.clear()
                    self._script_stall = 0
                    self._phase = "pad" if self.mod_tasks else "goal"
                    self._queue_next(curr)
            goal = self._current_goal()
            if goal and _adjacent_to_goal(self.player_xy, goal):
                entry = _entry_action_toward_goal(self.player_xy, goal)
                if entry and action == entry:
                    self._goal_entry_fails += 1
                    self._goal_wait_mode = True
                    if self._goal_entry_fails >= PLATFORM_CYCLE:
                        self._goal_wait_mode = False
                        self._goal_entry_fails = 0
                        if self.mod_task_idx < len(self.mod_tasks):
                            kind, _, target = self.mod_tasks[self.mod_task_idx]
                            if kind == "rot" and self.target_pad_entries < self.max_pad_entries:
                                self.target_pad_entries = min(self.max_pad_entries, target + 1)
                                self.mod_tasks[self.mod_task_idx] = (
                                    kind,
                                    self.mod_tasks[self.mod_task_idx][1],
                                    self.target_pad_entries,
                                )
                                self.pad_entries = 0
                                self._phase = "pad"
                            elif kind == "shape":
                                self.mod_tasks[self.mod_task_idx] = (
                                    kind,
                                    self.mod_tasks[self.mod_task_idx][1],
                                    target + 1,
                                )
                                self.shape_toggles = 0
                                self._phase = "pad"
                            elif kind == "color" and target < self.max_color_toggles:
                                self.mod_tasks[self.mod_task_idx] = (
                                    kind,
                                    self.mod_tasks[self.mod_task_idx][1],
                                    target + 1,
                                )
                                self.color_toggles = 0
                                self._phase = "pad"
                        self._platform_phase = 0
        elif action:
            self.stall = 0
            self._script_stall = 0
            if moved:
                self.wait_steps = 0

        if self._script_mode:
            # While script has remaining acts, do not replan over it.
            if self.queue:
                self._prev_player_xy = self.player_xy
                return
            self._script_mode = False
            # Script exhausted without level-up — resume heuristic planner.
            if not self.queue:
                self._phase = "pad" if self.mod_task_idx < len(self.mod_tasks) else "goal"
                self._queue_next(curr)
            self._prev_player_xy = self.player_xy
            return

        goal = self._current_goal()
        if goal and self.player_xy == goal:
            self._goal_wait_mode = False
            self._goal_entry_fails = 0
        if not self.queue and action and goal and self.player_xy == goal:
            if not self._modifier_satisfied(curr):
                kind, pad, target = self.mod_tasks[self.mod_task_idx] if self.mod_task_idx < len(self.mod_tasks) else ("none", (0, 0), 0)
                if kind == "rot" and target < self.max_pad_entries:
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.target_pad_entries = target + 1
                elif kind == "shape":
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.target_pad_entries = target + 1
                elif kind == "color" and target < self.max_color_toggles:
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.target_pad_entries = target + 1
                self.pad_entries = 0
                self.shape_toggles = 0
                self.color_toggles = 0
            elif self.mod_task_idx < len(self.mod_tasks):
                kind, pad, target = self.mod_tasks[self.mod_task_idx]
                if kind == "rot" and target < self.max_pad_entries:
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.target_pad_entries = target + 1
                    self.pad_entries = 0
            self.stall = 0
            self._phase = "pad"
            self._queue_next(curr)
        elif not self.queue and action:
            self._queue_next(curr)

        self._prev_player_xy = self.player_xy

    def plan(self, valid_actions: list[str]) -> Optional[str]:
        valid = {canonicalize(a) for a in valid_actions}
        if self._await_level_layout or self._layout_wait > 0:
            return None
        if not self.active:
            return None

        while self.queue:
            act = canonicalize(self.queue[0])
            if act in valid:
                self.queue.popleft()
                return act
            self.queue.popleft()

        g = self._last_grid
        if g is None:
            return None

        if self.wait_steps >= 2 and self.stall >= 2:
            self.wait_steps = 0
            self.stall = 0
            for act in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                if act in valid:
                    return act

        if self.stall >= 3:
            if self.mod_task_idx < len(self.mod_tasks):
                kind, pad, target = self.mod_tasks[self.mod_task_idx]
                if kind == "rot" and target < self.max_pad_entries:
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.target_pad_entries = target + 1
                    self.pad_entries = 0
                    self._phase = "pad"
                elif kind == "shape":
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.shape_toggles = 0
                    self._phase = "pad"
                elif kind == "color" and target < self.max_color_toggles:
                    self.mod_tasks[self.mod_task_idx] = (kind, pad, target + 1)
                    self.color_toggles = 0
                    self._phase = "pad"
            self.stall = 0
            self._queue_next(g)
            return self.plan(valid_actions)

        goal = self._current_goal()
        if goal and self.player_xy and _adjacent_to_goal(self.player_xy, goal):
            entry = _entry_action_toward_goal(self.player_xy, goal)
            if entry and entry in valid:
                return entry

        if goal and self.player_xy:
            use_push = self._use_push_nav(g)
            self._queue_greedy(g, goal, use_push)
            if self.queue:
                act = canonicalize(self.queue[0])
                if act in valid:
                    self.queue.popleft()
                    return act
        return None
