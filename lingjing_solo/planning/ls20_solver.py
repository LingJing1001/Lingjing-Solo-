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

# 实测确认「目标格本身在移动」的关卡，元素为 (levels_seen, goal_cell)。
# 只有登记在此的目标才需要"先到邻格等相位再进入"；其余关卡目标静止，直接踩上即可。
# 2026-09-18 探针 arc_adaptor/agents/strategies/probe_ls20_l2_platform.py 实测
# L2 目标格 (14,40) 在 22 个动作帧内坐标恒定不变，故不登记。
#
# 同批探针（probe_ls20_l2_frame.py / probe_ls20_l2_gate.py）实测到的两条引擎规则：
#   A. 换关当帧返回的画面仍是上一关 —— L1 在 (14,40) 有墙、L2 在同一格放目标，
#      所以 L2 第一帧该格被画成色 4 的墙，基于像素的 BFS 会判「目标不可达」。
#   B. ls20.py:1882-1887 —— 三元组不匹配时踩向目标格会被当作墙拒绝，
#      且该次动作**不扣步数**（实测 剩步 40→40）。所以静止目标「进不去」
#      等价于「修饰器还没调对」，应立刻回补台加次数，而不是原地等相位浪费步数。
MOVING_GOAL_CELLS: set[tuple[int, tuple[int, int]]] = set()


def _goal_is_moving(levels_seen: int, goal: Optional[tuple[int, int]]) -> bool:
    return goal is not None and (levels_seen, goal) in MOVING_GOAL_CELLS


# ---------------------------------------------------------------------------
# 画面 HUD 读数：步数 / 生命 / 步数补给，全部从帧里量，不按关卡硬编码。
#   ls20.py:1543-1546  步数条 = 第 61~62 行、x 从 13 开始、每"步"一列；
#                      剩余步用色 11 画，已花掉的用背景色 3 画 ⇒ 色 11 连续贴在条带右端。
#   ls20.py:1547-1551  生命 = 第 61 行 x∈{56,59,62}，活着时色 8。
#   ls20.py:341-347    npxgalaybz(步数补给) 是全游戏唯一使用色 11 的精灵，且 collidable=False，
#                      所以玩法区(rows 0..60)里出现的色 11 只可能是补给。
# ---------------------------------------------------------------------------
BAR_ROW = 61
BAR_X0, BAR_X1 = 13, 56          # [BAR_X0, BAR_X1)：步数条区，右边界让给生命指示
LIFE_XS = (56, 59, 62)
C_STEP_ON, C_STEP_OFF, C_LIFE = 11, 3, 8
C_REFILL = 11
PLAYFIELD_ROWS = 61              # 第 61 行起是 HUD，画玩法区时排除


def read_step_bar(grid: np.ndarray) -> Optional[tuple[int, int]]:
    """返回 (剩余步, 本关步数上限)；画面里没有步数条时返回 None。"""
    if grid.shape[0] <= BAR_ROW:
        return None
    row = np.asarray(grid[BAR_ROW, BAR_X0:BAR_X1])
    idx = np.flatnonzero(row == C_STEP_ON)
    if idx.size == 0:
        return None
    left, right = int(idx.min()), int(idx.max())
    return right - left + 1, right + 1


def read_lives(grid: np.ndarray) -> Optional[int]:
    if grid.shape[0] <= BAR_ROW:
        return None
    return sum(1 for x in LIFE_XS if int(grid[BAR_ROW, x]) == C_LIFE)


def _clusters(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """4 邻接聚类，返回每簇的左上角（= 精灵原点）。"""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for seed in points:
        if seed in seen:
            continue
        seen.add(seed)
        stack = [seed]
        comp: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (x + dx, y + dy)
                if q in points and q not in seen:
                    seen.add(q)
                    stack.append(q)
        out.append((min(c[0] for c in comp), min(c[1] for c in comp)))
    return out


def find_refill_cells(
    grid: np.ndarray, residue: tuple[int, int]
) -> list[tuple[int, int]]:
    """把画面上的 npxgalaybz 换算成"踩到它的那个玩家格"。

    引擎的拾取判定是"精灵原点落在目的地的 5x5 方格内"(ls20.py:1867-1876)，
    所以原点 (ox,oy) 对应的唯一玩家格是 X = ox - ((ox-rx) % 5)，Y 同理。
    """
    rx, ry = residue
    field = np.asarray(grid[:PLAYFIELD_ROWS, :]) == C_REFILL
    ys, xs = np.nonzero(field)
    origins = _clusters({(int(a), int(b)) for a, b in zip(xs, ys)})
    cells: list[tuple[int, int]] = []
    for ox, oy in origins:
        cx, cy = ox - ((ox - rx) % STEP), oy - ((oy - ry) % STEP)
        if 0 <= cx and 0 <= cy and _walkable(grid, cx, cy) and (cx, cy) not in cells:
            cells.append((cx, cy))
    return cells


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
    """仅对登记过的移动平台目标返回邻格（先待命再进入）；静止目标直接以目标格为终点。"""
    if (levels_seen, goal) not in MOVING_GOAL_CELLS:
        return goal
    gx, gy = goal
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

    # 动作 ID -> 字符串动作名（与 verified_solutions / ScriptBank 风格一致）
    _ID_TO_ACTION = {
        1: "ACTION1",  # UP
        2: "ACTION2",  # DOWN
        3: "ACTION3",  # LEFT
        4: "ACTION4",  # RIGHT
        5: "ACTION5",  # SWITCH
    }

    # 目标格拒绝进入、且修饰器次数已到上限后，"每台再踩一次" 的重试轮数上限。
    MAX_RETRY_LAPS = 2

    # 走完眼前这一段后还想保留的富余动作数；低于它就绕路去吃步数补给。
    REFILL_MARGIN = 3

    @classmethod
    def get_verified_route(cls, level: str) -> list[str] | None:
        """返回预存的 LS20 关卡路线（字符串动作形式）。

        Args:
            level: "L1" | "L2" | "L3" | "L4"

        Returns:
            动作字符串列表，如 ["ACTION3", "ACTION3", ...]，
            或 None（未知关卡）。
        """
        from .data.verified_solutions import LS20_SOLUTIONS

        key = level.upper()
        if key not in LS20_SOLUTIONS:
            return None
        ids = LS20_SOLUTIONS[key]
        return [cls._ID_TO_ACTION[i] for i in ids]

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
        self._retry_lap = 0
        self._steps_left: int | None = None
        self._step_cap = 0
        self._step_decrement = 0
        self.refill_target: tuple[int, int] | None = None
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
        self._retry_lap = 0
        self._layout_wait = 0
        bar = read_step_bar(g)
        self._steps_left = bar[0] if bar else None
        self._step_cap = bar[1] if bar else 0
        self._step_decrement = 0
        self.refill_target = None
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

    def _escalate_modifier(self):
        """踩目标被拒后，怀疑当前修饰器次数估错：先加次数，加无可加则换下一个修饰器。

        旋转台每次 rot_idx += 1 (mod 4)，所以 (目标索引-当前索引)%4 ∈ 0..3，
        次数永远不可能 >3；到 3 仍进不去说明问题不在旋转台。
        """
        self._platform_phase = 0
        if self.mod_task_idx >= len(self.mod_tasks):
            # 所有修饰器都加到上限仍进不去 → 绕回每台再踩一次，最多 MAX_RETRY_LAPS 轮，
            # 之后交回上层（否则会像 L2 实测那样原地反复踩同一格 500 步）。
            self._retry_lap += 1
            if not self.mod_tasks or self._retry_lap > self.MAX_RETRY_LAPS:
                self.active = False
                self.queue.clear()
                return
            self.mod_tasks = [(k, p, 1) for k, p, _ in self.mod_tasks]
            self.mod_task_idx = 0
            self.pad_entries = 0
            self.shape_toggles = 0
            self.color_toggles = 0
            self._sync_current_mod()
            self._phase = "pad"
            return
        kind, _, target = self.mod_tasks[self.mod_task_idx]
        if kind == "rot":
            if target >= 3:
                self._advance_mod_task()
                self._phase = "pad" if self.mod_tasks else "goal"
                return
            if self.target_pad_entries < self.max_pad_entries:
                self.target_pad_entries = min(3, target + 1)
                self.mod_tasks[self.mod_task_idx] = (
                    kind,
                    self.mod_tasks[self.mod_task_idx][1],
                    self.target_pad_entries,
                )
                self.pad_entries = 0
                self._phase = "pad"
            return
        if kind == "shape":
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

    def _estimate_rot_entries(self, grid: np.ndarray) -> int:
        # 旋转台每踩一次 rot_idx += 1 (mod 4)，所需次数 = (目标索引 - 当前索引) % 4。
        # L2: 起点 rot_idx=0，目标要求索引 3 → (3-0)%4 = 3 次。
        # 实测证据：arc_adaptor/agents/strategies/probe_ls20_l2_rot.py
        # （第 1 次踩台 0→1，第 2 次 1→2）。原值 1 系猜测，永远凑不齐三元组。
        if self.levels_seen == 1:
            return 3
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

    def _next_plan_waypoint(self, grid: np.ndarray) -> Optional[tuple[int, int]]:
        """当前这一段跑完之后要去的下一站：下一个修饰台，或最终目标格。"""
        for i in range(self.mod_task_idx + 1, len(self.mod_tasks)):
            return self.mod_tasks[i][1]
        return self._current_goal()

    def _leg_moves(self, grid: np.ndarray, src: tuple[int, int],
                   dst: tuple[int, int]) -> Optional[int]:
        if src == dst:
            return 0
        d = len(_bfs(grid, src, dst))
        return d if d else None

    def _leg_target(self, grid: np.ndarray) -> Optional[tuple[int, int]]:
        """"眼前这一段"的终点：还没上台就是台，在台上就是下一站（下一个台或目标格）。"""
        if self.mod_task_idx < len(self.mod_tasks):
            _, pad, _ = self.mod_tasks[self.mod_task_idx]
            if self.player_xy != pad:
                return pad
            return self._next_plan_waypoint(grid)
        return self._current_goal()

    def _next_leg_cost(self, grid: np.ndarray) -> Optional[int]:
        """走完"眼前这一段"要几个动作：赶到航点 + 台上剩余的踩次数（每次=离台+回台）。"""
        if not self.player_xy:
            return None
        nxt = self._leg_target(grid)
        if nxt is None:
            return None
        d = self._leg_moves(grid, self.player_xy, nxt)
        if d is None:
            return None
        if self.mod_task_idx < len(self.mod_tasks) and self.player_xy == self.mod_tasks[self.mod_task_idx][1]:
            kind, _, target = self.mod_tasks[self.mod_task_idx]
            done = (self.pad_entries if kind == "rot"
                    else self.shape_toggles if kind == "shape" else self.color_toggles)
            d += 2 * max(0, target - done)
        return d

    def _moves_left(self, grid: np.ndarray) -> Optional[int]:
        if self._steps_left is None or self._step_decrement <= 0:
            return None
        return self._steps_left // self._step_decrement

    def _refill_detour(self, grid: np.ndarray) -> Optional[tuple[int, int]]:
        """眼前这一段走不完时，挑一个"多走路最少"的 npxgalaybz 先吃掉。

        实测：吃补给把步数条复位（ls20.py:1888-1892），L2 上限 42 / 每动作扣 2 ⇒ 每次白赚
        21 个动作；而 L2 的 3 次旋转 + 往返目标最少要 46 个动作（probe_ls20_l2_budget.py），
        所以不吃补给根本不可能过。判据只看"这一段"，不看全程——按全程估会过早把远处的
        补给抓在手里，反而错过真正顺路的那一个。
        """
        if not self.player_xy:
            return None
        budget = self._moves_left(grid)
        leg = self._next_leg_cost(grid)
        if budget is None or leg is None or budget >= leg + self.REFILL_MARGIN:
            return None
        nxt = self._leg_target(grid)
        direct = self._leg_moves(grid, self.player_xy, nxt) if nxt else 0
        rx, ry = self.player_xy
        affordable: list[tuple[int, int, tuple[int, int]]] = []
        any_reach: list[tuple[int, int, tuple[int, int]]] = []
        for cell in find_refill_cells(grid, (rx % STEP, ry % STEP)):
            if cell == self.player_xy:
                continue
            d1 = self._leg_moves(grid, self.player_xy, cell)
            if d1 is None:
                continue
            d2 = 0
            if nxt and nxt != cell:
                d2 = self._leg_moves(grid, cell, nxt)
                if d2 is None:
                    continue
            extra = d1 + d2 - (direct or 0)
            any_reach.append((d1, extra, cell))
            if d1 <= budget:
                affordable.append((d1, extra, cell))
        if affordable:
            return min(affordable, key=lambda t: (t[1], t[0]))[2]
        # 一个都够不到：这一段反正走不完，先扑最近的补给。
        return min(any_reach, key=lambda t: (t[0], t[1]))[2] if any_reach else None

    def _queue_next(self, grid: np.ndarray):
        self.queue.clear()
        if not self.player_xy:
            return
        goal = self._current_goal()
        if not goal:
            return

        # ---- 预算不够走完这一段 → 先绕路吃最近的 npxgalaybz ----
        #      补给被吃掉后色 11 从画面上消失，这条规则自然失效。
        detour = self._refill_detour(grid)
        if detour:
            self._phase = "refill"
            self.refill_target = detour
            path = _bfs(grid, self.player_xy, detour)
            if path:
                self.queue.append(path[0])
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

        if (
            self._goal_wait_mode
            and _goal_is_moving(self.levels_seen, goal)
            and self.player_xy
            and goal
        ):
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

        nav_target = goal
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
                # 换关当帧渲染的还是上一关（probe_ls20_l2_frame.py 实测：L1 在 (14,40) 有墙、
                # L2 同一格是目标，所以首帧把目标格画成了墙）。再走 1 帧渲染就追上，
                # 等待期间驱动层会补一步 ACTION1 —— 每多等一步白扣 StepsDecrement，
                # L2 全程才 21 个动作，原来的 4 帧等于烧掉 8 步，故按实测降到 1。
                self._layout_wait = 1
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

        # ---- 预算标定：剩余步从画面读，每动作扣多少靠"确实动了且条子掉了"实测 ----
        #      撞墙不扣、被拒的目标格不扣、吃补给是往上跳，所以只认小的正向差值。
        bar = read_step_bar(curr)
        if bar:
            prev_steps = self._steps_left
            self._steps_left, self._step_cap = bar
            moved_now = bool(
                prev_player and self.player_xy and self.player_xy != prev_player
            )
            span = max(1, self._step_cap // 4)
            if moved_now and prev_steps is not None:
                delta = prev_steps - self._steps_left
                if 0 < delta <= span:
                    self._step_decrement = delta

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
                    if _goal_is_moving(self.levels_seen, goal):
                        self._goal_wait_mode = True
                        # 平台在动：先等相位，连续 PLATFORM_CYCLE 次仍进不去才怀疑次数估算。
                        if self._goal_entry_fails >= PLATFORM_CYCLE:
                            self._goal_wait_mode = False
                            self._goal_entry_fails = 0
                            self._escalate_modifier()
                    else:
                        # 目标静止 + 该次尝试不扣步数 ⇒ 进不去就是三元组不对，立刻补一次修饰器。
                        self._goal_entry_fails = 0
                        self._escalate_modifier()
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
