"""dc22 的 WorldView profile（团队规范《最小可运行闭环》§3.2.1 / §3.2.2）。

只消费线上真实可得的 FrameData 字段（`frame` / `state` / `levels_completed` /
`win_levels` / `available_actions`），不读引擎私有对象，不用快照回滚。

事实来源：`integrations/ARC-AGI-3-Kaggle-Starter/environment_files/dc22/fdcac232/dc22.py`
（本轮逐条核对过，非转抄）：

- `available_actions=[1, 2, 3, 4, 6]`；ACTION5 在 `step` 里是 `pass`，引擎不提供。
- 唯一的单调资源是"剩余步数条"：每发一个动作 `-1`，每次死亡 `-20`，只在换关时
  重置。它被画在**最后一行 row 63**，64 个格子按 `已消耗/总额` 的比例分两种颜色，
  尾部（`i/64 >= ratio`）是"剩余"色，头部是"已消耗"色。
- 剩余步数归零、或坠落检查拿不到落点 → `self.lose()` → `GameState.GAME_OVER`。
- 玩家 sprite 的 x/y 与目标 sprite 重合 → `self.next_level()`（最后一关走 `win()`），
  是官方机制，不是视觉推断。

§3.2.1 硬约束在本 profile 的落点：
1. 剩余步数是单调量 → 只进 `counters()`，`key()` 里把 row 63 整行遮掉。
2. `progress()` / `done()` 用 `levels_completed` / `state` 官方字段，不做视觉伪造。
3. `proposals()` 只返回 abstract action name；点击坐标放 `payload_hint`，
   由 boundary adapter 负责转 `GameAction.ACTION6`。
"""
from __future__ import annotations

import hashlib
from typing import Any, Hashable

import numpy as np

from ..core.utils import extract_grid
from .world_view import ActionSpec, ConfigSpace, WorldView

#: 引擎把剩余步数条画死的行号（camera 帧恒 64 行，故等价于"最后一行"）。
BUDGET_BAR_ROW = 63

#: 键盘四向；ACTION5 不在 available_actions 里，不提议。
KEYBOARD_ACTIONS: tuple[str, ...] = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")

#: 点击候选的上限——连通域再多也不值得让 P4 节点预算白烧。
MAX_CLICK_CANDIDATES = 8


def _bar_row(grid: np.ndarray) -> np.ndarray | None:
    """取步数条那一行；帧高不足时返回 None（调用方按"读不到"处理，不猜）。"""
    if grid is None or grid.ndim != 2 or grid.shape[0] <= BUDGET_BAR_ROW:
        return None
    return np.asarray(grid[BUDGET_BAR_ROW])


def _playfield(grid: np.ndarray) -> np.ndarray:
    """遮掉步数条行后的主网格副本（key 用它，避免单调量污染状态键）。"""
    out = np.array(grid, copy=True)
    if out.ndim == 2 and out.shape[0] > BUDGET_BAR_ROW:
        out[BUDGET_BAR_ROW, :] = 0
    return out


def _components(grid: np.ndarray, *, cap: int = 200) -> list[tuple[int, int, int, int]]:
    """4-连通同色域，返回 (面积, 首见行, 首见列, 颜色)，按 (行, 列, 颜色) 稳定排序。

    排除背景（出现最多的颜色）与步数条行。面积上限 `cap` 防止整片地板被当成一个"按钮"。
    """
    if grid is None or grid.ndim != 2:
        return []
    h, w = grid.shape
    flat = grid.reshape(-1)
    counts: dict[int, int] = {}
    for value in flat.tolist():
        counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return []
    background = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    seen = np.zeros((h, w), dtype=bool)
    if h > BUDGET_BAR_ROW:
        # 步数条行在 _playfield 里已被置 0；若 0 恰好吃掉背景色，整行会变成
        # 一个 size<=cap 的"伪按钮"连通域，所以这里无条件排除它。
        seen[BUDGET_BAR_ROW:, :] = True
    out: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            color = int(grid[r0, c0])
            if color == background:
                seen[r0, c0] = True
                continue
            # 迭代式 BFS，避免深域递归爆栈
            stack = [(r0, c0)]
            seen[r0, c0] = True
            size = 1
            cells = [(r0, c0)]
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        size += 1
                        stack.append((nr, nc))
                        cells.append((nr, nc))
            if 1 <= size <= cap:
                top = min(cells)
                out.append((size, top[0], top[1], color))
    out.sort(key=lambda item: (item[1], item[2], item[3]))
    return out


class DC22WorldView(WorldView):
    """dc22 的线上 profile：四向 + 小连通域点击，计数器与状态键分离。"""

    game_id = "dc22"
    profile_id = "dc22-v1-shadow"

    # ── 网格 ────────────────────────────────────────────────

    def grids(self, frame: Any) -> list[Any]:
        """返回 [完整顶层网格, 遮掉步数条行的主网格]。

        第 0 层给 `counters()` 读步数条，第 1 层给 `key()`/`proposals()` 用，
        这样单调量不可能混进状态键（§3.2.1 硬约束 1）。
        """
        grid = extract_grid(frame)
        if grid is None:
            return []
        grid = np.asarray(grid)
        return [grid, _playfield(grid)]

    def key(self, grids: list[Any]) -> Hashable:
        if not grids or len(grids) < 2:
            return ""
        payload = np.ascontiguousarray(grids[1]).tobytes()
        return hashlib.md5(payload).hexdigest()[:16]

    def counters(self, grids: list[Any]) -> dict[str, int]:
        """剩余步数条的近似格数（单调不增，只在换关时回填）。

        近似的原因：引擎在**同一行**还画了一个光标标记，会扰动几格。
        所以这个计数器只用于 P2 支配比较，不进 key、不当终局证据。
        """
        if not grids:
            return {}
        row = _bar_row(grids[0])
        if row is None or row.size == 0:
            return {}
        # 尾部最后一格恒为"剩余"色（i/64 >= ratio 对 ratio<=1 总成立）
        remaining_color = int(row[-1])
        remaining = int((row == remaining_color).sum())
        distinct = {int(v) for v in row.tolist()}
        return {
            "step_budget_cells_approx": remaining,
            "bar_row_distinct_colors": len(distinct),
        }

    def proposals(self, grids: list[Any], key: Hashable) -> list[ActionSpec]:
        """四向优先，再补小连通域的点击候选（坐标只作 payload_hint）。"""
        if not grids or len(grids) < 2:
            return []
        specs: list[ActionSpec] = [ActionSpec(name=name) for name in KEYBOARD_ACTIONS]
        try:
            comps = _components(grids[1], cap=64)
        except Exception:  # noqa: BLE001 — 候选生成不许把决策层带崩
            comps = []
        for _size, row, col, color in comps[:MAX_CLICK_CANDIDATES]:
            specs.append(
                ActionSpec(
                    name="ACTION6",
                    payload_hint={"x": int(col), "y": int(row), "color": int(color)},
                )
            )
        return specs

    def config_space(self, grids: list[Any]) -> ConfigSpace | None:
        """dc22 无配置层：直接走动作层（文档 §3.2.2 的"通常无配置层"一类）。"""
        return None

    def score(self, grids: list[Any]) -> float:
        """帧上读不出 dc22 的得分 → 恒 0，不拿视觉代理冒充启发式。

        内核当前不调用 `score()`；这条常量存在的意义是"拒绝伪造"，
        等真有多目标得分口径时再改。
        """
        return 0.0

    # ── 官方字段派生 ────────────────────────────────────────

    def progress(self, frame: Any) -> int:
        """官方 `levels_completed`，不用视觉变化冒充进度。"""
        try:
            return int(getattr(frame, "levels_completed", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def done(self, frame: Any) -> bool:
        """官方 `state == WIN`；或 levels_completed 追平 win_levels。"""
        if str(_state_value(frame)).upper() == "WIN":
            return True
        win_levels = getattr(frame, "win_levels", None)
        try:
            win_levels = int(win_levels or 0)
        except (TypeError, ValueError):
            win_levels = 0
        return win_levels > 0 and self.progress(frame) >= win_levels

    def dead(self, frame: Any) -> bool:
        """官方 `state == GAME_OVER`（步数耗尽或坠落）。"""
        return str(_state_value(frame)).upper() == "GAME_OVER"


def _state_value(frame: Any) -> Any:
    state = getattr(frame, "state", None)
    if state is None:
        return ""
    return getattr(state, "value", None) or getattr(state, "name", None) or state


__all__ = ["DC22WorldView", "BUDGET_BAR_ROW", "KEYBOARD_ACTIONS"]
