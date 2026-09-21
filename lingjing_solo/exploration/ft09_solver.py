"""FT09 翻转块谜题求解器。

机制（逆向自 ft09.py 官方源码）:
  - ACTION6 点击 "Hkx"/"NTi" 3x3 块，点击使颜色沿 gqb 表循环前进一位。
  - 每关有若干 "gig" 引导块（3x3 像素矩阵）作约束源：对 guide 中心
    (gx,gy) 的 8 个邻方向 (dx,dy)∈{±4,0}²，若 pixels[dy/4+1][dx/4+1]==0
    则要求 (gx+dx,gy+dy) 处的块颜色 == guide 中心色，否则 !=。
  - 全部约束满足（cgj() 为 True）即过关。

两层求解策略:
  1. 模型法（快）: 从引擎读 blocks/gqb/guides，把约束转成逐块目标色，
     直接算每块所需点击次数（模 len(gqb)），一次生成动作序列。
  2. 重放 BFS（回退）: 对模型不适用的关卡（结构变化、约束含混），
     reset+重放做宽度优先搜索，完整状态键 = 全部块 (x,y,像素)。

坐标约定与官方一致：x=列, y=行；sprite 网格坐标 *2 = 显示坐标。
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple

Click = Tuple[int, int]


def sprite_click_coords(sprites: Sequence) -> List[Click]:
    """从精灵列表提取 (x, y) 可点击坐标（显示坐标 = 网格坐标 * 2）。"""
    return [(int(s.x) * 2, int(s.y) * 2) for s in sprites]


def full_state_key(sprites: Sequence, level_index: int) -> tuple:
    """完整状态键：关卡索引 + 每块 (x, y, 全像素)。

    历史教训：单像素键（如 pixels[1][1]）会让同色块全部坍缩，
    BFS 第一层就死。
    """
    return (
        int(level_index),
        tuple(
            (int(s.x), int(s.y), tuple(int(v) for v in s.pixels.flatten()))
            for s in sprites
        ),
    )


def guide_constraints(guides: Sequence, block_pos: set) -> List[tuple]:
    """把 guide 3x3 像素矩阵翻译成逐块颜色约束。

    返回 [(block_pos, need_equal, guide_center_color), ...]。
    need_equal=True 表示该块颜色必须等于 guide 中心色。
    """
    constraints = []
    for guide in guides:
        center_color = int(guide.pixels[1][1])
        for dy in (-4, 0, 4):
            for dx in (-4, 0, 4):
                if dx == 0 and dy == 0:
                    continue
                flag = int(guide.pixels[dy // 4 + 1][dx // 4 + 1]) == 0
                pos = (int(guide.x) + dx, int(guide.y) + dy)
                if pos in block_pos:
                    constraints.append((pos, flag, center_color))
    return constraints


def solve_by_model(
    blocks: dict,
    gqb: Sequence[int],
    guides: Sequence,
) -> Optional[List[Click]]:
    """模型法：约束 → 每块点击次数，返回网格坐标点击序列。

    blocks: {(x, y): 当前颜色}；gqb: 颜色循环表；guides: 引导块精灵。
    每块独立求解：找到一个循环步数使全部涉及该块的约束同时满足；
    任一块无解（约束互相矛盾）则整体失败，返回 None。
    点击序列按块的网格坐标排序；每块重复出现 step 次
    （多色循环表下可能需要多次点击，如 gqb=[9,8,12] 时 9→12 需点 2 次）。
    """
    constraints = guide_constraints(guides, set(blocks))
    by_block: dict = {}
    for pos, need_equal, color in constraints:
        by_block.setdefault(pos, []).append((need_equal, color))

    clicks_per_block: dict = {}
    for pos, reqs in by_block.items():
        init_color = int(blocks[pos])
        try:
            init_idx = list(gqb).index(init_color)
        except ValueError:
            return None
        found = None
        for step in range(len(gqb)):
            color = list(gqb)[(init_idx + step) % len(gqb)]
            if all((color == c) == eq for eq, c in reqs):
                found = step
                break
        if found is None:
            return None
        if found:
            clicks_per_block[pos] = found
    path: List[Click] = []
    for pos in sorted(clicks_per_block):
        path.extend([pos] * clicks_per_block[pos])
    return path


class Ft09Solver:
    """FT09 关卡求解器：先模型法，失败再重放 BFS。

    参数:
        reset: 返回初始状态（无参回调，须把环境重置到当前关起点）。
        step: 执行一次显示坐标点击 (x, y)，返回最新状态。
        get_sprites: 从状态取可点击精灵序列（含 "Hkx" tag）。
        is_solved: 判断当前状态是否过关。
        read_model: 可选，从当前状态读 (blocks, gqb, guides)；
            提供则启用模型法。blocks 为 {(x,y): color}。
        is_dead: 判断当前状态是否已死（GAME_OVER），默认恒 False。
    """

    def __init__(
        self,
        reset: Callable[[], object],
        step: Callable[[Click], object],
        get_sprites: Callable[[object], Sequence],
        is_solved: Callable[[object], bool],
        read_model: Optional[Callable[[object], tuple]] = None,
        is_dead: Callable[[object], bool] = lambda state: False,
    ):
        self.reset = reset
        self.step = step
        self.get_sprites = get_sprites
        self.is_solved = is_solved
        self.read_model = read_model
        self.is_dead = is_dead

    def solve(self, t_limit: float = 60.0, max_depth: int = 20) -> Optional[List[Click]]:
        """返回过关点击序列（显示坐标）；两种方法都失败返回 None。"""
        # 1. 模型法
        if self.read_model is not None:
            state = self.reset()
            model = self.read_model(state)
            if model is not None:
                blocks, gqb, guides = model
                grid_path = solve_by_model(blocks, gqb, guides)
                if grid_path:
                    display_path = [(x * 2, y * 2) for x, y in grid_path]
                    self.reset()
                    for xy in display_path:
                        state = self.step(xy)
                    if self.is_solved(state):
                        return display_path
        # 2. 重放 BFS 回退
        return self._bfs(t_limit=t_limit, max_depth=max_depth)

    def _bfs(self, t_limit: float, max_depth: int) -> Optional[List[Click]]:
        initial = self.reset()
        sprites = self.get_sprites(initial)
        start = full_state_key(sprites, 0)
        coords = sprite_click_coords(sprites)
        queue: deque = deque([[]])
        seen = {start}
        t0 = time.monotonic()
        while queue:
            if time.monotonic() - t0 > t_limit:
                return None
            path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for xy in coords:
                self.reset()
                for prev in path:
                    self.step(prev)
                state = self.step(xy)
                if self.is_solved(state):
                    return path + [xy]
                if self.is_dead(state):
                    continue
                key = full_state_key(self.get_sprites(state), 0)
                if key not in seen:
                    seen.add(key)
                    queue.append(path + [xy])
        return None
