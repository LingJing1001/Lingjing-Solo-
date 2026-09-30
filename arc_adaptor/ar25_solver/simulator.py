"""AR25 状态模拟器 — 不依赖官方引擎，纯 Python 逆向实现。

从 ar25.py 源码逆向游戏规则，提供独立的状态模拟能力。

核心机制:
    - 21×21 棋盘
    - 可移动拼块: 有 MOVABLE_TAG 标签的 sprite
    - 镜像轴: 有 AXIS_TAG 标签，分横轴(y'=2ay-y)/竖轴(x'=2ax-x)
      - 轴是纯镜子: 不直接覆盖目标，只改变反射公式
      - 轴可移动(有 "sys_click" 标签且无 IMMOVABLE_TAG)
      - 横轴只竖直移动(只 y 变)，竖轴只水平移动(只 x 变)
    - 反射递归级联: 深度上限 MAX_CASCADE_DEPTH
    - 动作: 1=↑ 2=↓ 3=← 4=→ 5=切换
    - 通关: 所有目标格被拼块或反射覆盖

Example:
    >>> from ar25_solver.simulator import AR25Simulator
    >>> sim = AR25Simulator(level_data)
    >>> for action in solution:
    ...     sim.step(action)
    >>> sim.won
    True
"""
import copy

MAX_CASCADE_DEPTH = 12
BOARD_SIZE = 21

AXIS_TAG = "0003uqrdzdofso"
AXIS_TAG_VERTICAL = "0054kgxrvfihgm"
AXIS_TAG_HORIZONTAL = "0002nuguepuujf"
IMMOVABLE_TAG = "0056icpryeujyf"
MOVABLE_TAG = "0006lxjtqggkmi"

# 动作常量
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_SWITCH = 5


class AR25Simulator:
    """AR25 游戏状态模拟器。

    从关卡数据初始化，支持 step/snapshot/restore，
    可用于搜索算法或手动模拟游戏过程。
    """

    def __init__(self, level_data):
        self.board_size = level_data.get("board_size", BOARD_SIZE)
        self.budget = level_data["budget"]
        self.steps_used = 0
        self.won = False
        self.lost = False

        self.targets = {(t["x"], t["y"]) for t in level_data["targets"]}
        self.axes = [copy.deepcopy(a) for a in level_data["axes"]]

        all_sprites = [copy.deepcopy(s) for s in level_data["sprites"]]
        self.sprites = []
        for s in all_sprites:
            is_axis = any(
                s["x"] == ax["x"] and s["y"] == ax["y"] for ax in self.axes
            )
            if not is_axis:
                self.sprites.append(s)

        self.switch_list = []
        for i, ax in enumerate(self.axes):
            if IMMOVABLE_TAG not in ax.get("tags", []):
                self.switch_list.append(("axis", i))
        for i in range(len(self.sprites)):
            self.switch_list.append(("sprite", i))

        self.sel_idx = 0

    @property
    def selected_type(self):
        return self.switch_list[self.sel_idx]

    def _get_entity_pos(self):
        kind, idx = self.switch_list[self.sel_idx]
        if kind == "axis":
            return self.axes[idx]["x"], self.axes[idx]["y"]
        return self.sprites[idx]["x"], self.sprites[idx]["y"]

    def _set_entity_pos(self, x, y):
        kind, idx = self.switch_list[self.sel_idx]
        if kind == "axis":
            self.axes[idx]["x"] = x
            self.axes[idx]["y"] = y
        else:
            self.sprites[idx]["x"] = x
            self.sprites[idx]["y"] = y

    def _is_horizontal_axis(self):
        kind, idx = self.switch_list[self.sel_idx]
        if kind != "axis":
            return False
        return AXIS_TAG_HORIZONTAL in self.axes[idx].get("tags", [])

    def _is_vertical_axis(self):
        kind, idx = self.switch_list[self.sel_idx]
        if kind != "axis":
            return False
        return AXIS_TAG_VERTICAL in self.axes[idx].get("tags", [])

    # ─── 拼块格子 ─────────────────────────────
    def piece_cells(self, sprite_idx):
        s = self.sprites[sprite_idx]
        return [(s["x"] + c[0], s["y"] + c[1]) for c in s["cells"]]

    def all_piece_cells(self):
        cells = []
        for i in range(len(self.sprites)):
            cells.extend(self.piece_cells(i))
        return cells

    # ─── 镜像反射 ─────────────────────────────
    def get_reflections(self, cells):
        if not self.axes:
            return []
        result = set()
        frontier = set(cells)
        for _ in range(MAX_CASCADE_DEPTH):
            new_frontier = set()
            for ax in self.axes:
                is_v = AXIS_TAG_VERTICAL in ax.get("tags", [])
                is_h = AXIS_TAG_HORIZONTAL in ax.get("tags", [])
                for x, y in frontier:
                    if is_v:
                        rx, ry = 2 * ax["x"] - x, y
                        if (rx, ry) not in result and (rx, ry) not in frontier:
                            new_frontier.add((rx, ry))
                    if is_h:
                        rx, ry = x, 2 * ax["y"] - y
                        if (rx, ry) not in result and (rx, ry) not in frontier:
                            new_frontier.add((rx, ry))
            if not new_frontier:
                break
            result.update(new_frontier)
            frontier = new_frontier
        return list(result)

    def all_covered_cells(self):
        piece = self.all_piece_cells()
        refl = self.get_reflections(piece)
        return set(piece) | set(refl)

    # ─── 通关判定 ─────────────────────────────
    def is_won(self):
        return self.targets.issubset(self.all_covered_cells())

    # ─── 动作执行 ─────────────────────────────
    def move(self, dx, dy):
        if self.won or self.lost:
            return False
        if self._is_horizontal_axis():
            dx = 0
        elif self._is_vertical_axis():
            dy = 0
        if dx == 0 and dy == 0:
            return True

        x, y = self._get_entity_pos()
        new_x, new_y = x + dx, y + dy

        kind, idx = self.switch_list[self.sel_idx]
        if kind == "sprite":
            s = self.sprites[idx]
            cells = s["cells"]
            min_dx = min(c[0] for c in cells)
            max_dx = max(c[0] for c in cells)
            min_dy = min(c[1] for c in cells)
            max_dy = max(c[1] for c in cells)
            if new_x + min_dx < 0 or new_x + max_dx >= self.board_size:
                return False
            if new_y + min_dy < 0 or new_y + max_dy >= self.board_size:
                return False
        if kind == "axis":
            if self._is_horizontal_axis():
                if new_y < 0 or new_y >= self.board_size:
                    return False
            elif self._is_vertical_axis():
                if new_x < 0 or new_x >= self.board_size:
                    return False

        self._set_entity_pos(new_x, new_y)
        self.steps_used += 1
        if self.steps_used > self.budget:
            self.lost = True
            return False
        if self.is_won():
            self.won = True
        return True

    def switch(self):
        if self.won or self.lost:
            return False
        self.sel_idx = (self.sel_idx + 1) % len(self.switch_list)
        self.steps_used += 1
        if self.steps_used > self.budget:
            self.lost = True
            return False
        if self.is_won():
            self.won = True
        return True

    def step(self, action_id):
        """执行动作: 1=↑ 2=↓ 3=← 4=→ 5=切换"""
        if action_id == 1:
            return self.move(0, -1)
        if action_id == 2:
            return self.move(0, 1)
        if action_id == 3:
            return self.move(-1, 0)
        if action_id == 4:
            return self.move(1, 0)
        if action_id == 5:
            return self.switch()
        return False

    # ─── 快照/恢复 ────────────────────────────
    def snapshot(self):
        return {
            "sprites": copy.deepcopy(self.sprites),
            "axes": copy.deepcopy(self.axes),
            "sel_idx": self.sel_idx,
            "steps_used": self.steps_used,
            "won": self.won,
            "lost": self.lost,
        }

    def restore(self, snap):
        self.sprites = copy.deepcopy(snap["sprites"])
        self.axes = copy.deepcopy(snap["axes"])
        self.sel_idx = snap["sel_idx"]
        self.steps_used = snap["steps_used"]
        self.won = snap["won"]
        self.lost = snap["lost"]

    def state_key(self):
        return (
            tuple((s["x"], s["y"]) for s in self.sprites),
            tuple((ax["x"], ax["y"]) for ax in self.axes),
            self.sel_idx,
        )

    # ─── 启发函数 ─────────────────────────────
    def heuristic(self):
        """镜像对感知 + 全局轴代价 + 覆盖盈余惩罚。

        返回到达目标状态的最少步数下界。
        """
        covered = self.all_covered_cells()
        unc = self.targets - covered
        if not unc:
            return 0

        sprite_cells = [
            [(s["x"] + c[0], s["y"] + c[1]) for c in s["cells"]]
            for s in self.sprites
        ]

        best_h = self._h_direct(unc, sprite_cells)

        h_axes = [ax for ax in self.axes
                  if AXIS_TAG_HORIZONTAL in ax.get("tags", [])]
        if h_axes:
            for cand_ay in self._gen_h_candidates(unc, h_axes):
                h = self._h_with_h_axis(unc, sprite_cells, cand_ay, h_axes)
                if h < best_h:
                    best_h = h

        v_axes = [ax for ax in self.axes
                  if AXIS_TAG_VERTICAL in ax.get("tags", [])]
        if v_axes:
            for cand_ax in self._gen_v_candidates(unc, v_axes):
                h = self._h_with_v_axis(unc, sprite_cells, cand_ax, v_axes)
                if h < best_h:
                    best_h = h

        return best_h

    def _h_direct(self, unc, sprite_cells):
        refl_fn = self._current_reflection_fn()
        piece_max = [0] * len(self.sprites)
        piece_needed = [False] * len(self.sprites)

        for tx, ty in unc:
            best_cost = 999
            best_i = 0
            for i, cells in enumerate(sprite_cells):
                for cx, cy in cells:
                    d = abs(tx - cx) + abs(ty - cy)
                    if d < best_cost:
                        best_cost = d
                        best_i = i
                    if refl_fn:
                        for rx, ry in refl_fn(cx, cy):
                            d = abs(tx - rx) + abs(ty - ry)
                            if d < best_cost:
                                best_cost = d
                                best_i = i
            piece_needed[best_i] = True
            if best_cost > piece_max[best_i]:
                piece_max[best_i] = best_cost

        penalty = self._surplus_penalty(len(unc), sprite_cells)
        n_entities = sum(piece_needed)
        return sum(piece_max) + max(0, n_entities - 1) + penalty

    def _current_reflection_fn(self):
        if not self.axes:
            return None
        def refl(x, y):
            results = []
            for ax in self.axes:
                if AXIS_TAG_VERTICAL in ax.get("tags", []):
                    results.append((2 * ax["x"] - x, y))
                if AXIS_TAG_HORIZONTAL in ax.get("tags", []):
                    results.append((x, 2 * ax["y"] - y))
            return results
        return refl

    @staticmethod
    def _surplus_penalty(n_eff, sprite_cells):
        total_cells = sum(len(c) for c in sprite_cells)
        if n_eff > total_cells and sprite_cells:
            max_cells = max(len(c) for c in sprite_cells)
            extra = (n_eff - total_cells + max_cells - 1) // max_cells
            return extra * 2
        return 0

    def _gen_h_candidates(self, unc, h_axes):
        candidates = {ax["y"] for ax in h_axes}
        unc_list = list(unc)
        for i in range(len(unc_list)):
            x1, y1 = unc_list[i]
            for j in range(i + 1, len(unc_list)):
                x2, y2 = unc_list[j]
                if x1 == x2 and (y1 + y2) % 2 == 0:
                    candidates.add((y1 + y2) // 2)
        return candidates

    def _gen_v_candidates(self, unc, v_axes):
        candidates = {ax["x"] for ax in v_axes}
        unc_list = list(unc)
        for i in range(len(unc_list)):
            x1, y1 = unc_list[i]
            for j in range(i + 1, len(unc_list)):
                x2, y2 = unc_list[j]
                if y1 == y2 and (x1 + x2) % 2 == 0:
                    candidates.add((x1 + x2) // 2)
        return candidates

    def _h_with_h_axis(self, unc, sprite_cells, cand_ay, h_axes):
        effective_unc, _ = self._mirror_pairs(unc, cand_ay, "h")
        axis_cost = min(abs(cand_ay - ax["y"]) for ax in h_axes)
        piece_max, piece_needed = self._greedy_assign(
            effective_unc, sprite_cells, cand_ay, "h"
        )
        penalty = self._surplus_penalty(len(effective_unc), sprite_cells)
        n_entities = sum(piece_needed) + (1 if axis_cost > 0 else 0)
        return axis_cost + sum(piece_max) + max(0, n_entities - 1) + penalty

    def _h_with_v_axis(self, unc, sprite_cells, cand_ax, v_axes):
        effective_unc, _ = self._mirror_pairs(unc, cand_ax, "v")
        axis_cost = min(abs(cand_ax - ax["x"]) for ax in v_axes)
        piece_max, piece_needed = self._greedy_assign(
            effective_unc, sprite_cells, cand_ax, "v"
        )
        penalty = self._surplus_penalty(len(effective_unc), sprite_cells)
        n_entities = sum(piece_needed) + (1 if axis_cost > 0 else 0)
        return axis_cost + sum(piece_max) + max(0, n_entities - 1) + penalty

    @staticmethod
    def _mirror_pairs(unc, axis_val, axis_type):
        effective = set()
        paired = set()
        for t in unc:
            if t in paired:
                continue
            if axis_type == "h":
                mirror = (t[0], 2 * axis_val - t[1])
            else:
                mirror = (2 * axis_val - t[0], t[1])
            if mirror in unc and mirror != t:
                effective.add(t)
                paired.add(mirror)
            else:
                effective.add(t)
        return effective, paired

    @staticmethod
    def _greedy_assign(effective_unc, sprite_cells, axis_val, axis_type):
        piece_max = [0] * len(sprite_cells)
        piece_needed = [False] * len(sprite_cells)
        for tx, ty in effective_unc:
            best_cost = 999
            best_i = 0
            for i, cells in enumerate(sprite_cells):
                for cx, cy in cells:
                    d = abs(tx - cx) + abs(ty - cy)
                    if d < best_cost:
                        best_cost = d
                        best_i = i
                    if axis_type == "h":
                        ry = 2 * axis_val - ty
                        d = abs(tx - cx) + abs(ry - cy)
                    else:
                        rx = 2 * axis_val - tx
                        d = abs(rx - cx) + abs(ty - cy)
                    if d < best_cost:
                        best_cost = d
                        best_i = i
            piece_needed[best_i] = True
            if best_cost > piece_max[best_i]:
                piece_max[best_i] = best_cost
        return piece_max, piece_needed

    # ─── Dijkstra 寻路（用于 pathify）────────────────────────────

    def dijkstra_path(self, from_pos, to_pos, moving_sprite_idx=None,
                      moving_axis_idx=None):
        """从 from_pos 到 to_pos 的最短路径动作序列。

        障碍 = 其他 sprite 的格子（含轴）。
        动作: 1=↑ 2=↓ 3=← 4=→。

        Args:
            from_pos: (x, y) 起点
            to_pos: (x, y) 终点
            moving_sprite_idx: 正在移动的 sprite 索引（绕过时忽略）
            moving_axis_idx: 正在移动的 axis 索引（绕过时忽略）

        Returns:
            list[int] 动作序列，或 None（不可达）
        """
        fx, fy = from_pos
        tx, ty = to_pos
        if (fx, fy) == (tx, ty):
            return []

        # 预计算所有 sprite 的当前格子集合（用于碰撞检测）
        occupied = set()
        for i, s in enumerate(self.sprites):
            if i == moving_sprite_idx:
                continue
            for c in self.piece_cells(i):
                occupied.add(c)

        # 轴也占用其所在格（但通常轴在轴线上，占用一个格）
        for i, ax in enumerate(self.axes):
            if i == moving_axis_idx:
                continue
            occupied.add((ax["x"], ax["y"]))

        # BFS / Dijkstra（代价均为1）
        from collections import deque
        # state: (x, y)
        start = (fx, fy)
        goal = (tx, ty)
        queue = deque([(start, [])])  # (pos, actions)
        seen = {start}

        # 方向: (dx, dy, action_id)
        DIRS = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]

        # 对 sprite 做几何边界检查
        def sprite_fit(sx, sy, sprite_idx):
            """检查 sprite_idx 的所有 cell 在 (sx,sy) 是否都在棋盘内。"""
            s = self.sprites[sprite_idx]
            for c in s["cells"]:
                cx2 = sx + c[0]
                cy2 = sy + c[1]
                if cx2 < 0 or cx2 >= self.board_size or cy2 < 0 or cy2 >= self.board_size:
                    return False
            return True

        while queue:
            (cx, cy), actions = queue.popleft()
            for dx, dy, act in DIRS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen:
                    continue
                if (nx, ny) in occupied:
                    continue
                # Sprite 几何边界检查
                if moving_sprite_idx is not None and not sprite_fit(nx, ny, moving_sprite_idx):
                    continue
                new_actions = actions + [act]
                if (nx, ny) == goal:
                    return new_actions
                seen.add((nx, ny))
                queue.append(((nx, ny), new_actions))
        return None  # 不可达

    def can_place_at(self, sprite_idx, x, y):
        """检查 sprite_idx 能否放到 (x, y) 位置（无碰撞、出界）。"""
        s = self.sprites[sprite_idx]
        cells = s["cells"]
        for c in cells:
            cx, cy = x + c[0], y + c[1]
            if cx < 0 or cx >= self.board_size or cy < 0 or cy >= self.board_size:
                return False
            for j, other in enumerate(self.sprites):
                if j == sprite_idx:
                    continue
                for oc in other["cells"]:
                    ox, oy = other["x"] + oc[0], other["y"] + oc[1]
                    if (cx, cy) == (ox, oy):
                        return False
        return True

    def get_sprite_at(self, x, y, exclude_idx=None):
        """返回占据 (x, y) 的 sprite 索引，或 None。"""
        for i, s in enumerate(self.sprites):
            if i == exclude_idx:
                continue
            for c in s["cells"]:
                if (s["x"] + c[0], s["y"] + c[1]) == (x, y):
                    return i
        return None

    def switch_to(self, target_sel_idx):
        """从当前 sel_idx 切换到 target_sel_idx，返回需要的 switch 次数。"""
        n = len(self.switch_list)
        diff = (target_sel_idx - self.sel_idx) % n
        return diff
