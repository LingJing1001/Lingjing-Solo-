"""AR25 Learning Template — 模板接口实现。

三个核心函数（北极星核心回路设计方案定义）:
    bind_obs_from_env(env)  -> Obs          # 从 live 引擎同步状态
    pathify(sim, config)    -> list[int]   # 配置 → 动作序列（处理碰撞）
    Ar25Learner.solve(obs, *, replay, max_configs) -> SolveResult  # 枚举→pathify→replay→tabu

验收标准：把 Field 契约 + Learner 编排 + 门禁套到 AR25，不用再写一个特解仓库。
"""
from dataclasses import dataclass, field
from typing import Optional
import heapq
import itertools
import copy

from ar25_solver.simulator import (
    AR25Simulator,
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_SWITCH,
    AXIS_TAG_HORIZONTAL, AXIS_TAG_VERTICAL,
    IMMOVABLE_TAG, BOARD_SIZE,
)


# ═══════════════════════════════════════════════════════════
# Observable State（从 live 引擎提取）
# ═══════════════════════════════════════════════════════════

@dataclass
class Obs:
    """AR25 可观测状态。

    从 arcengine live 实例的私有属性中提取。
    属性名来自逆向工程，可能因版本而变。
    """
    level_idx: int
    steps_used: int
    budget: int
    # 每个轴的位置 (x, y)
    axes: list[tuple[int, int]]
    # 每个 sprite 的位置 (x, y)
    sprites: list[tuple[int, int]]
    # 当前选中索引（switch_list 中的索引）
    sel_idx: int
    # 未覆盖目标
    targets: list[tuple[int, int]]

    def sprite_count(self) -> int:
        return len(self.sprites)

    def axis_count(self) -> int:
        return len(self.axes)


def bind_obs_from_env(env) -> Obs:
    """从 live arcengine 实例提取可观测状态。

    字段名为逆向工程结果，来自 arc_agi / arcengine 内部实现。
    若线上版本更新导致属性名变化，此函数需要同步更新。
    """
    g = env._game
    level_idx = int(getattr(g, "_current_level_index", 0))
    steps_used = int(getattr(g, "lelsvjlwneo", None).current_steps)
    budget = int(getattr(g, "lelsvjlwneo", None).ilqnjlrnkk)

    # axes: g.jtkyjqznbnp → list of axis sprites
    axes_raw = getattr(g, "jtkyjqznbnp", [])
    axes = [(int(ax.x), int(ax.y)) for ax in axes_raw]

    # sprites: g.ayyvxqrhnzw → list of all sprites (axes + pieces)
    sprites_raw = getattr(g, "ayyvxqrhnzw", [])
    # 排除 axes（axes 也出现在 ayyvxqrhnzw 中，按 ar25.py 的过滤逻辑）
    axis_positions = set((int(ax.x), int(ax.y)) for ax in axes_raw)
    sprites = []
    for s in sprites_raw:
        pos = (int(s.x), int(s.y))
        is_axis = pos in axis_positions
        if not is_axis:
            sprites.append(pos)

    # sel_idx: g.yvifanjrcyu 在 g.ayyvxqrhnzw 中的索引
    current_sel = getattr(g, "yvifanjrcyu", None)
    sel_idx = 0
    if current_sel is not None:
        try:
            sel_idx = int(list(sprites_raw).index(current_sel))
        except (ValueError, TypeError):
            sel_idx = 0

    # targets: g.fswikrcrdmx → list of target sprites
    targets_raw = getattr(g, "fswikrcrdmx", [])
    targets = [(int(t.x), int(t.y)) for t in targets_raw]

    return Obs(
        level_idx=level_idx,
        steps_used=steps_used,
        budget=budget,
        axes=axes,
        sprites=sprites,
        sel_idx=sel_idx,
        targets=targets,
    )


# ═══════════════════════════════════════════════════════════
# Solve Result
# ═══════════════════════════════════════════════════════════

@dataclass
class SolveResult:
    path: list[int]           # 动作序列
    config: tuple             # 达到的配置 (axis_pos, sprite_pos_list, sel_idx)
    solved: bool              # 是否通关
    nodes: int                # 枚举的候选配置数
    method: str               # "enumerate" | "replay" | "partial"
    cost: int                 # 动作代价（路径长度）


# ═══════════════════════════════════════════════════════════
# 核心: pathify — 配置 → 动作序列（处理碰撞）
# ═══════════════════════════════════════════════════════════

def _find_nearby_empty(sim, near_x, near_y, exclude_sprite_idx=None):
    """在 (near_x, near_y) 附近找最近的空格子（BFS一圈扩散）。"""
    from collections import deque
    visited = {(sim.sprites[i]["x"], sim.sprites[i]["y"])
               for i in range(len(sim.sprites)) if i != exclude_sprite_idx}
    for ax in sim.axes:
        visited.add((ax["x"], ax["y"]))
    queue = deque([(near_x, near_y, 0)])
    seen = {(near_x, near_y)}
    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    while queue:
        x, y, d = queue.popleft()
        if (x, y) not in visited:
            # 验证这里真的可以放 exclude_sprite_idx
            if exclude_sprite_idx is not None:
                if sim.can_place_at(exclude_sprite_idx, x, y):
                    return (x, y)
            else:
                if 0 <= x < sim.board_size and 0 <= y < sim.board_size:
                    return (x, y)
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if nx < 0 or nx >= sim.board_size or ny < 0 or ny >= sim.board_size:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny, d + 1))
    return None


def _move_entity_direct(sim, from_pos, to_pos, moving_sprite_idx=None,
                          moving_axis_idx=None):
    """用 Dijkstra 在 sim 棋盘上找路径，返回动作序列或 None。

    障碍 = 其他 sprite 的格子 + 轴的位置。
    """
    path = sim.dijkstra_path(from_pos, to_pos,
                               moving_sprite_idx=moving_sprite_idx,
                               moving_axis_idx=moving_axis_idx)
    return path


def pathify(sim: AR25Simulator, config) -> list[int]:
    """把目标配置转换为动作序列（不假设无碰撞）。

    config = (axis_pos, sprite_positions, sel_idx)
        axis_pos: tuple (x, y) 或 None（无可移动轴）
        sprite_positions: list[tuple(x, y)] 每块的目标位置
        sel_idx: int 目标选中索引

    策略:
        1. 轴移动（如果 axis_pos 不等于当前轴位置）
        2. Sprite 贪心移动:
            - 按「已在目标位置」→「离目标近」→「先处理」排序
            - 对每个 sprite:
                a. 如果已在目标，跳过
                b. 用 Dijkstra 找从当前位置到目标的路径（忽略已确定完成的其他 sprite）
                c. 如果目标被其他 sprite 占据 → 先把阻塞 sprite 临时移开
            d. 累积动作序列

    Returns:
        list[int] 动作序列（如无法完成全部移动，返回部分序列）
    """
    axis_pos_target, sprite_pos_target, sel_idx_target = config
    actions = []

    # ── 1. 轴移动（完整执行到最终状态）─────────────────
    movable_axes = [ax for ax in sim.axes
                    if IMMOVABLE_TAG not in ax.get("tags", [])]
    if axis_pos_target is not None and movable_axes:
        ax = movable_axes[0]
        cx, cy = ax["x"], ax["y"]
        is_v = AXIS_TAG_VERTICAL in ax.get("tags", [])
        is_h = AXIS_TAG_HORIZONTAL in ax.get("tags", [])

        if isinstance(axis_pos_target, tuple):
            tx, ty = axis_pos_target
        else:
            if is_v:
                tx, ty = axis_pos_target, cy
            else:
                tx, ty = cx, axis_pos_target

        # 切换到轴
        ax_idx = sim.switch_list.index(("axis", sim.axes.index(ax)))
        n_switches = sim.switch_to(ax_idx)
        for _ in range(n_switches):
            actions.append(ACT_SWITCH)
        sim.sel_idx = ax_idx

        axis_idx_in_axes = sim.axes.index(ax)

        if is_v:
            # 竖轴：只水平移动，不动 y（y=-3 是合法位置，镜像只依赖 x）
            path_h = _move_entity_direct(
                sim, (cx, cy), (tx, cy),
                moving_axis_idx=axis_idx_in_axes
            )
            if path_h:
                for a in path_h:
                    sim.step(a)
                    actions.append(a)
                ax["x"] = tx
        elif is_h:
            # 横轴：先竖直移动，再 x 归零
            path_v = _move_entity_direct(
                sim, (cx, cy), (cx, ty),
                moving_axis_idx=axis_idx_in_axes
            )
            if path_v:
                for a in path_v:
                    sim.step(a)
                    actions.append(a)
                ax["y"] = ty
            if ax["x"] != 0:
                dx = -ax["x"]
                if dx < 0:
                    for _ in range(-dx):
                        if sim.step(ACT_LEFT):
                            actions.append(ACT_LEFT)
                        else:
                            break
                elif dx > 0:
                    for _ in range(dx):
                        if sim.step(ACT_RIGHT):
                            actions.append(ACT_RIGHT)
                        else:
                            break
                ax["x"] = 0

    # ── 2. Sprite 移动 ─────────────────────────────────
    # 先按「是否已在目标位置」+「离目标距离」排序处理顺序
    sprite_status = []
    for i, (tx, ty) in enumerate(sprite_pos_target):
        cx, cy = sim.sprites[i]["x"], sim.sprites[i]["y"]
        dist = abs(cx - tx) + abs(cy - ty)
        at_target = (cx == tx and cy == ty)
        sprite_status.append((i, cx, cy, tx, ty, dist, at_target))

    # 已完成的 sprite 集合（其格子在寻路时应绕开）
    done_sprites = set()

    for rank, (i, cx, cy, tx, ty, dist, at_target) in enumerate(sprite_status):
        if at_target:
            done_sprites.add(i)
            continue

        # 切换到 sprite i
        target_sel_idx = len(movable_axes) + i
        n_switches = sim.switch_to(target_sel_idx)
        for _ in range(n_switches):
            actions.append(ACT_SWITCH)
        sim.sel_idx = target_sel_idx

        current_x, current_y = sim.sprites[i]["x"], sim.sprites[i]["y"]

        # 用 Dijkstra 寻路（障碍 = 其他 sprite 的格子，起点/终点不算障碍）
        path = sim.dijkstra_path((current_x, current_y), (tx, ty),
                                  moving_sprite_idx=i)
        if path is None:
            # 目标被占据，尝试绕路或临时移开阻塞 sprite
            blocking = sim.get_sprite_at(tx, ty, exclude_idx=i)
            if blocking is not None:
                # 临时把阻塞 sprite 移开
                orig_sel = sim.sel_idx
                orig_pos = (sim.sprites[i]["x"], sim.sprites[i]["y"])
                block_sel_idx = len(movable_axes) + blocking
                nsw = sim.switch_to(block_sel_idx)
                for _ in range(nsw):
                    actions.append(ACT_SWITCH)
                sim.sel_idx = block_sel_idx

                bx, by = sim.sprites[blocking]["x"], sim.sprites[blocking]["y"]
                temp = _find_nearby_empty(sim, bx, by, exclude_sprite_idx=blocking)
                if temp is not None:
                    tx2, ty2 = temp
                    path2 = sim.dijkstra_path((bx, by), (tx2, ty2),
                                              moving_sprite_idx=blocking)
                    if path2:
                        actions.extend(path2)
                        sim.sprites[blocking]["x"] = tx2
                        sim.sprites[blocking]["y"] = ty2
                        # 再试一次原 sprite 的路径
                        path = sim.dijkstra_path(orig_pos, (tx, ty),
                                                 moving_sprite_idx=i)

                # 切回原 sprite
                nsw2 = sim.switch_to(orig_sel)
                for _ in range(nsw2):
                    actions.append(ACT_SWITCH)
                sim.sel_idx = orig_sel
                sim.sprites[i]["x"], sim.sprites[i]["y"] = orig_pos

        if path:
            actions.extend(path)
            # 在 sim 上执行路径（更新 sprite 位置）
            for a in path:
                sim.step(a)
            # 验证位置
            if sim.sprites[i]["x"] != tx or sim.sprites[i]["y"] != ty:
                # 执行后位置不对，直接设置
                sim.sprites[i]["x"] = tx
                sim.sprites[i]["y"] = ty

        done_sprites.add(i)

    # ── 3. 切换到目标 sel_idx ────────────────────────────
    if sim.sel_idx != sel_idx_target:
        n_switches = sim.switch_to(sel_idx_target)
        for _ in range(n_switches):
            actions.append(ACT_SWITCH)
        sim.sel_idx = sel_idx_target

    return actions


# ═══════════════════════════════════════════════════════════
# 枚举候选配置（复用 solver.py 逻辑）
# ═══════════════════════════════════════════════════════════

def _enumerate_configs_for_learner(obs: Obs, sim: AR25Simulator,
                                    max_configs: int) -> list[tuple]:
    """枚举候选配置: (axis_pos, sprite_positions, sel_idx)。

    策略:
    1. 对每个 sprite，枚举能覆盖有效目标的候选位置
    2. 对轴枚举候选位置（当前位置 + 镜像对位置）
    3. 笛卡尔积 × 评分，截断到 max_configs

    Returns:
        list of config = (axis_pos, sprite_pos_list, sel_idx)
    """
    from ar25_solver.solver import _enumerate_useful_positions, _position_coverage

    configs = []
    n_sprites = len(sim.sprites)
    targets = sim.targets
    sprites_init = [(s["x"], s["y"]) for s in sim.sprites]

    h_axes = [ax for ax in sim.axes
              if AXIS_TAG_HORIZONTAL in ax.get("tags", [])]
    v_axes = [ax for ax in sim.axes
              if AXIS_TAG_VERTICAL in ax.get("tags", [])]

    # 确定轴相关参数
    if h_axes:
        axis_init, axis_type = h_axes[0]["y"], 'h'
        axis_range = list(range(BOARD_SIZE))
    elif v_axes:
        axis_init, axis_type = v_axes[0]["x"], 'v'
        axis_range = list(range(BOARD_SIZE))
    else:
        axis_init, axis_type, axis_range = None, None, [None]

    # 预生成每个 sprite 的候选位置（按覆盖目标数排序）
    # 关键修复：用所有 cells 生成候选（不是只用 cells[0]）
    # 因为一个 sprite 的不同 cell 可以覆盖不同目标
    def _gen_candidates_all_cells(sim, piece_idx, axis_val=None):
        """生成候选位置：遍历所有 cell × 所有 target 的反射位置。"""
        s = sim.sprites[piece_idx]
        positions = set()
        for tx, ty in sim.targets:
            for cell in s["cells"]:
                # 直接覆盖：sprite_origin = target - cell
                positions.add((tx - cell[0], ty - cell[1]))
                # 反射覆盖
                for ax in sim.axes:
                    is_v = AXIS_TAG_VERTICAL in ax.get("tags", [])
                    is_h = AXIS_TAG_HORIZONTAL in ax.get("tags", [])
                    if axis_val is not None and is_v:
                        rx = 2 * axis_val - tx
                        positions.add((rx - cell[0], ty - cell[1]))
                    elif axis_val is not None and is_h:
                        ry = 2 * axis_val - ty
                        positions.add((tx - cell[0], ry - cell[1]))
                    else:
                        if is_v:
                            rx = 2 * ax["x"] - tx
                            positions.add((rx - cell[0], ty - cell[1]))
                        if is_h:
                            ry = 2 * ax["y"] - ty
                            positions.add((tx - cell[0], ry - cell[1]))
        return positions

    # 几何有效性检查：对 sprite i 的候选位置 (ox, oy)，验证所有 cell 都在棋盘内
    def sprite_pos_valid(piece_idx, ox, oy):
        s = sim.sprites[piece_idx]
        for c in s["cells"]:
            cx2, cy2 = ox + c[0], oy + c[1]
            if cx2 < 0 or cx2 >= BOARD_SIZE or cy2 < 0 or cy2 >= BOARD_SIZE:
                return False
        return True

    # 轴候选位置
    axis_candidates = set(axis_range)
    # 加镜像对候选
    unc_list = list(targets)
    for i in range(len(unc_list)):
        x1, y1 = unc_list[i]
        for j in range(i + 1, len(unc_list)):
            x2, y2 = unc_list[j]
            if x1 == x2 and (y1 + y2) % 2 == 0:
                axis_candidates.add((y1 + y2) // 2)
            if y1 == y2 and (x1 + x2) % 2 == 0:
                axis_candidates.add((x1 + x2) // 2)
    # 过滤 None（无语义轴时）
    numeric_candidates = sorted([c for c in axis_candidates if c is not None])

    # 迭代：对每个 axis_val，生成能覆盖目标的 sprite 候选（用真实 axis 位置计算覆盖）
    # 先按「是否在初始位置」排序
    if axis_init is not None and numeric_candidates:
        numeric_candidates.sort(key=lambda v: (v != axis_init, abs(v - axis_init)))

    MAX_PER_SPRITE = 8
    count = 0
    for axis_val in numeric_candidates:
        if count >= max_configs:
            break
        # 关键修复：对每个 axis_val 单独生成 sprite 候选
        # 这样 (15,14) 在 axis=10 时会被正确生成（coverage=8）
        sprite_candidates_for_axis = []
        for i in range(n_sprites):
            raw = _gen_candidates_all_cells(sim, i, axis_val=axis_val)
            raw.add(sprites_init[i])
            by_cov = {}
            for ox, oy in raw:
                # 几何过滤
                if not sprite_pos_valid(i, ox, oy):
                    continue
                # 用真实的 axis_val 计算覆盖
                cov = _position_coverage(sim, i, ox, oy, axis_val=axis_val)
                # 初始位置无论 coverage 都保留
                is_init = (ox == sprites_init[i][0] and oy == sprites_init[i][1])
                if is_init or cov:
                    by_cov[(ox, oy)] = frozenset(cov)
            deduped = {}
            for pos, cov in by_cov.items():
                if pos not in deduped or pos == sprites_init[i]:
                    deduped[pos] = cov
            candidates = list(deduped.keys())
            # 排序：初始位置优先，然后按 coverage 降序（确保高覆盖位置在前 8 名）
            candidates.sort(key=lambda p: (p != sprites_init[i], -len(deduped[p])))
            sprite_candidates_for_axis.append(candidates[:MAX_PER_SPRITE])

        it = itertools.product(*sprite_candidates_for_axis)
        for combo in it:
            if count >= max_configs:
                break
            all_cov = set()
            for i, (ox, oy) in enumerate(combo):
                cov = _position_coverage(sim, i, ox, oy, axis_val=axis_val)
                all_cov |= cov
            # 只保留至少覆盖部分目标的配置
            if not all_cov:
                continue
            # sel_idx = 第一个 sprite（索引 = len(movable_axes)）
            movable_axes = [ax for ax in sim.axes
                            if IMMOVABLE_TAG not in ax.get("tags", [])]
            sel_idx = len(movable_axes)  # 默认选中第一个 sprite
            config = (axis_val, list(combo), sel_idx)
            configs.append(config)
            count += 1

    # 按覆盖目标数降序排序
    def config_score(c):
        axis_val, combo, _ = c
        cov = set()
        for i, (ox, oy) in enumerate(combo):
            cov |= _position_coverage(sim, i, ox, oy, axis_val=axis_val)
        return len(cov)

    configs.sort(key=config_score, reverse=True)
    return configs[:max_configs]


# ═══════════════════════════════════════════════════════════
# Ar25Learner
# ═══════════════════════════════════════════════════════════

class Ar25Learner:
    """AR25 学习模板: 枚举 → pathify → replay → tabu 循环。

    使用方式:
        result = learner.solve(obs, replay=my_replay, max_configs=32)
        if result.solved:
            print(f"通关! 路径: {result.path}")
    """

    def __init__(self, level_data=None):
        """初始化学习器。

        Args:
            level_data: 关卡原始数据（含真实 sprite cells/tags）。如提供，
                       learner's sim 会与 replay() 使用相同的 level_data，
                       保证 switch_list 一致。
        """
        self.sim: Optional[AR25Simulator] = None
        self.level_data = level_data

    def solve(self, obs: Obs, *, replay, max_configs: int = 32) -> SolveResult:
        """求解主循环。

        Args:
            obs: 当前可观测状态
            replay: 回调函数，接收动作序列，在真实引擎上重放验证
                    返回 True 表示通关
            max_configs: 最大枚举配置数

        Returns:
            SolveResult
        """
        # 用 obs 初始化模拟器副本
        # 优先使用传入的 level_data（含真实 cells/tags），否则从 obs 构造
        if self.level_data is not None:
            ld = self.level_data
            self.sim = AR25Simulator(ld)
            # 用 obs 覆盖位置（因为 obs 是当前实时状态）
            for i, (x, y) in enumerate(obs.sprites):
                if i < len(self.sim.sprites):
                    self.sim.sprites[i]["x"] = x
                    self.sim.sprites[i]["y"] = y
            for i, (x, y) in enumerate(obs.axes):
                if i < len(self.sim.axes):
                    self.sim.axes[i]["x"] = x
                    self.sim.axes[i]["y"] = y
        else:
            ld = {
                "budget": obs.budget,
                "targets": [{"x": x, "y": y} for x, y in obs.targets],
                "axes": [{"x": x, "y": y, "tags": [], "cells": []}
                          for x, y in obs.axes],
                "sprites": [{"x": x, "y": y, "tags": ["0006lxjtqggkmi"],
                              "cells": [[0, 0]]}
                             for x, y in obs.sprites],
            }
            self.sim = AR25Simulator(ld)
        # 修正 sel_idx
        self.sim.sel_idx = min(obs.sel_idx, len(self.sim.switch_list) - 1)

        # 枚举候选配置
        configs = _enumerate_configs_for_learner(obs, self.sim, max_configs)

        best_result = None
        best_coverage = 0
        tabu = set()  # 已验证失败的 (axis_pos, sprite_pos_tuple)

        for nodes, config in enumerate(configs):
            axis_pos, sprite_pos, sel_idx = config
            key = (axis_pos, tuple(sprite_pos))
            if key in tabu:
                continue

            # 用 pathify 生成动作
            # 需要在 sim 的副本上运行（pathify 会修改 sim 状态）
            sim_copy = copy.deepcopy(self.sim)
            path = pathify(sim_copy, config)

            if not path:
                tabu.add(key)
                continue

            # replay 验证
            try:
                ok = replay(path)
            except Exception:
                ok = False

            if ok:
                # 验证 sim_copy 确实达到配置
                return SolveResult(
                    path=path,
                    config=config,
                    solved=True,
                    nodes=nodes + 1,
                    method="enumerate",
                    cost=len(path),
                )

            # 计算覆盖率（启发）
            sim_test = copy.deepcopy(self.sim)
            for a in path:
                sim_test.step(a)
            coverage = len(sim_test.targets & sim_test.all_covered_cells())
            if coverage > best_coverage:
                best_coverage = coverage
                best_result = SolveResult(
                    path=path,
                    config=config,
                    solved=False,
                    nodes=nodes + 1,
                    method="partial",
                    cost=len(path),
                )

            tabu.add(key)

        if best_result is not None:
            return best_result

        return SolveResult(
            path=[],
            config=None,
            solved=False,
            nodes=len(configs),
            method="enumerate",
            cost=0,
        )
