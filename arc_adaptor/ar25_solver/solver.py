"""AR25 求解器 — A* 搜索 + 宏动作搜索双策略。

双策略架构:
    1. A*搜索 (solve_astar) — 单步动作 + Dijkstra去重 + 预算剪枝
       适合: 目标少、拼块少的简单关卡
    2. 宏动作搜索 (solve_macro) — 枚举轴位置 × 拼块位置组合
       适合: 目标多、需要轴移动的复杂关卡

自动策略 (solve) 按目标数和budget选择，A*超时自动回退宏搜索。
"""
import heapq
import itertools
import json
import time
from pathlib import Path

from .simulator import (
    AR25Simulator,
    AXIS_TAG_HORIZONTAL,
    AXIS_TAG_VERTICAL,
    BOARD_SIZE,
    IMMOVABLE_TAG,
    ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_SWITCH,
)
from .advisor import Advisor, SearchSnapshot


# ═══════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════

def load_levels(path=None):
    """加载关卡数据。

    Args:
        path: JSON 文件路径。None 则使用内置数据。

    Returns:
        list[dict]: 关卡数据列表
    """
    if path is None:
        path = Path(__file__).parent / "data" / "ar25_levels.json"
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════
# A* 搜索
# ═══════════════════════════════════════════════════════

def solve_astar(level_data, t_limit=60.0, max_nodes=2000000, weight=None,
                verbose=True, advisor=None):
    """A*搜索: Dijkstra去重 + 预算剪枝 + 镜像对感知启发 + 顾问反思。

    Args:
        level_data: 关卡数据 dict
        t_limit: 时间限制(秒)
        max_nodes: 最大搜索节点数
        weight: 启发权重(越大越贪心)。None 自适应。
        verbose: 打印进度
        advisor: 反思顾问 (None 则不触发反思)

    Returns:
        list[int] 动作序列，或 None
    """
    sim = AR25Simulator(level_data)
    if sim.is_won():
        return []

    start_snap = sim.snapshot()
    t0 = time.time()
    level_idx = level_data["level"]

    ACTS = [1, 2, 3, 4, 5]
    seq = 0
    initial_h = sim.heuristic()

    if weight is None:
        if sim.budget >= 128:
            weight = 8
        elif sim.budget >= 64:
            weight = 5
        else:
            weight = 10

    heap = [(initial_h * weight, 0, seq, start_snap, [])]
    best_steps = {sim.state_key(): 0}
    nodes = 0
    max_steps = sim.budget - 1
    pruned = 0
    last_report = 0
    dead_ends = 0
    best_h_history = [initial_h]
    min_h = initial_h

    if verbose:
        print(f"  A* L{level_idx}: budget={sim.budget}, h={initial_h}, "
              f"w={weight}, sprites={len(sim.sprites)}, targets={len(sim.targets)}")

    while heap:
        if time.time() - t0 > t_limit or nodes > max_nodes:
            if verbose:
                print(f"  ❌ 超时: {nodes}节点, {time.time()-t0:.1f}s")
            return None

        f, depth, _, snap, path = heapq.heappop(heap)
        sim.restore(snap)
        k0 = sim.state_key()
        if depth > best_steps.get(k0, 999):
            continue
        if depth >= max_steps:
            continue

        # ── 顾问反思触发 ──
        if advisor is not None:
            cur_h = sim.heuristic()
            covered = len(sim.targets & sim.all_covered_cells())
            snap_ctx = SearchSnapshot(
                level=level_idx, strategy="astar",
                steps_used=depth, budget=sim.budget,
                coverage=covered, total_targets=len(sim.targets),
                heuristic=cur_h, nodes=nodes, dead_ends=dead_ends,
                best_h_history=best_h_history,
                elapsed=time.time() - t0,
            )
            refl = advisor.check_and_reflect(snap_ctx)
            if refl:
                if refl.action == "GIVE_UP":
                    if verbose:
                        print(f"  顾问建议放弃: {refl.advice}")
                    return None
                if refl.action == "ADJUST" and "weight" in refl.params:
                    old_w = weight
                    weight = refl.params["weight"]
                    if verbose:
                        print(f"  顾问调整权重: {old_w} → {weight}")
                if refl.action == "PRUNE" and "prune_ratio" in refl.params:
                    max_steps = max(int(max_steps * (1 - refl.params["prune_ratio"])),
                                   max_steps // 2)
                    if verbose:
                        print(f"  顾问收紧深度上限: {max_steps}")

        if verbose and nodes - last_report >= 50000:
            last_report = nodes
            print(f"    进展: {nodes}节点, depth={depth}, "
                  f"f={f:.0f}, {time.time()-t0:.1f}s")

        for a in ACTS:
            sim.restore(snap)
            if not sim.step(a):
                continue
            if sim.won:
                result = path + [a]
                if verbose:
                    print(f"  ✅ {len(result)}步 ({time.time()-t0:.1f}s, "
                          f"{nodes}节点)")
                return result
            if sim.lost:
                dead_ends += 1
                continue
            k = sim.state_key()
            new_depth = depth + 1
            if new_depth >= best_steps.get(k, 999):
                continue
            best_steps[k] = new_depth
            h_val = sim.heuristic()
            if h_val < min_h:
                min_h = h_val
            best_h_history.append(h_val)
            if new_depth + h_val > sim.budget:
                pruned += 1
                continue
            seq += 1
            heapq.heappush(heap, (new_depth + h_val * weight, new_depth, seq,
                                  sim.snapshot(), path + [a]))
            nodes += 1

    if verbose:
        print(f"  ❌ 搜索耗尽: {nodes}节点")
    return None


# ═══════════════════════════════════════════════════════
# 宏动作搜索
# ═══════════════════════════════════════════════════════

def _manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _enumerate_useful_positions(sim, piece_idx, axis_val=None):
    s = sim.sprites[piece_idx]
    positions = set()
    for tx, ty in sim.targets:
        for c in s["cells"]:
            positions.add((tx - c[0], ty - c[1]))
            for ax in sim.axes:
                h = AXIS_TAG_HORIZONTAL in ax.get("tags", [])
                v = AXIS_TAG_VERTICAL in ax.get("tags", [])
                if axis_val is not None and h:
                    ry = 2 * axis_val - ty
                    positions.add((tx - c[0], ry - c[1]))
                elif axis_val is not None and v:
                    rx = 2 * axis_val - tx
                    positions.add((rx - c[0], ty - c[1]))
                else:
                    if v:
                        rx = 2 * ax["x"] - tx
                        positions.add((rx - c[0], ty - c[1]))
                    if h:
                        ry = 2 * ax["y"] - ty
                        positions.add((tx - c[0], ry - c[1]))
    return positions


def _position_coverage(sim, piece_idx, ox, oy, axis_val=None):
    s = sim.sprites[piece_idx]
    cells = [(ox + c[0], oy + c[1]) for c in s["cells"]]
    all_cells = set(cells)
    if axis_val is not None:
        for ax in sim.axes:
            if AXIS_TAG_HORIZONTAL in ax.get("tags", []):
                all_cells.update((x, 2 * axis_val - y) for x, y in cells)
            elif AXIS_TAG_VERTICAL in ax.get("tags", []):
                all_cells.update((2 * axis_val - x, y) for x, y in cells)
    else:
        all_cells.update(sim.get_reflections(cells))
    return sim.targets & all_cells


def _compute_cost(axis_init, axis_target, sprites_init, combo):
    axis_cost = abs(axis_target - axis_init) if axis_init is not None else 0
    sprite_cost = sum(
        _manhattan((ox, oy), init)
        for (ox, oy, _), init in zip(combo, sprites_init)
    )
    return axis_cost + sprite_cost + len(combo)


def _generate_actions(axis_init, axis_target, sprites_init, combo, n_switch,
                     axis_type=None):
    actions = []
    current_sel = 0
    if axis_init is not None and axis_target is not None:
        delta = axis_target - axis_init
        if axis_type == 'v':
            if delta < 0:
                actions.extend([ACT_LEFT] * (-delta))
            elif delta > 0:
                actions.extend([ACT_RIGHT] * delta)
        else:
            if delta < 0:
                actions.extend([ACT_UP] * (-delta))
            elif delta > 0:
                actions.extend([ACT_DOWN] * delta)

    for i, (ox, oy, _) in enumerate(combo):
        target_idx = i + 1 if axis_init is not None else i
        while current_sel != target_idx:
            actions.append(ACT_SWITCH)
            current_sel = (current_sel + 1) % n_switch
        init = sprites_init[i]
        dx, dy = ox - init[0], oy - init[1]
        if dx > 0:
            actions.extend([ACT_RIGHT] * dx)
        elif dx < 0:
            actions.extend([ACT_LEFT] * (-dx))
        if dy > 0:
            actions.extend([ACT_DOWN] * dy)
        elif dy < 0:
            actions.extend([ACT_UP] * (-dy))
    return actions


def solve_macro(level_data, t_limit=60.0, verbose=True, advisor=None):
    """宏动作搜索: 枚举轴位置 + 拼块位置组合 + 顾问反思。

    Args:
        level_data: 关卡数据 dict
        t_limit: 时间限制(秒)
        verbose: 打印进度
        advisor: 反思顾问 (None 则不触发反思)

    Returns:
        list[int] 动作序列，或 None
    """
    sim = AR25Simulator(level_data)
    if sim.is_won():
        return []

    t0 = time.time()
    level_idx = level_data["level"]
    n_sprites = len(sim.sprites)
    targets = sim.targets
    sprites_init = [(s["x"], s["y"]) for s in sim.sprites]

    h_axes = [ax for ax in sim.axes if AXIS_TAG_HORIZONTAL in ax.get("tags", [])]
    v_axes = [ax for ax in sim.axes if AXIS_TAG_VERTICAL in ax.get("tags", [])]
    has_movable = any(IMMOVABLE_TAG not in ax.get("tags", []) for ax in sim.axes)

    if h_axes and has_movable:
        axis_init, axis_range, axis_type = h_axes[0]["y"], range(BOARD_SIZE), 'h'
    elif v_axes and has_movable:
        axis_init, axis_range, axis_type = v_axes[0]["x"], range(BOARD_SIZE), 'v'
    else:
        axis_init, axis_range, axis_type = None, [None], None

    if verbose:
        print(f"  宏搜索 L{level_idx}: {n_sprites}拼块, {len(targets)}目标, "
              f"轴={axis_type or '无'}, budget={sim.budget}")

    best_cost = 999
    best_config = None
    axis_tried = 0
    best_h_history = []

    for axis_val in axis_range:
        if time.time() - t0 > t_limit:
            if verbose:
                print(f"  ❌ 宏搜索超时 ({time.time()-t0:.1f}s)")
            break

        axis_tried += 1

        all_positions = []
        for i in range(n_sprites):
            raw = _enumerate_useful_positions(sim, i, axis_val)
            raw.add(sprites_init[i])
            by_cov = {}
            for ox, oy in raw:
                cov = frozenset(_position_coverage(sim, i, ox, oy, axis_val))
                if cov and cov not in by_cov:
                    by_cov[cov] = (ox, oy)
            all_positions.append([(p[0], p[1], c) for c, p in by_cov.items()])

        if any(not p for p in all_positions):
            continue

        total = 1
        for p in all_positions:
            total *= len(p)

        if total > 10_000_000:
            top = [sorted(p, key=lambda x: -len(x[2]))[:50] for p in all_positions]
            iterator = itertools.product(*top)
        else:
            iterator = itertools.product(*all_positions)

        for combo in iterator:
            all_cov = set()
            for _, _, cov in combo:
                all_cov |= cov
            if len(all_cov) == len(targets):
                cost = _compute_cost(axis_init, axis_val, sprites_init, combo)
                if cost < best_cost:
                    best_cost = cost
                    best_config = (axis_val, combo)
                    if verbose:
                        print(f"  axis={axis_val}: ✅ 全覆盖! cost={cost}")

        # ── 顾问反思触发 ──
        if advisor is not None:
            best_h = best_cost if best_config else 999
            best_h_history.append(best_h)
            if best_config:
                cov_count = len(targets)
                steps = best_cost
                snap_budget = sim.budget
            else:
                cov_count = len(all_cov) if all_cov else 0
                steps = axis_tried
                snap_budget = BOARD_SIZE
            snap_ctx = SearchSnapshot(
                level=level_idx, strategy="macro",
                steps_used=steps, budget=snap_budget,
                coverage=cov_count, total_targets=len(targets),
                heuristic=best_h,
                nodes=axis_tried,
                dead_ends=axis_tried - (1 if best_config else 0),
                best_h_history=best_h_history,
                elapsed=time.time() - t0,
            )
            refl = advisor.check_and_reflect(snap_ctx)
            if refl:
                if refl.action == "GIVE_UP" and not best_config:
                    if verbose:
                        print(f"  顾问建议放弃: {refl.advice}")
                    return None
                if refl.action == "SWITCH" and not best_config:
                    if verbose:
                        print(f"  顾问建议切换策略: {refl.advice}")
                    return None
                if refl.action == "GIVE_UP" and best_config:
                    if verbose:
                        print(f"  顾问建议停止搜索 (已有解): {refl.advice}")
                    break

    if verbose:
        print(f"  宏搜索完成 ({time.time()-t0:.1f}s)")

    if not best_config:
        if verbose:
            print("  ❌ 未找到解!")
        return None

    axis_val, combo = best_config
    if verbose:
        print(f"  最优: 轴={axis_val}, cost={best_cost}")

    return _generate_actions(
        axis_init, axis_val, sprites_init, combo, len(sim.switch_list),
        axis_type
    )


# ═══════════════════════════════════════════════════════
# 统一接口
# ═══════════════════════════════════════════════════════

def solve(level_data, strategy="auto", t_limit=120, verbose=True,
          advisor=None, **kwargs):
    """统一求解接口。

    Args:
        level_data: 关卡数据 dict
        strategy: "auto" | "astar" | "macro"
        t_limit: 总时间限制(秒)
        verbose: 打印进度
        advisor: 反思顾问实例。True 自动创建，None 不启用。
        **kwargs: 传递给具体求解器

    Returns:
        list[int] 动作序列，或 None

    Example:
        >>> from ar25_solver import load_levels, solve, Advisor
        >>> levels = load_levels()
        >>> adv = Advisor(budget=levels[0]["budget"], level=1)
        >>> actions = solve(levels[0], strategy="auto", advisor=adv)
        ✅ 15步
    """
    sim = AR25Simulator(level_data)
    level_idx = level_data["level"]
    n_targets = len(sim.targets)
    budget = sim.budget

    # 自动创建顾问
    if advisor is True:
        advisor = Advisor(budget=budget, level=level_idx, verbose=verbose)
        if verbose:
            print(f"  顾问已启用 (阈值: {advisor.thresholds})")

    if verbose:
        print(f"\n{'='*60}")
        print(f"求解 L{level_idx}: {len(sim.sprites)}拼块, "
              f"{n_targets}目标, budget={budget}, 策略={strategy}")
        print(f"{'='*60}")

    if strategy == "auto":
        if n_targets <= 10 or budget <= 64:
            strategy = "astar_first"
        else:
            strategy = "macro_first"

    half = t_limit // 2

    if strategy == "astar_first":
        if verbose:
            print(f"\n→ A*搜索 (limit={half}s)")
        actions = solve_astar(level_data, t_limit=half, verbose=verbose,
                              advisor=advisor, **kwargs)
        if actions:
            return actions
        if verbose:
            print(f"\n→ 回退宏搜索 (limit={half}s)")
        return solve_macro(level_data, t_limit=half, verbose=verbose,
                           advisor=advisor)

    if strategy == "macro_first":
        if verbose:
            print(f"\n→ 宏搜索 (limit={half}s)")
        actions = solve_macro(level_data, t_limit=half, verbose=verbose,
                              advisor=advisor)
        if actions:
            sim2 = AR25Simulator(level_data)
            for a in actions:
                sim2.step(a)
            if sim2.won:
                if verbose:
                    print(f"  ✅ 验证通过: {sim2.steps_used}步")
                return actions
            if verbose:
                print("  ⚠️ 验证失败, 尝试A*")
        if verbose:
            print(f"\n→ 回退A*搜索 (limit={half}s)")
        return solve_astar(level_data, t_limit=half, verbose=verbose,
                          advisor=advisor, **kwargs)

    if strategy == "astar":
        return solve_astar(level_data, t_limit=t_limit, verbose=verbose,
                           advisor=advisor, **kwargs)

    if strategy == "macro":
        return solve_macro(level_data, t_limit=t_limit, verbose=verbose,
                           advisor=advisor)

    return None


def solve_all(levels_data, strategy="auto", t_limit=120, verbose=True,
              advisor=None):
    """求解所有关卡。

    Args:
        advisor: True 为每关自动创建顾问，None 不启用，
                 或传入 Advisor 实例。

    Returns:
        dict: {level_number: actions_list or None}
    """
    results = {}
    t_total = time.time()

    for lv in levels_data:
        actions = solve(lv, strategy=strategy, t_limit=t_limit,
                        verbose=verbose, advisor=advisor)
        if actions:
            sim = AR25Simulator(lv)
            for a in actions:
                sim.step(a)
            results[lv["level"]] = actions
            if verbose:
                covered = len(sim.targets & sim.all_covered_cells())
                print(f"\n  {'✅' if sim.won else '❌'} "
                      f"{sim.steps_used}步, {covered}/{len(sim.targets)}")
        else:
            results[lv["level"]] = None
            if verbose:
                print("\n  ❌ 未找到解")

    if verbose:
        print(f"\n{'='*60}")
        print(f"汇总 ({time.time()-t_total:.1f}s)")
        print(f"{'='*60}")
        for lv in levels_data:
            idx = lv["level"]
            acts = results.get(idx)
            if acts:
                sim = AR25Simulator(lv)
                for a in acts:
                    sim.step(a)
                print(f"  L{idx}: ✅ {sim.steps_used}步")
            else:
                print(f"  L{idx}: ❌")

    return results


def verify(level_data, actions):
    """验证动作序列是否能通关。

    Args:
        level_data: 关卡数据
        actions: 动作序列 list[int]

    Returns:
        dict: {won, steps, covered, total, budget}
    """
    sim = AR25Simulator(level_data)
    for a in actions:
        sim.step(a)
    covered = len(sim.targets & sim.all_covered_cells())
    return {
        "won": sim.won,
        "steps": sim.steps_used,
        "covered": covered,
        "total": len(sim.targets),
        "budget": sim.budget,
    }
