"""R2+R3+R4 组合跑 AR25 闯关。

R2 感知: 从游戏内部状态提取轴/拼块/目标/覆盖信息
R3 搜索: 目标分解 → beam search → 状态空间搜索
R4 剪枝: 动作效果反馈 + 覆盖进度 + 反循环 (贪心回退)

策略:
  1. R2 感知关卡结构 → 决定搜索策略
  2. R3 目标分解 (单轴关卡, 秒级)
  3. R3 beam search (中复杂度)
  4. R3 arc_shadow (L6+)
  5. R4 贪心步进 (最终回退, 逐动作评分)
"""
import sys, os, time, heapq, itertools
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"F:/pro/Lingjing-Solo-/arc_adaptor")
sys.path.insert(0, r"F:/pro/Lingjing-Solo-backup-20260916-125530/arc_adaptor")

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState
from arc_shadow import _snapshot, _restore, _state_key, _heuristic

ACT_MAP = {n: getattr(GameAction, "ACTION%d" % n) for n in range(1, 8)}
ACT_NAMES = {1:'UP↑',2:'DOWN↓',3:'LEFT←',4:'RIGHT→',5:'TOGGLE⟳'}
H_AXIS_TAG = "0002nuguepuujf"
V_AXIS_TAG = "0054kgxrvfihgm"
BOARD = 21

def is_won(g):
    w = g.vplrhaovhr()
    return w is True or (hasattr(w, '__len__') and len(w) == 1 and bool(w))

def get_cells(obj):
    h, w = obj.pixels.shape
    return [(j, i) for i in range(h) for j in range(w) if obj.pixels[i, j] != -1]

# ═══════════════════════════════════════════════════════════
# R2: 感知 (从内部状态提取结构)
# ═══════════════════════════════════════════════════════════

def r2_perceive(g):
    """从游戏内部状态提取完整关卡信息。"""
    axis_ids = set(id(a) for a in g.jtkyjqznbnp)
    axes = []
    for ax in g.jtkyjqznbnp:
        tags = list(ax.tags) if hasattr(ax, "tags") else []
        atype = 'h' if H_AXIS_TAG in tags else ('v' if V_AXIS_TAG in tags else '?')
        axes.append({"x": int(ax.x), "y": int(ax.y), "type": atype, "cells": get_cells(ax)})
    switch_order = []
    for idx, obj in enumerate(g.ayyvxqrhnzw):
        is_axis = id(obj) in axis_ids
        switch_order.append({"idx": idx, "is_axis": is_axis,
                             "x": int(obj.x), "y": int(obj.y),
                             "cells": get_cells(obj) if not is_axis else None})
    sel_idx = g.ayyvxqrhnzw.index(g.yvifanjrcyu) if g.yvifanjrcyu in g.ayyvxqrhnzw else 0
    targets = {(int(t.x), int(t.y)) for t in g.fswikrcrdmx}
    budget = int(g.lelsvjlwneo.ilqnjlrnkk)
    steps_left = int(g.lelsvjlwneo.current_steps)
    grid = g.naxbskjmlg()
    uncovered = [(int(t.x), int(t.y)) for t in g.fswikrcrdmx if grid[t.y, t.x] < 0]
    movable = [s for s in switch_order if not s["is_axis"]]
    n_axes = len(axes)
    atype = axes[0]["type"] if n_axes == 1 else '?'
    return {
        "axes": axes, "switch_order": switch_order, "sel_idx": sel_idx,
        "targets": targets, "budget": budget, "steps_left": steps_left,
        "n_switch": len(switch_order), "movable": movable,
        "n_axes": n_axes, "atype": atype,
        "covered": len(targets) - len(uncovered), "total_targets": len(targets),
        "uncovered": uncovered, "won": is_won(g),
        "level": int(g._current_level_index),
        "state_key": _state_key(g),
    }

# ═══════════════════════════════════════════════════════════
# R3-A: 目标分解搜索
# ═══════════════════════════════════════════════════════════

def enum_useful_positions(sprite, axis_pos, targets, atype):
    cells = sprite["cells"]
    sx, sy = sprite["x"], sprite["y"]
    positions = set()
    for tx, ty in targets:
        for cx, cy in cells:
            positions.add((tx - cx, ty - cy))
            if atype == 'h':
                positions.add((tx - cx, 2 * axis_pos - ty - cy))
            else:
                positions.add((2 * axis_pos - tx - cx, ty - cy))
    by_cov = {}
    for ox, oy in positions:
        abs_cells = {(ox + cx, oy + cy) for cx, cy in cells}
        if atype == 'h':
            refl = {(x, 2 * axis_pos - y) for x, y in abs_cells}
        else:
            refl = {(2 * axis_pos - x, y) for x, y in abs_cells}
        cov = frozenset(targets & (abs_cells | refl))
        if cov:
            d = abs(ox - sx) + abs(oy - sy)
            if cov not in by_cov or d < by_cov[cov][2]:
                by_cov[cov] = (ox, oy, d)
    return [(v[0], v[1], cov) for cov, v in by_cov.items()]

def solve_goal_decomp(g, perc):
    """R3 目标分解: 枚举轴位置×拼块位置组合, 找全覆盖最小成本。"""
    targets = perc["targets"]
    axes = perc["axes"]
    movable = perc["movable"]
    if perc["n_axes"] != 1 or not movable or perc["atype"] == '?':
        return None
    ax = axes[0]
    atype = ax["type"]
    ax_idx = next((s["idx"] for s in perc["switch_order"] if s["is_axis"]), None)
    n_sw = perc["n_switch"]
    sel0 = perc["sel_idx"]
    print(f"  [R3 目标分解] 轴类型={atype} 轴初始=({ax['x']},{ax['y']}) "
          f"拼块数={len(movable)} 目标={len(targets)}")
    best = None
    axis_range = range(0, BOARD)
    for ap in axis_range:
        all_pos = [enum_useful_positions(s, ap, targets, atype) for s in movable]
        if any(not p for p in all_pos):
            continue
        total_combos = 1
        for p in all_pos:
            total_combos *= len(p)
        if total_combos == 0:
            continue
        lists = all_pos
        if total_combos > 500000:
            lists = [sorted(p, key=lambda x: -len(x[2]))[:30] for p in all_pos]
        for combo in itertools.product(*lists):
            all_cov = set()
            for _, _, cov in combo:
                all_cov |= cov
            if all_cov == targets:
                cost = _compute_cost(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw)
                if best is None or cost < best[0]:
                    best = (cost, ap, combo)
    if best is None:
        return None
    cost, ap, combo = best
    print(f"  [R3 目标分解] ✓ 全覆盖! 轴→{ap} cost={cost}")
    return _generate_actions(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw)

def _compute_cost(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw):
    if atype == 'h':
        axis_cost = abs(ap - ax["y"])
    else:
        axis_cost = abs(ap - ax["x"])
    spr_cost = sum(abs(ox - s["x"]) + abs(oy - s["y"]) for s, (ox, oy, _) in zip(movable, combo))
    cur = sel0
    switch_cost = 0
    if ax_idx is not None:
        switch_cost += (ax_idx - cur) % n_sw
        cur = ax_idx
    for s in movable:
        switch_cost += (s["idx"] - cur) % n_sw
        cur = s["idx"]
    return axis_cost + spr_cost + switch_cost

def _generate_actions(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw):
    actions = []
    cur = sel0
    if ax_idx is not None:
        for _ in range((ax_idx - cur) % n_sw):
            actions.append(5)
        cur = ax_idx
        if atype == 'h':
            dy = ap - ax["y"]
            actions.extend([2] * dy if dy > 0 else [1] * (-dy))
        else:
            dx = ap - ax["x"]
            actions.extend([4] * dx if dx > 0 else [3] * (-dx))
    for s, (ox, oy, _) in zip(movable, combo):
        for _ in range((s["idx"] - cur) % n_sw):
            actions.append(5)
        cur = s["idx"]
        dx, dy = ox - s["x"], oy - s["y"]
        actions.extend([4] * dx if dx > 0 else [3] * (-dx))
        actions.extend([2] * dy if dy > 0 else [1] * (-dy))
    return actions

# ═══════════════════════════════════════════════════════════
# R3-A2: 双轴目标分解 (L6+ 的 h+v 双镜像轴关卡)
# ═══════════════════════════════════════════════════════════

def _dual_reflect_chain(cells, vx, hy, max_depth=12):
    """双轴反射闭包: 与 ar25.py skqtojxvbv 相同的 BFS 级联 (深度≤12)。"""
    seen = set(cells)
    stack = [(x, y, 0) for x, y in cells]
    while stack:
        x, y, d = stack.pop()
        if d >= max_depth:
            continue
        for nx, ny in ((2 * vx - x, y), (x, 2 * hy - y)):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny, d + 1))
    return seen

def _dual_enum_positions(sprite, vx, hy, targets):
    """枚举拼块平移位置, 返回 {覆盖集: (ox, oy, 距离)} 的去重字典。"""
    cells = sprite["cells"]
    sx, sy = sprite["x"], sprite["y"]
    cand = {}
    for tx, ty in targets:
        for cx, cy in cells:
            ox, oy = tx - cx, ty - cy
            abs_cells = {(ox + dx, oy + dy) for dx, dy in cells}
            cov = frozenset(_dual_reflect_chain(abs_cells, vx, hy) & targets)
            if not cov:
                continue
            d = abs(ox - sx) + abs(oy - sy)
            if cov not in cand or d < cand[cov][2]:
                cand[cov] = (ox, oy, d)
    return cand

def solve_goal_decomp_dual(g, perc):
    """R3 双轴目标分解: 枚举 (h_y, v_x) × 拼块位置, 找全覆盖组合。

    覆盖判定与引擎 skqtojxvbv 的反射级联一致;
    拼块拼接用按覆盖大小的贪心, 组合数超阈值时截断。
    """
    if perc["n_axes"] != 2:
        return None
    h_axis = next((a for a in perc["axes"] if a["type"] == 'h'), None)
    v_axis = next((a for a in perc["axes"] if a["type"] == 'v'), None)
    movable = perc["movable"]
    if h_axis is None or v_axis is None or not movable:
        return None
    targets = perc["targets"]
    h_idx = next((s["idx"] for s in perc["switch_order"] if s["is_axis"] and s["x"] == h_axis["x"] and s["y"] == h_axis["y"]), None)
    v_idx = next((s["idx"] for s in perc["switch_order"] if s["is_axis"] and s["x"] == v_axis["x"] and s["y"] == v_axis["y"]), None)
    n_sw = perc["n_switch"]
    sel0 = perc["sel_idx"]
    budget = perc["budget"]
    print(f"  [R3 双轴分解] h@({h_axis['x']},{h_axis['y']}) v@({v_axis['x']},{v_axis['y']}) "
          f"拼块数={len(movable)} 目标={len(targets)}")
    best = None
    for hy in range(BOARD):
        for vx in range(BOARD):
            cand_list = [_dual_enum_positions(s, vx, hy, targets) for s in movable]
            if any(not c for c in cand_list):
                continue
            axis_cost = abs(hy - h_axis["y"]) + abs(vx - v_axis["x"])
            # 贪心拼接: 每块选剩余覆盖最大的放置
            uncovered = set(targets)
            picked = []
            spr_cost = 0
            feasible = True
            for s, cand in zip(movable, cand_list):
                ranked = sorted(cand.items(), key=lambda kv: -len(kv[0] & uncovered))
                cov, (ox, oy, d) = ranked[0]
                uncovered -= cov
                picked.append((s["idx"], ox, oy))
                spr_cost += d
            if uncovered:
                continue
            # 切换成本: 两轴 + 拼块按序
            order = []
            if h_idx is not None:
                order.append(h_idx)
            if v_idx is not None:
                order.append(v_idx)
            order.extend(p[0] for p in picked)
            switch_cost = 0
            cur = sel0
            for idx in order:
                switch_cost += (idx - cur) % n_sw
                cur = idx
            cost = axis_cost + spr_cost + switch_cost
            if best is None or cost < best[0]:
                best = (cost, hy, vx, picked, h_idx, v_idx)
    if best is None:
        return None
    cost, hy, vx, picked, h_idx, v_idx = best
    print(f"  [R3 双轴分解] ✓ 全覆盖! h_y→{hy} v_x→{vx} cost={cost} picks={picked}")
    return _dual_generate_actions(h_axis, v_axis, hy, vx, picked, h_idx, v_idx,
                                  perc, n_sw, sel0)

def _dual_generate_actions(h_axis, v_axis, hy, vx, picked, h_idx, v_idx,
                           perc, n_sw, sel0):
    actions = []
    cur = sel0
    for idx, target_y, sprite in ((h_idx, hy, h_axis), (v_idx, vx, v_axis)):
        if idx is None:
            continue
        for _ in range((idx - cur) % n_sw):
            actions.append(5)
        cur = idx
        if sprite is h_axis:
            dy = target_y - h_axis["y"]
            actions.extend([2] * dy if dy > 0 else [1] * (-dy))
        else:
            dx = vx - v_axis["x"]
            actions.extend([4] * dx if dx > 0 else [3] * (-dx))
    by_idx = {s["idx"]: s for s in perc["movable"]}
    for s_idx, ox, oy in picked:
        for _ in range((s_idx - cur) % n_sw):
            actions.append(5)
        cur = s_idx
        s = by_idx[s_idx]
        dx, dy = ox - s["x"], oy - s["y"]
        actions.extend([4] * dx if dx > 0 else [3] * (-dx))
        actions.extend([2] * dy if dy > 0 else [1] * (-dy))
    return actions

# ═══════════════════════════════════════════════════════════
# R3-B: 状态空间搜索
# ═══════════════════════════════════════════════════════════

def r3_state_search(env, t_limit=60, max_nodes=200000):
    g = env._game
    if is_won(g):
        return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try:
        g.next_level = lambda: None
    except:
        pass
    try:
        seq = 0
        heap = [(int(_heuristic(g)), 0, 0, seq, snap0, [])]
        seen = {_state_key(g)}
        t0 = time.time()
        while heap:
            if time.time() - t0 > t_limit or seq > max_nodes:
                return None
            h, _, _, _, snap, path = heapq.heappop(heap)
            mx = int(g.lelsvjlwneo.ilqnjlrnkk) - 1
            if len(path) >= mx:
                continue
            for a in [1, 2, 3, 4, 5]:
                _restore(g, snap)
                try:
                    g.perform_action(ActionInput(id=ACT_MAP[a], data={}, reasoning=None), raw=True)
                except:
                    continue
                if int(g._current_level_index) > li0 or is_won(g):
                    return path + [a]
                if g._state == GameState.GAME_OVER:
                    continue
                k = _state_key(g)
                if k in seen:
                    continue
                seen.add(k)
                seq += 1
                heapq.heappush(heap, (int(_heuristic(g)) * 10 + len(path) + 1, len(path) + 1, 0, seq, _snapshot(g), path + [a]))
        return None
    finally:
        try:
            del g.next_level
        except:
            pass
        _restore(g, snap0)

# ═══════════════════════════════════════════════════════════
# R3-C: Beam Search
# ═══════════════════════════════════════════════════════════

def r3_beam_search(env, t_limit=300, max_nodes=3000000, beam_width=8):
    g = env._game
    if is_won(g):
        return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try:
        g.next_level = lambda: None
    except:
        pass
    try:
        max_steps = int(g.lelsvjlwneo.ilqnjlrnkk) - 1
        seq = 0
        beam = [(int(_heuristic(g)) * 10, 0, seq, snap0, [])]
        seen = {_state_key(g)}
        nodes = 0
        best_sol = None
        best_len = 999
        t0 = time.time()
        while beam and time.time() - t0 < t_limit and nodes < max_nodes:
            next_beam = []
            for _ in range(min(beam_width, len(beam))):
                if not beam:
                    break
                f, depth, _, snap, path = heapq.heappop(beam)
                if depth >= max_steps or depth >= best_len:
                    continue
                for a in [1, 2, 3, 4, 5]:
                    _restore(g, snap)
                    try:
                        g.perform_action(ActionInput(id=ACT_MAP[a], data={}, reasoning=None), raw=True)
                    except:
                        continue
                    if int(g._current_level_index) > li0 or is_won(g):
                        result = path + [a]
                        if len(result) < best_len:
                            best_len = len(result)
                            best_sol = result
                            print(f"    [beam] 🎯 {best_len}步 ({time.time()-t0:.0f}s, {nodes}节点)")
                        continue
                    if g._state == GameState.GAME_OVER:
                        continue
                    k = _state_key(g)
                    if k in seen:
                        continue
                    seen.add(k)
                    h_val = int(_heuristic(g))
                    seq += 1
                    nodes += 1
                    next_beam.append((h_val * 10 + depth + 1, depth + 1, seq, _snapshot(g), path + [a]))
            for item in next_beam:
                heapq.heappush(beam, item)
            if len(beam) > beam_width * 4:
                beam = heapq.nsmallest(beam_width * 2, beam)
        return best_sol
    finally:
        try:
            del g.next_level
        except:
            pass
        _restore(g, snap0)

# ═══════════════════════════════════════════════════════════
# R4: 剪枝贪心步进 (动作效果反馈 + 覆盖进度 + 反循环)
# ═══════════════════════════════════════════════════════════

class R4Scorer:
    def __init__(self):
        self.stats = {a: {"uses": 0, "eff": 0, "ineff": 0} for a in [1, 2, 3, 4, 5]}
        self.visited = set()
        self.recent = []

    def score(self, action, perc, prev_perc):
        s = self.stats[action]
        total = s["eff"] + s["ineff"]
        success_rate = s["eff"] / total if total > 0 else 0.5
        cov_gain = 0.0
        if prev_perc:
            cov_gain = (perc["covered"] - prev_perc["covered"]) * 5.0
        goal_prox = 0.0
        if perc["movable"] and perc["covered"] < perc["total_targets"]:
            sel = perc["sel_idx"]
            movable = perc["movable"]
            if 0 <= sel < len(movable) and perc["targets"]:
                sx, sy = movable[sel]["x"], movable[sel]["y"]
                targets = list(perc["targets"])
                cx = sum(t[0] for t in targets) / len(targets)
                cy = sum(t[1] for t in targets) / len(targets)
                dx, dy = 0, 0
                if action == 1: dy = -1
                elif action == 2: dy = 1
                elif action == 3: dx = -1
                elif action == 4: dx = 1
                old_d = abs(sx - cx) + abs(sy - cy)
                new_d = abs(sx + dx - cx) + abs(sy + dy - cy)
                goal_prox = (old_d - new_d) * 0.3
        loop = self.recent[-5:].count(action) * 0.8 if len(self.recent) >= 5 else 0
        explore = 1.0 / (1 + s["uses"])
        return 2.0 * success_rate + cov_gain + goal_prox - loop + 0.3 * explore

    def choose(self, perc, prev_perc):
        cands = [(a, self.score(a, perc, prev_perc)) for a in [1, 2, 3, 4, 5]]
        cands.sort(key=lambda x: -x[1])
        return cands[0][0]

    def update(self, action, prev_perc, new_perc):
        eff = False
        if prev_perc and new_perc:
            if new_perc["covered"] > prev_perc["covered"]:
                eff = True
            if prev_perc["state_key"] != new_perc["state_key"]:
                eff = True
        self.stats[action]["eff" if eff else "ineff"] += 1
        self.stats[action]["uses"] += 1
        self.recent.append(action)
        self.visited.add(new_perc["state_key"])

def r4_greedy_play(env, t_limit=60, max_steps=200):
    """R4 贪心步进: 逐动作评分选择, 直到通关或超时。失败时恢复原状态。"""
    g = env._game
    if is_won(g):
        return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    scorer = R4Scorer()
    prev = r2_perceive(g)
    path = []
    t0 = time.time()
    while time.time() - t0 < t_limit and len(path) < max_steps:
        perc = r2_perceive(g)
        if perc["won"] or int(g._current_level_index) > li0:
            return path
        action = scorer.choose(perc, prev)
        try:
            g.perform_action(ActionInput(id=ACT_MAP[action], data={}, reasoning=None), raw=True)
        except:
            continue
        path.append(action)
        new_perc = r2_perceive(g)
        scorer.update(action, prev, new_perc)
        prev = new_perc
        if new_perc["won"] or int(g._current_level_index) > li0:
            return path
        if g._state == GameState.GAME_OVER:
            break
    _restore(g, snap0)
    return None

# ═══════════════════════════════════════════════════════════
# R2+R3+R4 组合入口
# ═══════════════════════════════════════════════════════════

def r234_solve(env, t_limit=60, level_idx=0):
    g = env._game
    li0 = int(g._current_level_index)

    # R2 感知
    perc = r2_perceive(g)
    print(f"  [R2] L{level_idx+1} 轴={perc['n_axes']}({perc['atype']}) "
          f"拼块={len(perc['movable'])} 目标={perc['total_targets']} "
          f"覆盖={perc['covered']}/{perc['total_targets']} "
          f"预算={perc['budget']}")

    # 1a. R3 目标分解 (单轴关卡)
    if perc["n_axes"] == 1 and perc["atype"] != '?':
        path = solve_goal_decomp(g, perc)
        if path is not None:
            snap = _snapshot(g)
            try:
                for a in path:
                    g.perform_action(ActionInput(id=ACT_MAP[a], data={}, reasoning=None), raw=True)
                if int(g._current_level_index) > li0 or is_won(g):
                    return path, "R3-目标分解"
                print(f"  [R3-目标分解] 验证失败({len(path)}步未通关)")
            except Exception as e:
                print(f"  [R3-目标分解] 执行异常: {e}")
            _restore(g, snap)

    # 1b. R3 双轴目标分解 (L6+ 的 h+v 关卡)
    if perc["n_axes"] == 2:
        path = solve_goal_decomp_dual(g, perc)
        if path is not None:
            snap = _snapshot(g)
            try:
                for a in path:
                    g.perform_action(ActionInput(id=ACT_MAP[a], data={}, reasoning=None), raw=True)
                if int(g._current_level_index) > li0 or is_won(g):
                    return path, "R3-双轴分解"
                print(f"  [R3-双轴分解] 验证失败({len(path)}步未通关)")
            except Exception as e:
                print(f"  [R3-双轴分解] 执行异常: {e}")
            _restore(g, snap)

    # 2. R3 arc_shadow (L6+)
    if level_idx >= 5:
        import arc_shadow
        arc_shadow._env = env
        arc_shadow._game = g
        arc_shadow._valid = True
        arc_shadow._num2act = ACT_MAP
        arc_shadow._act2num = {v: k for k, v in ACT_MAP.items()}
        arc_shadow._ActionInput = ActionInput
        arc_shadow._GameState = GameState
        arc_shadow._ACTS = [ACT_MAP[n] for n in [1, 2, 3, 4, 5]]
        print(f"  [R3-arc_shadow] L{level_idx+1} (t_limit=180s, max_nodes=600000)")
        path = arc_shadow.solve_level(t_limit=180.0, max_nodes=600000)
        if path is not None:
            return path, "R3-arc_shadow"

    # 3. R3 beam search
    beam_t = min(60, t_limit)
    print(f"  [R3-beam] 开始 (t_limit={beam_t}s, beam=8)")
    path = r3_beam_search(env, t_limit=beam_t, beam_width=8)
    if path is not None:
        return path, "R3-beam"

    # 4. R3 状态搜索
    search_t = min(30, t_limit)
    print(f"  [R3-搜索] 开始 (t_limit={search_t}s)")
    path = r3_state_search(env, t_limit=search_t)
    if path is not None:
        return path, "R3-搜索"

    # 5. R4 贪心步进 (最终回退)
    r4_t = min(30, t_limit)
    print(f"  [R4-贪心] 开始 (t_limit={r4_t}s)")
    path = r4_greedy_play(env, t_limit=r4_t)
    if path is not None:
        return path, "R4-贪心"

    return None, "失败"

# ═══════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════

def main():
    MAX_LEVELS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    arcade = Arcade(environments_dir=r"F:/pro/Lingjing-Solo-/environment_files",
                    operation_mode=OperationMode.OFFLINE)
    gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ar25")][0]
    print(f"game_id = {gid}")
    env = arcade.make(gid)
    env.reset()
    g = env._game

    print(f"\nAR25 闯关 (R2+R3+R4: 感知→目标分解→beam→搜索→贪心)")
    print(f"{'关':>3} {'方法':<16} {'步数':>4} {'耗时':>8} {'结果'}")
    print("-" * 55)

    total = 0
    for level in range(MAX_LEVELS):
        if int(g._current_level_index) < level:
            print(f"{level+1:3d} 未能进入")
            break
        t0 = time.time()
        result, method = r234_solve(env, t_limit=60, level_idx=level)
        elapsed = time.time() - t0
        if result is None:
            print(f"{level+1:3d} {method:<16} {'-':>4} {elapsed:7.1f}s 失败")
            break
        li0 = int(g._current_level_index)
        if int(g._current_level_index) == li0 and not is_won(g):
            for a in result:
                g.perform_action(ActionInput(id=ACT_MAP[a], data={}, reasoning=None), raw=True)
        total += len(result)
        ok = int(g._current_level_index) > level or is_won(g)
        print(f"{level+1:3d} {method:<16} {len(result):4d} {elapsed:7.1f}s {'✓通关' if ok else '✗未通关'}")
        if not ok:
            break
    print(f"\n总计: {total} 步, 到达 L{int(g._current_level_index)+1}")

if __name__ == '__main__':
    main()