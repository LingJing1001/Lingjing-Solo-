"""R3 重写: 目标分解搜索 + 状态空间搜索回退，跑 AR25 闯关。

R3 方法选择:
  1. 目标分解 (参考 ar25_l3_solver.py): 枚举轴位置×拼块位置组合, 检查全覆盖
     - 适合单轴关卡 (L1-L5), 秒级求解
  2. 状态空间搜索 (贪心最佳优先): 快照/恢复枚举动作序列
     - 回退方案, 适合复杂关卡或目标分解验证失败时
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
H_AXIS_TAG = "0002nuguepuujf"  # 横轴: y'=2*ay-y
V_AXIS_TAG = "0054kgxrvfihgm"  # 竖轴: x'=2*ax-x
BOARD = 21

def is_won(g):
    w = g.vplrhaovhr()
    return w is True or (hasattr(w, '__len__') and len(w) == 1 and bool(w))

def get_cells(obj):
    h, w = obj.pixels.shape
    return [(j, i) for i in range(h) for j in range(w) if obj.pixels[i, j] != -1]

# ═══════════════════════════════════════════════════════════
# 关卡数据提取
# ═══════════════════════════════════════════════════════════

def extract_level(g):
    """提取轴/拼块/目标/切换顺序。"""
    axis_ids = set(id(a) for a in g.jtkyjqznbnp)
    # 轴
    axes = []
    for ax in g.jtkyjqznbnp:
        tags = list(ax.tags) if hasattr(ax, "tags") else []
        atype = 'h' if H_AXIS_TAG in tags else ('v' if V_AXIS_TAG in tags else '?')
        axes.append({"x": int(ax.x), "y": int(ax.y), "type": atype, "cells": get_cells(ax)})
    # 可切换对象 (ayyvxqrhnzw), 标记是否是轴
    switch_order = []
    for idx, obj in enumerate(g.ayyvxqrhnzw):
        is_axis = id(obj) in axis_ids
        switch_order.append({"idx": idx, "is_axis": is_axis,
                             "x": int(obj.x), "y": int(obj.y),
                             "cells": get_cells(obj) if not is_axis else None})
    sel_idx = g.ayyvxqrhnzw.index(g.yvifanjrcyu) if g.yvifanjrcyu in g.ayyvxqrhnzw else 0
    targets = {(int(t.x), int(t.y)) for t in g.fswikrcrdmx}
    budget = int(g.lelsvjlwneo.ilqnjlrnkk)
    return {"axes": axes, "switch_order": switch_order, "sel_idx": sel_idx,
            "targets": targets, "budget": budget, "n_switch": len(switch_order)}

# ═══════════════════════════════════════════════════════════
# R3-A: 目标分解搜索 (参考 ar25_l3_solver.py)
# ═══════════════════════════════════════════════════════════

def enum_useful_positions(sprite, axis_pos, targets, atype):
    """枚举拼块在轴位置 axis_pos 下覆盖≥1个目标的位置。
    返回 [(ox, oy, frozenset(覆盖的目标))]"""
    cells = sprite["cells"]
    sx, sy = sprite["x"], sprite["y"]
    positions = set()
    for tx, ty in targets:
        for cx, cy in cells:
            # 直接覆盖: 拼块格(cx,cy)移到目标(tx,ty)
            positions.add((tx - cx, ty - cy))
            # 反射覆盖
            if atype == 'h':  # 横轴 y'=2*ay-y
                positions.add((tx - cx, 2 * axis_pos - ty - cy))
            else:  # 竖轴 x'=2*ax-x
                positions.add((2 * axis_pos - tx - cx, ty - cy))
    # 按覆盖集去重, 保留最近位置
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

def solve_goal_decomp(g):
    """目标分解: 枚举轴位置×拼块位置组合, 找全覆盖最小成本。返回动作序列或 None。"""
    lv = extract_level(g)
    targets = lv["targets"]
    axes = lv["axes"]
    if len(axes) != 1: return None  # 只处理单轴
    ax = axes[0]
    atype = ax["type"]
    if atype == '?': return None

    # 拼块 = switch_order 中非轴的
    movable = [s for s in lv["switch_order"] if not s["is_axis"]]
    if not movable: return None

    ax_idx = next((s["idx"] for s in lv["switch_order"] if s["is_axis"]), None)
    n_sw = lv["n_switch"]
    sel0 = lv["sel_idx"]

    print(f"  [目标分解] 轴类型={atype} 轴初始=({ax['x']},{ax['y']}) 拼块数={len(movable)} 目标={len(targets)}")

    best = None  # (cost, axis_pos, combo)
    # 枚举轴位置
    axis_range = range(0, BOARD) if atype == 'h' else range(0, BOARD)
    for ap in axis_range:
        all_pos = [enum_useful_positions(s, ap, targets, atype) for s in movable]
        if any(not p for p in all_pos): continue
        total_combos = 1
        for p in all_pos: total_combos *= len(p)
        if total_combos == 0: continue
        # 组合数太大时贪心剪枝: 每个拼块取覆盖最多的前30
        lists = all_pos
        if total_combos > 500000:
            lists = [sorted(p, key=lambda x: -len(x[2]))[:30] for p in all_pos]
        for combo in itertools.product(*lists):
            all_cov = set()
            for _, _, cov in combo: all_cov |= cov
            if all_cov == targets:
                cost = compute_cost(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw)
                if best is None or cost < best[0]:
                    best = (cost, ap, combo)

    if best is None: return None
    cost, ap, combo = best
    print(f"  [目标分解] ✓ 全覆盖! 轴→{ap} cost={cost}")
    return generate_actions(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw)

def compute_cost(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw):
    """总成本 = 轴移动 + Σ拼块移动 + 切换次数"""
    if atype == 'h':
        axis_cost = abs(ap - ax["y"])
    else:
        axis_cost = abs(ap - ax["x"])
    spr_cost = sum(abs(ox - s["x"]) + abs(oy - s["y"]) for s, (ox, oy, _) in zip(movable, combo))
    # 切换: 选中轴(1次) + 依次选中各拼块
    cur = sel0
    switch_cost = 0
    if ax_idx is not None:
        switch_cost += (ax_idx - cur) % n_sw; cur = ax_idx
    for s in movable:
        switch_cost += (s["idx"] - cur) % n_sw; cur = s["idx"]
    return axis_cost + spr_cost + switch_cost

def generate_actions(ax, ap, movable, combo, atype, ax_idx, sel0, n_sw):
    """生成动作序列: 移动轴 → 切换到各拼块 → 移动拼块"""
    actions = []
    cur = sel0
    # 1. 选中轴并移动
    if ax_idx is not None:
        for _ in range((ax_idx - cur) % n_sw): actions.append(5)
        cur = ax_idx
        if atype == 'h':
            dy = ap - ax["y"]
            actions.extend([2] * dy if dy > 0 else [1] * (-dy))  # DOWN=y+1, UP=y-1
        else:
            dx = ap - ax["x"]
            actions.extend([4] * dx if dx > 0 else [3] * (-dx))  # RIGHT=x+1, LEFT=x-1
    # 2. 依次选中拼块并移动
    for s, (ox, oy, _) in zip(movable, combo):
        for _ in range((s["idx"] - cur) % n_sw): actions.append(5)
        cur = s["idx"]
        dx, dy = ox - s["x"], oy - s["y"]
        actions.extend([4] * dx if dx > 0 else [3] * (-dx))
        actions.extend([2] * dy if dy > 0 else [1] * (-dy))
    return actions

# ═══════════════════════════════════════════════════════════
# R3-B: 状态空间搜索 (回退)
# ═══════════════════════════════════════════════════════════

def r3_state_search(env, t_limit=60, max_nodes=200000):
    g = env._game
    if is_won(g): return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try: g.next_level = lambda: None
    except: pass
    try:
        seq=0; heap=[(int(_heuristic(g)),0,0,seq,snap0,[])]; seen={_state_key(g)}; t0=time.time()
        while heap:
            if time.time()-t0 > t_limit or seq > max_nodes: return None
            h,_,_,_,snap,path = heapq.heappop(heap)
            mx = int(g.lelsvjlwneo.ilqnjlrnkk)-1
            if len(path) >= mx: continue
            for a in [1,2,3,4,5]:
                _restore(g, snap)
                try: g.perform_action(ActionInput(id=ACT_MAP[a],data={},reasoning=None),raw=True)
                except: continue
                if int(g._current_level_index) > li0 or is_won(g): return path+[a]
                if g._state == GameState.GAME_OVER: continue
                k = _state_key(g)
                if k in seen: continue
                seen.add(k); seq+=1
                heapq.heappush(heap,(int(_heuristic(g))*10+len(path)+1,len(path)+1,0,seq,_snapshot(g),path+[a]))
        return None
    finally:
        try: del g.next_level
        except: pass
        _restore(g, snap0)

# ═══════════════════════════════════════════════════════════
# R3-C: Beam Search (参考 optimize_ar25_l6.py, 适合 L6+)
# ═══════════════════════════════════════════════════════════

def r3_beam_search(env, t_limit=300, max_nodes=3000000, beam_width=8):
    """Beam search: 保留 top-K 状态并行搜索, 找到解后继续找更短解。"""
    g = env._game
    if is_won(g): return []
    li0 = int(g._current_level_index)
    snap0 = _snapshot(g)
    try: g.next_level = lambda: None
    except: pass
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
                if not beam: break
                f, depth, _, snap, path = heapq.heappop(beam)
                if depth >= max_steps or depth >= best_len: continue
                for a in [1, 2, 3, 4, 5]:
                    _restore(g, snap)
                    try: g.perform_action(ActionInput(id=ACT_MAP[a], data={}, reasoning=None), raw=True)
                    except: continue
                    if int(g._current_level_index) > li0 or is_won(g):
                        result = path + [a]
                        if len(result) < best_len:
                            best_len = len(result)
                            best_sol = result
                            print(f"    [beam] 🎯 {best_len}步 ({time.time()-t0:.0f}s, {nodes}节点)")
                        continue
                    if g._state == GameState.GAME_OVER: continue
                    k = _state_key(g)
                    if k in seen: continue
                    seen.add(k)
                    h_val = int(_heuristic(g))
                    seq += 1; nodes += 1
                    next_beam.append((h_val * 10 + depth + 1, depth + 1, seq, _snapshot(g), path + [a]))
            for item in next_beam:
                heapq.heappush(beam, item)
            if len(beam) > beam_width * 4:
                beam = heapq.nsmallest(beam_width * 2, beam)
        return best_sol
    finally:
        try: del g.next_level
        except: pass
        _restore(g, snap0)

# ═══════════════════════════════════════════════════════════
# R3 组合: 目标分解 → beam search → 状态搜索
# ═══════════════════════════════════════════════════════════

def r3_solve(env, t_limit=60, level_idx=0):
    g = env._game
    li0 = int(g._current_level_index)
    # 1. 目标分解
    path = solve_goal_decomp(g)
    if path is not None:
        snap = _snapshot(g)
        try:
            for a in path:
                g.perform_action(ActionInput(id=ACT_MAP[a],data={},reasoning=None),raw=True)
            if int(g._current_level_index) > li0 or is_won(g):
                return path, "目标分解"
            print(f"  [目标分解] 验证失败({len(path)}步未通关)")
        except Exception as e:
            print(f"  [目标分解] 执行异常: {e}")
        _restore(g, snap)
    # 2. L6+: 用 arc_shadow.solve_level (已验证能解 L6=197步)
    if level_idx >= 5:
        import arc_shadow
        arc_shadow._env = env; arc_shadow._game = g; arc_shadow._valid = True
        arc_shadow._num2act = ACT_MAP
        arc_shadow._act2num = {v: k for k, v in ACT_MAP.items()}
        arc_shadow._ActionInput = ActionInput; arc_shadow._GameState = GameState
        arc_shadow._ACTS = [ACT_MAP[n] for n in [1, 2, 3, 4, 5]]
        print(f"  [arc_shadow] L{level_idx+1} (t_limit=180s, max_nodes=600000)")
        path = arc_shadow.solve_level(t_limit=180.0, max_nodes=600000)
        if path is not None:
            return path, "arc_shadow"
    # 3. beam search (L1-L5 回退)
    beam_t = 60
    print(f"  [beam search] 开始 (t_limit={beam_t}s, beam=8)")
    path = r3_beam_search(env, t_limit=beam_t, beam_width=8)
    if path is not None:
        return path, "beam search"
    # 4. 状态搜索最终回退
    print(f"  [状态搜索] 开始 (t_limit={t_limit}s)")
    return r3_state_search(env, t_limit=t_limit), "状态搜索"

# ═══════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════

def main():
    MAX_LEVELS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    arcade = Arcade(environments_dir=r"F:/pro/Lingjing-Solo-/environment_files",
                    operation_mode=OperationMode.OFFLINE)
    gid = [e.game_id for e in arcade.get_environments() if e.game_id.startswith("ar25")][0]
    print(f"game_id = {gid}")
    env = arcade.make(gid); env.reset(); g = env._game

    print(f"\nAR25 闯关 (R3: 目标分解 + beam search + 状态搜索)")
    print(f"{'关':>3} {'方法':<12} {'步数':>4} {'耗时':>8} {'结果'}")
    print("-" * 55)

    total = 0
    for level in range(MAX_LEVELS):
        if int(g._current_level_index) < level:
            print(f"{level+1:3d} 未能进入"); break
        t0 = time.time()
        result = r3_solve(env, t_limit=60, level_idx=level)
        elapsed = time.time() - t0
        if result is None or result[0] is None:
            print(f"{level+1:3d} {'失败':<12} {'-':>4} {elapsed:7.1f}s 超时"); break
        path, method = result
        # 执行 (目标分解验证通过时已执行, 否则需执行)
        li0 = int(g._current_level_index)
        if int(g._current_level_index) == li0 and not is_won(g):
            for a in path: g.perform_action(ActionInput(id=ACT_MAP[a],data={},reasoning=None),raw=True)
        total += len(path)
        ok = int(g._current_level_index) > level or is_won(g)
        print(f"{level+1:3d} {method:<12} {len(path):4d} {elapsed:7.1f}s {'✓通关' if ok else '✗未通关'}")
        if not ok: break
    print(f"\n总计: {total} 步, 到达 L{int(g._current_level_index)+1}")

if __name__ == '__main__':
    main()
