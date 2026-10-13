# -*- coding: utf-8 -*-
"""SU15 通用求解器 v3：紧凑状态 A*（省内存）+ 引擎内转移 + 合并感知启发式

架构：
- 堆/seen 只存紧凑状态 (level, fruits, creatures, steps, penalty)，不存引擎
- 展开节点时 restore() 重建引擎（~5ms），候选间用 push快照/undo 转移（~1ms/次）
- 启发式：min_merges 合并下界 + 送货距离下界
"""
import sys, heapq, math, os, functools
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environment_files", "su15", "1944f8ab"))
import su15  # noqa: E402
from arcengine import GameAction, ActionInput, GameState  # noqa: E402

PLAY_Y_MIN, PLAY_Y_MAX = 10, 62
ATTR_R = 8
CREATURE_NAMES = ("0030xjmmfvfpqm", "0031xcwudgivus", "0032qekmtelwqi")
MAX_DISP = 11.0
DOWNGRADE_COST = 2.0


def make(level: int):
    g = su15.Su15()
    g.set_level(level)
    return g


def compact(g):
    fruits = tuple(sorted((g.kqywaxhmsb.get(s), s.x, s.y) for s in g.lkujttxgs))
    creatures = tuple(sorted((g.dfqhmningy(g.kcuphgwar.get(s)), s.x, s.y) for s in g.fezhhzhih))
    return (g.level_index, fruits, creatures, g.step_counter_ui.current_steps, g.step_counter_ui.penalty)


def restore(c):
    level, fruits, creatures, steps, penalty = c
    g = su15.Su15()
    g.set_level(level)
    for s in list(g.lkujttxgs):
        g.current_level.remove_sprite(s)
    g.lkujttxgs = []
    g.kqywaxhmsb = {}
    for s in list(g.fezhhzhih):
        g.current_level.remove_sprite(s)
    g.fezhhzhih = []
    g.kcuphgwar = {}
    for value, x, y in fruits:
        spr = su15.sprites[str(value)].clone()
        spr.set_position(x, y)
        g.current_level.add_sprite(spr)
        g.lkujttxgs.append(spr)
        g.kqywaxhmsb[spr] = value
    for tlevel, x, y in creatures:
        typename = g.wkwqpjaqpc(tlevel)
        spr = g.rhziiktjfu(typename).clone()
        spr.set_position(x, y)
        g.current_level.add_sprite(spr)
        g.fezhhzhih.append(spr)
        g.kcuphgwar[spr] = typename
    g.step_counter_ui.current_steps = steps
    g.step_counter_ui.penalty = penalty
    g.vsfwpngmx = False
    g.qygchysnh = "none"
    g.sqwdaqczb = False
    g.nscnqkkvg = []
    return g


def goal_boxes(g):
    return [(s.x, s.y, s.pixels.shape[1], s.pixels.shape[0]) for s in g.powykypsm]


def dist_point_bbox(px, py, x, y, w, h):
    nx = min(max(px, x), x + w - 1)
    ny = min(max(py, y), y + h - 1)
    return math.hypot(px - nx, py - ny)


def _fruit_reqs(g):
    data = g.dsqlbvwaj
    if data is None:
        return []
    items = data if isinstance(data[0], (list, tuple)) else [data]
    reqs = []
    for it in items:
        name, cnt = str(it[0]), int(it[1])
        if name in CREATURE_NAMES:
            continue
        reqs.append((int(name), cnt))
    return reqs


@functools.lru_cache(maxsize=None)
def min_merges(fr, req):
    cnt = Counter(fr)
    if all(cnt[v] >= c for v, c in req):
        return 0
    pot = sum(1 << v for v in fr)
    need = sum((1 << v) * c for v, c in req)
    if pot < need:
        return None
    best = None
    for w in sorted(set(fr)):
        if w >= 8 or cnt[w] < 2:
            continue
        lst = list(fr)
        lst.remove(w)
        lst.remove(w)
        lst.append(w + 1)
        lst.sort()
        r = min_merges(tuple(lst), req)
        if r is not None:
            cand = 1 + r
            best = cand if best is None else min(best, cand)
    return best


def _pair_dist_lb(g, weight_cluster=0.5):
    """聚簇项：下一步可合并（同值>=2 的最低值）的最近对距离 -> 定位下界"""
    from collections import defaultdict
    pos = defaultdict(list)
    for s in g.lkujttxgs:
        v = g.kqywaxhmsb.get(s, 0)
        pos[v].append((s.x + s.pixels.shape[1] // 2, s.y + s.pixels.shape[0] // 2))
    best_pair = 1e9
    for v in sorted(pos):
        lst = pos[v]
        if len(lst) < 2:
            continue
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                d = math.hypot(lst[i][0] - lst[j][0], lst[i][1] - lst[j][1])
                if d < best_pair:
                    best_pair = d
        break  # 只看最低的可合并值
    if best_pair == 1e9:
        return 0.0
    return weight_cluster * best_pair / 8.0


@functools.lru_cache(maxsize=None)
def _merge_work(fr, reqs):
    """贪心合并工作量（引导用）：反复合并最近同值对，估代价=定位点击+合并点击。
    需求满足后返回 (总代价, 最终果实列表)，供送货下界使用。
    fr: tuple[(value,x,y)], reqs: tuple[(value,count)]"""
    f = list(fr)
    total = 0
    finals = None
    for _ in range(len(f) + 1):
        cnt = Counter(v for v, _, _ in f)
        if all(cnt[v] >= c for v, c in reqs):
            finals = tuple(sorted(f))
            break
        # 找最低值中>=2个的最近对
        best = None
        for v in sorted(cnt):
            if cnt[v] < 2:
                continue
            group = [(x, y) for vv, x, y in f if vv == v]
            dmin = 1e18
            pi = pj = None
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    d = math.hypot(group[i][0] - group[j][0], group[i][1] - group[j][1])
                    if d < dmin:
                        dmin = d
                        pi, pj = group[i], group[j]
            best = (v, dmin, pi, pj)
            break
        if best is None:
            # 无同值对可合并且需求未满 -> 依赖怪物降级超值果实（每次降1级）
            hi = max(f, key=lambda t: t[0])
            hi_v, hi_x, hi_y = hi
            if hi_v <= 1:
                return (None, ())  # 无法再降级，死局
            total += DOWNGRADE_COST
            f.remove(hi)
            f.append((hi_v - 1, hi_x, hi_y))
            continue
        v, dmin, pi, pj = best
        pos_cost = max(0, math.ceil((dmin - 16.0) / 8.0))
        total += pos_cost + 1
        f.remove((v, pi[0], pi[1]))
        f.remove((v, pj[0], pj[1]))
        f.append((v + 1, (pi[0] + pj[0]) // 2, (pi[1] + pj[1]) // 2))
    return (total, finals)


MONSTER_MERGE_COST = 2.0


def _monster_work(g):
    """怪物目标工作量：(合并次数, 送货下界)。不可行返回 (None, 0)。
    同型怪物两两合并升级（0030→0031→0032），最终目标怪物需进区。"""
    data = g.dsqlbvwaj
    if data is None:
        return (0.0, 0.0)
    items = data if isinstance(data[0], (list, tuple)) else [data]
    reqs = {}
    for it in items:
        name, cnt = str(it[0]), int(it[1])
        if name in CREATURE_NAMES:
            reqs[g.dfqhmningy(name)] = int(cnt)
    if not reqs:
        return (0.0, 0.0)
    pool = Counter(g.dfqhmningy(g.kcuphgwar.get(s, su15.hytiizjifk)) for s in g.fezhhzhih)
    merges = 0
    for L in (1, 2, 3):
        have = pool.get(L, 0)
        need = reqs.get(L, 0)
        if have < need:
            return (None, 0.0)
        pairs = (have - need) // 2
        merges += pairs
        pool[L + 1] = pool.get(L + 1, 0) + pairs
    boxes = goal_boxes(g)
    if not boxes:
        return (float(merges), 0.0)
    d = 1e18
    for s in g.fezhhzhih:
        cx, cy = _center(s)
        d = min(d, min(dist_point_bbox(cx, cy, bx, by, bw, bh) for bx, by, bw, bh in boxes))
    return (float(merges), 0.0 if d == 1e18 else d / MAX_DISP)


def heuristic(g):
    reqs = tuple(_fruit_reqs(g))
    # 用果实中心坐标（与引擎进区判定一致）
    fr = tuple(sorted((g.kqywaxhmsb.get(s, 0), s.x + s.pixels.shape[1] // 2, s.y + s.pixels.shape[0] // 2)
                      for s in g.lkujttxgs))
    mw, finals = _merge_work(fr, reqs)
    if mw is None:
        return float("inf")
    mm, md = _monster_work(g)
    if mm is None:
        return float("inf")
    d = 0.0
    boxes = goal_boxes(g)
    if boxes and finals:
        # 送货下界：按贪心合并的最终落点，各需求果实到最近区
        for v, x, y in finals:
            d += min(dist_point_bbox(x, y, bx, by, bw, bh) for bx, by, bw, bh in boxes) / MAX_DISP
    return mw + d + MONSTER_MERGE_COST * mm + md


def _within(p, s):
    x, y = s.x, s.y
    w, h = s.pixels.shape[1], s.pixels.shape[0]
    return dist_point_bbox(p[0], p[1], x, y, w, h) <= ATTR_R


def _center(s):
    return (s.x + s.pixels.shape[1] // 2, s.y + s.pixels.shape[0] // 2)


def _in_bounds(px, py):
    return 0 <= px <= 63 and PLAY_Y_MIN <= py <= PLAY_Y_MAX


def candidate_clicks(g):
    """定向候选：朝同值伙伴移动 + 朝目标区移动 + 透镜合并点击（稀疏）"""
    pts = set()
    F = list(g.lkujttxgs)
    n = len(F)
    if n == 0:
        return pts
    vals = [g.kqywaxhmsb.get(s, 0) for s in F]
    centers = [_center(s) for s in F]
    zones = goal_boxes(g)
    zone_cs = [(z[0] + z[2] // 2, z[1] + z[3] // 2) for z in zones]

    # ---- 移动点击 ----
    for i in range(n):
        cx, cy = centers[i]
        # 找最近同值伙伴
        bd, bj = 1e9, None
        for j in range(n):
            if j == i or vals[j] != vals[i]:
                continue
            d = math.hypot(cx - centers[j][0], cy - centers[j][1])
            if d < bd:
                bd, bj = d, centers[j]
        targets = []
        if bj is not None and bd <= 48.0:
            # 朝伙伴移动 8px（若 d<=16 会变成合并点击，跳过，交给合并分支）
            if bd > 16.0:
                ux = (bj[0] - cx) / bd
                uy = (bj[1] - cy) / bd
                targets.append((round(cx + 8 * ux), round(cy + 8 * uy)))
        # 朝目标区移动：当前最大值果实，或孤立果实（对每个目标区）
        if zone_cs:
            v = vals[i]
            is_max = v == max(vals)
            isolated = bj is None
            if is_max or isolated:
                for zcx, zcy in zone_cs:
                    dx, dy = zcx - cx, zcy - cy
                    dd = math.hypot(dx, dy)
                    if dd > 1:
                        targets.append((round(cx + 8 * dx / dd), round(cy + 8 * dy / dd)))
        for tgt in targets:
            if not _in_bounds(*tgt):
                continue
            if not _within(tgt, F[i]):
                continue
            if any(j != i and _within(tgt, F[j]) for j in range(n)):
                continue  # 会吸到别的果实
            pts.add(tgt)
    # ---- 合并点击：同值对中心距<=16 才存在公共点击点 ----
    for i in range(n):
        for j in range(i + 1, n):
            if vals[i] != vals[j]:
                continue
            ax, ay = centers[i]
            bx, by = centers[j]
            d = math.hypot(bx - ax, by - ay)
            if d > 16.0:
                continue
            cand = []
            if d < 1e-6:
                cand.append((ax, ay))
            else:
                ux, uy = (bx - ax) / d, (by - ay) / d
                px, py = -uy, ux
                mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                half = math.sqrt(max(0.0, 64.0 - (d / 2.0) ** 2))
                cand.append((round(mx), round(my)))
                cand.append((round(mx + px * half), round(my + py * half)))
                cand.append((round(mx - px * half), round(my - py * half)))
                cand.append((round(mx + px * half * 0.5), round(my + py * half * 0.5)))
                cand.append((round(mx - px * half * 0.5), round(my - py * half * 0.5)))
            for qx, qy in cand:
                if not _in_bounds(qx, qy):
                    continue
                if not (_within((qx, qy), F[i]) and _within((qx, qy), F[j])):
                    continue
                if any(k != i and k != j and _within((qx, qy), F[k]) for k in range(n)):
                    continue
                pts.add((qx, qy))
    # ---- 怪物驱赶/牵引点击（不吸任何果实）----
    C = list(g.fezhhzhih)
    if C:
        for s in C:
            cx, cy = _center(s)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)):
                for step in (4, 8):
                    qx, qy = round(cx + dx * step), round(cy + dy * step)
                    if not _in_bounds(qx, qy):
                        continue
                    if not _within((qx, qy), s):
                        continue
                    if any(_within((qx, qy), f) for f in F):
                        continue  # 会吸到果实，弃
                    pts.add((qx, qy))
    # ---- 空闲点击：角落，不吸任何对象（让怪物自由追食）----
    for qx, qy in ((0, 10), (63, 10), (0, 62), (63, 62)):
        if any(_within((qx, qy), s) for s in F) or any(_within((qx, qy), s) for s in C):
            continue
        pts.add((qx, qy))
    return pts


def solve_level(level: int, max_nodes: int = 20000, weight: float = 1.3, verbose=False, time_budget=None):
    import time as _time
    t_start = _time.time()
    g0 = make(level)
    if not goal_boxes(g0):
        return []
    if g0.cbdhpcilgb():
        return []
    start = compact(g0)
    counter = 0
    open_heap = [(weight * heuristic(g0), 0, counter, (), start)]
    seen = {start: 0}
    nodes = 0
    best_h = 1e18

    while open_heap and nodes < max_nodes:
        if time_budget and _time.time() - t_start > time_budget:
            if verbose:
                print(f"  [A*] TIME OUT nodes={nodes}")
            return None
        f, g_cost, _, clicks, key = heapq.heappop(open_heap)
        if g_cost != seen.get(key):
            continue
        nodes += 1
        h = f - g_cost
        if h < best_h:
            best_h = h
            if verbose:
                print(f"  [A*] nodes={nodes} g={g_cost} h={h:.1f} steps={key[3]} fruits={key[1]}", flush=True)
        eng = restore(key)
        node_steps = key[3]
        node_penalty = key[4]
        if node_steps <= 0:
            continue
        for click in candidate_clicks(eng):
            eng.hlsimbcfgc()
            snap_before = len(eng.nscnqkkvg)
            before_lvl = eng.level_index
            eng.perform_action(
                ActionInput(id=GameAction.ACTION6, data={"x": click[0], "y": click[1]}, reasoning=None),
                raw=True,
            )
            if eng.level_index > before_lvl or eng._state == GameState.WIN:
                return list(clicks) + [click]
            nkey = compact(eng)
            child_h = heuristic(eng)
            # 还原引擎到本节点，继续下一个候选
            if len(eng.nscnqkkvg) < snap_before:
                # 坏合并内部已弹快照 -> 防御性全量恢复
                eng = restore(key)
            else:
                eng.vczehveskr()
            eng.step_counter_ui.current_steps = node_steps
            eng.step_counter_ui.penalty = node_penalty
            eng.nscnqkkvg.clear()
            ng = g_cost + 1
            if nkey[3] <= 0:
                continue  # 步数耗尽
            if nkey in seen and seen[nkey] <= ng:
                continue
            seen[nkey] = ng
            counter += 1
            heapq.heappush(open_heap, (ng + weight * child_h, ng, counter, tuple(list(clicks) + [click]), nkey))
    if verbose:
        print(f"  [A*] exhausted nodes={nodes}")
    return None


if __name__ == "__main__":
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sol = solve_level(lvl, verbose=True, time_budget=300)
    print(f"L{lvl+1}: {'NO SOLUTION' if sol is None else f'solution {len(sol)} clicks'}")
    if sol:
        for i, c in enumerate(sol):
            print(f"  {i+1:3d}. click {c}")
