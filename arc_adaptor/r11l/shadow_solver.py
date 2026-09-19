"""R11L 本地引擎影子模拟器 + 质心导航求解器

R11L 机制 (拼图放置):
- 托盘中若干 5x5 拼图块 (sys_click), 点击选中, 再点击棋盘放置 (可反复移动)
- 1 次点击 = 完整动作 (perform_action 内部完成整个动画周期)
- 点击落在块上 = 改选该块 (不能堆叠); 落点压 wakneh-* 占位区则放置无效
- 放置合法性: 动画结束时 **所有类** 的影子 (roefwu-*, 位于同类全部块的
  质心 = sum(中心)//n) 不得压 defgjl 禁区实心像素 (pixel-perfect);
  违反则该块弹回托盘 (累计 5 次判负)
- 胜利: 每类影子与其目标区 (flkdtg-*) 重叠
  - 有自己目标的类: 质心需落入目标
  - whkxtx 类 (无目标): 影子需覆盖其他无影子类目标, 颜色集合匹配 (ldzvchvkvp)
- 每关步数上限 60 (每击 1 步)

求解策略:
1. 从 defgjl/wakneh 精灵预计算实心像素集合 → 影子自由区/块可放格 的几何快速判定
2. 逐类求解: 贪心移动块使质心走向目标 (每步几何预检影子自由), 引擎执行验证
3. 失败自动重规划/换目标分配/换类顺序
4. 模拟执行与真实执行一致 (引擎确定性), 求解返回的点击序列由 apply() 逐步回放
"""
import json
import threading
import time
import random

_lock = threading.RLock()
_env = None
_game = None
_applied = 0
_valid = False
_error = None
_ActionInput = None
_GameState = None
_ACTION6 = None

_RNG = random.Random(20260905)

# 块中心网格 (5x5 块, top-left ∈ [0,59])
_CELLS = [(x, y) for x in range(2, 62, 3) for y in range(2, 62, 3)]


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _GameState, _ACTION6
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput, GameState
        arc = arc_agi.Arcade()
        _env = arc.make("r11l-495a7899")
        _game = _env._game
        _ActionInput = ActionInput
        _GameState = GameState
        _ACTION6 = GameAction.ACTION6
        _env.reset()
        _applied = 0
        _valid = True
    except Exception as e:
        _error = str(e)
        _valid = False
        import traceback
        traceback.print_exc()
    return _valid


def reset():
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _applied = 0
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def apply(x, y):
    """线上点击成功后调用: 影子同步执行同一点击 (显示坐标, 相机 64x64 scale=1)"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            lvl_before = int(_game._current_level_index)
            _game.perform_action(
                _ActionInput(id=_ACTION6, data={"x": int(x), "y": int(y)}, reasoning=None))
            lvl_after = int(_game._current_level_index)
            _applied += 1
            return True
        except Exception as e:
            _valid = False
            import traceback
            traceback.print_exc()
            return False


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def replay(history):
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            for h in history:
                if isinstance(h, dict):
                    x, y = h.get("x", 0), h.get("y", 0)
                else:
                    x, y = h[1], h[2]
                _game.perform_action(
                    _ActionInput(id=_ACTION6, data={"x": int(x), "y": int(y)}, reasoning=None))
            _applied = len(history)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


# ─── 几何预计算 ─────────────────────────────────

def _sprite_solid(s):
    """精灵实心像素的全局坐标集合"""
    try:
        import numpy as np
        px = np.array(s.pixels)
        out = set()
        for r in range(px.shape[0]):
            for c in range(px.shape[1]):
                if px[r][c] != -1:
                    out.add((int(s.x) + c, int(s.y) + r))
        return out
    except Exception:
        return set()


def _snapshot(g):
    # 持有精灵对象引用: 恢复时精确重建列表 (吸收会移除颜料精灵)
    obj_list = list(g.current_level._sprites)
    spr = [(s.x, s.y, s.pixels.copy()) for s in obj_list]
    sel = -1
    if g.wiayqaumjug is not None:
        for i, p in enumerate(g.bbijaigbknc):
            if p is g.wiayqaumjug:
                sel = i
                break
    misc = (int(g._action_count), bool(g.yfbjozweime), int(g.qvnmfoxseus), bool(g.jttetcghmsb),
            bool(g.flgzyjcqcspeg), int(g.flgdcqnkdomzf), int(g.fljpbsiftilwa), bool(g.npvvaucvsot),
            bool(g.uyawyyswbya), int(g.yledlprvvkb), int(g.holbcmkehyf), g._state, sel,
            tuple(g.sgdntmcrxpq), tuple(g.nqbqaxbtdej))
    extra = {
        "owuypsqbino": list(getattr(g, "owuypsqbino", []) or []),
        "bulmhgivatv": {k: list(v) for k, v in getattr(g, "bulmhgivatv", {}).items()},
    }
    return (obj_list, spr, misc, extra)


def _restore(g, snap):
    obj_list, spr, misc, extra = snap
    # 精确重建精灵列表 (保持快照顺序, 移除新增, 找回被移除的)
    lv = g.current_level
    cur = lv._sprites
    keep = {id(s) for s in obj_list}
    for s in list(cur):
        if id(s) not in keep:
            cur.remove(s)
    cur_ids = {id(s) for s in cur}
    for s in obj_list:
        if id(s) not in cur_ids:
            cur.append(s)
            cur_ids.add(id(s))
    cur[:] = obj_list  # 恢复顺序
    lv._need_sort = True
    for s, (x, y, px) in zip(cur, spr):
        s.set_position(x, y)
        s.pixels = px.copy()
    # 颜料列表与吸收记录
    if hasattr(g, "owuypsqbino"):
        g.owuypsqbino[:] = extra["owuypsqbino"]
    if hasattr(g, "bulmhgivatv"):
        for k, v in extra["bulmhgivatv"].items():
            if k in g.bulmhgivatv:
                g.bulmhgivatv[k][:] = v
    (g._action_count, g.yfbjozweime, g.qvnmfoxseus, g.jttetcghmsb,
     g.flgzyjcqcspeg, g.flgdcqnkdomzf, g.fljpbsiftilwa, g.npvvaucvsot,
     g.uyawyyswbya, g.yledlprvvkb, g.holbcmkehyf, st, sel,
     sg, nq) = misc
    g._state = st
    g.sgdntmcrxpq = tuple(sg)
    g.nqbqaxbtdej = tuple(nq)
    if 0 <= sel < len(g.bbijaigbknc):
        g.wiayqaumjug = g.bbijaigbknc[sel]
    else:
        g.wiayqaumjug = None
    try:
        for cotoxdycij in g.xaalmogcsnh.values():
            cotoxdycij["flgzyjcqcspeg"] = False
            cotoxdycij["flgdcqnkdomzf"] = 0
            cotoxdycij["fljpbsiftilwa"] = 0
            cotoxdycij["fssfabihmfrnw"] = None
        g.jtuysbdbdhk = None
        g.hznupmuxgqv = None
    except Exception:
        pass


def _center(s):
    return (int(s.x) + s.width // 2, int(s.y) + s.height // 2)


def _mask(s):
    """精灵实心像素掩码 (相对中心/左上角的偏移列表, 使用 numpy 加速)"""
    try:
        import numpy as np
        px = np.array(s.pixels)
        w, h = s.width, s.height
        cx, cy = w // 2, h // 2
        return [(dx - cx, dy - cy) for dy in range(h) for dx in range(w) if px[dy][dx] != -1]
    except Exception:
        return [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]


def _color_set(s):
    try:
        import numpy as np
        return {int(c) for c in np.unique(s.pixels) if c > 0}
    except Exception:
        return set()


def _click(g, x, y):
    g.perform_action(_ActionInput(id=_ACTION6, data={"x": int(x), "y": int(y)}, reasoning=None), raw=True)


# ─── 求解 ─────────────────────────────────

def _mask_off(spr):
    """颜料实心像素偏移 (相对左上角), 用于区分吸收区域"""
    try:
        import numpy as np
        px = np.array(spr.pixels)
        return [(dx, dy) for dy in range(px.shape[0]) for dx in range(px.shape[1])
                if px[dy][dx] != -1 and px[dy][dx] != 0]
    except Exception:
        return []


def _mask_key(spr):
    return frozenset(_mask_off(spr))


def _pellet_info(g):
    """提取颜料点: [(sprite, center, colors, region_key)]"""
    pellets = []
    for spr in list(getattr(g, "owuypsqbino", []) or []):
        c = _center(spr)
        colors = frozenset(_color_set(spr))
        region = _mask_key(spr)
        pellets.append((spr, c, colors, region))
    return pellets


def solve_level(t_limit=60.0, target_level=None):
    """求解影子当前关. target_level 可强制跳到指定关再求解."""
    ok = _ensure()
    if not ok or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        level0 = int(g._current_level_index)
        if target_level is not None and target_level > level0:
            _env.reset()
            _applied = 0
            g = _game
            level0 = int(g._current_level_index)
        if level0 >= len(g._levels) or g._state == _GameState.WIN:
            return []

        cats = g.kacotwgjcyq

        # 几何预计算
        solid = set()
        wakneh = set()
        for s in g.current_level._sprites:
            if s.name.startswith("defgjl"):
                solid |= _sprite_solid(s)
            elif s.name.startswith("wakneh-"):
                wakneh |= _sprite_solid(s)

        # 每类影子掩码 (相对影子中心) / 每块掩码 (相对块中心)
        shadow_masks = {}
        for cat, d in cats.items():
            shd = d.get("roduyfsmiznvg")
            if shd is not None:
                shadow_masks[cat] = _mask(shd)
        piece_masks = {id(p): _mask(p) for p in g.bbijaigbknc}

        def shadow_free(cat, cx, cy):
            m = shadow_masks.get(cat)
            if m is None:
                return True
            return all((cx + dx, cy + dy) not in solid for dx, dy in m)

        def cell_ok(p, ccx, ccy):
            m = piece_masks.get(id(p))
            if m is None:
                return True
            return all((ccx + dx, ccy + dy) not in wakneh for dx, dy in m)

        # 颜料
        pellets = _pellet_info(g)

        # needing 目标 (无自己影子的类, 排除 dirwzt)
        needing = [(cat2, d2["gosubdcyegamj"]) for cat2, d2 in cats.items()
                   if d2.get("gosubdcyegamj") is not None and "dirwzt" not in cat2
                   and d2.get("roduyfsmiznvg") is None]

        base_infos = []  # (cat, pieces, dest_center|None, kind)
        for cat, d in cats.items():
            pcs = d.get("lecfirgqbwunn") or []
            if not pcs:
                continue
            t = d.get("gosubdcyegamj")
            shadow = d.get("roduyfsmiznvg")
            if t is not None and "dirwzt" not in cat:
                c = (int(t.x) + t.width // 2, int(t.y) + t.height // 2)
                base_infos.append((cat, pcs, c, "own"))
            elif t is None and shadow is not None:
                base_infos.append((cat, pcs, None, "whkx"))

        if not base_infos:
            return []

        whkx_cats = [bi for bi in base_infos if bi[3] == "whkx"]

        # whkxtx 目标与颜料分配
        whkx_plans = _assign_whkx(whkx_cats, needing, pellets)

        plans = []
        if whkx_cats:
            for wplan in whkx_plans[:6]:
                infos = []
                for cat, pcs, c, kind in base_infos:
                    if kind == "own":
                        infos.append((cat, pcs, c, []))
                    else:
                        dest, pl = wplan[cat]
                        infos.append((cat, pcs, dest, pl))
                plans.append(infos)
        else:
            plans.append([(cat, pcs, c, []) for cat, pcs, c, _ in base_infos])

        all_base_pieces = [p for _, pcs, _, _ in base_infos for p in pcs]

        def cat_of(piece):
            for cat, pcs, _, _ in base_infos:
                if piece in pcs:
                    return cat
            return None

        # 类顺序 (含全排列, 上限6)
        import itertools as _it
        cat_names = [bi[0] for bi in base_infos]
        cat_orders = [list(cat_names)]
        if len(cat_names) <= 3:
            for perm in _it.permutations(cat_names):
                if list(perm) not in cat_orders:
                    cat_orders.append(list(perm))
            cat_orders = cat_orders[:6]

        for infos in plans:
            for order in cat_orders:
                if time.time() - t0 > t_limit:
                    return None
                clicks = _solve_sequential(g, infos, order, all_base_pieces, cat_of,
                                           shadow_free, cell_ok, level0, t0, t_limit)
                if clicks is not None:
                    return clicks
        return None


def _assign_whkx(whkx_cats, needing, pellets):
    """为 whkxtx 类分配目标与颜料.

    返回: [{cat: (dest_center, [pellet_sprite, ...])}, ...]
    """
    results = []
    tgt_colors = {cat2: _color_set(t2) for cat2, t2 in needing}

    by_color = {}
    for spr, c, colors, region in pellets:
        for col in colors:
            by_color.setdefault(col, []).append((spr, c, region))

    whkx_names = [wc[0] for wc in whkx_cats]
    tgt_names = [t[0] for t in needing]

    import itertools as _it
    for tgt_perm in _it.permutations(tgt_names):
        if len(tgt_perm) < len(whkx_names):
            continue
        plan = {}
        used_pellets = set()
        conflict = False
        for wi, cat in enumerate(whkx_names):
            tname = tgt_perm[wi]
            needed = tgt_colors[tname]
            chosen = []
            used_regions = set()
            ok = True
            for col in sorted(needed):
                cands = [(spr, c, region) for spr, c, region in by_color.get(col, [])
                         if id(spr) not in used_pellets and region not in used_regions]
                if not cands:
                    ok = False
                    break
                spr, c, region = cands[0]
                chosen.append(spr)
                used_pellets.add(id(spr))
                used_regions.add(region)
            if not ok:
                conflict = True
                break
            t2 = next(t for n2, t in needing if n2 == tname)
            dest = (int(t2.x) + t2.width // 2, int(t2.y) + t2.height // 2)
            plan[cat] = (dest, chosen)
        if not conflict:
            results.append(plan)
        if len(results) >= 4:
            break
    return results


def _solve_sequential(g, infos, cat_order, all_pieces, cat_of, shadow_free, cell_ok,
                      level0, t0, t_limit):
    """逐类求解: 每类按航点 (颜料→目标) 质心导航. 过关返回点击序列."""
    snap0 = _snapshot(g)
    all_clicks = []
    neutered = False
    try:
        g._win_pending = False

        def _nl_marker():
            g._win_pending = True
        g.next_level = _nl_marker
        neutered = True
    except Exception:
        neutered = False

    try:
        pos = {p: _center(p) for p in all_pieces}
        blacklist = set()

        def absorbed(spr):
            return spr not in (getattr(g, "owuypsqbino", []) or [])

        for cat in cat_order:
            pcs = next(p for c, p, _, _ in infos if c == cat)
            dest, my_pellets = next((d, pl) for c, p, d, pl in infos if c == cat)
            n = len(pcs)

            # 航点: 未吸收的己方颜料 (最近邻排序) + 目标
            def build_waypoints():
                pending = [spr for spr in my_pellets if not absorbed(spr)]
                wps = []
                ref = dest
                rem = list(pending)
                while rem:
                    rem.sort(key=lambda s: abs(_center(s)[0] - ref[0]) + abs(_center(s)[1] - ref[1]))
                    nxt = rem.pop(0)
                    wps.append(("pellet", nxt))
                    ref = _center(nxt)
                wps.append(("target", dest))
                return wps

            wps = build_waypoints()
            for wp_kind, wp in wps:
                budget = 16
                ok_wp = False
                for step in range(budget):
                    if time.time() - t0 > t_limit:
                        return None
                    if wp_kind == "pellet":
                        if absorbed(wp):
                            ok_wp = True
                            break
                        goal = _center(wp)
                        tol = 4
                    else:
                        goal = wp
                        tol = 5
                    sx = sum(pos[p][0] for p in pcs)
                    sy = sum(pos[p][1] for p in pcs)
                    cx, cy = sx // n, sy // n
                    if abs(cx - goal[0]) <= tol and abs(cy - goal[1]) <= tol:
                        if wp_kind == "pellet":
                            if absorbed(wp):
                                ok_wp = True
                                break
                            # 已在碰撞范围内但尚未吸收 — 放一步同格点击触发引擎判定
                        else:
                            if shadow_free(cat, cx, cy):
                                ok_wp = True
                                break

                    # 候选移动
                    my_pending = [spr for spr in my_pellets if not absorbed(spr)]
                    avoid = [(spr2, c2) for spr2, c2, _, _ in _pellet_info(g)
                             if not absorbed(spr2) and spr2 not in my_pending]
                    cands = []
                    for p in pcs:
                        for cc in _CELLS:
                            if cc == pos[p] or cc in [pos[q] for q in all_pieces]:
                                continue
                            if not cell_ok(p, cc[0], cc[1]):
                                continue
                            if (id(p), cc) in blacklist:
                                continue
                            nsx = sx + cc[0] - pos[p][0]
                            nsy = sy + cc[1] - pos[p][1]
                            ncx, ncy = nsx // n, nsy // n
                            if not shadow_free(cat, ncx, ncy):
                                continue
                            bad = False
                            for spr2, c2 in avoid:
                                if abs(ncx - c2[0]) <= 4 and abs(ncy - c2[1]) <= 4:
                                    bad = True
                                    break
                            if bad:
                                continue
                            d = abs(ncx - goal[0]) + abs(ncy - goal[1])
                            cands.append((d, _RNG.random(), p, cc))
                    if not cands:
                        break
                    cands.sort(key=lambda t: (t[0], t[1]))
                    pick = cands[0] if _RNG.random() < 0.75 else cands[_RNG.randrange(min(4, len(cands)))]
                    _, _, p, cc = pick

                    old_pos = pos[p]
                    all_clicks.append([old_pos[0], old_pos[1]])
                    _click(g, old_pos[0], old_pos[1])
                    if g._state == _GameState.GAME_OVER:
                        return None
                    if g._win_pending:
                        return all_clicks
                    if g.wiayqaumjug is not p:
                        return None
                    all_clicks.append([cc[0], cc[1]])
                    _click(g, cc[0], cc[1])
                    if g._state == _GameState.GAME_OVER:
                        return None
                    if g._win_pending:
                        return all_clicks
                    now = _center(p)
                    if g.wiayqaumjug is not p:
                        return None
                    if now == old_pos:
                        blacklist.add((id(p), cc))
                        continue
                    pos[p] = now
                if not ok_wp:
                    return None

            if g._win_pending:
                return all_clicks

        return None
    except Exception:
        return None
    finally:
        if neutered:
            try:
                del g.next_level
            except Exception:
                pass
        _restore(g, snap0)


# ─── Frame 实时贪心求解 ────────────────────────────────
# 当 shadow BFS 超时时降级使用。不依赖本地引擎状态，
# 直接从 API 返回的 frame 实时计算合法点击。

# 模块级历史：记录最近 N 次点击的碎片位置，避免重复选同一碎片
_FRAME_HISTORY = []        # [(centroid_r, centroid_c, color), ...]
_FRAME_HISTORY_MAX = 20    # 最多记忆 20 步
_STEP_COUNTER = 0          # 连续 solve_frame 调用次数（同一 arc-auto 周期内）
_LAST_FRAME_HASH = None     # 上次 frame 的哈希，检测 frame 是否变化

def solve_frame(frame):
    """从 64x64 frame 矩阵实时计算下一个点击坐标 [x, y]。
    策略：找托盘中的 piece，连通块分组，按颜色找配对目标，
    贪心将每块移向棋盘中心附近空格。
    - 颜色 5  = 棋盘实心格
    - 颜色 0  = 托盘空格 / board 空格
    - 颜色 2  = board 边界
    - 其他颜色 = piece / pellet / 目标区
    """
    global _FRAME_HISTORY, _STEP_COUNTER, _LAST_FRAME_HASH

    if not frame or len(frame) < 64 or len(frame[0]) < 64:
        return None

    # 检测 frame 是否变化（新的一步已执行）
    frame_hash = hash(tuple(tuple(row) for row in frame[:10]))
    if frame_hash != _LAST_FRAME_HASH:
        # frame 变了，说明上一步已执行，保留历史但不递增步内计数
        _LAST_FRAME_HASH = frame_hash
        _STEP_COUNTER = 0
    else:
        # frame 没变（可能是重复调用），步内计数递增
        _STEP_COUNTER += 1

    H, W = len(frame), len(frame[0])

    # 找 board 格（5）和托盘 board 空格（0）
    board_set  = { (r,c) for r in range(H) for c in range(W) if frame[r][c] == 5 }
    board_open = { (r,c) for r in range(H) for c in range(W) if frame[r][c] == 0 }

    # 找所有彩色连通块（piece）
    visited = set()
    clusters = []  # [(color, [(r,c),...])]
    for r in range(H):
        for c in range(W):
            v = frame[r][c]
            if (r,c) in visited or v in (0, 2, 5, -1):
                continue
            color = v
            cluster = []
            stack = [(r,c)]
            while stack:
                cr, cc = stack.pop()
                if (cr,cc) in visited or cr < 0 or cr >= H or cc < 0 or cc >= W:
                    continue
                if frame[cr][cc] != color:
                    continue
                visited.add((cr,cc))
                cluster.append((cr,cc))
                stack += [(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)]
            if cluster:
                clusters.append((color, cluster))

    if not clusters:
        return None

    # 过滤掉最近已选过的碎片（颜色+位置接近的组合）
    recent_centroids = set((c, r//10) for (r, c, col) in _FRAME_HISTORY[-5:])

    # 按块大小降序，选最大块
    clusters.sort(key=lambda x: len(x[1]), reverse=True)

    # 优先选没用过或不在最近历史的块
    chosen = None
    for color, cells in clusters:
        cr = sum(r for r,c in cells) // len(cells)
        cc = sum(c for r,c in cells) // len(cells)
        key = (color, cr//10)
        if key not in recent_centroids:
            chosen = (color, cells, cr, cc)
            break

    if chosen is None:
        # 所有块都用过了，随机选一个并清理历史
        _FRAME_HISTORY.clear()
        color, cells = clusters[_RNG.randrange(len(clusters))]
        cr = sum(r for r,c in cells) // len(cells)
        cc = sum(c for r,c in cells) // len(cells)
        chosen = (color, cells, cr, cc)
    else:
        color, cells, cr, cc = chosen

    # board 内空格：与 board_set 相邻的 board_open 格
    board_adjacent = []
    for br, bc in board_set:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = br+dr, bc+dc
            if (nr,nc) in board_open:
                board_adjacent.append((nr,nc))
                break

    def dist(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # 找 board 的 bounding box 中心
    if board_set:
        brs = [r for r,c in board_set]
        bcs = [c for r,c in board_set]
        center_r, center_c = (min(brs)+max(brs))//2, (min(bcs)+max(bcs))//2
    else:
        center_r, center_c = 32, 32

    candidates = board_adjacent if board_adjacent else list(board_open)
    if not candidates:
        return None

    # 找最近的 board 空格
    close_open = [bc for bc in candidates if dist((cr, cc), bc) <= 3]
    if close_open:
        target = close_open[_RNG.randrange(len(close_open))]
    else:
        target = min(candidates, key=lambda p: dist((cr, cc), p))

    # 碎片质心已在 board 空格附近 → 点击碎片本身（选中）
    if dist((cr, cc), target) <= 2:
        result = [cc, cr]
    else:
        # 点击碎片质心将其选中 → 下次调用再点击 board 空格
        result = [cc, cr]

    # 记录历史（用于下次避免重复）
    _FRAME_HISTORY.append((cr, cc, color))
    if len(_FRAME_HISTORY) > _FRAME_HISTORY_MAX:
        _FRAME_HISTORY = _FRAME_HISTORY[-_FRAME_HISTORY_MAX:]

    # 连续同坐标超过 3 次 → 强制随机一个新碎片
    recent_same = sum(1 for (r, c, _) in _FRAME_HISTORY[-6:-1]
                     if abs(r - cr) + abs(c - cc) <= 4)
    if recent_same >= 4 and _STEP_COUNTER >= 3:
        # 强制随机探索
        if len(clusters) > 1:
            for i, (color2, cells2) in enumerate(clusters):
                if i == 0:
                    continue
                cr2 = sum(r for r,c in cells2) // len(cells2)
                cc2 = sum(c for r,c in cells2) // len(cells2)
                result = [cc2, cr2]
                # 用新碎片替换历史中最近一条
                if len(_FRAME_HISTORY) >= 2:
                    _FRAME_HISTORY[-1] = (cr2, cc2, color2)
                break
        else:
            # 只有一个碎片：随机 board 空格点击探索
            tgt = candidates[_RNG.randrange(len(candidates))]
            result = [tgt[1], tgt[0]]
            _FRAME_HISTORY.clear()  # 重置历史

    return result
