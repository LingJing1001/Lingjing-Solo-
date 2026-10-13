# -*- coding: utf-8 -*-
"""CN04 影子求解器 —— 拼板谜题（选中/移动/旋转/换色，引擎判 win）。

机制 (cn04.py 直读, 2026-10-06):
- ACTION6 点击含非透明像素的可见块 → 选中 (swfljqaiqu)；点已选块的透明格 → 取消选择。
- ACTION1-4: 选中的块移动 1 格（网格边界内）。
- ACTION5: 多色变体的块 → 循环色变体 (eliexourzp)；单变体块 → rotate(90)。
- 每个动作 _action_count+1，>= MaxSteps(75-125) → lose。
- 胜利: sjwqloivve() 每块模板的 8/13 色格全部被解决 → rqolqpqwo=True，
  下一个 step 触发 next_level。引擎自己判定，求解器只管搜。
- uqlndqojuf 每动作重建目标映射（无陈旧引用）→ 浅快照（不克隆精灵）即可：
  保住 vausolnec/hlxyvcmpk 等按对象键的字典。
"""
import os
import threading
import time
from collections import deque

import itertools

import numpy as np

os.environ.setdefault("ARC_API_KEY", "f1d9f364-5ac8-4486-a4e9-6b9286de351f")

_lock = threading.RLock()
_env = None
_game = None
_valid = False
_error = None
_ActionInput = None
_GameState = None
_ACTION = None


def _ensure():
    global _env, _game, _valid, _error, _ActionInput, _GameState, _ACTION
    if _valid and _game is not None and _ActionInput is not None:
        return True
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput, GameState
        _ActionInput = ActionInput
        _GameState = GameState
        _ACTION = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5, 6)}
        if _game is None:
            arc = arc_agi.Arcade()
            _env = arc.make("cn04")
            _game = _env._game
            _env.reset()
        _valid = True
    except Exception as e:
        _error = str(e)
        _valid = False
    return _valid


def reset():
    global _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def apply(action):
    if not _ensure():
        return False
    with _lock:
        try:
            if isinstance(action, dict):
                aid = _ACTION.get(int(action.get("a", 6)), _ACTION[6])
                data = {"x": int(action.get("x", 0)), "y": int(action.get("y", 0))}
            else:
                aid = _ACTION[int(action)]
                data = {}
            _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
            return True
        except Exception:
            _valid = False
            return False


def is_synced(actions_taken):
    return _valid


def status():
    return {"valid": _valid, "error": _error, "loaded": _env is not None}


# ─── 回放基元: 引擎自带关初重置 ─────────────────────────────────
# 不做就地快照 —— cn04 的 on_set_level 会重建 hlxyvcmpk/vausolnec/uysylxuqw 等
# 游戏级字典并改写精灵像素, 就地还原追不上这类漂移。level_reset() 从
# _clean_levels 克隆原始关 + set_level(重置 _action_count/相机) + on_set_level,
# 是引擎自己定义的"精确关初", 每次回放从零重放天然确定性。

_TRANSITION_LOG = []


def _level_begin(g, level0):
    g._current_level_index = level0
    g.level_reset()
    nxt = level0 + 1
    if nxt < len(g._levels):
        # 引擎泵过关时会 set_level(下一关) 并就地改写其精灵 (color_remap 非
        # 幂等) —— 不重建克隆, 下一关第二次进入时 hlxyvcmpk 模板将永久失真
        g._levels[nxt] = g._clean_levels[nxt].clone()
    g._state = g._state.__class__.NOT_FINISHED if hasattr(g._state, "__class__") else g._state


def _bfs_solve(t_limit=150.0, max_nodes=300000):
    """求解当前关。返回动作序列 [("m",1-4) / ("r",) / ("c",x,y)]；已通关 []；失败 None。

    胜利语义 (cn04 step() + 引擎泵): 达成 sjwqloivve 的动作置 rqolqpqwo 且不调
    complete_action —— perform_action 的泵循环在同一动作内继续 step(), 顶部旗标分支
    完成 next_level。因此动作返回时旗标已复位、关卡已前进: 胜利信号 = 关卡下标
    变化, 且只许发生在路径最后一步。
    """
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        level0 = int(g._current_level_index)
        if level0 >= len(g._levels) or g._state == _GameState.WIN:
            return []
        budget = int(getattr(g, "ojcsxidcz", 150))

        seen = set()
        queue = deque([()])

        def _won():
            return int(g._current_level_index) > level0 or g._state == _GameState.WIN

        def _replay(path):
            """从原始关克隆重放整条路径。False = 异常/中途切关, 本路作废。"""
            _level_begin(g, level0)
            for j, a in enumerate(path):
                try:
                    if a[0] == "m":
                        g.perform_action(_ActionInput(id=_ACTION[a[1]], data={}, reasoning=None), raw=True)
                    elif a[0] == "r":
                        g.perform_action(_ActionInput(id=_ACTION[5], data={}, reasoning=None), raw=True)
                    else:
                        g.perform_action(_ActionInput(id=_ACTION[6],
                                                      data={"x": int(a[1]), "y": int(a[2])},
                                                      reasoning=None), raw=True)
                except Exception as e:
                    if len(_TRANSITION_LOG) < 5:
                        print(f"[cn04] replay exc at {j}: {type(e).__name__} path={path}", flush=True)
                        _TRANSITION_LOG.append(1)
                    return False
                if int(g._current_level_index) != level0:
                    if j != len(path) - 1:
                        # 非末步切关 = 路径中段就触发了过关, 尾部全是砸向下一关的废动作
                        if len(_TRANSITION_LOG) < 5:
                            print(f"[cn04] MID-TRANSITION at step {j}/{len(path)} path={path}", flush=True)
                            _TRANSITION_LOG.append(1)
                        return False
                    # 末步切关 = 正常胜利 (引擎泵在动作内部完成 next_level)
            return True

        def _pieces(g):
            return [s for s in g.current_level.get_sprites() if s.is_visible]

        def _click_cell(s):
            try:
                px = s.render()
                h, w = px.shape
                c0 = (w // 2, h // 2)
                if px[c0[1], c0[0]] >= 0:
                    return (int(s.x) + c0[0], int(s.y) + c0[1])
                best = None
                for yy in range(h):
                    for xx in range(w):
                        if px[yy, xx] >= 0:
                            d = (xx - w / 2) ** 2 + (yy - h / 2) ** 2
                            if best is None or d < best[0]:
                                best = (d, xx, yy)
                if best:
                    return (int(s.x) + best[1], int(s.y) + best[2])
            except Exception:
                pass
            return (int(s.x) + int(s.width) // 2, int(s.y) + int(s.height) // 2)

        def _state_key(g):
            items = tuple(sorted(
                (s.name, int(s.x), int(s.y), int(getattr(s, "rotation", 0)), bool(s.is_visible))
                for s in g.current_level.get_sprites()
            ))
            sel = g.xseexqzst
            return items + (sel.name if sel is not None else None, bool(getattr(g, "ztpxqonhr", False)))

        try:
            while queue and time.time() - t0 < t_limit and len(seen) < max_nodes:
                path = queue.popleft()
                if not _replay(path):
                    continue
                if _won():
                    _level_begin(g, level0)   # 干净退出, 实机回放由路径自己触发过关
                    return list(path)
                if g._state == _GameState.GAME_OVER:
                    continue
                if budget - int(getattr(g, "_action_count", 0)) <= 0:
                    continue

                sel = getattr(g, "xseexqzst", None)
                if sel is not None:
                    for d in (1, 2, 3, 4):
                        npath = path + (("m", d),)
                        if not _replay(npath):
                            continue
                        if _won():
                            _level_begin(g, level0)
                            return list(npath)
                        if g._state == _GameState.GAME_OVER:
                            continue
                        nk = _state_key(g)
                        if nk not in seen:
                            seen.add(nk)
                            queue.append(npath)
                    npath = path + (("r",),)
                    if not _replay(npath):
                        pass
                    else:
                        if _won():
                            _level_begin(g, level0)
                            return list(npath)
                        if g._state != _GameState.GAME_OVER:
                            nk = _state_key(g)
                            if nk not in seen:
                                seen.add(nk)
                                queue.append(npath)
                for p in _pieces(g):
                    if p is sel:
                        continue
                    cx, cy = _click_cell(p)
                    npath = path + (("c", cx, cy),)
                    if not _replay(npath):
                        continue
                    if _won():
                        _level_begin(g, level0)
                        return list(npath)
                    if g._state == _GameState.GAME_OVER:
                        continue
                    nk = _state_key(g)
                    if nk not in seen:
                        seen.add(nk)
                        queue.append(npath)
            return None
        finally:
            _level_begin(g, level0)




def _render_arr(template, rot):
    k = int((-int(rot) % 360) / 90)
    return np.rot90(template, k=k) if k else template


def _cells_at(template, rot, px, py):
    arr = _render_arr(template, rot)
    c8 = [(int(px) + int(x), int(py) + int(y)) for y, x in zip(*np.where(arr == 8))]
    c13 = [(int(px) + int(x), int(py) + int(y)) for y, x in zip(*np.where(arr == 13))]
    return c8, c13


def _cn04_groups(g):
    """组 = vausolnec 栈 (变体) 或单块。返回 (组列表, 精灵层序)。

    成员顺序必须用引擎自己的 vausolnec 列表 (on_set_level 按 layer 排序后注册,
    eliexourzp 的变体循环按该列表索引走) —— 用精灵表序自建会在 layer 非零的
    关卡 (如 L6) 停到错误变体上。
    """
    order = list(g.current_level.get_sprites())
    groups = []
    seen = set()
    for s in order:
        lst = g.vausolnec[s]
        key = id(lst)
        if key in seen:
            continue
        seen.add(key)
        vis = [i for i, m in enumerate(lst) if m.is_visible]
        groups.append({"members": list(lst), "vis_idx": vis[0] if vis else -1})
    return groups, order


def _csp_components(g):
    """解相对构型 (写时复制 DFS —— 就地 bump/rollback 有跨分支状态污染风险)。

    返回组件列表, 每组件 = {"assign": {gi: (mi, rx, ry, rot)}}。
    叶子条件: 每个组件内全部 8/13 世界格多重性恰为 2 (组件自配对)。
    """
    groups, _ = _cn04_groups(g)
    n = len(groups)

    # 每组形态: (member_idx, template, rot, w, h)
    # 单块组可 ACTION5 旋转 → 枚举 4 向; 栈组 ACTION5 用于换变体 → 成员旋转固定
    shapes = []
    for grp in groups:
        lst = grp["members"]
        if len(lst) == 1:
            s = lst[0]
            t = g.hlxyvcmpk[s.name]
            shapes.append([(0, t, r, int(s.width), int(s.height)) for r in (0, 90, 180, 270)])
        else:
            shapes.append([(mi, g.hlxyvcmpk[s.name], int(s.rotation), int(s.width), int(s.height))
                           for mi, s in enumerate(lst)])

    def cell_total(cand):
        return max(int((t == 8).sum()) + int((t == 13).sum()) for (_, t, _, _, _) in cand)

    order = sorted(range(n), key=lambda i: -cell_total(shapes[i]))

    def aligned_candidates(gi, occ):
        out = []
        ones8 = [k[1:] for k, v in occ.items() if k[0] == 8 and v == 1]
        ones13 = [k[1:] for k, v in occ.items() if k[0] == 13 and v == 1]
        for (mi, tem, rot, w, h) in shapes[gi]:
            c8, c13 = _cells_at(tem, rot, 0, 0)
            for (wx, wy) in ones8:
                for (ox, oy) in c8:
                    out.append((mi, wx - ox, wy - oy, rot))
            for (wx, wy) in ones13:
                for (ox, oy) in c13:
                    out.append((mi, wx - ox, wy - oy, rot))
        return set(out)

    def place_ok(occ, c8, c13):
        for p in c8:
            if occ.get((8,) + p, 0) >= 2:
                return False
        for p in c13:
            if occ.get((13,) + p, 0) >= 2:
                return False
        return True

    def apply_occ(occ, c8, c13, sign):
        out = dict(occ)
        for p in set(c8):
            k = (8,) + p
            v = out.get(k, 0) + sign * c8.count(p)
            if v > 0:
                out[k] = v
            else:
                out.pop(k, None)
        for p in set(c13):
            k = (13,) + p
            v = out.get(k, 0) + sign * c13.count(p)
            if v > 0:
                out[k] = v
            else:
                out.pop(k, None)
        return out

    def dfs(i, comps, assign):
        if i == n:
            return (assign, comps) if all(all(v == 2 for v in c["occ"].values()) for c in comps) else None
        gi = order[i]
        # 候选1: 加入已有组件 (对齐现有 count1 格)
        for ci, comp in enumerate(comps):
            for (mi, rx, ry, rot) in aligned_candidates(gi, comp["occ"]):
                tem = shapes[gi][mi][1]
                c8, c13 = _cells_at(tem, rot, rx, ry)
                if not place_ok(comp["occ"], c8, c13):
                    continue
                new_comp = {"occ": apply_occ(comp["occ"], c8, c13, +1),
                            "members": comp["members"] + [gi]}
                new_comps = comps[:ci] + [new_comp] + comps[ci + 1:]
                res = dfs(i + 1, new_comps, {**assign, gi: (mi, rx, ry, rot)})
                if res:
                    return res
        # 候选2: 新组件 (自身 8/13 格将来必须在组件内自配对)
        for (mi, tem, rot, w, h) in shapes[gi]:
            c8, c13 = _cells_at(tem, rot, 0, 0)
            if not all(v <= 2 for v in apply_occ({}, c8, c13, +1).values()):
                continue
            new_comps = comps + [{"occ": apply_occ({}, c8, c13, +1), "members": [gi]}]
            res = dfs(i + 1, new_comps, {**assign, gi: (mi, 0, 0, rot)})
            if res:
                return res
        return None

    res = dfs(0, [], {})
    if not res:
        return None
    assign, comps = res
    return [{"assign": {gi: assign[gi] for gi in comp["members"]}} for comp in comps]


def _build_plan(g, comps):
    """相对构型 → 平移 → 动作计划 (ACTION6 为 display 坐标)。

    组件内按排列枚举出块顺序: 先移动的块会盖住后点击块的格子, 需要找到
    每块点击时都有独占格的顺序 (引擎按层序取第一个覆盖者)。
    """
    gw, gh = g.current_level.grid_size
    groups, sprite_order = _cn04_groups(g)
    scale = min(int(64 / gw), int(64 / gh))
    padx = (64 - gw * scale) // 2
    pady = (64 - gh * scale) // 2

    start = {}
    for gi, grp in enumerate(groups):
        lst = grp["members"]
        vi = grp["vis_idx"]
        s0 = lst[vi]
        start[gi] = {"x": int(s0.x), "y": int(s0.y), "rot": int(s0.rotation), "member": vi}

    vis_sprites = [s for s in sprite_order if s.is_visible]
    sel_gi = None
    if vis_sprites:
        init_sel = min(vis_sprites, key=lambda s: int(s.x) ** 2 + int(s.y) ** 2)
        sel_gi = next((gi for gi, grp in enumerate(groups) if init_sel in grp["members"]), None)

    plan = []

    def emit(offs, dx, dy, state, sel_gi, out):
        """按给定顺序出块。成功返回 (True, sel_gi), 失败 (False, 原因)。"""
        for (gi, (mi, rx, ry, rot)) in offs:
            grp = groups[gi]
            st = state[gi]
            stack = len(grp["members"]) > 1
            tx, ty = rx + dx, ry + dy
            _need = ((stack and st["member"] != mi) or
                     (not stack and rot != st["rot"]) or
                     tx != st["x"] or ty != st["y"])
            if not _need:
                continue
            if sel_gi != gi:
                if st["member"] < 0:
                    return False, f"组 {gi} 无可见成员"
                member = grp["members"][st["member"]]
                arr = _render_arr(g.hlxyvcmpk[member.name], st["rot"])
                cells = [(int(st["x"]) + int(x), int(st["y"]) + int(y))
                         for y, x in zip(*np.where(arr >= 0))]

                def owner(wx, wy):
                    for s2 in sprite_order:
                        for gj, grp2 in enumerate(groups):
                            if s2 not in grp2["members"]:
                                continue
                            st2 = state[gj]
                            if st2["member"] < 0:
                                break
                            m2 = grp2["members"][st2["member"]]
                            a2 = _render_arr(g.hlxyvcmpk[m2.name], st2["rot"])
                            lx, ly = wx - st2["x"], wy - st2["y"]
                            if 0 <= ly < a2.shape[0] and 0 <= lx < a2.shape[1] and int(a2[ly, lx]) >= 0:
                                return gj
                            break
                    return None

                good = [c for c in cells if 0 <= c[0] < gw and 0 <= c[1] < gh and owner(*c) == gi]
                if not good:
                    return False, f"组 {gi} 点击无独占格"
                wx, wy = good[0]
                out.append(("c", wx * scale + scale // 2 + padx, wy * scale + scale // 2 + pady))
                sel_gi = gi
            if stack and st["member"] != mi:
                n_m = len(grp["members"])
                idx, ztp, steps = st["member"], True, 0
                while idx != mi and steps <= 4 * n_m + 4:
                    if ztp:
                        nxt = idx + 1
                        if nxt >= n_m:
                            ztp = False
                            nxt = idx - 1
                            if nxt < 0:
                                ztp = True
                                nxt = idx
                    else:
                        nxt = idx - 1
                        if nxt < 0:
                            ztp = True
                            nxt = idx + 1
                    idx, steps = nxt, steps + 1
                    out.append(("r",))
                if idx != mi:
                    return False, f"组 {gi} 变体循环不可达"
                st["member"] = idx
                st["rot"] = int(grp["members"][idx].rotation)
            want = rot if not stack else st["rot"]
            diff = ((want - st["rot"]) % 360) // 90
            for _ in range(diff):
                out.append(("r",))
            st["rot"] = want
            ddx, ddy = tx - st["x"], ty - st["y"]
            out.extend([("m", 4)] * ddx + [("m", 3)] * (-ddx) +
                       [("m", 2)] * ddy + [("m", 1)] * (-ddy))
            st["x"], st["y"] = tx, ty
        return True, sel_gi

    for comp in comps:
        offs = list(comp["assign"].items())
        lo_x = hi_x = lo_y = hi_y = None
        for (gi, (mi, rx, ry, rot)) in offs:
            grp = groups[gi]
            s = grp["members"][mi] if len(grp["members"]) > 1 else grp["members"][0]
            w, h = int(s.width), int(s.height)
            lo_x = max(lo_x, -rx) if lo_x is not None else -rx
            hi_x = min(hi_x, gw - w - rx) if hi_x is not None else gw - w - rx
            lo_y = max(lo_y, -ry) if lo_y is not None else -ry
            hi_y = min(hi_y, gh - h - ry) if hi_y is not None else gh - h - ry
        if lo_x is None or lo_x > hi_x or lo_y > hi_y:
            print(f"[cn04] plan: 平移区间空 x[{lo_x},{hi_x}] y[{lo_y},{hi_y}]", flush=True)
            return None

        def best_axis(pairs, lo, hi):
            ds = sorted(sp - rp for (sp, rp) in pairs)
            med = ds[len(ds) // 2]
            return max(lo, min(hi, med))

        dx = best_axis([(start[gi]["x"], rx) for (gi, (mi, rx, ry, rot)) in offs], lo_x, hi_x)
        dy = best_axis([(start[gi]["y"], ry) for (gi, (mi, rx, ry, rot)) in offs], lo_y, hi_y)

        done = False
        for perm in itertools.islice(itertools.permutations(offs), 24):
            st_try = {gi: dict(start[gi]) for gi in range(len(groups))}
            out = []
            ok, info = emit(perm, dx, dy, st_try, sel_gi, out)
            if ok:
                plan.extend(out)
                sel_gi = info
                # state 采纳本次尝试的终态 (后续组件的点击归属依赖它)
                for gi in st_try:
                    state_ok = True
                globals()  # noqa
                done = True
                break
        if not done:
            print(f"[cn04] plan: 所有排列均失败 (末次: {info})", flush=True)
            return None
    return plan


def _try_plan(g, plan):
    """关初重放计划; 返回是否触发过关 (末步引擎泵内完成 next_level)。"""
    level0 = int(g._current_level_index)
    g._current_level_index = level0
    g.level_reset()
    for j, a in enumerate(plan):
        try:
            if a[0] == "m":
                g.perform_action(_ActionInput(id=_ACTION[a[1]], data={}, reasoning=None), raw=True)
            elif a[0] == "r":
                g.perform_action(_ActionInput(id=_ACTION[5], data={}, reasoning=None), raw=True)
            else:
                g.perform_action(_ActionInput(id=_ACTION[6],
                                              data={"x": int(a[1]), "y": int(a[2])},
                                              reasoning=None), raw=True)
        except Exception:
            return False
        if int(g._current_level_index) != level0 and j != len(plan) - 1:
            print(f"[cn04] verify: 中途切关 step {j}", flush=True)
            _level_begin(g, level0)
            return False
    ok = int(g._current_level_index) > level0 or g._state == _GameState.WIN
    if not ok:
        occ8, occ13 = {}, {}
        for s in g.current_level.get_sprites():
            if not s.is_visible:
                continue
            c8, c13 = _cells_at(g.hlxyvcmpk[s.name], int(s.rotation), int(s.x), int(s.y))
            for p in c8:
                occ8[p] = occ8.get(p, 0) + 1
            for p in c13:
                occ13[p] = occ13.get(p, 0) + 1
        bad8 = {p: c for p, c in occ8.items() if c != 2}
        bad13 = {p: c for p, c in occ13.items() if c != 2}
        print(f"[cn04] verify: 未配对 8@{bad8} 13@{bad13} ac={getattr(g, '_action_count', '?')} "
              f"steps_left={getattr(g, 'ojcsxidcz', '?')}", flush=True)
    _level_begin(g, level0)
    return ok


def _agentify(plan):
    """交付 agent 的路径格式: ('r',) → ('i', 5) —— agent 转换器把 ('i', n) 映射为
    键盘 ACTIONn; 1 元组会砸了它的 int() 转换。"""
    return [("i", 5) if a == ("r",) else a for a in plan]


def solve_level(t_limit=150.0, max_nodes=300000):
    """CSP 定向求解优先, 失败回退 BFS。"""
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        if int(g._current_level_index) >= len(g._levels) or g._state == _GameState.WIN:
            return []
        try:
            # agent 漫游可能已把块挪到任意位置: 先重置到原始关, 保证
            # CSP 起点 / _try_plan 验真 / agent 回放三者基于同一状态
            # (level_reset 同时把 _action_count 归零, 预算满额)
            g.level_reset()
            comps = _csp_components(g)
            if comps:
                plan = _build_plan(g, comps)
                if plan:
                    budget = int(getattr(g, "ojcsxidcz", 150))
                    if _try_plan(g, plan) and len(plan) <= budget:
                        print(f"[cn04] CSP 命中: {len(plan)} 步 (budget {budget})", flush=True)
                        return _agentify(plan)
                    print(f"[cn04] CSP 计划未验真 (len={len(plan)}), 回退 BFS", flush=True)
        except Exception as e:
            print(f"[cn04] CSP 异常: {type(e).__name__}: {e}", flush=True)
        p = _bfs_solve(t_limit=t_limit, max_nodes=max_nodes)
        return _agentify(p) if p else p

def solve_level_enhanced(t_limit=150.0, **kw):
    return solve_level(t_limit=t_limit)


def get_frame():
    if not _ensure():
        return [[0] * 10 for _ in range(10)]
    try:
        import numpy as np
        sprites = list(_game.current_level.get_sprites())
        grid_cells = []
        for s in sprites:
            if not s.is_visible:
                continue
            px = np.array(s.pixels)
            for r in range(px.shape[0]):
                for c in range(px.shape[1]):
                    if px[r][c] not in (-1,):
                        grid_cells.append((int(s.x) + c, int(s.y) + r, int(px[r][c])))
        if not grid_cells:
            return [[0] * 10 for _ in range(10)]
        min_x = min(c[0] for c in grid_cells)
        max_x = max(c[0] for c in grid_cells)
        min_y = min(c[1] for c in grid_cells)
        max_y = max(c[1] for c in grid_cells)
        grid = [[0] * (max_x - min_x + 1) for _ in range(max_y - min_y + 1)]
        for x, y, color in grid_cells:
            grid[y - min_y][x - min_x] = color
        return grid
    except Exception:
        return [[0] * 10 for _ in range(10)]
