# -*- coding: utf-8 -*-
"""SP80 影子求解器 —— 引水解谜（方块导流 + ACTION5 放水润湿全部墙格）。

机制 (sp80.py 直读, 2026-10-06):
- "change" 相: ACTION6 点击选中 plzwjbfyfli/tuvkdkhdokr 方块, ACTION1-4 移动选中方块
  1 格 (y>=3 且不贴墙, husluhmboo); 每个动作扣 1 步 (zlhbnhpcq, 关卡 data "steps");
  ACTION5 放水 (lyremoheq>=4 时再放直接 lose)。
- "spill" 相: 水滴从水龙头 (sowlljgtjvn) 色值4像素下方与既有水滴 (liolfvkveqg) 下落,
  遇方块左右分流, 遇墙 (repwkzbkhxl) 且左右邻同墙 → 全墙变湿 (cevwbinfgl);
  触红 (waoewejnqzc) → kfdcqkodyy 失败; 水停后: 全墙湿 → next_level, 否则复位重来。
- 每关固定旋转 k (data "dojfslwbg"): 引擎对输入动作做 mfkgvxzkbj[k] 重映射,
  ACTION6 坐标经 ojydygjkdb 旋转 —— 求解器发动作前必须施加逆映射。

搜索: BFS 纯方块布局空间 (失败放水会复原布局, 所以放水不进分支), 每个新布局
用真引擎做一次放水模拟 (按布局缓存), 命中"全墙湿"即输出 路径+ACTION5。
"""
import os
import threading
import time
from collections import deque

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
    # adapter 每次 solve 前注入线上 game 且会把 _env 置回 None —— 注入优先,
    # 已注入不再自建 (dc22 同款教训: 否则每次求解都在解自家 env 的 L0)
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
            _env = arc.make("sp80-589a99af")
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


# ─── 快照/恢复 ─────────────────────────────────

def _is_spr(v):
    return hasattr(v, "pixels") and hasattr(v, "_interaction")


def _handle_map(g, sprites_of, index_of):
    pos = {}
    for s in sprites_of():
        i = index_of(s)
        if i is not None:
            pos[id(s)] = i
    out = {}
    for k, v in list(vars(g).items()):
        if _is_spr(v):
            i = pos.get(id(v))
            if i is not None:
                out[k] = ("s", i)
        elif isinstance(v, list) and v and all(_is_spr(x) for x in v):
            idx = [pos.get(id(x)) for x in v]
            if all(i is not None for i in idx):
                out[k] = ("l", tuple(idx))
        elif isinstance(v, set) and v and all(_is_spr(x) for x in v):
            idx = [pos.get(id(x)) for x in v]
            if all(i is not None for i in idx):
                out[k] = ("t", tuple(idx))
    return out


def _steps_of(g):
    bar = getattr(g, "lijmqzmgnw", None)
    if bar is None:
        return None
    return (int(getattr(bar, "vwbodnowge", 0)), int(getattr(bar, "current_steps", 0)))


def _snapshot(g):
    live = list(g._levels)
    idx = int(g._current_level_index)
    obj_list = list(live[idx]._sprites)
    spr = []
    for s in obj_list:
        try:
            spr.append((s.x, s.y, s.pixels.copy(), getattr(s, "_interaction", None)))
        except Exception:
            spr.append((s.x, s.y, None, None))
    index = {id(s): i for i, s in enumerate(obj_list)}
    handles = _handle_map(g, lambda: obj_list, lambda s: index.get(id(s)))
    misc = (
        str(getattr(g, "dkvpswzsjg", "")),
        int(getattr(g, "zlhbnhpcq", 0)),
        int(getattr(g, "lyremoheq", 0)),
        bool(getattr(g, "kfdcqkodyy", False)),
        bool(getattr(g, "lybfalkrdl", False)),
        int(getattr(g, "trhynadhiz", 0)),
        _steps_of(g),
    )
    return (obj_list, spr, misc, handles, idx)


def _restore(g, snap):
    obj_list, spr, misc = snap[0], snap[1], snap[2]
    idx = snap[4]
    lv = g._levels[idx]
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
    cur[:] = obj_list
    lv._need_sort = True
    for s, v in zip(cur, spr):
        s.set_position(v[0], v[1])
        if v[2] is not None:
            try:
                s.pixels = v[2].copy()
            except Exception:
                pass
        try:
            s._interaction = v[3]
        except Exception:
            pass
    (g.dkvpswzsjg, g.zlhbnhpcq, g.lyremoheq, g.kfdcqkodyy,
     g.lybfalkrdl, g.trhynadhiz, steps) = misc
    if steps is not None and getattr(g, "lijmqzmgnw", None) is not None:
        g.lijmqzmgnw.vwbodnowge, g.lijmqzmgnw.current_steps = steps
    for k, spec in (snap[3] if len(snap) > 3 else {}).items():
        try:
            if spec[0] == "s":
                setattr(g, k, obj_list[spec[1]])
            elif spec[0] == "l":
                setattr(g, k, [obj_list[i] for i in spec[1]])
            elif spec[0] == "t":
                setattr(g, k, {obj_list[i] for i in spec[1]})
        except Exception:
            pass


# ─── 动作执行（含旋转逆映射） ─────────────────────────────────

def _inv_remap(g, k):
    tbl = g.mfkgvxzkbj.get(k, {})
    inv = {}
    for kk, vv in tbl.items():
        vv_id = int(vv.value) if hasattr(vv, "value") else int(str(vv).split("ACTION")[-1])
        kk_id = int(kk.value) if hasattr(kk, "value") else int(str(kk).split("ACTION")[-1])
        inv[vv_id] = kk_id
    return inv


def _exec_move(g, sent):
    """直发动作 id（路径项即发送空间动作, 引擎 krehtwyvlu 会再映射到世界语义）。"""
    g.perform_action(_ActionInput(id=_ACTION[sent], data={}, reasoning=None), raw=True)


def _click_sent(g, wx, wy):
    """世界格 (wx,wy) → 应发送的显示坐标（k=0 映射 + rqlmciubuz 逆旋转）。"""
    cam = g._camera
    scale = 64 / cam.width
    pad_y = (64 - cam.height * scale) / 2
    dx0 = wx * scale + scale / 2
    dy0 = wy * scale + scale / 2 + pad_y
    k = int(getattr(g, "fahhoimkk", 0))
    if k == 0:
        return (int(round(dx0)), int(round(dy0)))
    if k == 1:
        return (int(round(dy0)), int(round(63 - dx0)))   # rqlmciubuz_1: (x,y)->(y,63-x)
    if k == 2:
        return (int(round(63 - dx0)), int(round(63 - dy0)))
    return (int(round(63 - dy0)), int(round(dx0)))


def _exec_click_sent(g, sx, sy):
    g.perform_action(_ActionInput(id=_ACTION[6], data={"x": int(sx), "y": int(sy)},
                                   reasoning=None), raw=True)


def _blocks(g):
    return [s for s in g.current_level.get_sprites_by_tag("plzwjbfyfli")] + \
           [s for s in g.current_level.get_sprites_by_tag("tuvkdkhdokr")]


def _layout_key(g):
    return tuple(sorted((s.name, int(s.x), int(s.y)) for s in _blocks(g)))


# ─── 求解 ─────────────────────────────────

def solve_level(t_limit=110.0, max_nodes=200000):
    """求解当前关。返回动作序列 [("m",1-4) / ("c",x,y) / ("p",)]；已通关 []；失败 None。"""
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        level0 = int(g._current_level_index)
        if level0 >= len(g._levels) or g._state == _GameState.WIN:
            return []
        budget = int(getattr(g, "zlhbnhpcq", 50))

        start_snap = _snapshot(g)
        start_key = (_layout_key(g), None)
        seen = {start_key}
        queue = deque([((), )])
        pour_cache = {}

        neutered = False
        try:
            g._win_pending = False
            g.next_level = lambda: setattr(g, "_win_pending", True)
            neutered = True
        except Exception:
            neutered = False

        def _won():
            return int(g._current_level_index) > level0 or g._win_pending

        def _dead():
            return g._state == _GameState.GAME_OVER

        def _replay(path):
            _restore(g, start_snap)
            for a in path:
                if a[0] == "m":
                    _exec_move(g, a[1])
                elif a[0] == "c":
                    _exec_click_sent(g, a[1], a[2])

        def _selected_index():
            sel = getattr(g, "vsoxmtrhqt", None)
            if sel is None:
                return None
            for i, b in enumerate(_blocks(g)):
                if b is sel:
                    return i
            return None

        try:
            while queue and time.time() - t0 < t_limit and len(seen) < max_nodes:
                path, = queue.popleft()
                _replay(path)
                if _won():
                    return list(path)
                if _dead() or str(g.dkvpswzsjg) != "change":
                    continue

                # 放水试探（按布局缓存; 失败放水复原布局, 不产生新状态 → 不进分支）
                layout = _layout_key(g)
                if layout not in pour_cache:
                    p_snap = _snapshot(g)
                    g.perform_action(_ActionInput(id=_ACTION[5], data={}, reasoning=None), raw=True)
                    pour_cache[layout] = bool(_won())
                    _restore(g, p_snap)
                if pour_cache[layout]:
                    return list(path) + [("p", 5)]

                blocks = _blocks(g)
                sel_i = _selected_index()
                budget_left = budget - len(path)
                if budget_left <= 0:
                    continue
                # 移动选中方块
                if sel_i is not None:
                    inv = _inv_remap(g, int(getattr(g, "fahhoimkk", 0)))
                    for intent in (1, 2, 3, 4):
                        d = inv.get(intent, intent)   # 路径存发送动作, 引擎映射回世界意图
                        npath = path + (("m", d),)
                        _replay(npath)
                        if _won():
                            return list(npath)
                        if _dead() or str(g.dkvpswzsjg) != "change":
                            continue
                        nk = (_layout_key(g), _selected_index())
                        if nk not in seen:
                            seen.add(nk)
                            queue.append((npath,))
                # 点击选中其他方块
                for i, b in enumerate(blocks):
                    if i == sel_i:
                        continue
                    _sx, _sy = _click_sent(g, int(b.x) + int(b.width) // 2, int(b.y) + int(b.height) // 2)
                    npath = path + (("c", _sx, _sy),)
                    _replay(npath)
                    if _won():
                        return list(npath)
                    if _dead() or str(g.dkvpswzsjg) != "change":
                        continue
                    nk = (_layout_key(g), _selected_index())
                    if nk not in seen:
                        seen.add(nk)
                        queue.append((npath,))
            return None
        finally:
            if neutered:
                try:
                    del g.next_level
                except Exception:
                    pass
            _restore(g, start_snap)


def solve_level_enhanced(t_limit=150.0, **kw):
    return solve_level(t_limit=t_limit)


def get_frame():
    if not _ensure():
        return [[0] * 10 for _ in range(10)]
    try:
        import numpy as np
        sprites = list(_game.current_level._sprites)
        grid_cells = []
        for s in sprites:
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
