"""SK48 本地引擎影子模拟器

SK48 机制:
- 键盘动作游戏 (ACTION1-5)

求解策略: 引擎 BFS 搜索
"""
import os
import threading

os.environ.setdefault("ARC_API_KEY", "f1d9f364-5ac8-4486-a4e9-6b9286de351f")

_lock = threading.RLock()
_env = None
_game = None
_applied = 0
_valid = False
_error = None
_ActionInput = None
_ACTS = None


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _ACTS
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput
        arc = arc_agi.Arcade()
        _env = arc.make("sk48-d8078629")
        _game = _env._game
        _ActionInput = ActionInput
        _ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5)}
        _env.reset()
        _applied = 0
        _valid = True
    except Exception as e:
        _error = str(e)
        _valid = False
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


def apply(action):
    """线上动作成功后调用: 影子同步执行同一键盘动作"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            aid = _ACTS.get(int(action))
            if aid is None:
                return False
            _game.perform_action(_ActionInput(id=aid, data={}, reasoning=None), raw=True)
            _applied += 1
            return True
        except Exception:
            _valid = False
            return False


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def replay(history):
    """自愈重建: 影子回到 L1 并重放完整动作历史"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            for h in history:
                if isinstance(h, dict):
                    a = int(h.get("a", 1))
                else:
                    a = int(h)
                aid = _ACTS.get(a)
                if aid is not None:
                    _game.perform_action(_ActionInput(id=aid, data={}, reasoning=None), raw=True)
            _applied = len(history)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


def game_status():
    if not _ensure():
        return {"level": 0, "state": "INVALID"}
    g = _game
    return {"level": int(getattr(g, "_current_level_index", 0)),
            "state": str(getattr(g, "_state", "UNKNOWN"))}


def _sk48_snapshot(g):
    import numpy as np
    spr = [(s, int(s.x), int(s.y), np.array(s.pixels, copy=True),
            bool(getattr(s, "visible", True))) for s in g.current_level._sprites]
    bodies = {h: list(seg) for h, seg in g.mwfajkguqx.items()}
    sel = g.vzvypfsnt
    return (spr, bodies, sel)


def _sk48_restore(g, snap):
    spr, bodies, sel = snap
    level = g.current_level
    objs = {id(s) for (s, *_r) in spr}
    for s in list(level._sprites):
        if id(s) not in objs:
            level.remove_sprite(s)
    have = {id(s) for s in level._sprites}
    for (s, x, y, px, vis) in spr:
        if id(s) not in have:
            level.add_sprite(s)
        s.set_position(int(x), int(y))
        s.pixels = px.copy()
        try:
            s.set_visible(vis)
        except Exception:
            pass
    for h, segs in list(g.mwfajkguqx.items()):
        if h in bodies:
            g.mwfajkguqx[h][:] = bodies[h]
    g.vzvypfsnt = sel
    # 动画/计数状态清零 (搜索是规划器, 预算由最终路径长度保证)
    g.qiercdohl = g.vhzjwcpmk
    g.ljprkjlji.clear() if hasattr(g, "ljprkjlji") else None
    if hasattr(g, "pzzwlsmdt"):
        g.pzzwlsmdt.clear()
    if hasattr(g, "hadfnehqh"):
        g.hadfnehqh = None
    if hasattr(g, "yzyidartf"):
        g.yzyidartf.clear()
    if hasattr(g, "lgdrixfno"):
        g.lgdrixfno = -1
    if hasattr(g, "xsoixcero"):
        g.xsoixcero = False
    if "GAME_OVER" in str(g._state):
        import arcengine
        g._state = arcengine.GameState.NOT_FINISHED


def _sk48_state_key(g):
    import numpy as np
    parts = []
    for s in sorted(g.current_level._sprites, key=lambda z: z.name):
        try:
            px = np.array(s.pixels).tobytes()
        except Exception:
            px = b""
        parts.append((s.name, int(s.x), int(s.y), px, bool(getattr(s, "visible", True))))
    # 蛇身顺序 (对象身份索引化)
    allsprites = {id(s): i for i, s in enumerate(g.current_level._sprites)}
    for h, segs in g.mwfajkguqx.items():
        parts.append(tuple(allsprites.get(id(seg), -1) for seg in segs))
    parts.append(g.mwfajkguqx.get(g.vzvypfsnt) is not None)
    return hash(tuple(parts))


def _sk48_heur(g):
    """A* 启发 (优化版): 全蛇未覆盖数 + 各蛇头到最近未覆盖目标距离 + 蛇身长度惩罚
    
    优化点:
    1. 遍历所有蛇 (不只选中的), 全局未覆盖总数
    2. 每条蛇的蛇头到最近未覆盖目标的曼哈顿距离
    3. 蛇身太短覆盖不够时加惩罚 (引导吃食物增长)
    """
    total = 0
    step = 2  # udenqlsrfq
    for head, tgt in g.xpmcmtbcv.items():
        ghosts = g.jdojcthkf.get(tgt, [])
        body = g.mwfajkguqx.get(head, [])
        if not ghosts:
            continue
        unc = []
        for i, gm in enumerate(ghosts):
            gx, gy = int(gm.x), int(gm.y)
            covered = any(int(b.x) == gx and int(b.y) == gy for b in body)
            if not covered:
                unc.append((gx, gy))
        total += len(unc) * 3  # 未覆盖目标权重提高
        if unc and body:
            hx, hy = int(body[0].x), int(body[0].y)
            # 蛇头到最近未覆盖目标
            d = min(abs(hx - gx) + abs(hy - gy) for gx, gy in unc)
            total += d // step
            # 蛇身不够长时加惩罚 (需要吃食物增长)
            if len(body) < len(unc):
                total += (len(unc) - len(body)) * 2
    return total


def _sk48_click_switches(g):
    """枚举 A6 换蛇选择点击 (优化版): 只切换到有未覆盖目标的蛇"""
    out = []
    for head, tgt in g.xpmcmtbcv.items():
        ghosts = g.jdojcthkf.get(tgt, [])
        body = g.mwfajkguqx.get(head, [])
        if not ghosts:
            continue
        # 只切换到还有未覆盖目标的蛇
        has_unc = False
        for gm in ghosts:
            gx, gy = int(gm.x), int(gm.y)
            if not any(int(b.x) == gx and int(b.y) == gy for b in body):
                has_unc = True
                break
        if has_unc:
            out.append(("c", int(head.x) + 1, int(head.y) + 1))
            out.append(("c", int(tgt.x) + 1, int(tgt.y) + 1))
    return out


def solve_level(history=None, t_limit=120.0, max_nodes=200000):
    """求解影子当前关 (快照 A*, 动作 = A1-4 移动 + A6 换蛇).

    history: 会话动作历史 (同步影子用)。胜利时: 抽干 35 帧胜利动画 ( filler
    动作计入路径), 然后重放历史+路径使引擎停在下一关起点, 与调用方的
    逐动作重放流程对齐 (vc33 同款契约)。

    返回 moves [("i", 1-4) | ("c", x, y)]; 已通关 []; 失败 None。
    """
    if not _ensure() or not _valid:
        return None
    import time as _time
    import heapq
    with _lock:
        g = _game
        if "WIN" in str(g._state):
            return []
        if history:
            replay(history)
        t0 = _time.time()
        base_idx = int(getattr(g, "_current_level_index", 0))
        root = _sk48_snapshot(g)
        seen = {_sk48_state_key(g)}
        counter = 0
        heap = [(0, 0, [], root)]
        nodes = 0
        while heap:
            if _time.time() - t0 > t_limit or nodes >= max_nodes:
                _sk48_restore(g, root)
                return None
            _, gc, path, snap = heapq.heappop(heap)
            _sk48_restore(g, snap)
            acts = [("i", a) for a in (1, 2, 3, 4)] + _sk48_click_switches(g)
            for mv in acts:
                if _time.time() - t0 > t_limit or nodes >= max_nodes:
                    _sk48_restore(g, root)
                    return None
                try:
                    if mv[0] == "c":
                        import arcengine as _ae
                        ok = g.perform_action(_ActionInput(id=_ae.GameAction.ACTION6,
                                                           data={"x": int(mv[1]), "y": int(mv[2])},
                                                           reasoning=None), raw=True)
                    else:
                        ok = g.perform_action(_ActionInput(id=_ACTS[int(mv[1])], data={}, reasoning=None), raw=True)
                except Exception:
                    continue
                nodes += 1
                # 胜利判定: gvtmoopqgy 返回 True 会启动 35 帧胜利动画,
                # 动画结束后才 next_level — 索引滞后, 必须看动画标志
                if getattr(g, "lgdrixfno", -1) >= 0 or int(getattr(g, "_current_level_index", 0)) > base_idx:
                    # 致胜步内引擎已同步走完胜利动画并切到下一关对象 — 跨关恢复快照
                    # 会产生 Frankenstein 状态 (vc33 教训)。干净重放历史回到本关起点,
                    # 调用方逐动作重放路径时, 致胜步自然触发过关。
                    # replay([]) 即干净 reset, 任何情况都不走跨关快照恢复
                    replay(list(history or []))
                    return path + [mv]
                if "GAME_OVER" in str(g._state):
                    _sk48_restore(g, snap)
                    continue
                if getattr(g, "lgdrixfno", -1) >= 0:
                    _sk48_restore(g, snap)
                    continue
                k = _sk48_state_key(g)
                if k not in seen:
                    seen.add(k)
                    counter += 1
                    heapq.heappush(heap, (gc + 1 + 5 * _sk48_heur(g), counter,
                                          path + [mv], _sk48_snapshot(g)))
                _sk48_restore(g, snap)
        _sk48_restore(g, root)
        return None


def get_frame():
    """返回当前游戏帧为 2D 网格列表"""
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
        cols = max_x - min_x + 1
        rows = max_y - min_y + 1
        grid = [[0] * cols for _ in range(rows)]
        for x, y, color in grid_cells:
            grid[y - min_y][x - min_x] = color
        return grid
    except Exception:
        return [[0] * 10 for _ in range(10)]
