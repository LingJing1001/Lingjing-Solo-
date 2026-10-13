"""DC22 本地引擎影子模拟器 + BFS 求解器

DC22 机制 (钥匙开门迷宫):
- 玩家 (tag jfva) 用 ACTION1-4 移动, 每步 2px; 撞到实体精灵则回退
- 胜利: 玩家位置与终点 (tag goknoi) 完全重合
- 钥匙 (piyqze, 单字符 tag): 玩家走过 → 同字符的门 (buezna) 变为 INTANGIBLE
  (可穿过/可站立), 钥匙消失
- 点击门 (buezna): 关联的 aybe 精灵移出屏幕 (机关门打开)
- 水: 玩家站在无实体支撑处 → 沉没计时 → 失败 (BFS 中剪枝)

求解策略: 引擎 BFS — 动作 = 4 方向移动 + 点击每个 buezna 门;
状态键 = 全精灵 (位置, 交互模式) 摘要; 过关 = next_level 标记 (搜索期屏蔽)。
"""
import os
import threading
import time
from collections import deque

os.environ.setdefault("ARC_API_KEY", "f1d9f364-5ac8-4486-a4e9-6b9286de351f")

_lock = threading.RLock()
_env = None
_game = None
_applied = 0
_valid = False
_error = None
_ActionInput = None
_GameState = None
_ACTS = None
_ACTION6 = None


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _GameState, _ACTS, _ACTION6
    # 已就绪: 有注入/自建的 game 且引擎类已加载即视为可用 (adapter 每次 solve 前注入 _game,
    # 且会把 _env 置回 None —— 此处不得因 _env 为空就重建并把注入的 _game 顶掉,
    # 否则每次求解都在解自家 env 的 L0 而不是线上游戏当前关, 2026-10-06 实证)。
    if _valid and _game is not None and _ActionInput is not None:
        return True
    try:
        from arcengine import GameAction, ActionInput, GameState
        _ActionInput = ActionInput
        _GameState = GameState
        _ACTS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
        _ACTION6 = GameAction.ACTION6
        if _game is None:
            import arc_agi
            arc = arc_agi.Arcade()
            _env = arc.make("dc22-fdcac232")
            _game = _env._game
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
    """线上动作成功后调用.

    action: int (ACTION1-4) 或 dict {"a": 6, "x":, "y":}
    """
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            if isinstance(action, dict):
                a = int(action.get("a", 6))
                data = {"x": int(action.get("x", 0)), "y": int(action.get("y", 0))}
            else:
                a = int(action)
                data = {}
            aid = _ACTION6 if a == 6 else _ACTS[a]
            _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
            _applied += 1
            return True
        except Exception:
            _valid = False
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
                    a = int(h.get("a", 6))
                    data = {"x": int(h.get("x", 0)), "y": int(h.get("y", 0))}
                else:
                    a = int(h)
                    data = {}
                aid = _ACTION6 if a == 6 else _ACTS[a]
                _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
            _applied = len(history)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


# ─── 内部: 快照与 BFS ─────────────────────────────────

def _is_spr(v):
    return hasattr(v, "pixels") and hasattr(v, "_interaction")


def _handle_map(g, sprites_of, index_of):
    """把 game 对象上"引擎自己攥着的精灵句柄"记成可回指的形式。

    dc22 的 kghadhkkby (dc22.py:10517-10519) 会在点击 buezna 门 / 踩空重生时**整关重建精灵**,
    并把 g.qnnpcoyzd(玩家) / g.hfuqkxulm(终点) / g.jrxnntmty / g.deajuohco 换成新造的那批对象。
    快照还原换回去的是精灵表, 句柄若不同步回指, `_player()` 就会读到上一支遗留对象的坐标
    —— 实测 (本轮 `_diag_20261005/dc22_l1_identity_trace_out.txt`): 还原起点后 `_player=(10,24)` 而非 (10,30)。
    sprites_of/index_of 让浅/深两种快照各自决定"一条精灵的定位方式"(下标 vs (关卡下标, 精灵下标))。
    """
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
    return out


def _snapshot(g):
    obj_list = list(g.current_level._sprites)
    spr = []
    for s in obj_list:
        try:
            spr.append((s.x, s.y, s.pixels.copy(), getattr(s, "_interaction", None)))
        except Exception:
            spr.append((s.x, s.y, None, None))
    misc = (int(g._action_count), g._state, _steps_of(g))
    index = {id(s): i for i, s in enumerate(obj_list)}
    handles = _handle_map(g, lambda: obj_list, lambda s: index.get(id(s)))
    return (obj_list, spr, misc, handles)


def _steps_of(g):
    """关卡自带计步器 (dc22.py:9898-9910) —— 挂在 Camera 的 interface 对象上, 不在 Level 里,
    所以任何快照/还原都必须单独带上它, 否则搜索期会把它耗干 (见 _snapshot 的说明)."""
    c = getattr(g, "ujotjblwn", None)
    if c is None:
        return None
    return (int(c.hethagvldv), int(c.current_steps))


def _restore(g, snap):
    obj_list, spr, misc = snap[0], snap[1], snap[2]
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
    g._action_count, st = misc[0], misc[1]
    g._state = st
    if len(misc) > 2 and misc[2] is not None and getattr(g, "ujotjblwn", None) is not None:
        g.ujotjblwn.hethagvldv, g.ujotjblwn.current_steps = misc[2]
    # 句柄回指: 点击/踩空会把 g.qnnpcoyzd 等换成新克隆, 不回指就等于"回起点"没回玩家
    for k, spec in (snap[3] if len(snap) > 3 else {}).items():
        try:
            if spec[0] == "s":
                setattr(g, k, obj_list[spec[1]])
            else:
                setattr(g, k, [obj_list[i] for i in spec[1]])
        except Exception:
            pass


def _state_key(g):
    """状态摘要: 玩家位置 + 各精灵位置/交互模式 (像素不变的部分不参与)"""
    p = g.qnnpcoyzd
    items = []
    for s in g.current_level._sprites:
        try:
            items.append((s.name, int(s.x), int(s.y), str(getattr(s, "_interaction", ""))))
        except Exception:
            pass
    return (int(p.x), int(p.y), tuple(sorted(items)))


def _player(g):
    return (int(g.qnnpcoyzd.x), int(g.qnnpcoyzd.y))


def _exec(g, act):
    if act[0] == "m":
        g.perform_action(_ActionInput(id=_ACTS[act[1]], data={}, reasoning=None), raw=True)
    else:
        g.perform_action(_ActionInput(id=_ACTION6,
                                      data={"x": int(act[1]), "y": int(act[2])}, reasoning=None), raw=True)


# 点击候选验证缓存: (精灵名, 格子) -> 命中结果。布局静态(钥匙浮动除外),
# 同名同格的 xodizggcom 验证只做一次, 省掉 A* 每节点上百次 render()
_click_verify_cache = {}


def _door_clicks(g):
    """所有可点击精灵的命中格 (display 坐标).

    包括: buezna (门), sys_click (方向钮), aybe (相位切换器), piyqze (钥匙)。
    相机 64x44: scale=1, y 方向 letterbox 偏移 +10。

    命中格用引擎自己的 xodizggcom(x, y, "buezna") 验证 (与点击处理 10674 行同一
    查找): 门框中心镂空、艺术像素逐格不同, 几何中心经常静默落空 (2026-10-06 探针
    实证: L1 钥匙候选差 1-2 行)。每扇门取最多 2 个验证通过的格, 控制点击分支。
    """
    cam = g._camera
    scale = max(1, int(64 / cam.width))
    pad_y = int((64 - cam.height * scale) / 2)
    out = []
    seen = set()
    claimed = set()
    for s in g.current_level._sprites:
        tags = set(s.tags)
        clickable = bool(tags & {"buezna", "sys_click", "aybe", "piyqze"}) or s.name.startswith("sprite-1")
        if not clickable or not s.is_visible:
            continue
        if getattr(s, "_interaction", None) is not None and s._interaction.name == "REMOVED":
            continue
        w, h = int(s.width), int(s.height)
        x0, y0 = int(s.x), int(s.y)
        # 中心向外螺旋扫格, 用引擎查找验证命中
        cands = sorted(((gx, gy) for gy in range(h) for gx in range(w)),
                       key=lambda c: (c[0] - w / 2) ** 2 + (c[1] - h / 2) ** 2)
        picked = 0
        for gx, gy in cands:
            if picked >= 2:
                break
            cell = (x0 + gx, y0 + gy)
            if cell in seen:
                continue
            ck = (s.name, cell)
            if ck in _click_verify_cache:
                hit = _click_verify_cache[ck]
            else:
                try:
                    hit = g.xodizggcom(cell[0], cell[1], "buezna")
                except Exception:
                    hit = None
                _click_verify_cache[ck] = hit
            if hit is None and "buezna" in tags:
                continue  # 门必须验证命中, 宁缺毋滥
            if "buezna" in tags and hit is not None and s not in (hit,) and hit is not s:
                pass  # 命中的是该格最高层 buezna —— 同格多门时也算有效点击
            seen.add(cell)
            claimed.add(id(s))
            out.append((cell[0], cell[1] * scale + pad_y))
            picked += 1
        if picked == 0 and id(s) not in claimed:  # 验证全失败时退回中心
            cx, cy = x0 + w // 2, y0 + h // 2
            disp = (cx, cy * scale + pad_y)
            if disp not in seen:
                seen.add(disp)
                out.append(disp)
    return out


# ─── 求解 ─────────────────────────────────

def solve_level(t_limit=120.0, max_states=200000):
    """求解影子当前关. 返回动作序列 [("m", a) / ("c", x, y), ...]; 已通关 []; 失败 None."""
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        level0 = int(g._current_level_index)
        if level0 >= len(g._levels) or g._state == _GameState.WIN:
            return []

        doors = _door_clicks(g)
        start_snap = _snapshot(g)
        start_key = _state_key(g)
        seen = {start_key}
        queue = deque([(start_key, [])])
        result = None

        neutered = False
        try:
            g._win_pending = False

            def _nl():
                g._win_pending = True
            g.next_level = _nl
            neutered = True
        except Exception:
            neutered = False

        try:
            while queue and time.time() - t0 < t_limit and len(seen) < max_states:
                key, acts = queue.popleft()
                _restore(g, start_snap)
                for a in acts:
                    _exec(g, a)

                # 死亡/沉没剪枝
                if g._state == _GameState.GAME_OVER or getattr(g, "guspipewt", False):
                    continue

                # 每节点重算点击候选: 钥匙收集会把 REMOVED 的门翻成 INTANGIBLE(可点),
                # 门集合随路径演变, 开局算一次会漏掉后半程才出现的门 (L1 实证)
                doors = _door_clicks(g)
                for a in (1, 2, 3, 4):
                    _exec(g, ("m", a))
                    if int(g._current_level_index) > level0 or g._win_pending:
                        result = acts + [("m", a)]
                        break
                    np_ = _player(g)
                    nk = _state_key(g)
                    if nk not in seen and g._state != _GameState.GAME_OVER \
                            and not getattr(g, "guspipewt", False):
                        seen.add(nk)
                        queue.append((nk, acts + [("m", a)]))
                    _restore(g, start_snap)
                    for a2 in acts:
                        _exec(g, a2)
                if result:
                    break

                # 点击门
                for (cx, cy) in doors:
                    _exec(g, ("c", cx, cy))
                    if int(g._current_level_index) > level0 or g._win_pending:
                        result = acts + [("c", cx, cy)]
                        break
                    nk = _state_key(g)
                    if nk not in seen and g._state != _GameState.GAME_OVER \
                            and not getattr(g, "guspipewt", False):
                        seen.add(nk)
                        queue.append((nk, acts + [("c", cx, cy)]))
                    _restore(g, start_snap)
                    for a2 in acts:
                        _exec(g, a2)
                if result:
                    break
            return result
        finally:
            if neutered:
                try:
                    del g.next_level
                except Exception:
                    pass
            _restore(g, start_snap)


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


# ─── 增强版求解 ─────────────────────────────────

import heapq as _heapq
import copy as _copy

def _snapshot_deep(g):
    """深度快照 — 只克隆当前关 (next_level 已屏蔽, 搜索期内其余关永不变化,
    整表克隆是纯浪费; 6 关里 5 关的克隆成本省掉, A* 提速 ~5 倍) + 关卡计步器 + 引擎句柄定位

    句柄定位用 (关卡下标, 关卡内精灵下标): `Level.clone()` 逐条 `sprite.clone()` 且**保序**
    (见 arcengine Level.clone 源码), 所以还原后按下标就能指回对应克隆。
    """
    live = list(g._levels)
    idx = int(g._current_level_index)

    def _sprites_of():
        for s in live[idx]._sprites:
            yield s

    index = {}
    for si, s in enumerate(live[idx]._sprites):
        index[id(s)] = (idx, si)
    handles = _handle_map(g, _sprites_of, lambda s: index.get(id(s)))
    return {
        "levels": live[:idx] + [live[idx].clone()] + live[idx + 1:],
        "idx": idx,
        "state": g._state,
        "action_count": int(getattr(g, "_action_count", 0)),
        "steps": _steps_of(g),
        "handles": handles,
    }

def _restore_deep(g, snap):
    """深度恢复 — 当前关从模板再 clone 一层, 其余关共享引用 (它们在搜索期内不变)"""
    idx = snap["idx"]
    new_levels = list(snap["levels"])
    new_levels[idx] = snap["levels"][idx].clone()
    g._levels = new_levels
    g._current_level_index = snap["idx"]
    g._state = snap["state"]
    if hasattr(g, "_action_count"):
        g._action_count = snap["action_count"]
    steps = snap.get("steps")
    if steps is not None and getattr(g, "ujotjblwn", None) is not None:
        g.ujotjblwn.hethagvldv, g.ujotjblwn.current_steps = steps
    for k, spec in (snap.get("handles") or {}).items():
        try:
            if spec[0] == "s":
                setattr(g, k, g._levels[spec[1][0]]._sprites[spec[1][1]])
            else:
                setattr(g, k, [g._levels[li]._sprites[si] for li, si in spec[1]])
        except Exception:
            pass

def _h_dc22(g):
    """启发: 玩家到终点的曼哈顿距离; 钥匙未收集时取
    min(直达, 绕钥匙) —— L4 型关卡必须先拿钥匙解锁 d 门, 纯目标距离会把
    搜索引向死路 (2026-10-06 L4 5366 节点穷尽教训)。"""
    try:
        p = g.qnnpcoyzd
        goal = key = None
        key_gone = True
        for s in g.current_level._sprites:
            tags = getattr(s, "tags", set())
            if "goknoi" in tags:
                goal = s
            elif "piyqze" in tags:
                key = s
                key_gone = False
        if goal is None:
            return 0
        d_goal = abs(int(p.x) - int(goal.x)) + abs(int(p.y) - int(goal.y))
        if key_gone or key is None:
            return d_goal
        d_key = abs(int(p.x) - int(key.x)) + abs(int(p.y) - int(key.y))
        d_detour = d_key + abs(int(key.x) - int(goal.x)) + abs(int(key.y) - int(goal.y))
        return min(d_goal, d_detour)
    except Exception:
        return 0

def solve_level_enhanced(t_limit=300.0, max_nodes=300000):
    """增强版求解 — 快照式 A* + 宏动作

    vs solve_level: 用 per-node 快照替代 replay-from-start, O(1) 恢复
    # 优先加载预存解法
    try:
        from arc_solver_utils import load_prestored_solution
        pre = load_prestored_solution("dc22")
        if pre:
            return pre
    except Exception:
        pass
    """
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = time.time()
        level0 = int(g._current_level_index)
        if level0 >= len(g._levels) or g._state == _GameState.WIN:
            return []

        doors = _door_clicks(g)
        start_snap = _snapshot_deep(g)
        seen = {_state_key(g)}
        ctr = 0

        # 屏蔽 next_level
        neutered = False
        try:
            g._win_pending = False
            g.next_level = lambda: setattr(g, "_win_pending", True)
            neutered = True
        except Exception:
            pass

        def _won():
            return int(g._current_level_index) > level0 or g._win_pending

        def _dead():
            return g._state == _GameState.GAME_OVER or getattr(g, "guspipewt", False)

        try:
            # ── 策略 1: 快照式 A* (含点击) ──
            q = [(2 * _h_dc22(g), 0, (), start_snap)]
            nodes = 0
            result = None

            while q:
                if time.time() - t0 > t_limit * 0.6 or nodes > max_nodes:
                    print("[dc22] A*超时: nodes=%d seen=%d" % (nodes, len(seen)))
                    break
                _f, _d, path, snap = _heapq.heappop(q)
                _restore_deep(g, snap)

                if _dead():
                    continue

                # 每节点重算点击候选: 钥匙收集会把 REMOVED 的门翻成 INTANGIBLE(可点),
                # 门集合随路径演变 (L1 实证: 点 a 门换 brixto 竖柱是通关必要动作)
                doors = _door_clicks(g)

                # 移动候选
                for a in (1, 2, 3, 4):
                    _restore_deep(g, snap)
                    _exec(g, ("m", a))
                    if _won():
                        result = list(path) + [("m", a)]
                        break
                    if _dead():
                        continue
                    k = _state_key(g)
                    if k not in seen:
                        seen.add(k)
                        ctr += 1
                        npath = path + (("m", a),)
                        _heapq.heappush(q, (len(npath) + 2 * _h_dc22(g), ctr, npath, _snapshot_deep(g)))
                        nodes += 1
                if result:
                    break

                # 点击候选
                for (cx, cy) in doors:
                    _restore_deep(g, snap)
                    _exec(g, ("c", cx, cy))
                    if _won():
                        result = list(path) + [("c", cx, cy)]
                        break
                    if _dead():
                        continue
                    k = _state_key(g)
                    if k not in seen:
                        seen.add(k)
                        ctr += 1
                        npath = path + (("c", cx, cy),)
                        _heapq.heappush(q, (len(npath) + 2 * _h_dc22(g), ctr, npath, _snapshot_deep(g)))
                        nodes += 1
                if result:
                    break

            if result:
                print("[dc22] A*命中: %d 步 nodes=%d" % (len(result), nodes))
                return result

            # ── 策略 2: 原始 BFS 兜底 ──
            print("[dc22] A*未解, 回退原始BFS (剩余 %.0fs)" % (t_limit - (time.time() - t0)))
            _restore_deep(g, start_snap)
            return solve_level(t_limit=t_limit - (time.time() - t0), max_states=max_nodes)

        finally:
            if neutered:
                try:
                    del g.next_level
                except Exception:
                    pass
            _restore_deep(g, start_snap)
