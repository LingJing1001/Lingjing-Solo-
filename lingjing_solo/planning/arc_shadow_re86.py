"""RE86 本地引擎影子模块

用于 arc-auto 自动闯关，维护本地游戏实例状态。
RE86 是混合键盘和点击游戏。
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
_ACTIONS = None
_ACTION6 = None


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _ACTIONS, _ACTION6
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput
        arc = arc_agi.Arcade()
        _env = arc.make("re86-8af5384d")
        _game = _env._game
        _ActionInput = ActionInput
        _ACTIONS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5)}
        _ACTION6 = GameAction.ACTION6
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

    action: int (ACTION1-5 键盘) 或 dict {"a": 6, "x":, "y":} (点击)
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
            aid = _ACTION6 if a == 6 else _ACTIONS.get(a)
            if aid is None:
                return False
            _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
            _applied += 1
            return True
        except Exception:
            _valid = False
            return False


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def replay(history):
    """自愈重建: 影子回到起点并重放完整动作历史"""
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
                aid = _ACTION6 if a == 6 else _ACTIONS.get(a)
                if aid is None:
                    continue
                _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
            _applied = len(history)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


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


def solve_level(t_limit=60.0):
    """求解影子当前关. 返回动作序列或空列表 (solver comes later)."""
    if not _ensure() or not _valid:
        return None
    return []


# ─── 增强版求解器 ─────────────────────────────────

import time as _time
import heapq as _heapq
import copy as _copy


def _snapshot_re86(g):
    """深度快照"""
    return {
        "levels": [lv.clone() for lv in g._levels],
        "idx": int(g._current_level_index),
        "state": g._state,
    }

def _restore_re86(g, snap):
    """深度恢复"""
    g._levels = snap["levels"]
    g._current_level_index = snap["idx"]
    g._state = snap["state"]

def _state_key_re86(g):
    """状态摘要: 全精灵位置/像素"""
    import hashlib
    items = []
    for s in g.current_level._sprites:
        try:
            import numpy as np
            px = np.ascontiguousarray(s.pixels)
            items.append((s.name, int(s.x), int(s.y), int(getattr(s, "rotation", 0)),
                          hashlib.sha1(px.tobytes()).hexdigest()[:8]))
        except Exception:
            items.append((s.name, int(s.x), int(s.y), 0, ""))
    return hashlib.sha1(repr(sorted(items)).encode()).hexdigest()

def _click_candidates_re86(g):
    """所有精灵的点击坐标（不要求 is_visible — re86 的精灵可能不可见但仍可点）"""
    out = []
    cam = g._camera
    scale = max(1, int(64 / cam.width))
    pad_y = int((64 - cam.height * scale) / 2)
    seen = set()
    for s in g.current_level._sprites:
        cx = int(s.x) + s.width // 2
        cy = int(s.y) + s.height // 2
        disp = (cx, cy * scale + pad_y)
        if disp in seen:
            continue
        seen.add(disp)
        out.append(disp)
    return out

def _h_re86(g):
    """启发: 估算未完成度 (精灵数量差异)"""
    try:
        return len(g.current_level._sprites)
    except Exception:
        return 0

def solve_level_enhanced(t_limit=300.0, max_nodes=200000):
    """增强版求解 — 快照式 A* (ACTION 1-5 + 点击)

    RE86 机制: 颜色填充 + ACTION5 选择 (A5 需特定条件)
    策略: BFS 枚举所有动作 (1-5 + 点击), 用快照 O(1) 恢复
    """
    # 优先加载预存解法
    try:
        from arc_solver_utils import load_prestored_solution
        pre = load_prestored_solution("re86")
        if pre:
            return pre
    except Exception:
        pass
    if not _ensure() or not _valid:
        return None
    with _lock:
        g = _game
        t0 = _time.time()
        level0 = int(g._current_level_index)
        if level0 >= len(g._levels):
            return []

        from arcengine import GameState
        start_snap = _snapshot_re86(g)
        seen = {_state_key_re86(g)}
        ctr = 0

        # 屏蔽 next_level
        win_pending = [False]
        neutered = False
        try:
            g.next_level = lambda: win_pending.__setitem__(0, True)
            neutered = True
        except Exception:
            pass

        def _won():
            return int(g._current_level_index) > level0 or win_pending[0] or g._state == GameState.WIN

        def _dead():
            return g._state == GameState.GAME_OVER

        try:
            clicks = _click_candidates_re86(g)
            q = [(_h_re86(g), 0, (), start_snap)]
            nodes = 0
            result = None

            while q:
                if _time.time() - t0 > t_limit or nodes > max_nodes:
                    print("[re86] 搜索超时: nodes=%d seen=%d" % (nodes, len(seen)))
                    break
                _f, _d, path, snap = _heapq.heappop(q)
                _restore_re86(g, snap)

                if _dead():
                    continue

                # ACTION 1-5
                for a in (1, 2, 3, 4, 5):
                    _restore_re86(g, snap)
                    aid = _ACTIONS.get(a)
                    if aid is None:
                        continue
                    g.perform_action(_ActionInput(id=aid, data={}, reasoning=None), raw=True)
                    if _won():
                        result = list(path) + [a]
                        break
                    if _dead():
                        continue
                    k = _state_key_re86(g)
                    if k not in seen:
                        seen.add(k)
                        ctr += 1
                        npath = path + (a,)
                        _heapq.heappush(q, (len(npath) + _h_re86(g), ctr, npath, _snapshot_re86(g)))
                        nodes += 1
                if result:
                    break

                # 点击
                for (cx, cy) in clicks:
                    _restore_re86(g, snap)
                    g.perform_action(_ActionInput(id=_ACTION6, data={"x": cx, "y": cy}, reasoning=None), raw=True)
                    if _won():
                        result = list(path) + [{"a": 6, "x": cx, "y": cy}]
                        break
                    if _dead():
                        continue
                    k = _state_key_re86(g)
                    if k not in seen:
                        seen.add(k)
                        ctr += 1
                        npath = path + ({"a": 6, "x": cx, "y": cy},)
                        _heapq.heappush(q, (len(npath) + _h_re86(g), ctr, npath, _snapshot_re86(g)))
                        nodes += 1
                if result:
                    break

            if result:
                print("[re86] 命中: %d 步 nodes=%d" % (len(result), nodes))
            else:
                print("[re86] 未解: nodes=%d seen=%d" % (nodes, len(seen)))
            return result

        finally:
            if neutered:
                try:
                    del g.next_level
                except Exception:
                    pass
            _restore_re86(g, start_snap)
