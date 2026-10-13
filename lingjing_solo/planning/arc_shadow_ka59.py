"""KA59 本地引擎影子模拟器

KA59 机制: 键盘+鼠标混合 (ACTION1-4 移动选中块, ACTION6 点击切换选中)
求解策略: ka59_solver 回放 BFS (直读游戏源码实证机制)
"""
import os
import threading
import time

os.environ.setdefault("ARC_API_KEY", "f1d9f364-5ac8-4486-a4e9-6b9286de351f")

_lock = threading.RLock()
_env = None
_game = None
_applied = 0
_valid = False
_error = None
_ActionInput = None
_Acts = None
_ACTION6 = None


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _Acts, _ACTION6
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput
        arc = arc_agi.Arcade()
        _env = arc.make("ka59-38d34dbb")
        _game = _env._game
        _ActionInput = ActionInput
        _Acts = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4)}
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


def apply(action, y=None):
    """执行动作: int (键盘1-4) / dict {"a":6,"x","y"} (点击) / 旧式 apply(x, y) 双参点击"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            if y is not None:
                data = {"x": int(action), "y": int(y)}
                aid = _ACTION6
            elif isinstance(action, dict):
                a = int(action.get("a", 6))
                if a == 6:
                    data = {"x": int(action.get("x", 0)), "y": int(action.get("y", 0))}
                    aid = _ACTION6
                else:
                    data = {}
                    aid = _Acts[a]
            else:
                a = int(action)
                data = {}
                aid = _Acts.get(a)
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
    """自愈重建: 影子回到 L1 并重放完整动作历史 (int 键盘 / dict 点击 / 旧式 [_,x,y])"""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            for h in history:
                if isinstance(h, dict):
                    a = int(h.get("a", 6))
                    if a == 6:
                        data = {"x": int(h.get("x", 0)), "y": int(h.get("y", 0))}
                        aid = _ACTION6
                    else:
                        data = {}
                        aid = _Acts.get(a)
                elif isinstance(h, (list, tuple)) and len(h) >= 3:
                    data = {"x": int(h[1]), "y": int(h[2])}
                    aid = _ACTION6
                else:
                    data = {}
                    aid = _Acts.get(int(h))
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


def solve_level(t_limit=60.0, max_nodes=100000):
    """求解影子当前关: 专用 ka59_solver (键盘 1-4 + 点击切块, 回放 BFS).

    返回动作序列 (int 1-4 / dict {"a":6,"x","y"}); 已通关返回 []; 失败返回 None。
    注意: 求解结束会把影子 level_reset 回本关起点, 仅应在刚进关(本关 0 步)时调用。
    """
    if not _ensure() or not _valid:
        return None
    import ka59_solver
    with _lock:
        return ka59_solver.solve(_game, t_limit=t_limit, max_nodes=max_nodes)


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
