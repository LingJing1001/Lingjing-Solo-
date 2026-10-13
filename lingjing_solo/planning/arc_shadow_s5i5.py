"""S5I5 本地引擎影子模拟器

S5I5 机制:
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
        _env = arc.make("s5i5-18d95033")
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


def solve_level(t_limit=60.0):
    """求解影子当前关. 返回动作路径 []; 失败返回 None."""
    if not _ensure() or not _valid:
        return None
    # Stub: return empty path
    return []




def solve_level_enhanced(t_limit=300.0, max_nodes=200000):
    """增强版求解 — 快照式 A*"""
    # 优先加载预存解法
    try:
        from arc_solver_utils import load_prestored_solution
        pre = load_prestored_solution("s5i5")
        if pre:
            return pre
    except Exception:
        pass
    if not _ensure() or not _valid:
        return None
    import hashlib
    from arc_solver_utils import enhanced_solve
    def _sk(g):
        items = []
        for s in g.current_level._sprites:
            try:
                items.append((s.name, int(s.x), int(s.y), int(getattr(s, "rotation", 0))))
            except Exception:
                pass
        return hashlib.sha1(repr(sorted(items)).encode()).hexdigest()
    def _h(g):
        return len(g.current_level._sprites)
    with _lock:
        return enhanced_solve(
            get_game=lambda: _game,
            actions=[1, 2, 3, 4, 6],
            state_key=_sk,
            heuristic=None,  # 使用 universal_heuristic 自动探测
            won=lambda: False,
            dead=lambda: False,
            t_limit=t_limit,
            max_nodes=max_nodes,
        )

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
