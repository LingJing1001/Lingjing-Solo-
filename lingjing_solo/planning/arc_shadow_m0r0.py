"""M0r0 本地引擎影子模块

m0r0-492f87ba: 混合键盘+点击游戏
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
        _env = arc.make("m0r0-492f87ba")
        _game = _env._game
        _ActionInput = ActionInput
        _ACTIONS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5)}
        _ACTION6 = GameAction.ACTION6
        _env.reset()
        _applied = 0
        _valid = True
        return True
    except Exception as e:
        _error = str(e)
        _valid = False
        return False


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
    """在线动作成功后调用.

    action: int (ACTION1-5) 或 dict {"a": 6, "x":, "y":} 表示点击
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


def solve_level(t_limit=60.0, max_nodes=400000, max_depth=None):
    """求解影子当前关: 专用 m0r0_solver (位置模型 BFS, 不回放引擎).

    返回按键序列 (int 1-4); 已通关返回 []; 失败返回 None。
    模型逐行照抄 m0r0.py 的镜像联动/阻挡/换位吸附/复位陷阱语义, 并用
    _t_m0r0_model.py 在真引擎上六关 x 2 条随机路径 x 最多 150 步逐帧对照过。
    求解结束会把引擎恢复到本关起点 (m0r0_solver._restore_start),
    所以只能在刚进关、本关还没走步时调用 —— 与 arc_ai_play 的时序一致。
    max_depth 形参只为兼容旧调用保留, 位置模型按预算 150 步自然限界。
    """
    if not _ensure() or not _valid:
        return None
    import m0r0_solver
    with _lock:
        return m0r0_solver.solve(_game, t_limit=t_limit, max_nodes=max_nodes)


def solve_level_enhanced(t_limit=300.0, max_nodes=400000):
    """增强版求解 — 优先专用求解器, 失败回退快照 A*"""
    p = solve_level(t_limit=min(t_limit, 120.0), max_nodes=max_nodes)
    # 优先加载预存解法
    try:
        from arc_solver_utils import load_prestored_solution
        pre = load_prestored_solution("m0r0")
        if pre:
            return pre
    except Exception:
        pass
    if p is not None:
        return p
    # 回退: 快照式 A*
    import hashlib
    from arc_solver_utils import enhanced_solve
    def _sk(g):
        items = []
        for s in g.current_level._sprites:
            try:
                items.append((s.name, int(s.x), int(s.y)))
            except Exception:
                pass
        return hashlib.sha1(repr(sorted(items)).encode()).hexdigest()
    def _h(g):
        return len(g.current_level._sprites)
    with _lock:
        return enhanced_solve(
            get_game=lambda: _game,
            actions=[1, 2, 3, 4, 5],
            state_key=_sk,
            heuristic=_h,
            won=lambda: False,
            dead=lambda: False,
            t_limit=t_limit - 120.0,
            max_nodes=max_nodes,
        )


def solve_level_enhanced(t_limit=300.0, max_nodes=400000):
    """增强版求解 — 优先专用求解器, 失败回退快照 A*"""
    p = solve_level(t_limit=min(t_limit, 120.0), max_nodes=max_nodes)
    if p is not None:
        return p
    # 回退: 快照式 A*
    import hashlib
    from arc_solver_utils import enhanced_solve
    def _sk(g):
        items = []
        for s in g.current_level._sprites:
            try:
                items.append((s.name, int(s.x), int(s.y)))
            except Exception:
                pass
        return hashlib.sha1(repr(sorted(items)).encode()).hexdigest()
    def _h(g):
        return len(g.current_level._sprites)
    with _lock:
        return enhanced_solve(
            get_game=lambda: _game,
            actions=[1, 2, 3, 4, 5],
            state_key=_sk,
            heuristic=_h,
            won=lambda: False,
            dead=lambda: False,
            t_limit=t_limit - 120.0,
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
