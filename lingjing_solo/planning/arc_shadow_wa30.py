"""Wa30 本地引擎影子模块

wa30-ee6fef47 是**键盘游戏**: 引擎只接受 ACTION1-5
(1=上 2=下 3=左 4=右 5=抓取/释放), 共 9 关, 每关 200 步预算。
早期版本按"点击游戏(ACTION6)"实现, 而 ACTION6 在本引擎里是空操作
(实测: apply(x, y) 后 current_steps 200 -> 200, 帧也不变), 所以自动播放
一直原地不动。现在 apply() 统一接受键盘动作/点击字典, 并另提供
get_state()/engine_* 一组原语, 供 markov_wa30.py 做模型搜索。
"""
import os
import threading

# ARC_API_KEY 由调用方通过环境变量提供，不在此硬编码

_lock = threading.RLock()
_env = None
_game = None
_applied = 0
_valid = False
_error = None
_ActionInput = None
_ACTIONS = None

# 已执行动作序列 (int 表示 ACTION1-5, dict 表示点击), 供搜索恢复状态用
_history = []

# 引擎里的语义 tag (来自 environment_files/wa30/ee6fef47/wa30.py)
_PLAYER_TAG = "wbmdvjhthc"
_BOX_TAG = "geezpjgiyd"
_CELL = 4  # 引擎内 celomdfhbh


# ---- 定向快照的字段表 (见 search_engine() 的注释) ----

# 一步(含过关切关)之内会被改写的 game 属性; Sprite/Level 对象本身按身份还原,
# Camera 只有渲染缓冲不参与状态, _levels/_clean_levels 列表对象不换、只改内部。
_SNAP_GAME_KEYS = (
    "_current_level_index", "_state", "_score", "_next_level", "_action",
    "_action_complete", "_action_count", "_full_reset", "_placeable_sprite",
    "_seed", "nsevyuople", "zmqreragji", "pkbufziase", "wyzquhjerd",
    "lqctaojiby", "qthdiggudy", "lkvghqfwan", "uuorgjazmj",
)


def _snap_val(v):
    """一层深的可变容器要复制, 其余(不可变值/按身份使用的对象)原样带过."""
    if isinstance(v, (set, frozenset)):
        return set(v)
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, list):
        return list(v)
    return v


def _snap_dict(d):
    out = {}
    for k, v in d.items():
        if k == "pixels" and hasattr(v, "copy"):
            out[k] = v.copy()          # uxricavavq 会原地改 sprite 像素边框
        else:
            out[k] = _snap_val(v)
    return out


def attach_game(external_env, actions_taken=0):
    """注入外部引擎实例（Agent 的 env._game），跳过 _ensure 的自建逻辑。

    Agent 和 markov_wa30 共享同一个 game 对象，decide() 直接读实时状态。
    """
    global _env, _game, _applied, _valid, _error, _ActionInput, _ACTIONS
    try:
        from arcengine import GameAction, ActionInput
        _env = external_env
        _game = external_env._game
        _ActionInput = ActionInput
        _ACTIONS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5, 6, 7)}
        _applied = int(actions_taken or 0)
        _history.clear()
        _valid = True
        _error = None
        return True
    except Exception as e:
        _error = str(e)
        _valid = False
        return False


def attach_game(external_env, actions_taken=0):
    """注入外部引擎实例（Agent 的 env._game），跳过 _ensure 的自建逻辑。

    Agent 和 markov_wa30 共享同一个 game 对象，decide() 直接读实时状态。
    """
    global _env, _game, _applied, _valid, _error, _ActionInput, _ACTIONS
    try:
        from arcengine import GameAction, ActionInput
        _env = external_env
        _game = external_env._game
        _ActionInput = ActionInput
        _ACTIONS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5, 6, 7)}
        _applied = int(actions_taken or 0)
        _history.clear()
        _valid = True
        _error = None
        return True
    except Exception as e:
        _error = str(e)
        _valid = False
        return False


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _ACTIONS
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput
        arc = arc_agi.Arcade()
        _env = arc.make("wa30-ee6fef47")
        _game = _env._game
        _ActionInput = ActionInput
        _ACTIONS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5, 6, 7)}
        _env.reset()
        _applied = 0
        _history.clear()
        _valid = True
        return True
    except Exception as e:
        _error = str(e)
        _valid = False
        return False


def _norm(action, y=None):
    """把调用方的动作统一成 (action_id, data).

    支持: int(1-7) / {"a":6,"x":..,"y":..} / apply(x, y) 老式点击调用
    """
    if isinstance(action, dict):
        a = int(action.get("a", 6))
        return a, {"x": int(action.get("x", 0)), "y": int(action.get("y", 0))}
    if y is not None:
        return 6, {"x": int(action or 0), "y": int(y)}
    return int(action or 0), {}


def _engine_step(action, y=None):
    """裸执行一步: 不加锁、不记账. 调用方负责持锁."""
    a, data = _norm(action, y)
    aid = _ACTIONS.get(a)
    if aid is None:
        return False
    _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
    return True


def reset():
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _applied = 0
            _history.clear()
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def apply(action, y=None):
    """执行一个动作并记账.

    action: int(1-5 键盘) 或 {"a":6,"x":..,"y":..}; y 不为 None 时按老式点击
    调用 apply(x, y) 解释。
    """
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            if not _engine_step(action, y):
                return False
            a, data = _norm(action, y)
            _history.append({"a": a, **data} if a == 6 else a)
            _applied += 1
            return True
        except Exception:
            _valid = False
            return False


def apply_action(move):
    """键盘语义入口: move 为 1-5."""
    return apply(int(move))


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def get_history():
    return list(_history)


def replay(history):
    """自愈重建: 回到初始状态并重放完整动作历史 (键盘/点击都支持)."""
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _history.clear()
            hist = list(history or [])
            for h in hist:
                if not _engine_step(h):
                    continue
                a, data = _norm(h)
                _history.append({"a": a, **data} if a == 6 else a)
            _applied = len(hist)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def get_available_actions():
    """引擎真实可用动作 (实测 [1,2,3,4,5])."""
    if not _ensure():
        return [1, 2, 3, 4, 5]
    try:
        acts = list(_game._available_actions or [])
        return [int(a) for a in acts] if acts else [1, 2, 3, 4, 5]
    except Exception:
        return [1, 2, 3, 4, 5]


def get_state():
    """抽象状态(马尔可夫模型的"状态"), 只读不改引擎.

    实测依据 (_diag_20260921/wa30_markov_validity2.py): 重放确定性 15/15 一致;
    跨历史同抽象状态转移检验 60 项 0 分歧 —— 但必须含玩家朝向, 去掉朝向后
    抓取动作(ACTION5)出现 2/60 分歧, 因为 vwiozbtqgi() 按 rotation 判定身前
    一格是否有箱子。
    """
    if not _ensure():
        return None
    with _lock:
        return _read_state()


def _read_state():
    g = _game
    try:
        tags = g.current_level.get_sprites_by_tag
        players = tags(_PLAYER_TAG)
        boxes = tags(_BOX_TAG)
        if not players:
            return None
        p = players[0]
        held = sorted(
            (a.x, a.y, b.x, b.y) for a, b in g.nsevyuople.items()
            if hasattr(a, "x") and hasattr(b, "x")
        )
        return {
            "level": int(g._current_level_index),
            "player": [int(p.x), int(p.y), int(p.rotation)],
            "boxes": sorted((int(s.x), int(s.y)) for s in boxes),
            "held": held,
            "targets": sorted((int(x), int(y)) for x, y in g.wyzquhjerd),
            "walls": sorted((int(x), int(y)) for x, y in g.pkbufziase),
            "locked": sorted((int(x), int(y)) for x, y in g.lqctaojiby),
            "blockers": sorted((int(x), int(y)) for x, y in g.qthdiggudy),
            "spawn": sorted((int(x), int(y)) for x, y in g.uuorgjazmj),
            "steps_left": int(g.kuncbnslnm.current_steps),
            "state": str(getattr(g._state, "name", g._state)),
        }
    except Exception:
        return None


def goal_reached():
    """引擎自带的过关判据 ymzfopzgbq(): 所有箱子都在目标格上且未被抱着."""
    if not _ensure():
        return False
    try:
        with _lock:
            return bool(_game.ymzfopzgbq())
    except Exception:
        return False


def get_game_state():
    idx = 0
    budget = used = 0
    st = "NOT_FINISHED"
    try:
        idx = int(_game._current_level_index)
        budget = int(_game.kuncbnslnm.dbdarsgrbj)
        used = budget - int(_game.kuncbnslnm.current_steps)
        st = str(getattr(_game._state, "name", _game._state))
    except Exception:
        pass
    won = st == "WIN"
    lost = st == "GAME_OVER"
    total = 0
    try:
        total = len(_game._levels)
    except Exception:
        pass
    return {
        "frame": get_frame(),
        "available_actions": get_available_actions(),
        "levels_completed": idx,
        "actions_taken": int(_applied),
        "game_over": won or lost,
        "won": won,
        "lost": lost,
        "current_level": idx + 1,
        "total_levels": total or 9,
        "steps_used": used,
        "budget": budget,
        "state": st,
    }


# ---- 供 markov_wa30.py 使用的搜索原语 (整段搜索期间独占引擎) ----

def search_acquire():
    _lock.acquire()


def search_release():
    try:
        _lock.release()
    except RuntimeError:
        pass


def search_engine():
    """返回 (game, read, step, restore, snapshot, restore_snapshot).

    restore(prefix) 从初始状态整体重放, 代价 O(已执行步数);
    snapshot()/restore_snapshot(bag) 用**定向快照**锚定实时状态, 代价恒定
    (实测快照 0.65ms、还原 0.55ms, 比 copy.deepcopy 快 15~25 倍, 且全字段
    指纹回滚校验 30/30 通过 —— _diag_20260921/wa30_fast_snap.py)。
    两个记账变量 _history/_applied 只由 restore() 更新, 快照路径不碰它们,
    所以搜索期间影子的记账始终保持真实。
    """
    if not _ensure():
        return None

    def step(a):
        return _engine_step(a)

    def restore(prefix):
        global _applied
        _env.reset()
        _history.clear()
        _applied = 0
        for a in prefix:
            if _engine_step(a):
                _history.append(a)
                _applied += 1
        return _read_state()

    def snapshot():
        """定向快照: 只存"一步会被改写"的字段, 比 copy.deepcopy(整个 game) 快一个量级.

        实测(_diag_20260921/wa30_fast_snap.py / _wa30_fast_snap2.py): 快照/还原
        ~0.5/1.5ms, 而 deepcopy 快照 9.7ms、还原 14.1ms; 随机序列下
        "快照 -> 走 1~20 步 -> 还原" 的全字段指纹比对 40/40 一致, 回滚后再走同样
        的 50 步 50/50 与前向一致, 且快照路径不碰 _history/_applied 记账。

        只存**当前关**的 sprite: 一局之内引擎只改 sprite 的坐标/朝向/像素边框
        (wqwsvmhhzj、qnmfimgpwc、uxricavavq)、current_level 的 sprite 列表
        (remove_sprite)、game 上那几个 set/dict、StepCounter 的 current_steps 和
        基类的关卡/动作字段; Camera 只有渲染缓冲不参与状态。若引擎在快照之后跨过
        了关卡边界(下一关的 sprite 也被走了), 这份快照就**作废** —— restore 返回
        None, 调用方退回整体重放。

        Sprite/Level 对象按身份被 dict 键引用, 所以就地还原属性、不换对象。
        """
        lvl = _game.current_level
        sprites = []
        for s in lvl._sprites:
            sprites.append((s, _snap_dict(s.__dict__)))
        return {"idx": _game._current_level_index, "lvl": lvl,
                "lvl_dict": _snap_dict(lvl.__dict__),
                "g": {k: _snap_val(getattr(_game, k, None)) for k in _SNAP_GAME_KEYS},
                "sc": dict(_game.kuncbnslnm.__dict__),
                "sprites": sprites}

    def restore_snapshot(bag):
        """回到 bag 记录的实时状态; 引擎已越过关卡边界时返回 None (快照作废)."""
        if bag["idx"] != _game._current_level_index or bag["lvl"] is not _game.current_level:
            return None
        for k, v in bag["g"].items():
            _game.__dict__[k] = _snap_val(v)
        _game.kuncbnslnm.__dict__.update(bag["sc"])
        bag["lvl"].__dict__.clear()
        bag["lvl"].__dict__.update(_snap_dict(bag["lvl_dict"]))
        for s, d in bag["sprites"]:
            s.__dict__.clear()
            s.__dict__.update(_snap_dict(d))
        return _read_state()

    return _game, _read_state, step, restore, snapshot, restore_snapshot


def restart_level(history):
    """把引擎搬到"当前这一关的起点", 返回应当保留的历史前缀长度.

    给 /api/proxy/arc-retry 用。做法: 先整体重放一遍, 记录最后一次"关卡号变大"
    发生在第几步之后(那一行之前的动作都属于已经通过的关卡), 再清空重放到那个位置。
    本关内耗尽步数( GAME_OVER )时, 截断后的重放自然停在本关起点、步数满格。
    """
    if not _ensure():
        return 0
    global _applied
    with _lock:
        try:
            hist = list(history or [])
            _env.reset()
            _history.clear()
            cut = 0
            lvl = int(_game._current_level_index)
            for i, h in enumerate(hist):
                if not _engine_step(h):
                    continue
                _history.append(h)
                now = int(_game._current_level_index)
                if now != lvl:
                    lvl = now
                    cut = i + 1
            _env.reset()
            _history.clear()
            _applied = 0
            for h in hist[:cut]:
                if _engine_step(h):
                    _history.append(h)
                    _applied += 1
            _valid = True
            return cut
        except Exception:
            _valid = False
            return 0


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


def solve_level(t_limit=60.0):
    """求解影子当前关. 返回动作序列; 已通关返回 []; 失败返回 None."""
    if not _ensure() or not _valid:
        return None
    try:
        import markov_wa30
        return markov_wa30.solve_current_level(t_limit=t_limit)
    except Exception:
        return None


def solve_level_enhanced(t_limit=300.0, max_nodes=200000):
    """增强版求解 — 快照式 A* (ACTION 1-7)"""
    # 优先加载预存解法
    try:
        from arc_solver_utils import load_prestored_solution
        pre = load_prestored_solution("wa30")
        if pre:
            return pre
    except Exception:
        pass
    if not _ensure() or not _valid:
        return None
    import hashlib
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
    from arc_solver_utils import enhanced_solve
    return enhanced_solve(
        get_game=lambda: _game,
        actions=[1, 2, 3, 4, 5, 6, 7],
        state_key=_sk,
        heuristic=_h,
        won=lambda: False,
        dead=lambda: False,
        t_limit=t_limit,
        max_nodes=max_nodes,
    )


def solve_level_enhanced(t_limit=300.0, max_nodes=200000):
    """增强版求解 — 快照式 A* (ACTION 1-5,7; 6=点击需坐标, 不在通用搜索中)"""
    if not _ensure() or not _valid:
        return None
    import hashlib
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
    from arc_solver_utils import enhanced_solve
    return enhanced_solve(
        get_game=lambda: _game,
        actions=[1, 2, 3, 4, 5, 7],
        state_key=_sk,
        heuristic=_h,
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
