"""AR25 影子模块 —— 直接驱动官方引擎 (arc_shadow)

背景: 本模块原先是纯 Python 逆向仿真器 (ar25_solver.simulator) 的绑定, 已归档为
arc_shadow_ar25_legacy.py。实测它与官方引擎机制不一致 —— 官方引擎上已验证的 L1
15 步解法拿到它上面执行只覆盖 1/5 目标格; ar25_solver.solver 对 L1/L2/L3 也求不出
解 (宏搜索日志显示"轴=无")。所以 AR25 的自动闯关改为直接驱动官方 arcengine 实例。

解法来源 = arc_shadow.KNOWN_SOLUTIONS, 载入时经"回放验证再接受"把关
(见 arc_shadow._load_solutions), 不会把搜索假解当作可用解法。

对外接口与此前的仿真器绑定保持兼容, app.py 无需改动。
"""
import arc_shadow as _sh

TOTAL_LEVELS = 8  # AR25 关卡总数 (KNOWN_SOLUTIONS 已覆盖 L1-L8)

# 帧颜色语义, 与 arc_shadow_ar25_legacy 的 21x21 帧保持一致
C_EMPTY, C_SPRITE, C_WALL, C_AXIS, C_TARGET = 0, 5, 9, 10, 11
BOARD = 21


def _ok():
    return bool(_sh._ensure())


def _legacy():
    import arc_shadow_ar25_legacy as lg
    return lg


def _level_index():
    g = _sh._game
    return int(g._current_level_index) if g is not None else 0


def _step_counters():
    """返回 (预算, 已用步数)。官方引擎的 current_steps 从 budget 递减。"""
    try:
        c = _sh._game.lelsvjlwneo
        budget = int(c.ilqnjlrnkk)
        return budget, budget - int(c.current_steps)
    except Exception:
        return 0, 0


# ─── 生命周期 ─────────────────────────────────────────

def reset():
    """重置到 L1 起点"""
    if not _ok():
        return _legacy().reset()
    return _sh.reset()


def apply(action):
    """执行动作, 返回 {success, won, level_advanced}（与原绑定同构）

    官方引擎在通关时会在同一次调用里跳关, 所以"关卡索引推进"即"本步通关"。
    """
    if not _ok():
        return _legacy().apply(action)
    before = _level_index()
    success = bool(_sh.apply(action))
    after = _level_index()
    advanced = after > before
    return {"success": success, "won": advanced, "level_advanced": advanced}


def replay(actions):
    """从 L1 起点重放动作历史, 自愈影子状态"""
    if not _ok():
        return _legacy().replay(actions)
    return _sh.replay(actions)


def is_synced(actions_taken):
    if not _ok():
        return _legacy().is_synced(actions_taken)
    return _sh.is_synced(actions_taken)


def current_solution():
    """当前关的已验证解法 (动作序列, 从本关起点算), 没有则 None

    给"智能播放"用: 按 steps_used() 取下一条即可, 无需现场搜索。
    """
    if not _ok():
        return None
    return _sh.KNOWN_SOLUTIONS.get(_level_index())


def solve_level(level_idx=None, t_limit=60.0, max_nodes=None):
    """求解当前关。官方引擎不可用时返回 None —— 绝不给出未经验证的动作。"""
    if not _ok():
        print("[arc_shadow_ar25] 官方引擎不可用, 跳过求解(避免给出未验证动作)")
        return None
    cur = _level_index()
    if level_idx is not None and int(level_idx) != cur:
        print(f"[arc_shadow_ar25] 只支持求解当前关: 请求 L{int(level_idx) + 1}, 当前 L{cur + 1}")
        return None
    if max_nodes is None:
        return _sh.solve_level(t_limit=t_limit)
    return _sh.solve_level(t_limit=t_limit, max_nodes=max_nodes)


# ─── 状态查询 ─────────────────────────────────────────

def current_level_index():
    if not _ok():
        return _legacy().current_level_index()
    return _level_index()


def levels_completed():
    if not _ok():
        return _legacy().levels_completed()
    return _level_index()


def steps_used():
    if not _ok():
        return _legacy().steps_used()
    return _step_counters()[1]


def get_available_actions():
    if not _ok():
        return _legacy().get_available_actions()
    return [1, 2, 3, 4, 5]


def _build_frame():
    """把官方引擎的关卡数据渲染成 21x21 帧 (含镜像反射)。

    颜色优先级与旧绑定一致: 目标格 > 轴 > 拼块, 拼块只填空白格。
    """
    ld = _sh._extract_level_data() or {}
    grid = [[C_EMPTY] * BOARD for _ in range(BOARD)]
    for i in range(BOARD):
        grid[0][i] = grid[BOARD - 1][i] = C_WALL
        grid[i][0] = grid[i][BOARD - 1] = C_WALL

    for t in ld.get("targets") or []:
        x, y = int(t["x"]), int(t["y"])
        if 0 <= x < BOARD and 0 <= y < BOARD:
            grid[y][x] = C_TARGET

    axes = ld.get("axes") or []
    for ax in axes:
        ax_x, ax_y = int(ax["x"]), int(ax["y"])
        for cell in ax.get("cells") or [[0, 0]]:
            x, y = ax_x + int(cell[0]), ax_y + int(cell[1])
            if 0 <= x < BOARD and 0 <= y < BOARD:
                grid[y][x] = C_AXIS

    def put(x, y):
        if 0 <= x < BOARD and 0 <= y < BOARD and grid[y][x] == C_EMPTY:
            grid[y][x] = C_SPRITE

    for sp in ld.get("sprites") or []:
        sx, sy = int(sp["x"]), int(sp["y"])
        for cell in sp.get("cells") or []:
            x, y = sx + int(cell[0]), sy + int(cell[1])
            put(x, y)
            for ax in axes:
                cells = ax.get("cells") or [[0, 0]]
                xs = {int(c[0]) for c in cells}
                ys = {int(c[1]) for c in cells}
                if len(xs) == 1:      # 竖轴: 沿 x 镜像
                    put(2 * int(ax["x"]) - x, y)
                elif len(ys) == 1:    # 横轴: 沿 y 镜像
                    put(x, 2 * int(ax["y"]) - y)
    return grid


def get_frame():
    if not _ok():
        return _legacy().get_frame()
    try:
        return _build_frame()
    except Exception as e:
        print(f"[arc_shadow_ar25] 构建帧失败: {e!r}")
        return [[C_EMPTY] * BOARD for _ in range(BOARD)]


def get_game_state():
    if not _ok():
        return _legacy().get_game_state()
    budget, used = _step_counters()
    idx = _level_index()
    return {
        "frame": get_frame(),
        "available_actions": get_available_actions(),
        "levels_completed": idx,
        "actions_taken": int(getattr(_sh, "_applied", 0)),
        "game_over": False,
        "won": False,
        "lost": False,
        "current_level": idx + 1,  # 1-based, 供展示
        "total_levels": TOTAL_LEVELS,
        "steps_used": used,
        "budget": budget,
    }


def __getattr__(name):
    """把 _game / _applied / _extract_level_data 等透传给 arc_shadow"""
    try:
        return getattr(_sh, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
