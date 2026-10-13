"""Lf52 本地引擎影子模块

lf52-271a04aa: 键盘 + 鼠标混合游戏 (动作集实测 [1,2,3,4,6,7]: 1-4 滑子板、
6=点击、7=撤销; 5 不可用)。规则与 ka59 同一族 —— 光靠点击推不动局面,
两条通道必须一起搜: 第 2 关只走点击通道, 全空间只有 22 个可达状态、节点最少压到
2 个 (过关要 1 个), 见 _diag_20261001/lf52_l2_click_only_closure.py。

机制 (来源 environment_files/lf52/271a04aa/lf52.py, 逐条实测核对):
  - 节点 = tag 以 "fozwvlovdui" 开头的 sprite, 格坐标 chahdtpdoz (:3667);
    ndtvadsrqf() 是 startswith 匹配 (:4108-4110), 所以 red/blue 变体也算节点。
  - 点节点 A (:5334-5341 → xpcuvjyrgu :5353): 按四方向 vjafsffahp 检查
    qikmikecdf(A,d) —— 邻格 A+d 有节点 且 再外格 A+2d 是净空路格 —— 满足就把
    连线 sprite 挂在 A 上, 偏移 cdpcbbnfdp = (dx*2*6, dy*2*6) (:5361)。
    连线池 zpbguihjnf 固定 4 个、顺序与 vjafsffahp 对齐 (:5151-5154)。
  - 点连线 → cfilhtifcb (:5374): 目标格 = A + 偏移, 中点 = A+d; 中点上的**非 blue**
    节点若与 A 同名 (tag 相同, :5393-5397) 就被移除, 同时 A 滑到 A+2d (:5411)。
    即"滑动并吃掉中间节点", 每成功一次节点数 -1。
  - 过关判定 (:5572): 剩余节点数 == 1 (第 6/7 关为 == 2) → tdcblgbfxw → win (:2088)。
    第 8 关起先扣掉 blue 节点数 (:5376-5377), 所以启发值按"非 blue 节点数"取。
  - 键盘通道 (:5811-5826): ACTION1-4 → tmhxwcojkh(dx,dy) (:5276-5313), 按推送序
    整体平移 hupkpseyuim2 那批子板格, 踩在上面的节点跟着走 (xxlsofmxas)。
    它改变的是 qikmikecdf 要的"再外格净空路", 所以是能不能继续合并的开关。
  - 格↔像素: 网格原点 hncnfaqaddg.cdpcbbnfdp **逐关不同** (:5149/5157/5159/5161/5163),
    格边长 6 (:5191-5194) → 中心 = (6*gx + ox + 3, 6*gy + oy + 3)。
  - 判负 (:5771-5782): 动作计数 asqvqzpfdi 第 6 关前 >=320、之后 >=3200。

求解: **两阶段** —— 加权 best-first 队列先占 `_QUEUE_SHARE` 的时间片, 没解出来再把
**剩余**时间交给 `_dive` 俯冲 (深度优先 + `frames[d]` 兄弟挂起栈); 第 4 关只有俯冲解得动
(队列 75s 铺 242 状态无解, 俯冲 107s 出 84 步)。队列的评分函数是
f = 原子动作数 + 40*还差几次合并 + 2*节点最近邻距离和。
展开单位是**算子**而不是单步: 键盘滑子板 (1,)/(2,)/(3,)/(4,) —— 只在当前有
hupkpseyuim2 子板格时才生成, 第 1 关没有子板格, 键盘只会白扣判负计数; 合并算子
(点节点 A, 点 A+2d) 直接用引擎自己的闸口 qikmikecdf(A,d) 解析出来 (:5359-5361
挂线、:5383-5386 点线取的目标格就是 A+2d), 一个引擎步都不多花; 属性名失效时才
退回 _expand_ops_probe 的"真点一次再复位"; 板上已经挂着连线时额外给"单点连线"
算子 (_pending_connector_ops) —— 一个合并是两步点击, 调用方可能停在两步之间,
不给这一步的话解照样成立但会多算一步 (实测真实走过 5 击时返回 4 步、真正剩余 3 步)。
状态指纹 = (节点格, 挂出的连线格, 可滑子板格), 少任一项都会把不同局面误剪。
返回前还会从干净起点把整条解重放一次自证, 过关信号不成立就返回 None。

复位纪律 (实测踩过的坑, 见 _restore_start): 起搜一律 full_reset + 重放 _history
(不能用 _env.reset(), 它在 _action_count!=0 时只 level_reset 当前关); 每次回到起点
要把 关卡索引 + _score + _state 三者一起复原 —— 过关那一步在同一次 perform_action
内就换了关, 只 level_reset 会复位的其实是下一关, 搜索会解到别的关卡上去。搜索跨关界
展开时后面几关的 sprite 会被写脏, 所以 solve_level 收尾再调一次 _scrub_future_levels
把它们从 _clean_levels 重新 clone 回来。重放逐位可复现 (队尾帧与线上引擎 13/13 一致),
所以搜索不进 _history/_applied 记账。

诚实说明: 状态读取用的是引擎内部对象 (节点格、连线偏移), 属于特权信息, 不是从
像素反推的; 换到别的关卡包或重混淆命名后这些属性名会失效 —— 失效时 _read_state()
返回 None, solve_level() 明确返回 None 而不是乱点。
"""
import ast
import heapq
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
_solve_note = None      # 上一次 solve_level 的解出自哪条通道 (模型 / 引擎两阶段); 成功时 _error 被清空, 归因只能记在这里
_ActionInput = None
_ACTIONS = None
_ACTION6 = None
_last_frame = None

# 真实执行过的动作序列 (int=键盘, dict=点击), 搜索期间的重放以它为前缀
_history = []

# ---- 搜索期间的引擎游标 (不记账, 见 solve_level) ----
_cur_path = None          # 引擎当前反映的"真实状态之外的动作序列"
_level_prefix = []        # _history 中属于"当前这一关"的那段, 用 level_reset 复位时要重放
_search_steps = 0         # 本次搜索消耗的引擎步数
_search_level = 0         # 搜索起点关卡索引 (胜利着会把引擎带到下一关, 复位必须钉回这里)
_search_score = 0         # 搜索起点 _score (next_level() 会 +1, 不清掉过关信号会一直为真)
_search_state = None      # 搜索起点 _state (WIN/GAME_OVER 时 perform_action 直接返回空帧)

_NODE_TAG = "fozwvlovdui"
_PANEL_TAG = "hupkpseyuim2"   # ACTION1-4 搬的就是这批格 (lf52.py:5278)
_CELL = 6                 # 引擎内格边长 (lf52.py:5191-5194)
_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # 引擎模块级常量 vjafsffahp (lf52.py:5002)
_H_MERGE = 40             # 启发: 还差一次合并的权重
_H_DIST = 2               # 启发: 节点最近邻曼哈顿距离的权重
_MAX_ACTIONS = 120        # 兜底深度上限: 只在读不到引擎判负阈值时才用
_ACTION_CEIL = 640        # 引擎阈值本身的最大值 (第 6 关后 64*10), 防混淆名失效时失控

_MAX_SEARCH_STEPS = 200000  # 单关引擎步上限 (L1 实测 247 步 / 0.7s)
_MAX_STATES = 6000        # 搜索展开的状态上限
# 队列 best-first 占 t_limit 的比例, 其余留给俯冲兜底。
# 0.3 是实测定的 (lf52_dive_apply_ladder_out.txt, 2026-10-02):
#   - 队列解第 3 关要 46s, 给的是 46 步; 俯冲解同一关只要 33s, 但给 54 步 ——
#     分数按动作数计, 所以"能短就短"的队列必须留够时间片, 不能让俯冲全代打;
#   - 第 4 关队列 75s 铺 242 状态无解, 俯冲要 107s —— 时间片给过头就把俯冲挤死,
#     调用方 t_limit=240 时这一分成 72s/168s, 两关都够且留了机器抖动的余量。
# 诊断脚本要单跑某一阶段就直接改这个模块变量 (0.0 = 纯俯冲, 1.0 = 纯队列),
# solve_level 每次都读它, 不需要环境变量或改源码。
_QUEUE_SHARE = 0.3


def _depth_cap():
    """本关还剩多少原子动作可走 = 引擎自己的判负阈值 - 已消耗。

    阈值不是我们定的, 是 lf52.py:5769-5780 那三个分支:
      whtqurkphir == 1 → `asqvqzpfdi >= 64` 判负;
      whtqurkphir >= 6 → `>= 64 * 10` (=640);
      其余 (2~5)      → `>= 64 * 5` (=320)。
    `whtqurkphir` 是 **1 基**关卡号, `asqvqzpfdi` 在每次换关重建时归零 (:5140),
    键盘 (:5277) 和点击 (:5335) 两条通道都往它加数 —— 所以深度上限必须按关读,
    原先写死 120 会让第 5 关"潜到 119 就贴顶" (实测症状: 俯冲 150s 未解但最深 119)。

    **只能在搜索起点求值一次**: 循环里读的话, 已消耗量会把搜索自己走的路也算进去,
    深度越探上限越缩水, 等于又造一个隐性闸门。
    """
    try:
        w = _game.ikhhdzfmarl
        lv = int(w.whtqurkphir)
        lim = 64 if lv == 1 else (640 if lv >= 6 else 320)
        return max(8, min(_ACTION_CEIL, lim - int(w.asqvqzpfdi) - 2))
    except Exception:
        return _MAX_ACTIONS


def _ensure():
    global _env, _game, _applied, _valid, _error, _ActionInput, _ACTIONS, _ACTION6
    if _env is not None:
        return _valid
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput
        arc = arc_agi.Arcade()
        _env = arc.make("lf52-271a04aa")
        _game = _env._game
        _ActionInput = ActionInput
        # lf52 真实可用动作 [1,2,3,4,6,7]; 旧实现只建到 5, ACTION7(撤销) 取不到
        _ACTIONS = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5, 6, 7)}
        _ACTION6 = GameAction.ACTION6
        _env.reset()
        _applied = 0
        del _history[:]
        _valid = True
        return True
    except Exception as e:
        _error = str(e)
        _valid = False
        return False


def _norm(action, y=None):
    """把调用方的动作统一成 (action_id, data)。

    支持 int(1-7) / {"a":6,"x":..,"y":..} / [x, y] / 老式点击调用 apply(x, y)。
    """
    if isinstance(action, dict):
        a = int(action.get("a", 6))
        return a, {"x": int(action.get("x", 0)), "y": int(action.get("y", 0))}
    if isinstance(action, (list, tuple)):
        if len(action) == 2 and y is None:
            return 6, {"x": int(action[0]), "y": int(action[1])}
        return int(action[0]), {}
    if y is not None:
        return 6, {"x": int(action or 0), "y": int(y)}
    return int(action or 0), {}


def _engine_step(action, y=None):
    """裸执行一步: 不加锁、不记账、不进 _history。返回 FrameDataRaw 或 None。"""
    global _last_frame
    a, data = _norm(action, y)
    aid = _ACTIONS.get(a)
    if aid is None:
        return None
    _last_frame = _game.perform_action(
        _ActionInput(id=aid, data=data, reasoning=None), raw=True)
    return _last_frame


def reset():
    """影子回到整局起点 (第 1 关、历史清空)。

    记账清空了就得连关卡一起清空: _env.reset() 在 _action_count!=0 时只 level_reset
    当前关 (base_game:305-315), 那会留下"零历史却停在第 N 关"的错位状态。
    """
    global _applied, _valid, _cur_path, _last_frame
    if not _ensure():
        return False
    with _lock:
        try:
            _full_start()
            if _last_frame is None:
                _last_frame = _env.reset()
            _applied = 0
            del _history[:]
            _cur_path = None
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def apply(action, y=None):
    """执行一个动作并记账 (在线动作成功后调用)。

    action: int(1-7) 或 {"a":6,"x":..,"y":..}; y 不为 None 时按老式点击 apply(x, y) 解释。
    """
    global _applied, _valid
    if not _ensure():
        return False
    with _lock:
        try:
            if _engine_step(action, y) is None:
                return False
            a, data = _norm(action, y)
            _history.append({"a": a, **data} if a == 6 else a)
            _applied += 1
            return True
        except Exception:
            _valid = False
            return False


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def get_history():
    return list(_history)


def replay(history):
    """自愈重建: 从**整局起点**重放完整动作历史 (键盘 int / 点击 dict / 旧式 [x, y])。

    这里必须 full_reset 而不是 _env.reset(): 后者在 _action_count!=0 时只 level_reset
    当前关 (base_game:305-315), 拿整段历史去重放就会落在错误的关卡上。
    """
    global _applied, _valid, _cur_path
    if not _ensure():
        return False
    with _lock:
        try:
            _full_start()
            if _last_frame is None:
                _env.reset()
            del _history[:]
            _cur_path = None
            hist = list(history or [])
            for h in hist:
                if _engine_step(h) is None:
                    continue
                a, data = _norm(h)
                _history.append({"a": a, **data} if a == 6 else a)
            _applied = len(hist)
            _valid = True
            return True
        except Exception:
            _valid = False
            return False


def status():
    return {"valid": _valid, "applied": _applied, "error": _error, "loaded": _env is not None}


def get_available_actions():
    try:
        return [int(a) for a in (_game._available_actions or [])]
    except Exception:
        return [1, 2, 3, 4, 6, 7]


# ---- 引擎内部状态读取 (特权信息, 见模块头的诚实说明) ----

def _world():
    return _game.ikhhdzfmarl


def _grid():
    return _world().hncnfaqaddg


def _node_cells():
    return tuple(sorted((int(s.chahdtpdoz[0]), int(s.chahdtpdoz[1]))
                        for s in _grid().ndtvadsrqf(_NODE_TAG)
                        if getattr(s, "chahdtpdoz", None) is not None))


def _live_node_cells():
    """参与过关判定的节点格: 排除 blue 变体 (lf52.py:5375-5377/5393)。"""
    out = []
    for s in _grid().ndtvadsrqf(_NODE_TAG):
        if getattr(s, "chahdtpdoz", None) is None:
            continue
        if "blue" in str(getattr(s, "name", "") or ""):
            continue
        out.append(tuple(s.chahdtpdoz))
    return tuple(sorted(out))


def _pending_nodes():
    return len(_live_node_cells())


def _selected_cell():
    """当前被选中(挂着连线)的节点格; 没选中时 None (lf52.py:5378-5382)。"""
    sel = _world().wpwvsglgmb.qoifrofmiu
    cell = getattr(sel, "chahdtpdoz", None) if sel is not None else None
    return tuple(cell) if cell is not None else None


def _connector_cells():
    """连线的绝对格 = A + off//6 —— 与 cfilhtifcb 算 fldpqdkmge 同一套
    (lf52.py:5383-5386), 点这格才走合并分支。"""
    sc = _selected_cell()
    if sc is None:
        return ()
    out = []
    for s in _world().zpbguihjnf:
        off = getattr(s, "cdpcbbnfdp", (0, 0))
        if off and tuple(off) != (0, 0):
            out.append((sc[0] + int(off[0]) // _CELL, sc[1] + int(off[1]) // _CELL))
    return tuple(sorted(set(out)))


def _panel_cells():
    """可滑子板的格子集合: ACTION1-4 就是按推送序搬它们(以及踩着的节点)。

    证据 lf52.py:5276-5313 (tmhxwcojkh 取 whdmasyorl("hupkpseyuim2") 排序后
    xxlsofmxas(dx,dy)), 第 2 关实测 (4,4) 随 ACTION3/4 移到 (3,4)/(5,4)。
    """
    return tuple(sorted((int(s.grid_x), int(s.grid_y))
                        for s in _grid().whdmasyorl(_PANEL_TAG)))


def _center(cell):
    gx, gy = cell
    ox, oy = _grid().cdpcbbnfdp
    return (int(_CELL * gx + ox + _CELL // 2), int(_CELL * gy + oy + _CELL // 2))


def _read_state():
    """状态指纹 = (节点格, 挂出的连线格, 可滑子板格); 读不到内部对象返回 None。

    连线格和子板格都必须进键: 前者决定下一击能不能合并, 后者决定
    qikmikecdf 的净空路格在哪 —— 少一项就会把不同局面当同一个状态剪掉。
    """
    try:
        return (_node_cells(), _connector_cells(), _panel_cells())
    except Exception:
        return None


def _level_index():
    try:
        return int(_game._current_level_index)
    except Exception:
        return 0


def _lost():
    """引擎已判负 (步数/点击数超限, lf52.py:5771-5782)。"""
    try:
        return str(getattr(_game, "_state", "")).split(".")[-1] == "GAME_OVER"
    except Exception:
        return False


def _heuristic():
    """加权和: 还差几次合并 + 节点彼此的最近距离 (滑子板的意义就是把节点靠拢)。"""
    cells = [c for c in _node_cells()]
    n = len(cells)
    if n <= 1:
        return 0
    nn = 0
    for i, a in enumerate(cells):
        nn += min(abs(a[0] - b[0]) + abs(a[1] - b[1])
                  for j, b in enumerate(cells) if j != i)
    return _H_MERGE * (n - 1) + _H_DIST * nn


def _engine_act(a):
    """搜索用的原子动作: int=键盘滑动, (x,y)/dict=点击。"""
    global _search_steps
    _search_steps += 1
    if isinstance(a, tuple):
        return _engine_step({"a": 6, "x": int(a[0]), "y": int(a[1])})
    return _engine_step(a)


def _full_start():
    """回到整局起点 (第 1 关、_score=0)。

    不能用 _env.reset(): base_game.handle_reset (:305-315) 在 _action_count!=0 时
    只做 level_reset(当前关), 于是"把整段历史重放一遍"会落在错误的关卡上 ——
    影子过了第 1 关后起点就是错的, 搜索会顺带解到别的关上去。
    """
    global _last_frame
    try:
        _game.full_reset()
    except Exception:
        _last_frame = _env.reset()
        return
    try:
        from arcengine import GameAction, ActionInput
        _last_frame = _game.perform_action(
            ActionInput(id=GameAction.RESET, data={}, reasoning=None), raw=True)
    except Exception:
        _last_frame = None


def _mark_start():
    """记下当前引擎位置 = 本次搜索的起点 (关卡索引 / _score / _state)。"""
    global _search_level, _search_score, _search_state
    _search_level = _level_index()
    try:
        _search_score = int(_game._score)
    except Exception:
        _search_score = 0
    _search_state = getattr(_game, '_state', None)


def _replay_prefix():
    """整局 full_reset + 重放 _history 回到影子真实状态, 顺带量出"本关之内的前缀"。"""
    global _cur_path, _search_steps, _last_frame, _level_prefix
    _full_start()
    _search_steps += 1
    if _last_frame is None:
        _last_frame = _env.reset()
    idx = _level_index()
    _level_prefix = []
    for h in _history:
        if _engine_step(h) is None:
            continue
        _search_steps += 1
        now = _level_index()
        if now != idx:          # 跨关: 本关前缀从这一步之后重算
            idx = now
            _level_prefix = []
            continue
        _level_prefix.append(h)
    _cur_path = ()
    _mark_start()


def _restore_start():
    """把引擎带回本次搜索的起点 (关卡 + _level_prefix), 并清掉跳关/胜利残留。

    必须三样一起复原, 都实测过后果:
      - 关卡索引: next_level() 之后同一次 perform_action 里就 _really_set_next_level()
        换关了 (base_game:228-229/414-416), 只 level_reset "当前关" = 复位到下一关;
      - _score: next_level() 立刻 +1, 不清掉的话"过关"信号会一直为真;
      - _state: win()/lose() 之后 perform_action 直接返回空帧 FrameData(frame=[])
        (base_game:207-217), 后面每个动作都是空操作。
    """
    global _cur_path
    try:
        _game._next_level = False
        if _search_state is not None:
            _game._state = _search_state
        if _level_index() != _search_level:
            _game.set_level(_search_level)
        _game._score = _search_score
        _game.level_reset()
    except Exception:
        _replay_prefix()
        return
    _cur_path = ()
    for a in _level_prefix:
        _engine_act(a)
    # 约定: _cur_path 只记"搜索起点之外"的原子动作, _level_prefix 属于起点本身
    _cur_path = ()


def _rewind(path):
    """回到 "搜索起点 + path" 的局面 (比整局 full_reset 便宜得多)。"""
    global _cur_path
    _restore_start()
    for a in path:
        _engine_act(a)
    _cur_path = tuple(path)


def _scrub_future_levels():
    """把搜索期间可能被写脏的"后面几关"从 _clean_levels 重新克隆回来。

    _restore_start 只重建当前关 (level_reset 干的事), 但搜索一旦跨过关界就会在
    下一关上继续展开, 那几关的 sprite 对象就带着搜索留下的改动。真实玩家之后
    推进过去拿到的必须是干净关卡, 否则影子和线上就不是同一局了。
    只在 solve_level 收尾调一次 —— 每次 rewind 都克隆整摞关卡太贵。
    """
    try:
        clean = _game._clean_levels
        lv = _game._levels
        for i in range(_search_level + 1, min(len(clean), len(lv))):
            lv[i] = clean[i].clone()
    except Exception:
        pass


def _goto(path):
    """把引擎带到 "真实状态 + path", 返回 (状态指纹, 过关数)。

    path 是原子动作序列: int 1-4 = 键盘滑子板, (x, y) = 点击 (两步合并算子由
    _expand_ops 成对给出)。引擎已停在 path 的前缀上时只补走差额, 否则 _rewind ——
    即 _restore_start 把 关卡/_score/_state 复原后重放。重放逐位可复现是本地能当
    线上 oracle 的前提(已实测)。
    """
    global _cur_path
    if _cur_path is None or len(_cur_path) > len(path) or path[:len(_cur_path)] != _cur_path:
        _rewind(path)
    for a in path[len(_cur_path):]:
        _engine_act(a)
    _cur_path = tuple(path)
    st = _read_state()
    fr = _last_frame
    lc = int(getattr(fr, "levels_completed", 0) or 0) if fr is not None else 0
    return st, lc


def _node_list():
    """(节点格, 节点名) —— 名字决定 cfilhtifcb 能不能吃掉中点 (lf52.py:5393-5397)。"""
    out = []
    for s in _grid().ndtvadsrqf(_NODE_TAG):
        cell = getattr(s, "chahdtpdoz", None)
        if cell is None:
            continue
        out.append(((int(cell[0]), int(cell[1])), str(getattr(s, "name", "") or "")))
    return out


def _merge_ops():
    """解析式合并算子: 点节点 A + 点 A+2d。

    xpcuvjyrgu (:5353-5361) 就是拿 qikmikecdf(A, d) 当唯一闸口, 满足才把连线挂在
    偏移 (dx*12, dy*12) 上 —— 而点那条连线时 cfilhtifcb 的目标格 = A + off//6 = A+2d
    (:5383-5386)。所以直接问引擎自己的谓词就知道点下去会怎样, 一个引擎步都不用多花
    (逐节点真点一次再复位的做法, 每次展开要付 节点数 × 深度 步重放, 第 3 关 14 个节点
    时直接把预算吃光)。

    谓词读不到 (关卡包换了混淆名) 返回 None, 由调用方回退到试探式展开。
    """
    try:
        w = _world()
        ops = []
        for cell, _name in _node_list():
            for d in _DIRS:
                if not w.qikmikecdf(cell, d):
                    continue
                tgt = (cell[0] + 2 * d[0], cell[1] + 2 * d[1])
                ops.append((_center(cell), _center(tgt)))
        return ops
    except Exception:
        return None


def _pending_connector_ops():
    """半算子补完: 已经选中节点并挂着连线时, "点连线"本身就是一步可走的算子。

    一个合并 = 两步点击 (点节点 A, 点连线 A+2d), 而调用方可能在两步之间停下
    (arc-auto 逐着提交、或者上次解只走了一半)。这时只生成完整对等于把已经付出的
    那半击当没发生 —— 解仍然成立但多一步: 实测真实走过 5 击后返回 4 步, 而真正
    剩余是 3 步。连线格由 _connector_cells 直接读引擎的选中态, 不用试探。
    """
    try:
        return [(_center(c),) for c in _connector_cells()]
    except Exception:
        return []


def _expand_ops(path):
    """当前局面的展开算子 (每个算子 = 一串原子动作), 引擎须已停在 path 上。

    - 有可滑子板格时才有键盘算子 [1,2,3,4]: 第 1 关实测 hupkpseyuim2 集合为空,
      tmhxwcojkh 只会白扣 asqvqzpfdi (判负计数), 列进算子等于把分支因子从 5 撑到 9。
    - 合并算子 (点节点, 点连线) 由 _merge_ops 解析得到; 解析不了才退回 _expand_ops_probe。
    - 板上有现成连线时先给"单点连线"算子 (_pending_connector_ops), 让半截合并能一步收尾。
    - "单击选中"不单独成算子: 节点没有可挂方向时 xpcuvjyrgu 走的是 ebslbrugfo (:5671),
      那只是原地抖一下 (1,0)+(-2,0)+(1,0), 除了白扣一步什么也没改。
    """
    ops = []
    if _panel_cells():
        ops += [(1,), (2,), (3,), (4,)]
    ops += _pending_connector_ops()
    merged = _merge_ops()
    if merged is None:
        merged = _expand_ops_probe(path)
    out = list(ops)
    for m in merged:
        if m not in out:
            out.append(m)
    return out


def _expand_ops_probe(path):
    """回退路径: 真点一次节点、读引擎挂出的连线格, 试探完必须 _rewind(path) 复原游标。"""
    merged = []
    for cell, _name in _node_list():
        click = _center(cell)
        _engine_act(click)
        for cc in _connector_cells():
            pair = (click, _center(cc))
            if pair not in merged:
                merged.append(pair)
        _rewind(path)
    return merged


def _goal_hit(idx0, lc0, lc):
    """过关信号: 换关 / _score 增长 / 跳关标记 / 终局 win 都算。

    next_level() 里 _score 立即 +1 并置 _next_level (base_game:414-416),
    最后一次过关直接 win() 置 WIN 不换关 —— 只看关卡索引会漏掉它。
    队列搜索和俯冲搜索共用这一份判据, 两条通道对"过关"的定义必须逐字一致。
    """
    if _level_index() > idx0 or lc > lc0:
        return True
    if getattr(_game, '_next_level', False):
        return True
    return str(getattr(_game, '_state', '')).split('.')[-1] == 'WIN'


def _dive(idx0, lc0, st0, t_end, cap, stat=None):
    """俯冲搜索 (深度优先 + 兄弟挂起栈): 沿一条链推进, 只在死胡同回退一层。

    `cap` 是调用方在搜索起点按引擎判负阈值算出的深度上限 (见 _depth_cap), 单位是
    **原子动作数**。诊断要看 `stat["max_depth"]`: 贴着 cap = 上限错了(该抬),
    深度很浅就停 = 拓扑/启发错了(该换搜索), 这两种"未解"完全不是一回事。

    为什么队列 best-first 到第 3 关就到顶: 它每弹出一个节点, 都要为每个候选算子
    各重放一次整条前缀 (_goto 走到非前缀分支就是 _restore_start + 重放), 成本
    O(分支 × 深度)。第 3 关 14 个节点时解析式算子就有四十多个, 第 4 关实测 75s
    只铺开 242 个状态 / 10926 引擎步 —— 时间和引擎步都被"反复重铺浅层"吃掉,
    而不是被深度吃掉。俯冲把预算花在往深处走。

    完整性坑 (g50t/ka59 实测, 见 [[arc-source-first-replay-search]]):
    "每层只承诺评价最优的候选、落选兄弟只进 seen"的俯冲**不完整** —— 落选兄弟既已
    进 seen 又被当作已处理, 回退时该层直接判穷尽, 整棵兄弟子树被永久丢弃 (g50t L1
    21 节点、ka59 L1 29 节点就"穷尽根节点")。所以 frames[d] 挂起深度 d 全部已评估
    的新节点, 回退到该层时按 f 从小到大逐个下潜。

    代价是完整俯冲会空转: 启发值一路下降时它能贪心潜到深度上限而撞不到解,
    所以调用方让它当**兜底**, 队列先跑 (浅关队列又快又短)。

    stat 是诊断计数器 (max_depth / seen_hit / dead / expanded), 由调用方打印。
    返回原子动作序列 (int 1-4 键盘 / (x, y) 点击), 未解返回 None;
    引擎残留由调用方统一收尾 (_rewind + _scrub_future_levels), 这里不碰记账。
    """
    def goal(lc):
        return _goal_hit(idx0, lc0, lc)

    frames = [[]]                 # frames[d] = 深度 d 已评估、尚未下潜的兄弟 [(f, op), ...]
    seen = {st0}
    path = ()

    def unwind():
        """退到最近一层还有挂起兄弟的节点; 一路退空返回 False (搜索穷尽)。

        frames[len(path)] 是当前节点自己的挂起列表, 退到它时要保留, 只作废更深的帧。
        """
        nonlocal path
        while True:
            if frames[len(path)]:
                del frames[len(path) + 1:]
                return True
            if not path:
                del frames[1:]
                return False
            path = path[:-1]

    while True:
        if time.time() > t_end or _search_steps > _MAX_SEARCH_STEPS:
            if stat is not None:
                stat["stop"] = "时间/引擎步耗尽"
            return None
        d = len(path)
        st, lc = _goto(path)
        if st is None:
            if stat is not None:
                stat["stop"] = "状态读取失败"
            return None
        if goal(lc):
            return list(path)
        if d >= cap:
            if not unwind():
                if stat is not None:
                    stat["stop"] = "深度上限内穷尽"
                return None
            continue

        if not frames[d]:
            # 本层还没评估过: 每个算子只走一次, 新状态挂进 frames[d] 排队下潜
            for op in _expand_ops(path):
                npath = path + op
                st2, lc2 = _goto(npath)
                if st2 is None or _lost():
                    if stat is not None:
                        stat["dead"] = stat.get("dead", 0) + 1
                    continue
                if goal(lc2):
                    if stat is not None:
                        stat["max_depth"] = max(stat.get("max_depth", 0), len(npath))
                    return list(npath)
                if st2 in seen:
                    if stat is not None:
                        stat["seen_hit"] = stat.get("seen_hit", 0) + 1
                    continue
                seen.add(st2)
                if stat is not None:
                    stat["expanded"] = stat.get("expanded", 0) + 1
                frames[d].append((len(npath) + _heuristic(), tuple(op)))
            if stat is not None:
                stat["max_depth"] = max(stat.get("max_depth", 0), d)
                stat["seen"] = len(seen)

        if not frames[d]:
            if stat is not None:
                stat["exhausted"] = stat.get("exhausted", 0) + 1
            if not unwind():
                if stat is not None:
                    stat["stop"] = "兄弟挂起栈退空(穷尽)"
                return None
            continue

        # 只按 f 排: op 的结构异构 (键盘单步 (1,) vs 点击对 ((x,y),(x2,y2))),
        # 裸 sort() 在 f 相等时会去比 op → int 和 tuple 相比直接 TypeError,
        # 表现是 solve_level 报"搜索异常"、线上求解按钮失败 (2026-10-02 实测)。
        # sort 稳定, 所以同 f 的兄弟保持生成顺序。
        frames[d].sort(key=lambda e: e[0])
        _f, op = frames[d].pop(0)
        path = path + op
        while len(frames) < len(path) + 1:
            frames.append([])


# ---------------------------------------------------------------------------
# 关卡模型通道 (2026-10-02 晚新增): 把搜索从引擎里搬到关卡字符图上
# ---------------------------------------------------------------------------
# 为什么需要它: 引擎内两阶段搜索在第 5 关 601s 只铺开 951 个状态 (100295 引擎步,
# 平均**每铺开 1 个新状态花 105 个引擎步**), 俯冲最深 157 而深度上限 318 —— 预算全花在
# "每弹一个节点要为每个候选算子重放一次前缀"上, 既不是深度闸门也不是启发闸门
# (_diag_20261001/lf52_l5_fulllevel_macro_out.txt)。而这一关的全部规则可以从
# 关卡数据 + 三个引擎谓词静态闭合, 于是同一关在纯 Python 里 22.2s / 109492 状态出 84 步解,
# 第 6 关 6.6s / 33542 状态出 85 步解, 两份计划都在真引擎上逐手对拍通过
# (_diag_20261001/lf52_l5_plan_verify_out.txt、lf52_model_ladder_l5_l10_out.txt)。
#
# 规则出处 (lf52.py, 271a04aa):
#   字符图 + 映射表  grid5 图 :4787-4795, 映射 :4797-4812 (每关一份, 第 6 关起有新字符)
#   走法            qikmikecdf(A,d) = pymqmlkgzs(A+d) and posalhhmjq(A+2d)  :5649-5669
#                   住户 = 名含 fozwvlovdui/dgxfozncuiz; 落点 = 恰 1 个 sprite 且名含
#                   hupkpseyuim (净空垫), 或恰 2 个且其中有名**恰好** hupkpseyuim2 (子板格)
#   吃子            中点必须找得到**非 blue 且与移动者逐字同名**的 fozwvlovdui      :5393-5397
#                   → 红节点只吃红节点, 蓝节点谁都吃不掉也吃不了谁
#   按键            全体子板按推送序同向挪一格, 目标格含 kraubslpehi 家族即可;
#                   源格除轨道外的 sprite (子板 + 踩着的住户) 一起走              :5278-5294
#   过关            节点数 == 1 (第 6/7 关 == 2)                                   :5572
# 状态 = (节点 (格,名) 集, 诱饵格集, 子板格集); 静态层 (轨道/垫) 从映射表读, 活物从引擎读,
# 所以中途调用 (上一段只走了一半) 也能接上 —— 起点永远取引擎当前停住的位置。
_MODEL_TAG_NODE = "fozwvlovdui"
_MODEL_TAG_DECOY = "dgxfozncuiz"
_MODEL_TAG_SLIDER = "hupkpseyuim2"
_MODEL_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_MODEL_KEY = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
_MODEL_KD = {v: k for k, v in _MODEL_KEY.items()}
_MODEL_W = 8                 # 加权 A*: h 只有 2*(还差几次合并), 不加权就是等代价 BFS
_MODEL_TIME_SHARE = 0.4      # 模型通道占 t_limit 的比例, 剩下的仍归原来的两阶段引擎搜索
_model_ctx_cache = {}
_model_ctx_error = {}        # 关卡号 -> 解析失败原因, 只缓存成功解析 (失败可重试)


_model_src_cache = {}


def _model_src_text():
    """关卡源码正文 (整份 lf52.py)。

    不能用 `inspect.getsourcefile(type(_game))`: arc_agi 的加载器
    (site-packages/arc_agi/local_wrapper.py:126-129) 是
    `spec_from_loader(module_name, loader=None)` + `exec` 源码造出模块的, 这种类在
    inspect 眼里就是 built-in, 直接 TypeError("<class 'arc_agi_3.lf52-...>.Lf52'>
    is a built-in class") —— 出厂阶梯 2026-10-02 实测, 模型通道因此 L1~L5 每一关都
    在第一步静默失败 (旧代码整段包在 try 里, 只留下"关卡包换了结构?"这种误导话)。

    改按加载器自己的算法取路径, 两条通道依次试, 每条都要"文件打得开且正文里真有
    `\"grid`"才算数, 拿到就用不上的路径不会污染结果:
      1) environment_info.local_dir / <class_name 小写>.py  (和加载器同一套判据);
      2) 从类模块名 arc_agi_3.<game_id> 反推本仓库目录
         environment_files/<abbr>/<hash>/<abbr>.py。
    """
    if _model_src_cache:
        return next(iter(_model_src_cache.values()))
    cands = []
    try:
        info = getattr(_env, 'environment_info', None)
        ld = getattr(info, 'local_dir', None) if info is not None else None
        cn = (getattr(info, 'class_name', '') or '') if info is not None else ''
        if ld and cn:
            for nm in (str(cn).lower(), str(cn)):
                cands.append(os.path.join(str(ld), nm + '.py'))
    except Exception:
        pass
    try:
        gid = str(getattr(type(_game), '__module__', '')).split('.')[-1]
        if '-' in gid:
            abbr, rid = gid.split('-', 1)
            root = os.path.dirname(os.path.abspath(__file__))
            cands.append(os.path.join(root, 'environment_files', abbr, rid, abbr + '.py'))
    except Exception:
        pass
    for p in cands:
        try:
            with open(p, encoding='utf-8') as f:
                txt = f.read()
        except Exception:
            continue
        if '"grid' in txt:
            _model_src_cache[p] = txt
            return txt
    _model_ctx_error['src'] = '候选路径全部读不到: %s' % (cands or '(一条都没凑出来)')
    return None


def _model_ctx(level_no):
    """读关卡字符图与映射表 → 静态层 (轨道格 / 净空垫格) + 棋盘窗口尺寸; 解析不了返回 None。"""
    if level_no in _model_ctx_cache:
        return _model_ctx_cache[level_no]
    out = None
    try:
        import re as _re
        txt = _model_src_text()
        if txt is None:
            raise ValueError('关卡源码取不到 (%s)' % _model_ctx_error.get('src', '未记录'))
        i = txt.index('"grid%s": eollalrjeg(' % level_no)
        seg = txt[i:i + 3600]
        lines = seg.splitlines()[1:]
        rows, mapping = [], None
        for idx, line in enumerate(lines):
            s = line.strip()
            if mapping is None:
                m = _re.match(r'^"(.*)",$', s)
                if m:
                    rows.append(m.group(1))
                    continue
                if s.startswith('{'):
                    body = '\n'.join(l.strip() for l in lines[idx:])
                    depth = 0
                    for k, chr_ in enumerate(body):
                        if chr_ == '{':
                            depth += 1
                        elif chr_ == '}':
                            depth -= 1
                            if depth == 0:
                                break
                    mapping = ast.literal_eval(body[:k + 1])
                    break
        if not rows or not isinstance(mapping, dict):
            raise ValueError('关卡图解析失败')
        wd = max(len(r) for r in rows)
        rows = [r.ljust(wd) for r in rows]
        static, nodes0, decoys0, sliders0 = {}, set(), set(), set()
        for y, r in enumerate(rows):
            for x, chn in enumerate(r):
                names = mapping.get(chn)
                if names is None:
                    continue
                c = (x, y)
                static[c] = [n for n in names if n != _MODEL_TAG_SLIDER
                             and not n.startswith(_MODEL_TAG_SLIDER)
                             and _MODEL_TAG_NODE not in n and _MODEL_TAG_DECOY not in n]
                for n in names:
                    if _MODEL_TAG_NODE in n:
                        nodes0.add((c, n))
                    elif _MODEL_TAG_DECOY in n:
                        decoys0.add(c)
                    elif n == _MODEL_TAG_SLIDER:
                        sliders0.add(c)
        out = dict(level=int(level_no), w=wd, h=len(rows), static=static,
                   start_nodes=frozenset(nodes0), start_decoys=frozenset(decoys0),
                   start_sliders=frozenset(sliders0),
                   track=frozenset(c for c, ns in static.items()
                                   if any(n.startswith("kraubslpehi") for n in ns)),
                   start=(frozenset(nodes0), frozenset(decoys0), frozenset(sliders0)))
    except Exception as e:
        # 解析失败必须留下原因: 整段包在 try 里, 光返回 None 的话连 `import ast` 漏了
        # 这种自身的 bug 都会被读成"关卡包换了结构", 模型通道就悄无声息地不跑了
        _model_ctx_error[str(level_no)] = repr(e)
        out = None
    if out is not None:
        _model_ctx_cache[level_no] = out
    return out


def _model_names(ctx, state, c):
    """该格此刻的 sprite 名多重集 (静态层 + 活物) —— 谓词只看个数与成员, 顺序无关。"""
    out = list(ctx['static'].get(c, ()))
    for cc, n in state[0]:
        if cc == c:
            out.append(n)
    for cc in state[1]:
        if cc == c:
            out.append(_MODEL_TAG_DECOY)
    if c in state[2]:
        out.append(_MODEL_TAG_SLIDER)
    return out


def _model_occupant(ctx, state, c):
    return any(_MODEL_TAG_NODE in n or _MODEL_TAG_DECOY in n
               for n in _model_names(ctx, state, c))


def _model_landable(ctx, state, c):
    ns = _model_names(ctx, state, c)
    if len(ns) == 1 and "hupkpseyuim" in ns[0]:
        return True
    return len(ns) == 2 and _MODEL_TAG_SLIDER in ns


def _model_slide(ctx, state, d):
    """一次按键: 全体子板同向挪一格 (推送序 = 顺移动轴排), 踩着子板的住户跟着走。"""
    nodes, decoys, sliders = state
    order = sorted(sliders, key=lambda c: (c[0] if d[0] else c[1]),
                   reverse=(d[0] > 0 or d[1] > 0))
    cur, moves = set(sliders), []
    for c in order:
        t = (c[0] + d[0], c[1] + d[1])
        if t in ctx['track'] and t not in cur:
            cur.discard(c)
            cur.add(t)
            moves.append((c, t))
    if not moves:
        return state, []
    nn, nd, ns = set(nodes), set(decoys), set(sliders)
    for a, b in moves:
        ns.discard(a); ns.add(b)
        nn = {(c, n) if c != a else (b, n) for (c, n) in nn}
        nd = {c if c != a else b for c in nd}
    return (frozenset(nn), frozenset(nd), frozenset(ns)), moves


def _model_jump(ctx, state, A, d):
    """一次点击走法 (点节点 + 点落点, 2 个原子动作)。返回 (新状态, 是否真合并) 或 None。"""
    nodes, decoys, _sliders = state
    name = dict(nodes).get(A)
    if name is None:                 # 移动者必须是节点; blue 节点照样能"跳"过诱饵,
        return None                  # 只是吃不掉任何东西 —— 由下面的 eat 判定管
    mid = (A[0] + d[0], A[1] + d[1])
    land = (A[0] + 2 * d[0], A[1] + 2 * d[1])
    if not _model_occupant(ctx, state, mid) or not _model_landable(ctx, state, land):
        return None
    midname = dict(nodes).get(mid)
    eat = midname is not None and midname == name and 'blue' not in name
    nn = {(c, n) for (c, n) in nodes if c != A and not (eat and c == mid)}
    nn.add((land, name))
    return (frozenset(nn), decoys, _sliders), eat


def _model_legal(ctx, state):
    out = []
    for d in _MODEL_DIRS:
        ns, mv = _model_slide(ctx, state, d)
        if mv:
            out.append(('k', _MODEL_KEY[d]))
    for A in sorted(c for c, _n in state[0]):
        for d in _MODEL_DIRS:
            if _model_jump(ctx, state, A, d):
                out.append(('c', A, (A[0] + 2 * d[0], A[1] + 2 * d[1])))
    return out


def _model_move(ctx, state, mv):
    if mv[0] == 'k':
        ns, m = _model_slide(ctx, state, _MODEL_KD[mv[1]])
        return ns, bool(m), False
    A, land = mv[1], mv[2]
    dd = (land[0] - A[0], land[1] - A[1])
    if abs(dd[0]) + abs(dd[1]) != 2:
        return state, False, False
    r = _model_jump(ctx, state, A, (dd[0] // 2, dd[1] // 2))
    if not r:
        return state, False, False
    return r[0], True, r[1]


def _model_astar(ctx, start, goal, cap, t_end, stat):
    """纯模型加权 A*; 返回走法序列 (格坐标语义) 或 None。零引擎步。"""
    seen = {start: 0}
    heap = [(_MODEL_W * 2 * max(0, len(start[0]) - goal), 0, 0, start, ())]
    ctr = 0
    t0 = time.time()
    while heap:
        if time.time() > t_end:
            stat['stop'] = '模型时间片耗尽'
            return None
        _f, neg_g, _, s, path = heapq.heappop(heap)
        g = -neg_g
        stat['expanded'] = stat.get('expanded', 0) + 1
        stat['min_nodes'] = min(stat.get('min_nodes', 99), len(s[0]))
        if len(s[0]) <= goal:
            stat['atom'] = g
            return path
        if g >= cap:
            continue
        for mv in _model_legal(ctx, s):
            ng = g + (1 if mv[0] == 'k' else 2)
            if ng > cap:
                continue
            ns, ok, _eat = _model_move(ctx, s, mv)
            if not ok or seen.get(ns, 1 << 30) <= ng:
                continue
            seen[ns] = ng
            ctr += 1
            heapq.heappush(heap, (ng + _MODEL_W * 2 * max(0, len(ns[0]) - goal),
                                  -ng, -ctr, ns, path + (mv,)))
    stat['stop'] = '模型穷尽(状态空间铺完)'
    stat['states'] = len(seen)
    return None


def _model_live_state(ctx):
    """模型起点从**引擎当前停住的位置**读, 不从字符图读 —— 中途调用才接得上。"""
    try:
        nodes = frozenset(((int(s.chahdtpdoz[0]), int(s.chahdtpdoz[1])), str(s.name))
                          for s in _grid().ndtvadsrqf(_MODEL_TAG_NODE)
                          if getattr(s, "chahdtpdoz", None) is not None)
        w = _world()
        occ = set()
        for x in range(ctx['w']):
            for y in range(ctx['h']):
                if w.pymqmlkgzs((x, y)):
                    occ.add((x, y))
        decoys = frozenset(occ - {c for c, _n in nodes})
        return (nodes, decoys, frozenset(_panel_cells()))
    except Exception:
        return None


def _model_channel(goal, cap, t_end):
    """模型通道: 搜走法 → 沿路把格坐标换算成像素 → 顺手在引擎上重放自证。

    返回 (原子动作序列 | None, 说明)。换算与自证是同一次: 键盘 = 1 个原子动作,
    点击走法 = 先点移动者格、再点落点格 (像素中心每手现读, 因为相机会跟着走位移动)。
    任一手引擎状态与模型预测不符 (节点格 / 子板格 / 住户数) 就整体放弃、返回 None,
    由调用方退回引擎两阶段搜索 —— 模型只能被打脸, 不能反驳引擎。
    """
    lv = None
    try:
        lv = int(_world().whtqurkphir)
    except Exception:
        lv = _search_level + 1
    ctx = _model_ctx(lv)
    if ctx is None:
        return None, ('第 %d 关字符图读不到 (关卡包换了结构?): %s' % (
            lv, _model_ctx_error.get(str(lv), '(未记录原因)')))
    start = _model_live_state(ctx)
    if start is None:
        return None, '模型起点读取失败'
    if len(start[0]) <= goal:
        return [], '本关目标已满足'
    stat = {}
    t0 = time.time()
    path = _model_astar(ctx, start, goal, cap, t_end, stat)
    if path is None:
        return None, '%s: 展开 %d 状态 / 节点最少 %d / %.1fs' % (
            stat.get('stop', '无解'), stat.get('expanded', 0),
            stat.get('min_nodes', len(start[0])), time.time() - t0)
    actions = []
    state = start
    _rewind(())
    idx0 = _level_index()
    score0 = int(getattr(_game, '_score', 0) or 0)
    won = False
    for mv in path:
        if mv[0] == 'k':
            atom, acts = ('k', mv[1]), [mv[1]]
            actions.append(mv[1])
        else:
            A, L = mv[1], mv[2]
            atom, acts = ('c', A, L), [_center(A), _center(L)]
            actions += [tuple(_center(A)), tuple(_center(L))]
        for a in acts:
            _engine_act(a)
        # 过关那一手引擎当场把棋盘换成下一关 (base_game:228-229/414-416),
        # 之后再拿 _node_cells()/_panel_cells() 和模型对拍就是在比两张不同的棋盘 ——
        # 末手"不符"的假报警实测过一次 (_diag_20261001/lf52_l5_plan_verify.py)。
        if (_level_index() > idx0 or int(getattr(_game, '_score', 0) or 0) > score0
                or getattr(_game, '_next_level', False)
                or str(getattr(_game, '_state', '')).split('.')[-1] == 'WIN'):
            won = True
            break
        ns, ok, _eat = _model_move(ctx, state, atom)
        if not ok:
            return None, '模型自证: 走法 %s 在模型里就不合法' % (atom,)
        state = ns
        if len(_node_cells()) != len(state[0]) or set(_panel_cells()) != set(state[2]):
            return None, ('模型自证: 第 %d 个走法之后引擎与模型不符 (节点 引擎%d/模型%d, '
                          '子板 引擎%d/模型%d)' % (
                              len(actions), len(_node_cells()), len(state[0]),
                              len(_panel_cells()), len(state[2])))
        if _lost():
            return None, '模型自证: 第 %d 个原子动作引擎判负' % len(actions)
    if not won:
        return None, ('模型自证: %d 个走法走完没有过关信号 (节点 引擎%d/模型%d)' % (
            len(path), len(_node_cells()), len(state[0])))
    stat['seconds'] = round(time.time() - t0, 1)
    stat['atoms'] = len(actions)
    return actions, '模型通道解出 %d 个走法 / %d 个原子动作 / %.1fs' % (
        len(path), len(actions), stat['seconds'])


def solve_level(t_limit=60.0):
    """求解影子当前关: 返回动作序列 —— int 1-4 = 键盘滑子板, {"a":6,"x":..,"y":..} = 点击。

    已过关返回 []; 读不到内部状态 / 预算内搜不到解返回 None。
    搜索只在引擎内部重放 (level_reset + 本关前缀), 不进 _history / _applied 记账,
    结束时把影子复位回真实状态, 调用方按返回路径逐个 apply() 即可。

    动作模型 = 键盘 + 鼠标两条通道, 与 ka59 同一规则 (metadata tags=keyboard_click,
    ka59.py:41133 available_actions=[1,2,3,4,6]; 1-4 移动选中块、6 切换选中)。
    lf52 这边是 ACTION1-4 滑 hupkpseyuim2 子板 (lf52.py:5811-5826 → :5276-5313)、
    ACTION6 选节点/点连线合并; 第 2 关只走点击通道搜完全空间只有 22 个可达状态、
    最少压到 2 个节点 (过关要 1 个) —— 见 _diag_20261001/lf52_l2_click_only_closure.py,
    所以两条通道必须一起搜。

    搜索 = 加权 best-first, f = 原子动作数 + (40*还差几次合并 + 2*节点最近邻距离和);
    状态指纹去重 (节点格, 连线格, 子板格)。预算: 时间 t_limit、引擎步、原子动作深度、
    状态数四个上限, 任一耗尽就返回 None, 不返回半条路径。

    调度 = **模型通道 → 队列 → 俯冲** 三段, 预算按绝对时间戳切: 模型通道先拿
    `_MODEL_TIME_SHARE`, 队列再拿**模型实际用完之后剩余**时间的 `_QUEUE_SHARE`,
    最后剩下的交给 `_dive` (深度优先 + 兄弟挂起栈)。
    为什么模型要排在最前 (2026-10-02 晚): 深关卡在引擎里搜不动 —— 第 5 关 601s 铺开
    951 个状态 = 每个新状态 105 个引擎步, 而同一关的关卡字符图模型 22s 出 84 步解
    (第 6 关 7s/85 步), 两份计划都在真引擎逐手对拍通过。模型的输出还要沿路在引擎上
    重放自证, 任一手格数对不上就整体作废, 所以它只会交真解或让位给引擎搜索。
    引擎两阶段保留的理由是浅关它又快又短、且是唯一不依赖关卡字符图结构的通道:
    队列按 f 排序, 浅关一步到位且路径短 (L1 0.25s/8 步、L3 51s/46 步); 深关它每弹一个
    节点都要为每个候选算子重放一次整条前缀 (O(分支×深度)), 预算全花在重铺浅层,
    于是靠俯冲走得动。截止点一律是绝对时间戳 `t0 + t_limit`, 不传"剩余秒数" ——
    g50t 实测那样会把 420s 的限制实际跑成 252s。
    诊断要单跑某一段: `_MODEL_TIME_SHARE = 0.0` 关掉模型通道, `_QUEUE_SHARE` 调队列
    (0.0 = 纯俯冲, 1.0 = 纯队列)。

    实测 (offline; 证据 _diag_20261001/lf52_crawl_v3.txt、lf52_l1_path_exact_check.py、
    lf52_midlevel_recheck.py、lf52_dive_l4_out.txt、lf52_dive_apply_ladder_out.txt):
    —— 前四条是 2026-10-02 上午**引擎两阶段时代**的读数, 保留是为了说明两阶段各自的能力边界;
    现在出厂由模型通道出解, 请看下面"出厂配置 L1~L6 阶梯重跑"那条 (L4 已从 84 降到 50)。
      - 第 1 关 8 击 / 0.25s / 41 引擎步, 与已上线重放通过的那条 8 击逐位一致;
      - 第 2 关 34 步 (键盘 26 + 点击 8);
      - 第 3 关 (14 节点) 46 步 (键盘 20 + 点击 22 + 画面外点击 4) —— 队列 46s,
        纯俯冲 33s 但给的是 54 步 (更长), 这就是两阶段要保留队列时间片的原因;
      - **第 4 关 (6 节点 3 子板) 俯冲解出: 84 步 (键盘 52 + 点击 32, 无出画面点击)
        / 107s / 19581 引擎步**; 队列阶段 75s 只铺 242 状态, 完全够不着;
      - 上面四关的解逐个走真实 apply 通道后, 引擎关卡索引都**正好 +1** (1/2/3/4),
        收尾残留为 0 (_next_level=False、state=NOT_FINISHED、历史步数 == applied),
        把整段历史 replay 回同一位置也一致 (历史 180 步 -> 关卡索引 4);
      - 第 5 关: 深度上限写死 120 时纯俯冲 150s 未解且"最深 119"贴顶; 换成 `_depth_cap`
        直读引擎阈值后复测 (_diag_20261001/lf52_l5_depthcap_out.txt), 最深走到 161/上限318
        —— 深度闸门解除了, 但 300s / 64624 引擎步只铺开 380 个新节点, 仍未解。
        当时据此判定"L5 卡的是**启发引导**, 下一个杠杆是分阶段子目标"。
        **10-02 晚复核: 这条判断错了。** 真正的闸门是引擎内搜索的**重放成本**: 第 5 关
        601s 只铺开 951 个状态、100295 个引擎步 = 每个新状态 105 个引擎步, 启发还没来得及
        被检验预算就没了 (_diag_20261001/lf52_l5_fulllevel_macro_out.txt)。把搜索搬到关卡
        字符图上之后同一关 22s/109492 状态出解, 所以杠杆是**换搜索空间**, 不是加启发、
        也不是加时间。
      - 第 5/6 关由模型通道解出 (对拍证据 lf52_l5_plan_verify_out.txt、
        lf52_model_ladder_l5_l10_out.txt): L5 = 84 个原子动作 / 68 个走法 / 22s,
        L6 = 85 个原子动作 / 55 个走法 / 7s, 两份计划都在真引擎上逐手对拍通过、
        关卡索引正好 +1; 第 7 关 (3 节点 15 诱饵 6 子板) 模型 120s 未解, 最少压到 3 个节点。
      - **出厂配置 L1~L6 阶梯重跑** (offline, `_MODEL_TIME_SHARE=0.4 _QUEUE_SHARE=0.3
        t_limit=240`, 逐手走真实 apply 通道; 证据 _diag_20261001/lf52_shippath_ladder_v5_model_out.txt):
        六关**全部由模型通道解出**, 引擎两阶段一次都没被调用 ——
        L1 8步/0.0s, L2 34步/0.1s, L3 46步/0.4s, **L4 50步/0.6s**, L5 84步/22.4s, L6 85步/7.1s;
        每关关卡索引都正好 +1, 合计 307 步, 收尾 `_next_level=False / state=NOT_FINISHED /
        历史=307=applied / is_synced=True`, 把 307 步独立 replay 回同一位置一致。
        比 v3 出厂配置 (引擎两阶段, 4 关 172 步 / 全程约 250s) 快了约 7 倍, 并且**第 4 关
        从 84 步降到 50 步** —— 省下的 34 个动作是直接计入线上成绩的动作数。
        L1~L3 的步数与 v3 逐关相同, 只有路径尾部的选点不同 (第 1 关前 6 击与已上线那条一致,
        后 2 击换了一步; 等长, 未上线复验)。
      - 上一轮 (同配置, 但 `_model_ctx` 还在用 `inspect.getsourcefile(type(_game))` 取源码)
        实测模型通道**一次都没跑起来**: L1~L5 每关都在第一步静默失败, 报
        `TypeError("<class 'arc_agi_3.lf52-271a04aa.Lf52'> is a built-in class")`,
        表面上"结果和以前一样"其实是新代码等于死代码
        (_diag_20261001/lf52_shippath_ladder_v4_out.txt)。原因与正确取法见 `_model_src_text`。
      - 中途调用 (实测于 10-01, 那时 solve_level 还只有引擎两阶段): 真实走过 2 击返回剩余
        6 击、走过 5 击返回剩余 3 击, 都与整解尾部逐位相同 (前提是 _pending_connector_ops
        允许半截合并一步收尾)。**模型通道接管主路径以后这条没有复测过** ——
        `_model_live_state` 是按"引擎当前停住的位置"起算的, 设计上支持中途调用,
        但真半截合并 (停在两步点击之间) 时模型会把已挂出的连线当成一次完整走法,
        返回的剩余步数可能比真实剩余多一步。用之前要单独量一次。
    注意: 第 3 关的解里有 4 个落在 64x64 画面外的点击 (本地引擎按自己的相机换算接受,
    计数上仍是动作) —— 要上线重放得先把这类坐标核一遍。
    """
    global _valid, _error, _solve_note
    if not _ensure() or not _valid:
        return None
    with _lock:
        global _search_steps
        _search_steps = 0
        t0 = time.time()
        # 先把引擎带回影子的真实位置 (full_reset + 重放 _history), 再取起点判据;
        # 顺序反了会读到上一次运行残留的关卡索引, 过关判定就整个错掉
        _replay_prefix()
        idx0 = _search_level
        st0 = _read_state()
        if st0 is None:
            _error = "读不到引擎内部状态 (属性名失效或引擎未就绪)"
            return None
        # levels_completed 是**全局累计**值, 不能拿 >0 当过关信号 (第 2 关起会误判)
        lc0 = _search_score
        goal = (2 if idx0 + 1 in (6, 7) else 1)
        if _pending_nodes() <= goal:
            return []          # 本关目标已满足

        # 深度上限在**起点**求一次 (引擎判负阈值 - 本关已消耗), 两个搜索阶段共用同一个数;
        # 见 _depth_cap: 循环里再读会把搜索自己走的路计进"已消耗", 深度越探上限越缩水
        cap = _depth_cap()

        # 截止点一律用**绝对时间戳**: 曾把"剩余秒数"当绝对时间传给兜底阶段,
        # 420s 的关卡限制实际只跑到 252s 就收工 (g50t run5 实测)
        t_deadline = t0 + t_limit
        t_model_end = t0 + (t_deadline - t0) * _MODEL_TIME_SHARE
        t_queue_end = t0 + (t_deadline - t0) * _QUEUE_SHARE

        def goal_hit(lc):
            return _goal_hit(idx0, lc0, lc)

        seen = {st0: 0}
        heap = [(_heuristic(), 0, 0, ())]
        ctr = 0
        result = None
        stop = None
        qstat = {}
        model_why = None
        from_model = False
        try:
            # **模型通道优先** (2026-10-02 晚): 先在关卡字符图上搜, 解出来就沿路在引擎上
            # 重放自证。理由是实测的成本差: 第 5 关在引擎里 601s 只铺开 951 个状态
            # (每 1 个新状态 105 个引擎步), 同一关在模型上 22s / 109492 状态出 84 步解,
            # 第 6 关 7s / 33542 状态出 85 步解, 两份计划都在真引擎逐手对拍通过。
            # 模型只有两种下场: 交出**已被引擎验证过**的路径, 或者返回 None 退回下面的
            # 两阶段引擎搜索 —— 它不能反驳引擎 (见 _model_channel 的自证循环)。
            m_path, model_why = _model_channel(goal, cap, t_model_end)
            if m_path:
                result = list(m_path)
                from_model = True
            elif m_path is not None:
                _solve_note = '模型通道: 本关目标已满足'
                return []
            # 引擎两阶段只吃模型**实际**用完之后剩下的时间; 模型的重放自证不计入它自己的
            # 时间片, 所以这里按墙上时钟重算, 而不是复用公式里的名义值
            t_now = time.time()
            t_queue_end = t_now + max(0.0, t_deadline - t_now) * _QUEUE_SHARE

            while heap and result is None:
                if (time.time() > t_queue_end or _search_steps > _MAX_SEARCH_STEPS
                        or len(seen) > _MAX_STATES):
                    stop = "队列阶段预算耗尽(时间/引擎步/状态数)"
                    break
                _f, _c, g, path = heapq.heappop(heap)
                st, lc = _goto(path)
                if st is None:
                    stop = "状态读取失败"
                    break
                if goal_hit(lc):
                    result = list(path)
                    break
                if g >= cap:
                    continue
                for op in _expand_ops(path):
                    npath = path + op
                    ng = len(npath)
                    st2, lc2 = _goto(npath)
                    if st2 is None or _lost():
                        continue
                    if goal_hit(lc2):
                        result = list(npath)
                        break
                    if st2 in seen:
                        continue
                    seen[st2] = ng
                    ctr += 1
                    heapq.heappush(heap, (ng + _heuristic(), ctr, ng, npath))

            # 俯冲兜底: 队列铺不到解就把**剩余**预算交给深度优先。
            # 顺序是实测定的 —— 浅关队列又快又短 (L1 0.25s/8 步、L3 51s/46 步),
            # 完整俯冲在浅关会贪心空转到深度上限; 深关只有俯冲走得动。
            if result is None and time.time() < t_deadline - 5:
                _rewind(())
                d0 = _read_state()
                if d0 is None:
                    stop = "俯冲起点状态读取失败"
                else:
                    result = _dive(idx0, lc0, d0, t_deadline, cap, qstat)
                    if result is None:
                        stop = "俯冲" + (qstat.get("stop") or "未解")
        except Exception as e:
            stop = "搜索异常: %s" % e
        finally:
            # 复位回影子真实位置; _restore_start 已清 关卡/_score/_state 残留
            try:
                _rewind(())
                _scrub_future_levels()
            except Exception:
                _valid = False
        if result is None:
            _error = "%s: 展开 %d 状态 / 引擎步 %d / 俯冲%s / %.1fs | 模型通道: %s" % (
                stop or "无解(搜索穷尽)", len(seen), _search_steps,
                "最深%d/上限%d 新节点%d seen命中%d 死局%d 穷尽层%d" % (
                    qstat.get("max_depth", 0), cap, qstat.get("expanded", 0),
                    qstat.get("seen_hit", 0), qstat.get("dead", 0),
                    qstat.get("exhausted", 0)) if qstat else "(未跑)",
                time.time() - t0, model_why or "(未跑)")
            return None
        # 自证: 从干净起点把返回路径整个重放一遍, 过关信号必须在末步再现。
        # 只在搜索游标里成立的"假解"交出去就是线上白烧动作数, 宁可返回 None。
        try:
            _rewind(tuple(result))
            if not goal_hit(int(getattr(_game, '_score', 0) or 0)):
                _rewind(())
                _scrub_future_levels()
                _error = "解不可复现: 干净起点重放 %d 步后过关信号不成立" % len(result)
                return None
            _rewind(())
            _scrub_future_levels()
        except Exception as e:
            _error = "解的自证重放异常: %s" % e
            return None
        _error = None
        _solve_note = (('模型通道 | %s' % model_why) if from_model else
                       '引擎两阶段 | 展开%d状态/引擎步%d' % (len(seen), _search_steps))
        return [a if not isinstance(a, tuple) else {"a": 6, "x": a[0], "y": a[1]}
                for a in result]


def get_frame():
    """当前显示帧 (64x64 网格列表)。

    lf52 的 FrameDataRaw.frame 是**变长帧栈** (动画队列 frames_to_render,
    lf52.py:5720-5730): 只有队尾帧 frame[-1] 是确定性的棋盘, 中间帧是
    np.random 噪声。旧实现用 sprite.pixels 自己拼 bbox, 拼出来的既不是引擎
    真正送出的画面、也无法和线上逐位比对 (线上/本地实测 13/13 一致的正是队尾帧)。
    """
    if not _ensure():
        return [[0] * 64 for _ in range(64)]
    try:
        import numpy as np
        fr = _last_frame
        if fr is None:
            fr = getattr(_env, "_last_response", None)
        planes = getattr(fr, "frame", None)
        if planes is None:
            raise AttributeError("no frame")
        if isinstance(planes, (list, tuple)):
            tail = np.asarray(planes[-1])
        else:
            arr = np.asarray(planes)
            tail = arr[-1] if arr.ndim == 3 else arr
        return [[int(v) for v in row] for row in tail]
    except Exception:
        return [[0] * 64 for _ in range(64)]
