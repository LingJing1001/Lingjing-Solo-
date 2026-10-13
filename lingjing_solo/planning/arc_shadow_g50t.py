"""G50t 本地引擎影子模拟器

游戏机制: keyboard_click 混合游戏
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
_Acts = None
_ACTION6 = None
# 2026-10-04 新增"方案表通道"用: _plans 是懒加载缓存, _plan_note 记录上一次为什么没给方案
_plans = None
_plan_note = None


def _ensure():
    """确保引擎已加载; 返回 True 表示**引擎可用**。

    2026-10-04 更正: 旧实现在 `_env` 已建好时把 `_valid` 一起返回, 而 `_valid` 会被
    apply()/reset()/replay() 里**任何**一次异常永久钉成 False(`arc_shadow_g50t.py:75-77`),
    且只有本函数的 try 会写 `_error` —— 于是一次越界动作(实测 `apply(0)`)就把整个进程
    锁死成"本地引擎未就绪: "(冒号后空白, 10-04 用户实见), 连 `get_frame()` 也只能给
    10x10 占位帧, 页面表现就是"背景自己缩小"。加载状态与"上一手动作落得干净不干净"
    是两件事: 后者归 `_valid`, 由调用方按需检查(如 `solve_level`)。
    """
    global _env, _game, _applied, _valid, _error, _ActionInput, _Acts, _ACTION6
    if _env is not None:
        return True
    try:
        import arc_agi
        from arcengine import GameAction, ActionInput
        arc = arc_agi.Arcade()
        _env = arc.make("g50t-5849a774")
        _game = _env._game
        _ActionInput = ActionInput
        _Acts = {a: getattr(GameAction, f"ACTION{a}") for a in (1, 2, 3, 4, 5)}
        _ACTION6 = GameAction.ACTION6
        _env.reset()
        _applied = 0
        _valid = True
        _error = None
    except Exception as e:
        _error = "%s: %s" % (type(e).__name__, e)
        _valid = False
    return _valid


def reset():
    global _applied, _valid, _error
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _applied = 0
            _valid = True
            _error = None
            return True
        except Exception as e:
            _error = "%s: %s" % (type(e).__name__, e)
            _valid = False
            return False


def apply(action):
    """执行动作: int (ACTION1-5) 或 dict {"a": 6, "x":, "y":}"""
    global _applied, _valid, _error
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
            if a == 6:
                aid = _ACTION6
            elif a in _Acts:
                aid = _Acts[a]
            else:
                # 越界按键先挡住: 引擎状态一点没被动过, 不该因此把整局判脏。
                # 旧写法走 `_Acts[a]` 抛 KeyError -> `_valid=False` 永久锁死,
                # 表现是"本地引擎未就绪: "后面什么都没有、画面塌成 10x10。
                _error = "非法按键 %r (g50t 可用 1-5, 点击是 6)" % (action,)
                return False
            _game.perform_action(_ActionInput(id=aid, data=data, reasoning=None), raw=True)
            _applied += 1
            _error = None
            return True
        except Exception as e:
            _error = "%s: %s" % (type(e).__name__, e)
            _valid = False
            return False


def is_synced(actions_taken):
    return _valid and _applied == int(actions_taken or 0)


def replay(history):
    """整局重放到 history 末尾; 任一步落不下去就返回 False 并标脏。

    2026-10-04: 原来这里自己拼 `_Acts[a]`, 越界按键会抛 KeyError 把 `_valid` 钉死;
    现在统一走 `apply()` —— 它有按键校验和 `_error` 记账。
    """
    global _applied, _valid, _error
    if not _ensure():
        return False
    with _lock:
        try:
            _env.reset()
            _applied = 0
        except Exception as e:
            _error = "%s: %s" % (type(e).__name__, e)
            _valid = False
            return False
        for h in (history or []):
            if not apply(h):
                _valid = False
                return False
        _valid = True
        _error = None
        return True


def restart_level(history):
    """把引擎搬到"当前这一关的起点", 返回应当保留的历史前缀长度。

    给 /api/proxy/arc-retry 用, 语义与 arc_shadow_wa30.restart_level:370 一致:
    先整体重放一遍, 记下最后一次"关卡号变大"发生在第几步之后(那之前的动作
    都属于已通过的关卡), 再 reset + 只重放到那个位置。

    为什么 g50t 需要它: `g50t_solver.solve()` 收尾必然用 `_restore_start` 把
    引擎拉回本关起点, 所以 arc-auto 只能在本关第 0 步调用它(app.py:3902 的
    ucorwtereb 守卫)。原来的"重试本关"对 g50t 只清 solve_path 缓存、引擎仍停在
    本关中途 ⇒ 守卫会一直拒绝, 用户只剩"从头重开整局"一条路。
    """
    if not _ensure():
        return 0
    global _applied, _valid, _error
    with _lock:
        try:
            hist = list(history or [])
            # 必须 full_reset: arcengine handle_reset(base_game.py:305-316) 只在
            # _action_count==0 或 WIN 时才整局重置, 否则 RESET 只复位**当前关**,
            # 关卡号原地不动 —— 用它当起点会从中途开始重放, 永远观察不到进关,
            # cut 恒为 0 (10-04 实测: 22 步历史 cut=0)。
            _game.full_reset()
            _applied = 0
            cut = 0
            lvl = int(_game._current_level_index)
            for i, h in enumerate(hist):
                if not apply(h):
                    continue
                now = int(_game._current_level_index)
                if now != lvl:
                    lvl = now
                    cut = i + 1
            _game.full_reset()
            _applied = 0
            for h in hist[:cut]:
                if not apply(h):
                    break
            _valid = True
            _error = None
            return cut
        except Exception as e:
            _error = "%s: %s" % (type(e).__name__, e)
            _valid = False
            return 0


def status():
    return {"valid": _valid, "applied": _applied, "error": _error,
            "loaded": _env is not None, "plan_note": _plan_note,
            "plan_channel": plan_channel_enabled(),
            "model_channel": model_channel_enabled(),
            "legacy_fallback": legacy_fallback_enabled(),
            "solve_note": _solve_note}


# 2026-10-04 晚口径: 表里那七条序列的来源是**外部公开答案表**(Kaggle 内核的 OPTIMAL_PLANS),
# 用户明确"不要再接公开答案了" ⇒ 这条通道**默认关闭**, 端点拿不到方案, 只会走本仓库自己的
# solve_level。要做离线验收靶子(比长度、比逐步状态轨迹)时, 显式设 G50T_PLAN_CHANNEL=1 再读。
# 口径与开关集中在 arc_plan_channel.py(2026-10-04 晚已把 ls20/sb26 的预存解支一并纳入)。
_PLAN_CHANNEL_ENV = "G50T_PLAN_CHANNEL"


def plan_channel_enabled():
    """方案表通道是否允许被读取 —— 默认 False(见上面 `_PLAN_CHANNEL_ENV` 的注释)。"""
    import arc_plan_channel
    return arc_plan_channel.enabled("g50t")


def _load_plans():
    """懒加载 `g50t_plans.json` —— 七关**已逐手验证过**的动作序列表。

    文件结构: `{"0": [4,4,...], "1": [...], ...}`, 键是关卡索引(0 起), 值是按键 int 列表。
    来源与口径写在 docs/arc-play-methods-and-decisions.md 的"2026-10-04 g50t 方案表通道"一节
    (Kaggle 公开答案表 + 我们影子引擎逐手重放自证), **不是**本仓库求解器搜出来的。
    读不到/格式不对 ⇒ 返回 None, 原因进 `_plan_note`, 调用方回退到搜索。
    """
    global _plans, _plan_note
    if _plans is not None:
        return _plans or None
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g50t_plans.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            _plan_note = "方案表为空或结构不对: %s" % path
            _plans = {}
            return None
        _plans = data
    except Exception as e:
        _plan_note = "方案表读取失败 %s: %s" % (type(e).__name__, e)
        _plans = {}
        return None
    return _plans


def plan_for_current_level():
    """返回**本关起点**的已验证动作序列; 任何条件不满足都返回 None(调用方回退搜索)。

    返回值与 `solve_level()` 同格式(list of int 1-5), 所以 app.py arc-auto 的执行段
    (`session["solve_path"]` 缓存 + 每请求走一步)不用动就能直接吃这张表。

    第 0 道闸(2026-10-04 晚新增, **默认拦死**): 通道开关。这张表的来源是外部公开答案,
    用户口径"不要再接公开答案了" ⇒ `plan_channel_enabled()` 默认 False, 这里直接返回
    `None`, 调用方只会去走本仓库自己的 `solve_level()`。离线拿它当验收靶子才设
    `G50T_PLAN_CHANNEL=1`。

    另外三道闸:
      1) 引擎已加载;
      2) 当前关卡索引在表里(键 "0".."6");
      3) **本关一步都没走** —— `ucorwtereb == 0`。这张表是"从本关起点"的序列,
         中途端上去会和 session 的 `actions_taken` 脱钩(与 g50t_solver 收尾
         `_restore_start` 的约束同源, 见 app.py 那条守卫的注释)。

    为什么不直接让它去搜: 2026-10-04 实测, 第 2 关起点调 `solve_level(200.0)` 跑满
    **200.5 秒返回 None**(`_g50t_l2_auto_repro.txt`), 前端在这 200 秒里 loading=true、
    `arc-state` 轮询被 loadingRef 跳过(ArcGame.tsx:613) ⇒ 用户看到的就是"点了没反应"。
    """
    global _plan_note
    _plan_note = None
    if not plan_channel_enabled():
        _plan_note = ("方案表通道已按 2026-10-04 口径关闭(外部公开答案不进产品通道); "
                      "离线验收才需设 %s=1" % _PLAN_CHANNEL_ENV)
        return None
    if not _ensure() or _game is None:
        _plan_note = "引擎未加载"
        return None
    plans = _load_plans()
    if not plans:
        return None
    idx = int(getattr(_game, "_current_level_index", -1))
    seq = plans.get(str(idx))
    if not seq:
        _plan_note = "表里没有第%d关(索引%s)的方案" % (idx + 1, idx)
        return None
    stepped = int(getattr(_game, "ucorwtereb", 0) or 0)
    if stepped != 0:
        _plan_note = "本关已走到第%d步, 方案表只能从关卡起点放" % stepped
        return None
    _plan_note = "第%d关方案 %d 手, 起点核对通过(ucorwtereb=0)" % (idx + 1, len(seq))
    return [int(a) for a in seq]


# 2026-10-05 新增"纯 python 关卡模型通道": g50t_model 把关卡抽成整数格模型(几何/机关/录像/计时),
# 在模型上搜出方案后**先在这台引擎上逐手对拍**, 0 分歧且同手过关才交出去。
# 为什么要它: 老通道 g50t_solver.solve 是"每节点从头重放引擎"的回放式搜索, 10-04 实测第 2 关起点
# 跑满 200.5s 返回 None(_g50t_l2_auto_repro.txt) => 前端那 200 秒 loading=true、arc-state 轮询被
# loadingRef 跳过(ArcGame.tsx:613), 用户看到的就是"点了没反应"。模型侧实测 L1 17 手 0.03s /
# L2 31 手 0.18s / L3 64 手 1.71s, 且三关逐手 0 差异(_g50t_latch.txt, 10-05 12:24:31 档)。
# 与外部答案表无关: 那条通道(g50t_plans.json)按 10-04 口径仍然默认关闭, 这里搜出来的是我们自己的解。
_MODEL_CHANNEL_ENV = "G50T_MODEL_CHANNEL"       # =0 关掉模型通道
_LEGACY_FALLBACK_ENV = "G50T_LEGACY_FALLBACK"   # =1 才允许模型没解时回落 200s 老搜索(默认不回落)
_solve_note = None


def model_channel_enabled():
    return os.environ.get(_MODEL_CHANNEL_ENV, "1") != "0"


def legacy_fallback_enabled():
    """模型没解时要不要再烧一次老的引擎内搜索 —— 默认**不要**(它就是那次 200s 冻结)。"""
    return os.environ.get(_LEGACY_FALLBACK_ENV, "") == "1"


def solve_note():
    """上一次 solve_level 的口径标注(模型自搜/对拍不过/未建模缺口); 给 app.py 显示用。"""
    return _solve_note


def solve_level(t_limit=60.0, max_nodes=200000):
    """求解影子当前关: 优先纯 python 关卡模型 (g50t_model), 失败再按开关回落 g50t_solver.

    返回动作序列 (int 1-5); 已通关返回 []; 失败返回 None。
    注意: 两条通道收尾都会把影子恢复到**本关起点**, 所以只有当前关尚未走步时使用 ——
    与 app.py arc-auto 的 `ucorwtereb == 0` 守卫同一契约。
    模型的方案在交出前已在这台引擎上逐手对拍过(见 g50t_model.verify), 对拍不过就当没解。
    """
    global _solve_note, _plan_note
    _solve_note = None
    if not _ensure() or not _valid:
        return None
    if model_channel_enabled():
        import g50t_model
        # 模型自带更小的时间闸(默认 20s), 别让调用方的 200s 变成新的冻结时长
        tl = min(g50t_model.DEFAULT_T_LIMIT, max(1.0, float(t_limit or 1.0)))
        with _lock:
            plan, note = g50t_model.solve(_game, t_limit=tl)
        _solve_note = note
        if plan:
            _plan_note = None
            return plan
        if not legacy_fallback_enabled():
            _plan_note = note
            return None
    import g50t_solver
    with _lock:
        return g50t_solver.solve(_game, t_limit=t_limit, max_nodes=max_nodes)


def _composite_from_sprites():
    """回退路径: 手工合成, 但**尺寸固定为相机视口**, 不再按精灵包围盒裁剪。

    规则出处 (arcengine):
      - camera.py:227 `np.full((height, width), self._background)` —— 底色取相机的
        background (g50t 传的是 BACKGROUND_COLOR=0, g50t.py:1845), 不是"补 0"的巧合;
      - camera.py:231 按 layer 升序、只画 is_visible 的精灵;
      - camera.py:236-253 相对相机取位置, 视口外的部分裁掉 (精灵 x 会是负数 ——
        计时条每 2 手 -1 格就是这么走的);
      - camera.py:263 `visible_mask = sprite_region >= 0` —— 所有负值都是透明
        (不只是 -1; 精灵里还出现 -2, 旧实现把它当颜色画出来就成了非法色);
      - sprites.py:456 `Sprite.render()` —— 旋转/缩放/镜像后的位图, 不是 s.pixels。

    2026-10-04 改掉的关键一点: 旧实现拿 min_x..max_x / min_y..max_y 的**包围盒**当帧,
    于是帧尺寸跟着精灵位置变。计时条往左走 x 变负 ⇒ 包围盒每 2 手宽 1 列(实测第 1 关
    一路 64 -> 72 列、高度被裁成 63 行), 而前端是按帧的行列数算 cellSize 并重建 canvas
    (ArcGame.tsx:204-217), 画面表现就是"背景每两手自动缩一档"。现在尺寸恒等于相机视口。
    """
    import numpy as np

    cam = _game.camera
    w = int(cam.width)
    h = int(cam.height)
    grid = [[int(cam.background)] * w for _ in range(h)]
    game_sprites = sorted(
        (s for s in _game.current_level._sprites
         if getattr(s, "is_visible", True) is not False),
        key=lambda s: getattr(s, "layer", 0),
    )
    for s in game_sprites:
        px = np.array(s.render())
        rel_x = int(s.x) - int(cam.x)
        rel_y = int(s.y) - int(cam.y)
        for r in range(px.shape[0]):
            y = rel_y + r
            if y < 0 or y >= h:
                continue
            row = grid[y]
            for c in range(px.shape[1]):
                x = rel_x + c
                if x < 0 or x >= w:
                    continue
                v = int(px[r][c])
                if v < 0:
                    continue
                row[x] = v
    return grid


def get_frame():
    """返回当前游戏帧为 2D 网格列表

    取引擎合成帧本体 —— `camera.render(current_level.get_sprites())` 正是
    arcengine 构造 FrameData 时调用的那一次 (base_game.py:232), 因此与官方
    `frame[-1]` 逐格一致 (10-04 实测: 第 1 关 17 手逐格差全 0)。
    旧实现从 s.pixels 自己铺网格, 漏了上面 4 条规则中的 3 条并把 -2 当颜色画,
    实测 7 关逐格差 826~1042 格 (约 23%), 表现就是"画面颜色和官方不一样"。

    2026-10-04: 这里不再拿 `_valid` 当门槛。`_valid=False` 只说明"上一手动作没干净地
    落到引擎上", 与"能不能画出当前状态"无关; 旧写法一次越界按键就让 64x64 棋盘塌成
    10x10 全零 —— 前端按帧尺寸重算 cellSize 并重建 canvas, 用户看到的就是"背景唰地
    缩小"。现在只有"引擎根本没加载起来"才给占位帧, 原因写进 `_error`。
    不加 `_lock`: `solve_level` 搜索时持锁可达 200s, 只读轮询若排队等锁会整页卡住,
    取到瞬时状态(可能半成品帧)比让 arc-state 超时更可用。
    """
    global _error
    if not _ensure() or _game is None:
        return [[0] * 10 for _ in range(10)]
    try:
        frame = _game.camera.render(_game.current_level.get_sprites())
        grid = [[int(v) for v in row] for row in frame]
        if grid:
            return grid
    except Exception as e:
        _error = "camera.render %s: %s" % (type(e).__name__, e)
    try:
        grid = _composite_from_sprites()
        if grid:
            return grid
    except Exception as e:
        _error = "合成回退 %s: %s" % (type(e).__name__, e)
    return [[0] * 10 for _ in range(10)]
