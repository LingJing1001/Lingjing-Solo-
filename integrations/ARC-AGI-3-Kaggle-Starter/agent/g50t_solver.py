# -*- coding: utf-8 -*-
"""G50T 专用求解器 — 机制来自直读游戏源码 environment_files/g50t/5849a774/g50t.py

源码实证机制:
- 玩家 = 容器.dzxunlkwxt (5x5 色9块), 目标 = 容器.whftgckbcu;
  通关条件 goal.x+1==player.x 且 goal.y+1==player.y (g50t.py:2517)。
- A1=上 A2=下 A3=左 A4=右 一格=pitch 6px (g50t.py:2816-2823); 撞墙/危险区=无效但仍耗1步计时。
- A5=回放键 (g50t.py:2698-2726): 历史非空整段回卷; 历史空录幽灵进下一阶段 (多阶段机制)。
- 失败: 计时条每 2 步左移 1, 移出屏=lose (g50t.py:2844-2849) => 每关预算 130 步。
- 引擎自带关卡重置: g.level_reset() 从 _clean_levels 重建当前关并重跑 on_set_level
  (base_game.py:326-329) — 零 deepcopy 快照污染。
- 移动的合法性可被引擎自己的预言函数直接判定: 容器.rhvduhvfwn(sprite, x, y)
  = 落点与地板遮罩 afbbgvkpip 有重叠 且 不与可见危险区 uwxkstolmf 重叠 (g50t.py:2644-2658)。

求解分两级:
1) 快速通道: 纯位置 BFS (预言函数判墙/危险, 零重放) -> 找最短路径 -> 真实重放验证;
2) 验证不过 (有机关联动等副作用) 则退回通用回放式 BFS: 每节点 level_reset+重放路径+试一步,
   状态键 = 容器对象图摘要 (剔除移动历史等纯簿记字段, 否则去重失效 — 首轮实测根因)。
"""
import hashlib
import sys
import time
from collections import deque

from arcengine import ActionInput, GameAction, GameState

try:
    import numpy
except Exception:  # pragma: no cover - arcengine 依赖 numpy, 正常不会失败
    numpy = None

# 移动历史/动画队列是纯簿记字段, 入键会使去重彻底失效 (实测);
# 但 A5 录像带 areahjypvy/uocsatwnyt 决定幽灵下一步怎么走, 属于"状态"而不是簿记。
# 两档并存的理由 (_tape.log 实测, 同一 250s 预算):
#   快档(剔录像带) L1 17 步/11s 解出, L2 俯冲只铺到深度 41;
#   精档(录像带入键) L1 121s 无解, L2 俯冲能铺到深度 121。
# 于是队列用快档(浅关靠它), 俯冲兜底切精档(深关靠它), 不做一刀切。
_TAPE_ATTRS = {"areahjypvy", "uocsatwnyt"}
_SKIP_ATTRS = {"vynnrceibs", "vgwycxsxjz", "areahjypvy", "uocsatwnyt", "hjvvibklzv"}
_EXACT_SKIP_ATTRS = {"vynnrceibs", "vgwycxsxjz", "hjvvibklzv"}
_SKIP = _SKIP_ATTRS
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # ACTION1..4


def _budget(g):
    """本关动作预算: 计时条每 2 步左移 1, -x>width 即 lose (g50t.py:2844-2849)"""
    try:
        t = g.twyixucrqi
        return max(10, 2 * (int(t.x) + int(t.width)) + 2)
    except Exception:
        return 130


def _sprite_sig(s):
    import numpy as np
    px = np.ascontiguousarray(s.pixels)
    return (int(s.x), int(s.y), int(s.rotation or 0), int(bool(s.is_visible)),
            hashlib.sha1(px.tobytes()).hexdigest()[:12])


def _keytag(k):
    """字典键的确定性描述: repr(对象) 含内存地址, 直接入键会让同一路径两次重放得到不同键"""
    if isinstance(k, (bool, int, float, str)):
        return repr(k)
    if hasattr(k, "x") and hasattr(k, "y"):
        try:
            return "%s@%d,%d" % (type(k).__name__, int(k.x), int(k.y))
        except Exception:
            return type(k).__name__
    return type(k).__name__


def _collect(obj, sprites, scalars, seen, depth=0, tag=""):
    """容器图扁平遍历: Sprite 位姿/像素摘要 + 实例标量机关标志 (递归 _dig 的快版本)。

    tag 保留属性/键身份, 否则 {A=1,B=2} 与 {A=2,B=1} 排序后同键 (假相等 -> 漏解)。
    注意: 类对象 (枚举等) 只取类型名, 不进 __dict__ — 否则一堆常量属性会淹没实例差异,
    实测把 6 个不同状态压成 4 个键, 聚焦搜索 35 节点就"穷尽"。
    """
    if obj is None or depth > 5:
        return
    if isinstance(obj, (bool, int, float, str)):
        scalars.append(tag + repr(obj))
        return
    if isinstance(obj, type):
        scalars.append(tag + "T:" + obj.__name__)
        return
    if numpy is not None and isinstance(obj, numpy.ndarray):
        scalars.append(tag + "nd" + hashlib.sha1(numpy.ascontiguousarray(obj).tobytes()).hexdigest()[:12])
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if hasattr(obj, "pixels") and hasattr(obj, "x") and hasattr(obj, "y"):
        sprites.append(_sprite_sig(obj))
    if isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            _collect(obj[k], sprites, scalars, seen, depth + 1,
                     "%s#%s." % (tag, _keytag(k)))
        return
    if isinstance(obj, (list, tuple, set)):
        if isinstance(obj, set):
            for x in obj:
                _collect(x, sprites, scalars, seen, depth + 1, tag)
        else:
            for i, x in enumerate(obj):
                _collect(x, sprites, scalars, seen, depth + 1, "%s[%d]." % (tag, i))
        return
    d = getattr(obj, "__dict__", None)
    if d is None:
        return
    nt = tag + type(obj).__name__
    for k in sorted(d):
        if k in _SKIP:
            continue
        _collect(d[k], sprites, scalars, seen, depth + 1, "%s.%s=" % (nt, k))


def _state_key(g):
    sprites, scalars = [], []
    _collect(g.vgwycxsxjz, sprites, scalars, set())
    return hashlib.sha1(
        repr((sorted(sprites), sorted(scalars))).encode("utf-8")).hexdigest()


def _pitch(g):
    mod = sys.modules.get(type(g).__module__)
    return int(getattr(mod, "jarvstobjt", 6))


def _h(g):
    """启发: 玩家块到目标嵌入位 (goal.x+1, goal.y+1) 的曼哈顿距离 (格为单位)"""
    pitch = _pitch(g) or 1
    c = g.vgwycxsxjz
    return (abs(int(c.dzxunlkwxt.x) - (int(c.whftgckbcu.x) + 1))
            + abs(int(c.dzxunlkwxt.y) - (int(c.whftgckbcu.y) + 1))) // pitch


def _restore_start(g, idx0):
    """把引擎恢复到搜索开始那一关的起点。

    不能用 level_reset(): 它重置的是 _current_level_index, 而搜索中的胜利着已经把它推进到
    下一关 (实测: 调用方重放首个动作时直接打印"关卡2", 路径根本没走过)。
    """
    g._levels = [lv.clone() for lv in g._clean_levels]
    g._next_level = False
    g._score = idx0
    g.set_level(idx0)
    g._state = GameState.NOT_FINISHED


def solve(game, t_limit=120.0, max_nodes=200000):
    """求 game 当前关: 返回按键序列 (int 1-4); 失败/超时 None; 已胜 [].

    结束时游戏总是 level_reset 回当前关起点, 调用方按路径重放即可。
    """
    g = game
    t0 = time.time()
    idx0 = int(g._current_level_index)
    if int(g._score) > idx0:
        return []

    AID = {a: getattr(GameAction, "ACTION%d" % a) for a in (1, 2, 3, 4, 5)}

    def act(a):
        return g.perform_action(ActionInput(id=AID[a], data={}, reasoning=None), raw=True)

    def won(r):
        return (r.levels_completed > idx0 or r.state == GameState.WIN
                or int(g._current_level_index) > idx0)

    def dead(r):
        return r.state == GameState.GAME_OVER

    try:
        path = _bfs_oracle(g, act)
        if path is not None:
            g.level_reset()
            r = None
            ok = True
            for a in path:
                r = act(a)
                if won(r):
                    return list(path)
                if dead(r):
                    ok = False
                    break
            else:
                ok = False  # 走完路径未胜 -> 预言模型漏了副作用
            if not ok:
                print("[g50t_solver] 快速通道验证失败, 退回回放搜索 (路径长 %d)" % len(path))
        now = time.time()
        # 顺序依据实测: 队列搜索 14s 就能解 L1(17 步), 而修好完整性的俯冲在 L1 上
        # 会一路贪心潜到深度 90+ 空转 60s 无解 —— 浅关交给队列, 俯冲只打深关兜底。
        # 截止点一律用绝对时间戳: 曾把"剩余秒数"当绝对时间传给俯冲, 420s 的关卡限制
        # 实际只跑到 252s 就收工 (run5 实测)。
        path = _bfs_replay(g, act, won, dead, t0, now + (t0 + t_limit - now) * 0.4, max_nodes)
        if path is not None:
            return path
        if t0 + t_limit - time.time() < 5:
            return None
        return _dive_exact_key(g, act, won, dead, lambda: _h(g), _budget(g), t0, t0 + t_limit)
    finally:
        try:
            _restore_start(g, idx0)
        except Exception:
            pass


def _bfs_oracle(g, act):
    """纯位置 BFS: 用引擎预言函数判移动合法性, 不执行任何动作; 返回 int 动作序列或 None."""
    g.level_reset()
    c = g.vgwycxsxjz
    pitch = _pitch(g)
    p = c.dzxunlkwxt
    start = (int(p.x), int(p.y))
    goal = (int(c.whftgckbcu.x) + 1, int(c.whftgckbcu.y) + 1)
    budget = _budget(g)
    seen = {start}
    q = deque([(start, ())])
    while q:
        (x, y), path = q.popleft()
        if len(path) >= budget - 1:
            continue
        for i, (dx, dy) in enumerate(_DIRS, start=1):
            nx, ny = x + dx * pitch, y + dy * pitch
            np_ = (nx, ny)
            if np_ in seen:
                continue
            if not c.rhvduhvfwn(c.dzxunlkwxt, nx, ny):
                continue
            seen.add(np_)
            npath = path + (i,)
            if np_ == goal:
                return list(npath)
            q.append((np_, npath))
    return None


def _bfs_replay(g, act, won, dead, t0, t_end, max_nodes):
    """回放式聚焦搜索: 每节点 reset+重放路径+试 5 动作, f=步数+2*曼哈顿(块,目标).

    (L2 基线 175 > FIFO BFS 在 300s 内能穷尽的深度; 用目标距离加权优先队列直达深区,
    去重后仍是真最短优先级别的完全搜索——只放弃"步数最优", 保留"可达即找到")
    """
    import heapq
    seen = {_state_key(g)}
    ctr = 0

    def h():
        return _h(g)

    budget = _budget(g)
    q = [(2 * h(), 0, (), None)]  # (f, tie, path, 缓存的节点h)

    def reach(path):
        g.level_reset()
        r = None
        for i, a in enumerate(path):
            r = act(a)
            if won(r):
                return ("won", i)
            if dead(r):
                return ("dead", r)
        return ("ok", r)

    nodes = 0
    while q:
        if time.time() > t_end or nodes > max_nodes:
            print("[g50t_solver] 聚焦搜索超时/超限: nodes=%d seen=%d q=%d" % (nodes, len(seen), len(q)))
            return None
        _f, _d, path, _hz = heapq.heappop(q)
        if len(path) >= budget - 1:
            continue
        for a in (1, 2, 3, 4, 5):  # A5=幽灵录像, 是过关必需机制
            st, info = reach(path)
            if st == "won":
                return list(path[:info]) + [path[info]]
            if st == "dead":
                break
            r = act(a)
            if won(r):
                return list(path) + [a]
            if dead(r):
                continue
            k = _state_key(g)
            if k in seen:
                continue
            seen.add(k)
            npath = path + (a,)
            hn = h()
            ctr += 1
            heapq.heappush(q, (len(npath) + 2 * hn, ctr, npath, hn))
            nodes += 1
    print("[g50t_solver] 聚焦搜索穷尽: nodes=%d seen=%d" % (nodes, len(seen)))
    return None


def _dive_exact_key(g, act, won, dead, h, budget, t0, t_end, w=2.0, stat=None):
    """俯冲专用: 把状态键切到"录像带入键"的精档再下潜。

    实测 (_tape.log, 同一 250s): 快档俯冲在 L2 只铺到深度 41 就因假合并判穷尽,
    精档能铺到 121; 但精档解 L1 要 121s 仍无解 (快档 11s 出 17 步), 所以只在兜底阶段切。
    """
    global _SKIP
    old = _SKIP
    _SKIP = _EXACT_SKIP_ATTRS
    try:
        return _dive(g, act, won, dead, h, budget, t0, t_end, w, stat)
    finally:
        _SKIP = old


def _dive(g, act, won, dead, h, budget, t0, t_end, w=2.0, stat=None):
    """俯冲搜索 (深度优先 + 兄弟挂起栈): 每候选只评估一次, 未下潜的兄弟回头再潜.

    队列版聚焦搜索的问题: 弹出一个节点要为每个候选各重放一次整条路径 (O(分支*深度)),
    于是 300s 只够铺到深度 ~30, 而 L2 基线 175 步。俯冲沿一条链推进, 只在死胡同回退一层,
    能真正走到 budget 深度。
    早期"每层只承诺最优候选"的写法不完整: 其余候选的状态既已进 seen 又被标记 tried,
    回退时该层直接判穷尽, 兄弟子树被永久丢弃 (实测连 L1 都在 21 节点"穷尽根节点",
    那 17 步解是队列兜底找到的)。现在 frames[d] 挂起本层全部新节点, 按 f 排序逐个下潜。
    """
    def reach(path):
        g.level_reset()
        for a in path:
            r = act(a)
            if won(r):
                return "won"
            if dead(r):
                return "dead"
        return "ok"

    path = ()
    eng = [None]
    frames = [[]]        # frames[d] = 深度 d 已评估、尚未下潜的子节点 [(f, action), ...]
    seen = set()
    nodes = 0

    def at(p):
        if eng[0] != p:
            st = reach(p)
            eng[0] = p
            return st
        return "ok"

    def unwind():
        """回退到最近一层还有挂起兄弟的节点; 一路退空返回 None (搜索穷尽).

        frames[len(path)] 是"当前节点"自己的挂起列表, 退到它时要保留, 只作废更深层的帧。
        """
        nonlocal path
        while True:
            if frames[len(path)]:
                del frames[len(path) + 1:]
                return path
            if not path:
                del frames[1:]
                return None
            path = path[:-1]

    while True:
        if time.time() > t_end:
            print("[g50t_solver] 俯冲超时: %.0fs 深度=%d 节点=%d seen=%d"
                  % (time.time() - t0, len(path), nodes, len(seen)))
            return None
        d = len(path)
        if d >= budget - 1 or at(path) != "ok":
            if unwind() is None:
                print("[g50t_solver] 俯冲穷尽: 节点=%d" % nodes)
                return None
            continue
        if stat is not None:
            stat["max_depth"] = max(stat.get("max_depth", 0), d)
        if not frames[d]:
            for a in (1, 2, 3, 4, 5):
                if at(path) != "ok":
                    break
                r = act(a)
                eng[0] = path + (a,)
                if won(r):
                    print("[g50t_solver] 俯冲命中: 深度=%d 节点=%d" % (d + 1, nodes))
                    return list(path) + [a]
                if dead(r):
                    if stat is not None:
                        stat["dead"] = stat.get("dead", 0) + 1
                    continue
                k = _state_key(g)
                if k in seen:
                    if stat is not None:
                        stat["seen_hit"] = stat.get("seen_hit", 0) + 1
                    continue
                seen.add(k)
                nodes += 1
                frames[d].append((d + 1 + w * h(), a))
        if not frames[d]:
            if stat is not None:
                stat["exhausted"] = stat.get("exhausted", 0) + 1
            if unwind() is None:
                print("[g50t_solver] 俯冲穷尽: 节点=%d" % nodes)
                return None
            continue
        frames[d].sort()
        _f, a = frames[d].pop(0)
        path = path + (a,)
        while len(frames) < len(path) + 1:
            frames.append([])
