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

求解分两级 (搜索基建在 solver_common, 教训清单见其模块头):
1) 快速通道: 纯位置 BFS (预言函数判墙/危险, 零重放) -> 找最短路径 -> 真实重放验证;
2) 验证不过 (有机关联动等副作用) 则退回回放式搜索: 队列(浅关快解) -> 俯冲(深关兜底)。
"""
import hashlib
import sys
import time
from collections import deque

from arcengine import ActionInput, GameAction

from solver_common import (dive as _dive, replay_bfs as _replay_bfs,
                           restore_start as _restore_start,
                           make_won, is_dead, sprite_sig as _sprite_sig)

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


def solve(game, t_limit=120.0, max_nodes=200000):
    """求 game 当前关: 返回按键序列 (int 1-4); 失败/超时 None; 已胜 [].

    结束时游戏总是 restore_start 回当前关起点, 调用方按路径重放即可。
    """
    g = game
    t0 = time.time()
    idx0 = int(g._current_level_index)
    if int(g._score) > idx0:
        return []

    won = make_won(g, idx0)
    dead = is_dead
    AID = {a: getattr(GameAction, "ACTION%d" % a) for a in (1, 2, 3, 4, 5)}

    def act(a):
        return g.perform_action(ActionInput(id=AID[int(a)], data={}, reasoning=None), raw=True)

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
        # 截止点一律用绝对时间戳 (教训 2, 见 solver_common 模块头)。
        path = _replay_bfs(
            g, act, won, dead, lambda: _h(g), _budget(g), t0,
            now + (t0 + t_limit - now) * 0.4, max_nodes,
            lambda _g: (1, 2, 3, 4, 5), _state_key, tag="g50t_solver")
        if path is not None:
            return path
        if t0 + t_limit - time.time() < 5:
            return None
        return _dive_exact_key(g, act, won, dead, t0, t0 + t_limit)
    finally:
        try:
            _restore_start(g, idx0)
        except Exception:
            pass


def _dive_exact_key(g, act, won, dead, t0, t_end):
    """俯冲专用: 把状态键切到"录像带入键"的精档再下潜。

    实测 (_tape.log, 同一 250s): 快档俯冲在 L2 只铺到深度 41 就因假合并判穷尽,
    精档能铺到 121; 但精档解 L1 要 121s 仍无解 (快档 11s 出 17 步), 所以只在兜底阶段切。
    """
    global _SKIP
    old = _SKIP
    _SKIP = _EXACT_SKIP_ATTRS
    try:
        return _dive(g, act, won, dead, lambda: _h(g), _budget(g), t0, t_end,
                     lambda _g: (1, 2, 3, 4, 5), _state_key, tag="g50t_solver")
    finally:
        _SKIP = old
