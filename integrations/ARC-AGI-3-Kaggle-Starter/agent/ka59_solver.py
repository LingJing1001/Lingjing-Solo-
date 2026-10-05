# -*- coding: utf-8 -*-
"""KA59 专用求解器 — 机制来自直读游戏源码 environment_files/ka59/38d34dbb/ka59.py

源码实证机制 (行号均指 ka59.py):
- 可用动作 [1,2,3,4,6] (41133): 1-4 把"当前选中的 0022vrxelxosfy 方块"移动一格 3px
  (41380-41447, zsqdfmgyjo=3); ACTION6 点击 = display_to_grid -> get_sprite_at(0022 标签)
  -> 切换选中块 (41448-41457, qfachkyqvv 41153)。
- 推动对象 = 0022 方块 / 0001uqqokjrptk 本体 / 0003umnkyodpjp 爆炸桶 (wspfaiigqs 41089),
  推挤为链式动画且引擎在 perform_action 内循环到收敛 (41319-41334) — 确定性、无随机。
- 敌人: 每次玩家动作后 Enemy 向 0001 本体逐像素追 6 步, 走到同格 lose (41203-41225)。
- 预算: 关卡 data StepCounter = 100/127/100/127/100/150/200 (40946-41036),
  任何动作 (含点击) 扣 1 (umlpkcmkwm), 归零 lose (41461)。
- 通关: 每个 0010xzmuziohuf 槽恰好嵌入一个 0022 方块, 且每个 0027jbgxilrocf 槽
  嵌入 0001 块 (dbmlcqbquh 41262-41273 + dujiampjkx 41093)。
- 关卡重置: g.level_reset() (base_game.py:326) 从 _clean_levels 重建当前关。

搜索 = 回放 BFS (同 g50t_solver 框架): 每节点 level_reset + 重放路径 + 试候选
([1,2,3,4] + 每个 0022 块一个点击坐标, 由 grid->display 反变换并用 display_to_grid
回读校验); 状态键 = 当前关全部 sprite 的 (位置/旋转/可见/像素摘要) 排序集合 + 选中块位置
+ 爆炸推挤偏置表 ooneovlmbq。StepCounter 余量=路径长度, 不入键。
"""
import hashlib
import time
from collections import deque

from arcengine import ActionInput, GameAction, GameState

_BOX_TAG = "0022vrxelxosfy"


def _budget(g):
    try:
        lv = g._clean_levels[int(g._current_level_index)]
        return max(5, int(lv.get_data("StepCounter")))
    except Exception:
        return 100


def _cell_to_display(cam, wx, wy):
    """grid(世界)坐标 -> display 像素候选列表 (display_to_grid 的整数反变换)"""
    scale = min(int(64 / cam.width), int(64 / cam.height))
    xpad = int((64 - cam.width * scale) / 2)
    ypad = int((64 - cam.height * scale) / 2)
    gx, gy = wx - cam.x, wy - cam.y
    if gx < 0 or gy < 0 or gx >= cam.width or gy >= cam.height:
        return []
    return [(gx * scale + xpad + i, gy * scale + ypad + j)
            for i in range(scale) for j in range(scale)]


def _click_candidates(g):
    """每个可选 0022 块产出一个经过双重校验的点击动作 dict {"a":6,"x","y"}."""
    lv = g.current_level
    cam = g.camera
    out, used = [], set()
    for s in lv.get_sprites_by_tag(_BOX_TAG):
        if not s.is_visible:
            continue
        hit = None
        cx, cy = int(s.x) + s.width // 2, int(s.y) + s.height // 2
        cells = [(cx, cy)]
        cells += [(cx + d, cy) for d in range(1, max(s.width // 2, 1))]
        cells += [(cx, cy + d) for d in range(1, max(s.height // 2, 1))]
        for (wx, wy) in cells:
            if lv.get_sprite_at(wx, wy, _BOX_TAG) is not s:
                continue
            for (dx, dy) in _cell_to_display(cam, wx, wy):
                if cam.display_to_grid(dx, dy) != (wx, wy):
                    continue
                if (dx, dy) in used:
                    continue
                hit = {"a": 6, "x": dx, "y": dy}
                used.add((dx, dy))
                break
            if hit:
                break
        if hit:
            out.append(hit)
    return out


def _sprite_sig(s):
    import numpy as np
    px = np.ascontiguousarray(s.pixels)
    return (int(s.x), int(s.y), int(s.rotation or 0), int(bool(s.is_visible)),
            hashlib.sha1(px.tobytes()).hexdigest())


def _state_key(g):
    sigs = sorted(_sprite_sig(s) for s in g.current_level.get_sprites())
    sel = (int(g.prkgpeyexo.x), int(g.prkgpeyexo.y))
    offs = sorted((int(k.x), int(k.y), int(v[0]), int(v[1]))
                  for k, v in g.ooneovlmbq.items())
    return hashlib.sha1(repr((sigs, sel, offs)).encode("utf-8")).hexdigest()


def _restore_start(g, idx0):
    """恢复到搜索开始那一关的起点。

    level_reset() 重置的是"当前" _current_level_index, 而搜索中的胜利着已把它推进到下一关
    (g50t 首轮实测踩坑: 调用方首步直接跳关, 解法路径一步没走过), 故先整体还原再 set_level。
    """
    g._levels = [lv.clone() for lv in g._clean_levels]
    g._next_level = False
    g._score = idx0
    g.set_level(idx0)
    g._state = GameState.NOT_FINISHED


def _dive(g, act, won, dead, h, budget, t0, t_end, w=2.0):
    """俯冲搜索 (深度优先 + 兄弟挂起栈): 每候选只评估一次, 未下潜的兄弟回头再潜.

    队列搜索每弹一个节点要为每个候选各重放一次整条路径 (O(分支*深度)), 240s 只铺到深度 ~20;
    KA59 第 2 关基线 109 步。俯冲沿一条链推进, 死胡同才回退一层, 能走到 budget 深度。
    与 g50t 同步修掉的完整性 bug: 旧写法只承诺最优候选, 其余候选既进 seen 又被标记 tried,
    回退时该层直接判穷尽, 兄弟子树永久丢弃 (实测 L1 也在 29 节点"穷尽根节点").
    """
    def do(a):
        return act(a if not isinstance(a, tuple) else {"a": 6, "x": a[1], "y": a[2]})

    def reach(path):
        g.level_reset()
        for a in path:
            r = do(a)
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
            print("[ka59_solver] 俯冲超时: %.0fs 深度=%d 节点=%d seen=%d"
                  % (time.time() - t0, len(path), nodes, len(seen)))
            return None
        d = len(path)
        if d >= budget - 1 or at(path) != "ok":
            if unwind() is None:
                print("[ka59_solver] 俯冲穷尽: 节点=%d" % nodes)
                return None
            continue
        if not frames[d]:
            cands = [1, 2, 3, 4] + [("C", c["x"], c["y"]) for c in _click_candidates(g)]
            for a in cands:
                if at(path) != "ok":
                    break
                r = do(a)
                eng[0] = path + (a,)
                if won(r):
                    print("[ka59_solver] 俯冲命中: 深度=%d 节点=%d" % (d + 1, nodes))
                    return list(path) + [a]
                if dead(r):
                    continue
                k = _state_key(g)
                if k in seen:
                    continue
                seen.add(k)
                nodes += 1
                frames[d].append((d + 1 + w * h(), a))
        if not frames[d]:
            if unwind() is None:
                print("[ka59_solver] 俯冲穷尽: 节点=%d" % nodes)
                return None
            continue
        frames[d].sort(key=lambda t: (t[0], str(t[1])))
        _f, a = frames[d].pop(0)
        path = path + (a,)
        while len(frames) < len(path) + 1:
            frames.append([])


def solve(game, t_limit=180.0, max_nodes=100000):
    """回放 BFS 求当前关: 返回动作序列 (int 1-4 / dict {"a":6,"x","y"}); 失败 None; 已胜 []."""
    g = game
    t0 = time.time()
    idx0 = int(g._current_level_index)
    if int(g._score) > idx0:
        return []
    budget = _budget(g)
    AID = {a: getattr(GameAction, "ACTION%d" % a) for a in (1, 2, 3, 4, 6)}

    def act(a):
        if isinstance(a, dict):
            inp = ActionInput(id=AID[6], data={"x": int(a["x"]), "y": int(a["y"])},
                              reasoning=None)
        else:
            inp = ActionInput(id=AID[int(a)], data={}, reasoning=None)
        return g.perform_action(inp, raw=True)

    def won(r):
        return (r.levels_completed > idx0 or r.state == GameState.WIN
                or int(g._current_level_index) > idx0)

    def dead(r):
        return r.state == GameState.GAME_OVER

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

    def h():
        """启发: 每个槽到"最近可嵌块目标"的曼哈顿距离和 (嵌入偏移 2)。

        试过把"未填满的槽数"做成字典序主项 (unmet*1000+dist)，实测无收益：
        第 2 关仍 300s 无解，而第 1 关从 6.3s 涨到 20s —— f = len + 2h 里的
        深度信号被 1000 量级的主项淹没，队列退化成贪心。
        """
        lv = g.current_level
        boxes = lv.get_sprites_by_tag(_BOX_TAG)
        mains = lv.get_sprites_by_tag("0001uqqokjrptk")
        total = 0
        for s in lv.get_sprites_by_tag("0010xzmuziohuf"):
            cand = [max(0, abs(s.x - b.x) + abs(s.y - b.y) - 2) for b in boxes]
            total += min(cand, default=0)
        for s in lv.get_sprites_by_tag("0027jbgxilrocf"):
            cand = [max(0, abs(s.x - m.x) + abs(s.y - m.y) - 2) for m in mains]
            total += min(cand, default=0)
        return total

    import heapq

    def _out(path):
        return [a if not isinstance(a, tuple) else {"a": 6, "x": a[1], "y": a[2]}
                for a in path]

    def _queue_search(t_end):
        """回放式加权队列搜索 (实测 L1 11 步 / 6s): 浅关首选, 深关铺不满深度."""
        reach(())
        seen = {_state_key(g)}
        ctr = 0
        q = [(2 * h(), 0, ())]
        nodes = 0
        while q:
            if time.time() > t_end or nodes > max_nodes:
                print("[ka59_solver] 队列搜索超时/超限: nodes=%d seen=%d q=%d" % (nodes, len(seen), len(q)))
                return None
            _f, _c, path = heapq.heappop(q)
            if len(path) >= budget - 1:
                continue
            clicks = None
            for a in (1, 2, 3, 4, "C"):
                st, info = reach(path)
                if st == "won":
                    return list(path[:info]) + [path[info]]
                if st == "dead":
                    break
                if a == "C":
                    if clicks is None:
                        clicks = _click_candidates(g)
                    for c in clicks:
                        r = act(c)
                        if won(r):
                            return list(path) + [c]
                        if dead(r):
                            continue
                        k = _state_key(g)
                        if k in seen:
                            continue
                        seen.add(k)
                        npath = path + (c,)
                        ctr += 1
                        heapq.heappush(q, (len(npath) + 2 * h(), ctr, npath))
                        nodes += 1
                    continue
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
                ctr += 1
                heapq.heappush(q, (len(npath) + 2 * h(), ctr, npath))
                nodes += 1
        print("[ka59_solver] 队列搜索穷尽: nodes=%d seen=%d" % (nodes, len(seen)))
        return None

    try:
        now = time.time()
        # 截止点必须是绝对时间戳: 曾把"剩余秒数"当截止时间交给俯冲,
        # 360s 的关卡限制实际只跑到 216s 就收工 (run4 实测)。
        p = _queue_search(now + (t0 + t_limit - now) * 0.4)
        if p is not None:
            return _out(p)
        # 顺序依据实测: 队列解浅关又快又短 (L1 11 步/6s), 俯冲只在深关接手
        # (修好完整性的俯冲在 L1 上会一路潜到深度 92 空转 60s 无解)
        if t0 + t_limit - time.time() < 5:
            return None
        p = _dive(g, act, won, dead, h, budget, t0, t0 + t_limit)
        return _out(p) if p is not None else None
    finally:
        try:
            _restore_start(g, idx0)
        except Exception:
            pass
