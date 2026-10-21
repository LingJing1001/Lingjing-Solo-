# -*- coding: utf-8 -*-
"""Lingjing 专用求解器公共框架 —— 从 g50t/ka59 solver 抽出的共享搜索基建。

只放"两关实测都验证过"的通用机制; 游戏专属知识 (机关标签/启发式/候选动作/状态键
入口) 一律留在各 solver 文件里。改这里的代码前先读注释: 每条"奇怪"的写法都对应
一次实测翻车, 不要顺手简化。

实测教训 (勿删):
1. restore_start 不能用 level_reset(): 搜索中的胜利着已把 _current_level_index
   推进到下一关, 调用方按返回路径重放首步会直接跳关。必须整体还原 _levels 后
   再 set_level(idx0)。
2. 搜索截止时间一律绝对时间戳: 曾把"剩余秒数"当截止传给搜索, 420s 预算实际
   只跑了 60% (run5/run4 实测)。
3. 俯冲(dive)必须把整层兄弟挂起后再逐个下潜: 早期"每层只承诺最优候选"的写法,
   其余候选既进 seen 又被跳过, 回退时该层直接判穷尽, 兄弟子树永久丢失
   (实测连 L1 都在 21/29 节点"穷尽根节点")。
4. 队列(replay_bfs)每弹一个节点要为每个候选 reset+重放 (O(分支*深度)),
   铺不满深关 — 只做浅关快解; 深关交给 dive 兜底。
5. 状态键绝不进纯簿记字段 (移动历史/动画队列), 否则去重彻底失效;
   影响后续行为的字段 (录像带/选中块/推挤偏置) 是"状态", 按档位决定去留。
6. 每个候选动作执行前必须 reach() 重放到节点状态: ka59 旧版队列一批点击只
   重放一次, 第 2 个点击起在"上一点击之后"的错误状态上执行, won() 命中假解
   且错误状态键污染 seen (2026-10-21 抽框架时修正)。
"""
import hashlib
import time

from arcengine import GameState


def sprite_sig(s):
    """Sprite 的确定性签名: 位姿/旋转/可见性 + 像素 sha1。

    不能用 repr(s): 含内存地址, 同一状态两次重放得到不同键。
    """
    import numpy as np
    px = np.ascontiguousarray(s.pixels)
    return (int(s.x), int(s.y), int(s.rotation or 0), int(bool(s.is_visible)),
            hashlib.sha1(px.tobytes()).hexdigest())


def make_won(g, idx0):
    """胜负判定闭包: levels 推进 / 引擎 WIN / 关卡索引推进, 任一即胜。"""
    def won(r):
        return (r.levels_completed > idx0 or r.state == GameState.WIN
                or int(g._current_level_index) > idx0)
    return won


def is_dead(r):
    return r.state == GameState.GAME_OVER


def restore_start(g, idx0):
    """把引擎恢复到搜索开始那一关的起点 (教训 1)。

    solve() 的 finally 必须调它: 不管搜没搜到, 引擎都回到当前关起点,
    调用方按返回路径重放即可; 不污染引擎状态。
    """
    g._levels = [lv.clone() for lv in g._clean_levels]
    g._next_level = False
    g._score = idx0
    g.set_level(idx0)
    g._state = GameState.NOT_FINISHED


def make_reach(g, act, won, dead):
    """reset + 重放整条路径; 中途胜/死立即报告 (调用方据此截取前缀解)。"""
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
    return reach


def replay_bfs(g, act, won, dead, h, budget, t0, t_end, max_nodes,
               cands_fn, state_key, w=2.0, tag="solver"):
    """回放式加权队列搜索 (浅关快解): f = 步数 + w*h。

    每弹一个节点: 逐候选 reach 重放 -> act 试步 -> 按状态键去重入队。
    cands_fn(g) 在"引擎已重放到节点状态"之后才调用 — 点击类候选的生成依赖
    节点状态, 提前调用会在错误状态上取候选 (教训 6)。
    去重后仍是按 f 最短优先的完全搜索 — 只放弃"步数最优", 保留"可达即找到"。
    """
    import heapq
    reach = make_reach(g, act, won, dead)
    reach(())                      # 先回到关起点, 根状态键才有意义
    seen = {state_key(g)}
    ctr = 0
    nodes = 0
    q = [(w * h(), 0, ())]
    while q:
        if time.time() > t_end or nodes > max_nodes:
            print("[%s] 队列搜索超时/超限: nodes=%d seen=%d q=%d"
                  % (tag, nodes, len(seen), len(q)))
            return None
        _f, _c, path = heapq.heappop(q)
        if len(path) >= budget - 1:
            continue
        cands = None               # 引擎重放到节点状态后才生成候选 (教训 6)
        idx = 0
        while True:
            st, info = reach(path)
            if st == "won":
                return list(path[:info]) + [path[info]]
            if st == "dead":
                break
            if cands is None:
                cands = list(cands_fn(g))
            if idx >= len(cands):
                break
            a = cands[idx]
            idx += 1
            r = act(a)
            if won(r):
                return list(path) + [a]
            if dead(r):
                continue
            k = state_key(g)
            if k in seen:
                continue
            seen.add(k)
            npath = path + (a,)
            ctr += 1
            heapq.heappush(q, (len(npath) + w * h(), ctr, npath))
            nodes += 1
    print("[%s] 队列搜索穷尽: nodes=%d seen=%d" % (tag, nodes, len(seen)))
    return None


def dive(g, act, won, dead, h, budget, t0, t_end, cands_fn, state_key,
         w=2.0, stat=None, tag="solver"):
    """俯冲搜索 (深度优先 + 兄弟挂起栈): 沿一条链推进, 死胡同才回退一层,
    能真正走到 budget 深度 (队列版铺不满深关时用它兜底, 教训 3/4)。

    frames[d] 挂起深度 d 已评估、尚未下潜的全部子节点, 按 f 排序逐个下潜;
    回退时只作废更深层, 当前层的挂起兄弟保留 (unwind)。
    """
    reach = make_reach(g, act, won, dead)
    path = ()
    eng = [None]         # 引擎当前停在哪个路径前缀上 (避免无谓重放)
    frames = [[]]        # frames[d] = 深度 d 的挂起子节点 [(f, action), ...]
    seen = set()
    nodes = 0

    def at(p):
        if eng[0] != p:
            st = reach(p)[0]
            eng[0] = p
            return st
        return "ok"

    def unwind():
        """回退到最近一层还有挂起兄弟的节点; 一路退空返回 None (穷尽)。

        frames[len(path)] 是"当前节点"自己的挂起列表, 退到它时要保留,
        只作废更深层的帧。
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
            print("[%s] 俯冲超时: %.0fs 深度=%d 节点=%d seen=%d"
                  % (tag, time.time() - t0, len(path), nodes, len(seen)))
            return None
        d = len(path)
        if d >= budget - 1 or at(path) != "ok":
            if unwind() is None:
                print("[%s] 俯冲穷尽: 节点=%d" % (tag, nodes))
                return None
            continue
        if stat is not None:
            stat["max_depth"] = max(stat.get("max_depth", 0), d)
        if not frames[d]:
            # cands_fn 在 at(path)=="ok" 之后调用: 引擎已处于节点状态 (教训 6)
            for a in cands_fn(g):
                if at(path) != "ok":
                    break
                r = act(a)
                eng[0] = path + (a,)
                if won(r):
                    print("[%s] 俯冲命中: 深度=%d 节点=%d" % (tag, d + 1, nodes))
                    return list(path) + [a]
                if dead(r):
                    if stat is not None:
                        stat["dead"] = stat.get("dead", 0) + 1
                    continue
                k = state_key(g)
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
                print("[%s] 俯冲穷尽: 节点=%d" % (tag, nodes))
                return None
            continue
        frames[d].sort(key=lambda t: (t[0], str(t[1])))
        _f, a = frames[d].pop(0)
        path = path + (a,)
        while len(frames) < len(path) + 1:
            frames.append([])
