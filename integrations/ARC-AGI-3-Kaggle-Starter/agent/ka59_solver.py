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

搜索 = solver_common 的队列/俯冲基建 (教训清单见其模块头): 每节点 level_reset +
重放路径 + 试候选 ([1,2,3,4] + 每个 0022 块一个点击坐标, 由 grid->display 反变换
并用 display_to_grid 回读校验); 状态键 = 当前关全部 sprite 的 (位置/旋转/可见/
像素摘要) 排序集合 + 选中块位置 + 爆炸推挤偏置表 ooneovlmbq。
StepCounter 余量=路径长度, 不入键。

2026-10-21 修正: 旧版队列搜索一批点击只 reach 重放一次, 第 2 个点击起在
"上一点击之后"的错误状态上执行 (won() 命中假解、错误状态键污染 seen);
现改为每个点击都由框架逐候选重放 (教训 6)。
"""
import hashlib
import time

from arcengine import ActionInput, GameAction

from solver_common import (dive as _dive, replay_bfs as _replay_bfs,
                           restore_start as _restore_start,
                           make_won, is_dead, sprite_sig as _sprite_sig)

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
    """每个可选 0022 块产出一个经过双重校验的点击动作 dict {"a":6,"x","y"}.

    必须在引擎处于目标节点状态时调用 (选中块/可见性是节点状态的函数)。
    """
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


def _state_key(g):
    sigs = sorted(_sprite_sig(s) for s in g.current_level.get_sprites())
    sel = (int(g.prkgpeyexo.x), int(g.prkgpeyexo.y))
    offs = sorted((int(k.x), int(k.y), int(v[0]), int(v[1]))
                  for k, v in g.ooneovlmbq.items())
    return hashlib.sha1(repr((sigs, sel, offs)).encode("utf-8")).hexdigest()


def solve(game, t_limit=180.0, max_nodes=100000):
    """回放 BFS 求当前关: 返回动作序列 (int 1-4 / dict {"a":6,"x","y"}); 失败 None; 已胜 []."""
    g = game
    t0 = time.time()
    idx0 = int(g._current_level_index)
    if int(g._score) > idx0:
        return []
    budget = _budget(g)
    won = make_won(g, idx0)
    dead = is_dead
    AID = {a: getattr(GameAction, "ACTION%d" % a) for a in (1, 2, 3, 4, 6)}

    def act(a):
        if isinstance(a, dict):
            inp = ActionInput(id=AID[6], data={"x": int(a["x"]), "y": int(a["y"])},
                              reasoning=None)
        else:
            inp = ActionInput(id=AID[int(a)], data={}, reasoning=None)
        return g.perform_action(inp, raw=True)

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

    def cands(_g):
        # 点击候选必须在"引擎处于节点状态"时生成; 框架保证 cands_fn 的调用时机 (教训 6)。
        return [1, 2, 3, 4] + list(_click_candidates(g))

    try:
        now = time.time()
        # 截止点必须是绝对时间戳 (教训 2)。顺序依据实测: 队列解浅关又快又短
        # (L1 11 步/6s), 俯冲只在深关接手 (修好完整性的俯冲在 L1 上会一路
        # 潜到深度 92 空转 60s 无解)。
        p = _replay_bfs(g, act, won, dead, h, budget, t0,
                        now + (t0 + t_limit - now) * 0.4, max_nodes,
                        cands, _state_key, tag="ka59_solver")
        if p is not None:
            return list(p)
        if t0 + t_limit - time.time() < 5:
            return None
        p = _dive(g, act, won, dead, h, budget, t0, t0 + t_limit,
                  cands, _state_key, tag="ka59_solver")
        return list(p) if p is not None else None
    finally:
        try:
            _restore_start(g, idx0)
        except Exception:
            pass
