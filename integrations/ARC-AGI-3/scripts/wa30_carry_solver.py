"""wa30 L0 定向求解器（env ee6fef47 实证通过）：按已解码机制规划 grab-and-carry 序列，在真实引擎上执行并回放验证。

机制依据（wa30.py ee6fef47）:
- 移动 4 格跳（celomdfhbh=4），仅检查落点；未携带时每次移动都更新朝向（pjedoipwee），
  落点被挡则原地转向（wqwsvmhhzj 无 else 分支，转向不消耗额外判定）。
- ACTION5 未携带: 抓正前方恰 4 格的 geezpjgiyd 块（vwiozbtqgi，按 rotation 换算）。
- 抓取锁定块相对偏移; 携带期间朝向冻结（qnmfimgpwc 只在未携带时 set_rotation），
  偏移随移动保持（wqwsvmhhzj dx,dy 保序）。
- 可行性 fuykgiiwit: 玩家落点 t 非障碍 或 t==块当前格; 块落点 b 非障碍 或 b==玩家当前格。
- ACTION5 携带中: 就地放下（kqrtstlzkg 无前置），块停在当前位置。
- 胜利 ymzfopzgbq: 所有块左上角 ∈ wyzquhjerd（fsjjayjoeg 足迹）且无任何块被携带。
- 放下末块时 step() 内直接 next_level()，故最后一动之后读到的精灵属于下一关。

构建稳定接口: 精灵/标签/足迹/is_collidable/_score/_state 来自 arcengine 基类，跨 build 稳定;
唯一 build 专属读取是携带映射 nsevyuople（ee6fef47 实名，缺失时回退到模型自证）。
2026-09-23 实测: 26/200 步通关，逐动作断言零偏差，全新 reset 回放复现。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
STARTER = PKG.parent / "ARC-AGI-3-Kaggle-Starter"
os.chdir(STARTER)
for p in (str(PKG), str(STARTER)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput, GameAction

CEIL = 4  # celomdfhbh
DIRV = {1: (0, -CEIL), 2: (0, CEIL), 3: (-CEIL, 0), 4: (CEIL, 0)}  # 动作→位移
ROTD = {1: 0, 2: 180, 3: 270, 4: 90}  # 动作→朝向（pjedoipwee: 北0 东90 南180 西270）
NAMES = {1: "A1↑", 2: "A2↓", 3: "A3←", 4: "A4→", 5: "A5抓/放"}

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                     environments_dir=str(STARTER / "environment_files"))
env = arc.make("wa30")
env.reset()
g = env._game
budget = g.kuncbnslnm.dbdarsgrbj


def lv():
    return g.current_level


def player():
    return lv().get_sprites_by_tag("wbmdvjhthc")[0]


def blocks():
    return lv().get_sprites_by_tag("geezpjgiyd")


def carry_count():
    m = getattr(g, "nsevyuople", None)
    return len(m) if m is not None else None


def footprint(sp):
    return {(sp.x + i, sp.y + j) for i in range(sp.width) for j in range(sp.height)}


def ring():
    r = set()
    for i in range(0, 64, CEIL):
        r |= {(-CEIL, i), (64, i), (i, -CEIL), (i, 64)}
    return r


def level_sets():
    """区域格（fsjjayjoeg 足迹）、墙体格（bnzklblgdk 左上角）、实心格（collidable 左上角）。"""
    zone: set = set()
    for sp in lv().get_sprites_by_tag("fsjjayjoeg"):
        zone |= footprint(sp)
    qth = {(sp.x, sp.y) for sp in lv().get_sprites_by_tag("bnzklblgdk")}
    solid = {(sp.x, sp.y) for sp in lv().get_sprites() if sp.is_collidable}
    return zone, qth, solid


ZONE, QTH, SOLID = level_sets()
RING = ring()


def walk_bfs(start, rot0, goal, goal_rot, blocked):
    """未携带走位 BFS。状态 (x,y,rot)；落点可行→(落点,新朝向)，被挡→原地转向。
    blocked 为冻结障碍集，须已扣除玩家起点格（起点让出后可重入）。返回动作列表或 None。"""
    s = (start[0], start[1], rot0)
    t = (goal[0], goal[1], goal_rot)
    if s == t:
        return []
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        for a, (dx, dy) in DIRV.items():
            nxt_cell = (cur[0] + dx, cur[1] + dy)
            if nxt_cell in blocked:
                nxt = (cur[0], cur[1], ROTD[a])            # 撞墙 → 原地转向
            else:
                nxt = (nxt_cell[0], nxt_cell[1], ROTD[a])  # 正常跳
            if nxt not in prev:
                prev[nxt] = (cur, a)
                if nxt == t:
                    path = []
                    it = nxt
                    while prev[it] is not None:
                        it, act = prev[it]
                        path.append(act)
                    return path[::-1]
                q.append(nxt)
    return None


def carry_bfs(p0, off, static):
    """携带移动 BFS。状态 (px,py)，偏移 off 固定、朝向冻结。
    static: 冻结障碍（RING|实心格，扣除玩家与块的初始格——两格随行让出后可重入）。
    目标: 块落点 (p+off) ∈ ZONE。返回动作列表或 None。"""
    b0 = (p0[0] + off[0], p0[1] + off[1])
    if b0 in ZONE:
        return []
    prev = {p0: None}
    q = deque([p0])
    while q:
        cur = q.popleft()
        bcur = (cur[0] + off[0], cur[1] + off[1])
        for a, (dx, dy) in DIRV.items():
            t = (cur[0] + dx, cur[1] + dy)      # 玩家落点
            b = (t[0] + off[0], t[1] + off[1])  # 块落点
            ok_t = (t not in static or t == bcur) and t not in QTH   # fuykgiiwit 前半
            ok_b = (b not in static or b == cur)                     # fuykgiiwit 后半
            if not (ok_t and ok_b):
                continue
            if b in ZONE:
                prev[t] = (cur, a)
                path = []
                it = t
                while prev[it] is not None:
                    it, act = prev[it]
                    path.append(act)
                return path[::-1]
            if t not in prev:
                prev[t] = (cur, a)
                q.append(t)
    return None


root_score, root_idx = g._score, g.level_index
SOLVED = False


def transitioned():
    return (g._score > root_score or g.level_index > root_idx
            or str(g._state) != "GameState.NOT_FINISHED")


def do(aid, pred):
    """执行动作并断言引擎结果与预测一致；换关（胜利）则置 SOLVED 跳过断言。
    pred: (期望玩家xy, 期望朝向, 期望携带数|None)"""
    global SOLVED
    fr = g.perform_action(ActionInput(id=GameAction.from_id(aid), data={}), raw=True)
    if transitioned():
        SOLVED = True
        return fr
    p = player()
    got = ((p.x, p.y), p.rotation, carry_count())
    want = pred[2] if pred[2] is not None else got[2]
    if got != (pred[0], pred[1], want):
        print(f"!! 断言失败 动作{NAMES[aid]} 预测{pred} 实际{got}")
        sys.exit(2)
    return fr


def phase_costs():
    """对每个剩余块×每个站位，算 walk+grab+carry+drop 总代价。"""
    p = player()
    solid = {(sp.x, sp.y) for sp in lv().get_sprites() if sp.is_collidable}
    blocked = (RING | solid | QTH) - {(p.x, p.y)}  # 扣玩家起点格：起点让出后可重入
    opts = []
    for bi, blk in enumerate(blocks()):
        if (blk.x, blk.y) in ZONE:
            continue
        others = [(b.x, b.y) for j, b in enumerate(blocks()) if j != bi]
        for d, (dx, dy) in DIRV.items():
            stance = (blk.x - dx, blk.y - dy)  # 面向 d 时前方 4 格 = 块
            w = walk_bfs((p.x, p.y), p.rotation, stance, ROTD[d], blocked)
            if w is None:
                continue
            off = (blk.x - stance[0], blk.y - stance[1])
            static = ((RING | solid) - {(p.x, p.y), (blk.x, blk.y)}) | QTH
            c = carry_bfs(stance, off, static)
            if c is None:
                continue
            opts.append({"block": (blk.x, blk.y), "block_idx": bi, "dir": d,
                         "stance": stance, "off": off, "walk": w, "carry": c,
                         "total": len(w) + 1 + len(c) + 1})
    return opts


t0 = time.perf_counter()
seq_full: list[int] = []
phases = []
while not SOLVED:
    if all((b.x, b.y) in ZONE for b in blocks()) and carry_count() in (0, None):
        print("全部块就位且无携带 — WIN 条件成立")
        break
    opts = phase_costs()
    if not opts:
        print("!! 无可行方案")
        sys.exit(3)
    best = min(opts, key=lambda o: o["total"])
    print(f"选块 {best['block']} 站位 {best['stance']} 朝向{ROTD[best['dir']]} "
          f"walk={len(best['walk'])} carry={len(best['carry'])} 总代价={best['total']}")
    p = player()
    for a in best["walk"]:
        dx, dy = DIRV[a]
        nxt = (p.x + dx, p.y + dy)
        live_solid = {(sp.x, sp.y) for sp in lv().get_sprites() if sp.is_collidable}
        if nxt in RING | live_solid | QTH:
            pred = ((p.x, p.y), ROTD[a], 0)   # 撞墙原地转向
        else:
            pred = (nxt, ROTD[a], 0)
        do(a, pred)
        if SOLVED:
            break
        p = player()
        seq_full.append(a)
    if not SOLVED:
        blk = next(b for b in blocks() if (b.x, b.y) == best["block"])
        do(5, ((p.x, p.y), p.rotation, None))  # 抓（携带数走模型：块应仍在原位随行确认）
        assert (blk.x, blk.y) == (
            p.x + best["off"][0], p.y + best["off"][1]), "抓取后块未按偏移随行"
        seq_full.append(5)
        p = player()
        for a in best["carry"]:
            dx, dy = DIRV[a]
            pred = ((p.x + dx, p.y + dy), p.rotation, None)  # 携带中朝向冻结
            do(a, pred)
            if SOLVED:
                break
            p = player()
            want_blk = (p.x + best["off"][0], p.y + best["off"][1])
            blk = next((b for b in blocks() if (b.x, b.y) == want_blk), None)
            assert blk is not None, f"携带偏移漂移：无块在预测随行位 {want_blk}"
            seq_full.append(a)
    if not SOLVED:
        p = player()
        do(5, ((p.x, p.y), p.rotation, None))  # 放
        seq_full.append(5)
    phases.append({"block": list(best["block"]), "stance": list(best["stance"]),
                   "walk": len(best["walk"]), "carry": len(best["carry"])})

dt = time.perf_counter() - t0
print(f"动作数={len(seq_full)}/{budget} 用时{dt:.1f}s score={g._score} "
      f"level={g.level_index} state={g._state} SOLVED={SOLVED}")

# —— 回放验证：全新 reset 后重放同一序列 ——
env.reset()
g = env._game
replay_ok = True
try:
    for a in seq_full:
        g.perform_action(ActionInput(id=GameAction.from_id(a), data={}), raw=True)
except Exception as e:  # noqa: BLE001
    replay_ok = False
    print("回放异常:", e)
replay_transition = (g._score > root_score or g.level_index > root_idx
                     or str(g._state) != "GameState.NOT_FINISHED")
print(f"回放: ok={replay_ok} score={g._score} level={g.level_index} "
      f"state={g._state} 换关={replay_transition}")

report = {
    "game": "wa30", "level": 0, "env_version": "ee6fef47",
    "mechanics": "grab-and-carry: 4格跳+朝向; A5抓正前4格块/携带中放下; 偏移锁定; 朝向携带期间冻结",
    "budget": budget, "n_actions": len(seq_full),
    "sequence": seq_full, "sequence_names": [NAMES[a] for a in seq_full],
    "phases": phases,
    "final_score": g._score, "final_level": g.level_index,
    "final_state": str(g._state),
    "solved": SOLVED, "replay_ok": replay_ok, "replay_transition": replay_transition,
    "generated_at": datetime.now(timezone.utc).isoformat(),
}
dst = PKG / "bench" / ("wa30_solved_carry_" +
                       datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json")
dst.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("JSON ->", dst)
print("序列:", " ".join(NAMES[a] for a in seq_full))
if not (SOLVED and replay_ok and replay_transition):
    sys.exit(1)
