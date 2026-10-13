# -*- coding: utf-8 -*-
"""CN04 A* 求解器 — 紧凑状态 A* + 引擎回放。

不依赖任何预存计划，纯 A* 搜索。
"""
import sys, os, heapq, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput, GameState
import numpy as np


def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)


def apply_action(env, action):
    kind = action[0]
    if kind == "click":
        return env.step(GameAction.ACTION6, data={"x": action[1], "y": action[2]})
    elif kind == "move":
        return env.step(GameAction.from_id(action[1]), data={})
    elif kind == "rotate":
        return env.step(GameAction.ACTION5, data={})
    elif kind == "undo":
        return env.step(GameAction.ACTION7, data={})


def replay(env, actions, level):
    frame = env.reset()
    for a in actions:
        frame = apply_action(env, a)
        if frame.levels_completed > level or frame.state.name == "WIN":
            break
    return frame


def compact_state(env):
    """紧凑状态：所有 sprite 的 (name, x, y, rotation, visible, layer) + 选中。"""
    g = env._game
    sprites = []
    for s in g.current_level.get_sprites():
        sprites.append((s.name, s.x, s.y, s.rotation, s.is_visible, s.layer))
    sel = g.xseexqzst.name if g.xseexqzst else None
    return (g.level_index, tuple(sorted(sprites)), sel)


def restore_state(env, key):
    """从紧凑状态恢复：reset + 重建 sprite 状态。"""
    level, sprites, sel = key
    env.reset()
    g = env._game
    # 如果 level 不对，跳到目标关
    while g.level_index < level:
        g.next_level()
    # 恢复 sprite 状态
    current = {s.name: s for s in g.current_level.get_sprites()}
    for name, x, y, rot, vis, layer in sprites:
        if name in current:
            s = current[name]
            s.set_position(x, y)
            while s.rotation != rot:
                s.rotate(90)
            s.set_visible(vis)
            s.set_layer(layer)
    # 恢复选中
    if sel and sel in current:
        g.xseexqzst = current[sel]
    else:
        g.xseexqzst = None
    g.uqlndqojuf()
    return g


def heuristic(env):
    """启发式：未匹配的特殊像素数。"""
    g = env._game
    count = 0
    for s in g.current_level.get_sprites():
        if not s.is_visible:
            continue
        orig = g.hlxyvcmpk.get(s.name)
        if orig is None:
            continue
        # 数原始 8/13 像素在当前 rotation 下是否被匹配
        h, w = orig.shape
        for y in range(h):
            for x in range(w):
                if orig[y, x] in (8, 13):
                    # 应用旋转
                    rot = s.rotation
                    if rot == 0:
                        rx, ry = x, y
                    elif rot == 90:
                        rx, ry = h - 1 - y, x
                    elif rot == 180:
                        rx, ry = w - 1 - x, h - 1 - y
                    elif rot == 270:
                        rx, ry = y, w - 1 - x
                    else:
                        rx, ry = x, y
                    gx, gy = int(s.x + rx), int(s.y + ry)
                    # 检查是否被其他 sprite 覆盖
                    matched = False
                    for s2 in g.current_level.get_sprites():
                        if s2 == s or not s2.is_visible:
                            continue
                        if s2.x <= gx < s2.x + s2.width and s2.y <= gy < s2.y + s2.height:
                            rendered = s2.render()
                            lx, ly = gx - s2.x, gy - s2.y
                            if 0 <= ly < rendered.shape[0] and 0 <= lx < rendered.shape[1]:
                                if rendered[ly, lx] in (8, 13):
                                    matched = True
                                    break
                    if not matched:
                        count += 1
    return count / 2.0


def candidate_actions(env):
    """生成候选动作。"""
    g = env._game
    actions = []
    seen = set()

    # 预计算 grid->display 映射
    g2d = {}
    for dy in range(64):
        for dx in range(64):
            res = g.camera.display_to_grid(dx, dy)
            if res is not None and res not in g2d:
                g2d[res] = (dx, dy)

    # 1. 点击每个可见 sprite 的非透明像素
    for s in g.current_level.get_sprites():
        if not s.is_visible:
            continue
        rendered = s.render()
        if rendered is None:
            continue
        non_tp = np.argwhere(rendered >= 0)
        if len(non_tp) == 0:
            continue
        indices = list(range(0, len(non_tp), max(1, len(non_tp) // 5)))[:5]
        for idx in indices:
            y, x = non_tp[idx]
            gx, gy = int(s.x + x), int(s.y + y)
            if (gx, gy) in g2d:
                d = g2d[(gx, gy)]
                if d not in seen:
                    seen.add(d)
                    actions.append(("click", d[0], d[1]))

    # 2. 如果有选中的 sprite，生成移动和旋转
    if g.xseexqzst:
        for act_id in (1, 2, 3, 4):
            actions.append(("move", act_id))
        actions.append(("rotate",))

    return actions


def solve_level(env, level, max_nodes=50000, weight=1.5, time_budget=120):
    """A* 搜索单关。"""
    t0 = time.time()
    frame = env.reset()
    if frame.levels_completed > level or frame.state.name == "WIN":
        return []

    key0 = compact_state(env)
    h0 = heuristic(env)

    counter = 0
    heap = [(weight * h0, 0, counter, [], key0)]
    seen = {key0: 0}
    nodes = 0
    best_h = h0

    while heap and nodes < max_nodes:
        if time.time() - t0 > time_budget:
            print(f"  [A*] TIME OUT nodes={nodes}", flush=True)
            return None

        f, g_cost, _, actions, key = heapq.heappop(heap)
        if g_cost != seen.get(key):
            continue
        nodes += 1

        h = f - g_cost
        if h < best_h:
            best_h = h
            if nodes % 50 == 1:
                print(f"  [A*] nodes={nodes} g={g_cost} h={h:.1f} heap={len(heap)}", flush=True)

        # 恢复到此节点
        restore_state(env, key)
        if env._game.level_index > level or env._game._state.name == "WIN":
            return actions

        # 展开候选
        cands = candidate_actions(env)
        for action in cands:
            fr = apply_action(env, action)
            if fr.levels_completed > level or fr.state.name == "WIN":
                return list(actions) + [action]

            new_key = compact_state(env)
            new_h = heuristic(env)
            new_g = g_cost + 1

            if new_key in seen and seen[new_key] <= new_g:
                continue
            seen[new_key] = new_g
            counter += 1
            heapq.heappush(heap, (new_g + weight * new_h, new_g, counter, list(actions) + [action], new_key))

    print(f"  [A*] exhausted nodes={nodes}", flush=True)
    return None


def main():
    env = make_env("cn04")
    if env is None:
        raise SystemExit("无法创建 cn04 环境")

    total = 0
    all_ok = True
    for lvl in range(6):
        t0 = time.time()
        sol = solve_level(env, lvl, max_nodes=50000, weight=1.5, time_budget=120)
        dt = time.time() - t0
        if sol is None:
            print(f"L{lvl+1}: NO SOLUTION  {dt:.1f}s")
            all_ok = False
            break
        frame = replay(env, sol, lvl)
        ok = frame.levels_completed > lvl or frame.state.name == "WIN"
        total += len(sol)
        if not ok:
            all_ok = False
        print(f"L{lvl+1}: {len(sol)} actions -> levels={frame.levels_completed}  "
              f"{'OK' if ok else 'FAIL'}  {dt:.1f}s")

    print(f"\nTotal: actions={total}")
    print("ALL_PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
