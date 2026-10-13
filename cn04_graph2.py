# -*- coding: utf-8 -*-
"""cn04 图探索求解器 — 自定义 BFS 搜索，支持点击位置候选。

策略:
  1. 用 generic_snapshot/generic_restore 管理状态
  2. 候选生成：点击特殊像素位置 + 键盘动作
  3. BFS 搜索，状态去重

用法:
    python cn04_graph2.py <level>
"""
import os, sys, json, time, heapq
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

# 导入通用搜索工具
from lingjing_solo.planning.search.generic_shadow import (
    generic_snapshot, generic_restore, generic_state_key, _level_index
)


def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)


def get_special_pixels(game, sprite):
    """获取 sprite 的特殊像素全局坐标。"""
    orig = game.hlxyvcmpk.get(sprite.name)
    if orig is None:
        return []
    h, w = orig.shape
    rot = sprite.rotation
    pts = []
    for y in range(h):
        for x in range(w):
            if orig[y, x] in (8, 13):
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
                pts.append((int(sprite.x + rx), int(sprite.y + ry)))
    return pts


def generate_candidates(game):
    """生成候选动作列表。

    返回: [(aid, x, y), ...]  其中 x,y=0 表示无坐标动作
    """
    acts = []

    # 预计算 grid->display 映射
    g2d = {}
    for dy in range(64):
        for dx in range(64):
            res = game.camera.display_to_grid(dx, dy)
            if res is not None and res not in g2d:
                g2d[res] = (dx, dy)

    # 获取所有 sprite 的特殊像素
    all_special = []
    for s in game.current_level.get_sprites():
        if not s.is_visible:
            continue
        sp = get_special_pixels(game, s)
        for px, py in sp:
            all_special.append((px, py))

    # 生成点击候选：点击特殊像素位置
    click_positions = set()
    for px, py in all_special:
        if (px, py) in g2d:
            click_positions.add(g2d[(px, py)])

    # 也生成一些偏移位置
    for px, py in all_special:
        for dx in (-1, 0, 1, 2, -2):
            for dy in (-1, 0, 1, 2, -2):
                gx, gy = px + dx, py + dy
                if (gx, gy) in g2d:
                    click_positions.add(g2d[(gx, gy)])

    # 添加点击候选
    for dx, dy in click_positions:
        acts.append((6, dx, dy))

    # 添加键盘动作
    for aid in (1, 2, 3, 4, 5):
        acts.append((aid, 0, 0))

    return acts


def apply_action(env, aid, x=0, y=0):
    """执行动作。"""
    if aid == 6:
        return env.step(GameAction.ACTION6, data={"x": x, "y": y})
    else:
        return env.step(GameAction.from_id(aid), data={})


def solve_level(level, max_nodes=50000, time_budget=120):
    """求解单关。"""
    env = make_env("cn04")
    frame = env.reset()
    g = env._game

    # 跳到目标关
    for _ in range(level):
        g.next_level()

    # 检查是否已过关
    if g.sjwqloivve():
        return []

    level_idx0 = _level_index(g)
    start_snap = generic_snapshot(g)

    # BFS 搜索
    heap = [(0, 0, 0, start_snap, [])]  # (heuristic, depth, seq, snapshot, path)
    root_key = generic_state_key(g)
    if root_key is None:
        return None
    seen = {root_key}
    nodes = 0
    seq = 0
    t0 = time.time()

    while heap:
        if time.time() - t0 > time_budget or nodes > max_nodes:
            return None

        _, depth, _, snap, path = heapq.heappop(heap)

        if depth >= 100:
            continue

        # 恢复状态
        generic_restore(g, snap)

        # 生成候选
        candidates = generate_candidates(g)

        for aid, x, y in candidates:
            if time.time() - t0 > time_budget:
                return None

            # 执行动作
            try:
                frame = apply_action(env, aid, x, y)
            except Exception:
                continue

            # 检查是否过关
            if _level_index(g) > level_idx0 or g.sjwqloivve():
                return path + [(aid, x, y)]

            # 检查是否 game over
            try:
                if g._state == GameState.GAME_OVER:
                    continue
            except Exception:
                pass

            # 状态去重
            k = generic_state_key(g)
            if k is None or k in seen:
                continue
            seen.add(k)

            # 计算启发式：未匹配的特殊像素数
            h = 0
            for s in g.current_level.get_sprites():
                if not s.is_visible:
                    continue
                sp = get_special_pixels(g, s)
                h += len(sp)

            seq += 1
            heapq.heappush(
                heap, (h, depth + 1, seq, generic_snapshot(g), path + [(aid, x, y)])
            )
            nodes += 1

    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("level", type=int, default=0, help="关卡编号 0-5")
    p.add_argument("--nodes", type=int, default=50000, help="最大节点数")
    p.add_argument("--time", type=int, default=120, help="超时秒数")
    args = p.parse_args()

    t0 = time.time()
    sol = solve_level(args.level, max_nodes=args.nodes, time_budget=args.time)
    dt = time.time() - t0

    if sol:
        print(f"L{args.level+1}: SOLVED {len(sol)} actions in {dt:.1f}s")
        for i, (aid, x, y) in enumerate(sol):
            if aid == 6:
                print(f"  {i+1}: click ({x},{y})")
            else:
                print(f"  {i+1}: ACTION{aid}")
    else:
        print(f"L{args.level+1}: NO SOLUTION in {dt:.1f}s")


if __name__ == "__main__":
    main()
