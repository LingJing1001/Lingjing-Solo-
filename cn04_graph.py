# -*- coding: utf-8 -*-
"""cn04 图探索求解器 — 用 r3_generic_search 解 cn04。

策略:
  1. 用 r3_generic_search 做 BFS 搜索
  2. 候选生成：点击特殊像素附近的位置 + 键盘动作
  3. 状态去重：用 game.__dict__ 快照

用法:
    python cn04_graph.py <level>
"""
import os, sys, json, time
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

# 导入通用搜索
from lingjing_solo.planning.search.generic_shadow import r3_generic_search


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


def make_cn04_action(game, env):
    """为 cn04 生成候选动作。"""
    g = game
    acts = []

    # 预计算 grid->display 映射
    g2d = {}
    for dy in range(64):
        for dx in range(64):
            res = g.camera.display_to_grid(dx, dy)
            if res is not None and res not in g2d:
                g2d[res] = (dx, dy)

    # 获取所有 sprite 的特殊像素
    all_special = []
    for s in g.current_level.get_sprites():
        if not s.is_visible:
            continue
        sp = get_special_pixels(g, s)
        for px, py in sp:
            all_special.append((px, py, s.name))

    # 生成点击候选：点击特殊像素位置
    click_positions = set()
    for px, py, _ in all_special:
        if (px, py) in g2d:
            click_positions.add(g2d[(px, py)])

    # 也生成一些偏移位置（点击特殊像素附近）
    for px, py, _ in all_special:
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

    # 构建 act_map
    act_map = {n: getattr(GameAction, f"ACTION{n}") for n in range(1, 8)}

    # 构建 make_action
    def make_action(aid):
        if aid == 6:
            # 点击动作需要 x,y
            # 这里用默认值，实际由候选生成器提供
            return ActionInput(id=aid, data={"x": 0, "y": 0}, reasoning=None)
        return ActionInput(id=aid, data={}, reasoning=None)

    # 用 r3_generic_search 搜索
    path = r3_generic_search(
        env,
        t_limit=time_budget,
        max_nodes=max_nodes,
        max_depth=100,
        act_map=act_map,
        make_action=make_action,
        game_over_state=GameState.GAME_OVER,
        win_state=GameState.WIN,
    )

    return path


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
        print(f"  Actions: {sol}")
    else:
        print(f"L{args.level+1}: NO SOLUTION in {dt:.1f}s")


if __name__ == "__main__":
    main()
