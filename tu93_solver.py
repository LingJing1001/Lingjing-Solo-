# -*- coding: utf-8 -*-
"""tu93 黑盒 BFS 求解器 — 快照优化版。

不猜内部机制，只用 env.step() + generic_snapshot/generic_restore。
每步保存快照，回溯时恢复，避免全路径重放。
"""
import sys, os, time
from pathlib import Path
from collections import deque

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

from lingjing_solo.planning.search.generic_shadow import (
    generic_snapshot, generic_restore, generic_state_key, _level_index
)


def make_env(gid="tu93"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)


def get_player_pos(g):
    for s in g.current_level.get_sprites():
        if "0017" in str(s.tags):
            return (s.x, s.y)
    return None


def solve_level(env, level, max_nodes=50000, time_budget=120):
    """快照优化 BFS。"""
    frame = env.reset()
    g = env._game
    for _ in range(level):
        g.next_level()

    level_idx0 = _level_index(g)
    start_snap = generic_snapshot(g)
    start_pos = get_player_pos(g)

    # BFS with snapshots — 用全状态键去重（包含敌人位置）
    start_key = generic_state_key(g)
    queue = deque([(start_key, [], start_snap)])
    visited = {start_key}
    nodes = 0
    t0 = time.time()

    while queue:
        if time.time() - t0 > time_budget or nodes > max_nodes:
            return None

        key, path, snap = queue.popleft()

        for aid in [1, 2, 3, 4]:
            if time.time() - t0 > time_budget:
                return None
            generic_restore(g, snap)
            env.step(GameAction.from_id(aid), data={})

            if _level_index(g) > level_idx0:
                return path + [aid]

            new_key = generic_state_key(g)
            if new_key and new_key not in visited:
                visited.add(new_key)
                new_snap = generic_snapshot(g)
                queue.append((new_key, path + [aid], new_snap))
                nodes += 1

        if nodes % 500 == 0 and nodes > 0:
            elapsed = time.time() - t0
            print(f"  BFS: {nodes} nodes, {len(visited)} states, {len(queue)} queue, {elapsed:.1f}s", flush=True)

    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("level", type=int, default=0)
    p.add_argument("--nodes", type=int, default=50000)
    p.add_argument("--time", type=int, default=120)
    args = p.parse_args()

    env = make_env("tu93")
    t0 = time.time()
    sol = solve_level(env, args.level, max_nodes=args.nodes, time_budget=args.time)
    dt = time.time() - t0

    if sol:
        print(f"L{args.level+1}: SOLVED {len(sol)} actions in {dt:.1f}s")
        # 验证
        env.reset()
        g = env._game
        for _ in range(args.level):
            g.next_level()
        for a in sol:
            env.step(GameAction.from_id(a), data={})
            if _level_index(g) > args.level:
                break
        ok = _level_index(g) > args.level
        print(f"  Verify: {'OK' if ok else 'FAIL'}")
    else:
        print(f"L{args.level+1}: NO SOLUTION in {dt:.1f}s")


if __name__ == "__main__":
    main()
