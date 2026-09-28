"""Probe L2 moving platform phase at goal (14,40)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import lingjing_solo.planning.ls20_solver as s


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
    for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
        resp = env.step(getattr(GameAction, act))
    g = s._as_grid(np.array(resp.frame[0]))
    for act in s._bfs(g, s._find_player(g), (49, 45)) + ["ACTION3", "ACTION4", "ACTION3", "ACTION4"]:
        resp = env.step(getattr(GameAction, act))
    goal = (14, 40)
    for _ in range(50):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        if p is None:
            resp = env.step(GameAction.ACTION1)
            continue
        if p == (14, 35):
            break
        path = s._bfs(g, p, (14, 35))
        resp = env.step(getattr(GameAction, path[0] if path else "ACTION1"))

    print("at", s._find_player(s._as_grid(np.array(resp.frame[0]))))
    for i in range(32):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        gx, gy = goal
        block = g[gy : gy + 5, gx : gx + 5]
        wall14 = int(np.sum(block == 14))
        walk = s._walkable(g, gx, gy)
        resp = env.step(GameAction.ACTION2)
        g2 = s._as_grid(np.array(resp.frame[0]))
        p2 = s._find_player(g2)
        moved = p2 != p if p2 and p else False
        lv = int(resp.levels_completed or 0)
        print(f"{i:2d} walk={walk} wall14={wall14} p={p2} moved={moved} L={lv}")
        if lv > 1:
            print("WIN")
            break


if __name__ == "__main__":
    main()
