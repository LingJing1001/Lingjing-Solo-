"""Test L2 rotation count vs goal entry."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState
import lingjing_solo.planning.ls20_solver as s


def beat_l1(env):
    g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
    for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
        resp = env.step(getattr(GameAction, act))
    return resp


def pad_entries(env, n_entries: int):
    """n_entries total entries onto rot pad at (49,45)."""
    resp = beat_l1(env)
    g = s._as_grid(np.array(resp.frame[0]))
    p = s._find_player(g)
    pad = (49, 45)
    for act in s._bfs(g, p, pad):
        resp = env.step(getattr(GameAction, act))
    entries = 1  # arrival
    while entries < n_entries:
        resp = env.step(getattr(GameAction, "ACTION3"))  # to (44,45)
        resp = env.step(getattr(GameAction, "ACTION4"))  # re-enter
        entries += 1
    return resp


def main():
    for n in range(1, 6):
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make("ls20")
        resp = pad_entries(env, n)
        g = s._as_grid(np.array(resp.frame[0]))
        print(f"entries={n} at_pad={s._find_player(g)}")
        p = s._find_player(g)
        for act in s._bfs(g, p, (14, 35)):
            resp = env.step(getattr(GameAction, act))
        for i in range(12):
            g = s._as_grid(np.array(resp.frame[0]))
            p = s._find_player(g)
            resp = env.step(GameAction.ACTION2)
            g2 = s._as_grid(np.array(resp.frame[0]))
            p2 = s._find_player(g2)
            lv = int(resp.levels_completed or 0)
            if p2 != p or lv > 1:
                print(f"  step {i}: {p} -> {p2} L={lv}")
            if lv > 1:
                print("  WIN")
                break
        else:
            print(f"  fail at {s._find_player(s._as_grid(np.array(resp.frame[0])))}")


if __name__ == "__main__":
    main()
