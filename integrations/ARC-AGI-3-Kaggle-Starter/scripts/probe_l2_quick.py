"""Quick: test rotation counts vs goal entry on L2."""
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


def beat_l1(env):
    g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
    for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
        resp = env.step(getattr(GameAction, act))
    return resp


def rot_toggles(env, n: int):
    resp = beat_l1(env)
    g = s._as_grid(np.array(resp.frame[0]))
    pad = (49, 45)
    for act in s._path_to(g, s._find_player(g), pad):
        resp = env.step(getattr(GameAction, act))
    # n toggles = exit+enter pairs while on pad
    for _ in range(n):
        resp = env.step(GameAction.ACTION3)
        resp = env.step(GameAction.ACTION4)
    return resp


def try_goal(env):
    g = s._as_grid(np.array(resp.frame[0]))
    p = s._find_player(g)
    for act in s._path_to(g, p, (14, 35)):
        resp = env.step(getattr(GameAction, act))
    for i in range(16):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        resp = env.step(GameAction.ACTION2)
        g2 = s._as_grid(np.array(resp.frame[0]))
        p2 = s._find_player(g2)
        lv = int(resp.levels_completed or 0)
        if p2 != p or lv > 1:
            return i, p2, lv
    return -1, s._find_player(s._as_grid(np.array(resp.frame[0]))), 1


if __name__ == "__main__":
    for n in range(0, 6):
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make("ls20")
        resp = rot_toggles(env, n)
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        for act in s._path_to(g, p, (14, 35)):
            resp = env.step(getattr(GameAction, act))
        ok = False
        for i in range(16):
            g = s._as_grid(np.array(resp.frame[0]))
            p = s._find_player(g)
            resp2 = env.step(GameAction.ACTION2)
            g2 = s._as_grid(np.array(resp2.frame[0]))
            p2 = s._find_player(g2)
            lv = int(resp2.levels_completed or 0)
            if lv > 1:
                ok = True
                print(f"toggles={n} WIN at try {i} p={p2}")
                break
            if p2 != p:
                print(f"toggles={n} moved at {i}: {p}->{p2} L={lv}")
                break
        if not ok:
            print(f"toggles={n} stuck at {s._find_player(s._as_grid(np.array(resp.frame[0])))}")
