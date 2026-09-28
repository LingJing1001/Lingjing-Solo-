"""L2 win: 2 toggles (match=True) + phase wait at (14,35)."""
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


def game(env):
    g = env._game
    return g.bejndxqqzf(0), g.cklxociuu, g.dhksvilbb[g.cklxociuu]


def setup(env):
    g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
    for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
        resp = env.step(getattr(GameAction, act))
    g = s._as_grid(np.array(resp.frame[0]))
    for act in s._path_to(g, s._find_player(g), (49, 45)):
        resp = env.step(getattr(GameAction, act))
    for _ in range(2):
        resp = env.step(GameAction.ACTION3)
        resp = env.step(GameAction.ACTION4)
    for _ in range(80):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        if p is None:
            resp = env.step(GameAction.ACTION1)
            continue
        if p == (14, 35):
            break
        path = s._path_to(g, p, (14, 35)) or s._bfs(g, p, (14, 35))
        resp = env.step(getattr(GameAction, path[0] if path else "ACTION3"))
    return resp


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    resp = setup(env)
    m, idx, deg = game(env)
    print(f"at adj match={m} rot={deg}")
    for i in range(64):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        if i % 8 == 0:
            act = GameAction.ACTION2
        else:
            act = GameAction.ACTION3 if i % 2 else GameAction.ACTION4
        resp = env.step(act)
        g2 = s._as_grid(np.array(resp.frame[0]))
        p2 = s._find_player(g2)
        lv = int(resp.levels_completed or 0)
        if p2 != p or lv > 1:
            print(f"  {i} {act.name}: {p}->{p2} L={lv}")
        if lv > 1:
            print("WIN")
            return
    print("fail", s._find_player(s._as_grid(np.array(resp.frame[0]))))


if __name__ == "__main__":
    main()
