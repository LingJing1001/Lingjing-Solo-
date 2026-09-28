"""L2: 2 toggles + avoid rot pad + goal entry."""
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

PAD = (49, 45)
GOAL_ADJ = (14, 35)


def game(env):
    g = env._game
    return g.bejndxqqzf(0), g.dhksvilbb[g.cklxociuu]


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
    for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
        resp = env.step(getattr(GameAction, act))
    g = s._as_grid(np.array(resp.frame[0]))
    for act in s._path_to(g, s._find_player(g), PAD):
        resp = env.step(getattr(GameAction, act))
    for _ in range(2):
        resp = env.step(GameAction.ACTION3)
        resp = env.step(GameAction.ACTION4)
    print("after rot", game(env))
    avoid = set(s._rot_pad_cells(s._as_grid(np.array(resp.frame[0]))))
    for _ in range(80):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        if p is None:
            resp = env.step(GameAction.ACTION1)
            continue
        if p == GOAL_ADJ:
            break
        path = s._path_to(g, p, GOAL_ADJ, avoid)
        resp = env.step(getattr(GameAction, path[0] if path else "ACTION3"))
    print("at adj", game(env), s._find_player(s._as_grid(np.array(resp.frame[0]))))
    for i in range(32):
        g = s._as_grid(np.array(resp.frame[0]))
        p = s._find_player(g)
        act = GameAction.ACTION2 if i % 8 == 0 else (GameAction.ACTION3 if i % 2 else GameAction.ACTION4)
        resp = env.step(act)
        p2 = s._find_player(s._as_grid(np.array(resp.frame[0])))
        lv = int(resp.levels_completed or 0)
        if p2 != p or lv > 1:
            print(f"  {i} {act.name}: {p}->{p2} L={lv} match={game(env)[0]}")
        if lv > 1:
            print("WIN")
            return
    print("fail")


if __name__ == "__main__":
    main()
