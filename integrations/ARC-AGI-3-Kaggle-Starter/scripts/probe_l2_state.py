"""Inspect L2 internal rotation state after pad toggles."""
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


def game_state(env):
    g = env._game if hasattr(env, "_game") else env.game
    rot = g.dhksvilbb[g.cklxociuu]
    return {
        "rot_idx": g.cklxociuu,
        "rot_deg": rot,
        "shape": g.fwckfzsyc,
        "color": g.hiaauhahz,
        "goal_rot": g.ehwheiwsk,
        "goal_shape": g.ldxlnycps,
        "goal_color": g.yjdexjsoa,
        "match": g.bejndxqqzf(0) if g.plrpelhym else None,
    }


def main():
    for n in range(4):
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make("ls20")
        resp = beat_l1(env)
        g = s._as_grid(np.array(resp.frame[0]))
        pad = (49, 45)
        for act in s._path_to(g, s._find_player(g), pad):
            resp = env.step(getattr(GameAction, act))
        st = game_state(env)
        print(f"after arrive: {st}")
        for t in range(n):
            resp = env.step(GameAction.ACTION3)
            resp = env.step(GameAction.ACTION4)
            print(f"  toggle {t+1}: {game_state(env)}")


if __name__ == "__main__":
    main()
