"""Debug L2 pad leave / soft-reset."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import lingjing_solo.planning.ls20_solver as s


def dump(env, resp, tag: str) -> None:
    g = env._game
    grid = s._as_grid(np.array(resp.frame[0])) if resp.frame else None
    p = s._find_player(grid) if grid is not None else None
    sc = g._step_counter_ui
    print(
        f"{tag}: p={p} pos=({g.gudziatsk.x},{g.gudziatsk.y}) "
        f"rot={g.dhksvilbb[g.cklxociuu]} m={g.bejndxqqzf(0)} "
        f"aqy={g.aqygnziho} ebf={g.ebfuxzbvn} ako={g.akoadfsur} "
        f"L={resp.levels_completed} sc={getattr(sc, 'osgviligwp', None)}"
    )


def main() -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
    for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
        resp = env.step(getattr(GameAction, act))
    for _ in range(4):
        resp = env.step(GameAction.ACTION1)
    dump(env, resp, "L2start")
    g = s._as_grid(np.array(resp.frame[0]))
    p = s._find_player(g)
    path = s._path_to(g, p, (49, 45))
    print("path_len", len(path))
    for i, act in enumerate(path):
        resp = env.step(getattr(GameAction, act))
        if i % 5 == 0 or i == len(path) - 1:
            dump(env, resp, f"nav{i}")
    for t in range(2):
        resp = env.step(GameAction.ACTION3)
        dump(env, resp, f"tog{t}a")
        resp = env.step(GameAction.ACTION4)
        dump(env, resp, f"tog{t}b")
    resp = env.step(GameAction.ACTION1)
    dump(env, resp, "leaveN")
    for i in range(6):
        resp = env.step(GameAction.ACTION1)
        dump(env, resp, f"wait{i}")
        if env._game.ebfuxzbvn <= 0 and env._game.akoadfsur <= 0:
            break


if __name__ == "__main__":
    main()
