"""Trace rotation after each step from pad to goal."""
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


def state(env, resp):
    gg = env._game
    p = s._find_player(s._as_grid(np.array(resp.frame[0])))
    return p, gg.bejndxqqzf(0), gg.dhksvilbb[gg.cklxociuu]


arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ls20")
g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
    resp = env.step(getattr(GameAction, act))
g = s._as_grid(np.array(resp.frame[0]))
for act in s._path_to(g, s._find_player(g), (49, 45)):
    resp = env.step(getattr(GameAction, act))
for _ in range(2):
    resp = env.step(GameAction.ACTION3)
    resp = env.step(GameAction.ACTION4)
print("start", state(env, resp))
avoid = set(s._rot_pad_cells(s._as_grid(np.array(resp.frame[0]))))
g = s._as_grid(np.array(resp.frame[0]))
path = s._path_to(g, s._find_player(g), (14, 35), avoid)
print("path len", len(path), "first", path[:8])
for i, act in enumerate(path):
    resp = env.step(getattr(GameAction, act))
    p, m, rot = state(env, resp)
    if not m or i < 12 or i > len(path) - 4:
        print(f"  {i:2d} {act} p={p} match={m} rot={rot}")
