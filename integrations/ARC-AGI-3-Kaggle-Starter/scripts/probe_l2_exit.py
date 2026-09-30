"""L2: rot + exit pad + goal."""
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

def gs(env, resp):
    g = env._game
    return s._find_player(s._as_grid(np.array(resp.frame[0]))), g.bejndxqqzf(0), g.dhksvilbb[g.cklxociuu]

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
print("on pad", gs(env, resp))
g = s._as_grid(np.array(resp.frame[0]))
ex = s._pad_exit_only(g, s._find_player(g), (49, 45), avoid_rot=True)
print("exit act", ex)
resp = env.step(getattr(GameAction, ex))
print("after exit", gs(env, resp))
avoid = set(s._rot_pad_cells(s._as_grid(np.array(resp.frame[0]))))
g = s._as_grid(np.array(resp.frame[0]))
for act in s._path_to(g, s._find_player(g), (14, 35), avoid):
    resp = env.step(getattr(GameAction, act))
print("at adj", gs(env, resp))
for i in range(32):
    resp = env.step(GameAction.ACTION2)
    p, m, r = gs(env, resp)
    if int(resp.levels_completed or 0) > 1:
        print("WIN", i)
        break
    if i < 5 or i % 8 == 0:
        print(f"  {i}: p={p} match={m} rot={r}")
