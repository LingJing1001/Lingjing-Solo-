"""List all rot pad cells on L2."""
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

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ls20")
g = s._as_grid(np.array(env.step(GameAction.RESET).frame[0]))
for act in s._bfs(g, (34, 45), (19, 30)) + s._bfs(g, (19, 30), (34, 10)) + ["ACTION1"]:
    resp = env.step(getattr(GameAction, act))
g = s._as_grid(np.array(resp.frame[0]))
print("rot cells", s._rot_pad_cells(g))
print("shape", s._find_shape_pad(g, s._find_player(g)))
