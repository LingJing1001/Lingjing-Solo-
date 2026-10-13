import os, sys
from pathlib import Path
KAGGLE = Path('integrations/ARC-AGI-3-Kaggle-Starter')
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault('LINGJING_SRC', str(Path('lingjing_solo')))
sys.path.insert(0, str(Path('lingjing_solo')))
sys.path.insert(0, str(KAGGLE))
VENDOR = KAGGLE / 'vendor' / 'ARC-AGI-3-Agents'
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('cn04')

sys.path.insert(0, str(Path('lingjing_solo/planning')))
import arc_shadow_cn04 as cn04_mod

# 也导入 cn04_direct
sys.path.insert(0, str(Path('.')))
import cn04_direct as cn04_dir

# 也导入 cn04_direct
sys.path.insert(0, str(Path('.')))
import cn04_direct as cn04_dir

for lvl in range(6):
    env.reset()
    g = env._game
    cn04_mod._game = g
    cn04_mod._env = env

    sol = cn04_dir.solve_level_direct(g, 30)
    if sol:
        env.reset()
        for a in sol:
            if isinstance(a, tuple) and len(a) == 3 and a[0] == 'click':
                frame = env.step(GameAction.ACTION6, data={'x': a[1], 'y': a[2]})
            elif isinstance(a, tuple) and len(a) == 1 and a[0] == 'rotate':
                frame = env.step(GameAction.ACTION5, data={})
            elif isinstance(a, tuple) and len(a) == 2 and a[0] == 'move':
                frame = env.step(GameAction.from_id(a[1]), data={})
            else:
                print('Unknown action:', a)
                break
            if frame.levels_completed > lvl or frame.state.name == 'WIN':
                break
        ok = frame.levels_completed > lvl or frame.state.name == 'WIN'
        status = 'OK' if ok else 'FAIL'
        print('L%d: direct %d steps -> levels=%d %s' % (lvl+1, len(sol), frame.levels_completed, status))
    else:
        print('L%d: direct no solution' % (lvl+1))
