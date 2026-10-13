import os, sys
from pathlib import Path
KAGGLE = Path('integrations/ARC-AGI-3-Kaggle-Starter')
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault('LINGJING_SRC', str(Path('lingjing_solo')))
sys.path.insert(0, str(Path('lingjing_solo')))
sys.path.insert(0, str(KAGGLE))
VENDOR = KAGGLE / 'vendor' / 'ARC-AGI-3-Agents'
if VENDOR.exists(): sys.path.insert(0, str(VENDOR))
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make('cn04')
frame = env.reset()
g = env._game

plan = [
    {'a': 6, 'x': 24, 'y': 36},
    4, 4, 4, 4,
    {'a': 6, 'x': 45, 'y': 15},
    2, 2, 2, 2, 2, 2, 2, 2,
    {'a': 6, 'x': 12, 'y': 12},
    4, 4, 4, 4, 2, 2, 2, 2, 2, 2,
    {'a': 6, 'x': 57, 'y': 51},
    5, 5, 5, 1, 1,
]

print('L1 initial:')
print('  selected:', g.xseexqzst.name if g.xseexqzst else None)

for i, action in enumerate(plan):
    if isinstance(action, dict):
        frame = env.step(GameAction.ACTION6, data={'x': action['x'], 'y': action['y']})
        print('  step %d: click (%d,%d) -> selected=%s' % (i+1, action['x'], action['y'], g.xseexqzst.name if g.xseexqzst else None))
    else:
        frame = env.step(GameAction.from_id(action), data={})
        print('  step %d: ACTION%d -> selected=%s' % (i+1, action, g.xseexqzst.name if g.xseexqzst else None))

    if g.sjwqloivve():
        print('  WIN at step %d!' % (i+1))
        break

print()
print('Final:')
print('  selected:', g.xseexqzst.name if g.xseexqzst else None)
print('  win:', g.sjwqloivve())
print('  levels:', frame.levels_completed)
