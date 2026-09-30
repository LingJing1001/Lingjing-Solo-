"""Mine route clicks as guaranteed-positive training samples.

Replays the three validated routes (tests/routes/*.json) and captures the frame
before every click together with the clicked cell — a click that is part of a
route which clears its level is effective by construction. Output merges into
the same npz schema as collect_click_labels.py (n_changed=999 marks route-mined).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages')
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

ROUTES = sorted(Path('F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/tests/routes').glob('*.json'))
OUT = Path('F:/pro2/state/route_clicks.npz')
lg = logging.getLogger('mine')
lg.setLevel(logging.ERROR)

arc = Arcade(environments_dir='F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/environment_files',
             logger=lg, operation_mode=OperationMode.OFFLINE)


def grid_of(frame):
    return np.asarray(frame.frame)[-1]


frames, pxs, pys, gnames = [], [], [], []
for rp in ROUTES:
    route = json.loads(rp.read_text(encoding='utf-8'))
    env = arc.make(route['game_id'], seed=0, save_recording=False)
    fr = env.reset()
    n_captured = 0
    for spec in route['actions']:
        if spec['a'] == 'ACTION6':
            before = grid_of(fr).copy()
            fr = env.step(GameAction.ACTION6, data={'x': spec['x'], 'y': spec['y']})
            frames.append(before)
            pxs.append(spec['x'])
            pys.append(spec['y'])
            gnames.append(route['game_id'])
            n_captured += 1
        else:
            fr = env.step(getattr(GameAction, spec['a']))
        if fr.state.name != 'NOT_FINISHED':
            break
    print(f"{route['game_id']}: {n_captured} route clicks mined", flush=True)

np.savez_compressed(
    OUT,
    frames=np.asarray(frames, dtype=np.int8),
    px=np.asarray(pxs, dtype=np.int8),
    py=np.asarray(pys, dtype=np.int8),
    games=np.asarray(gnames),
)
print('DONE', len(frames), 'route-click samples')
