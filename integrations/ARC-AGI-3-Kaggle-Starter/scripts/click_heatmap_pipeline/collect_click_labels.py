"""Collect click-effect labels across all 25 practice games (entry state).

For every game: reset, enumerate generic probe cells (centroid of every connected
region of every non-background colour + an 8px coarse grid), reset-and-click each
probe, record whether the picture actually changed (>=5 cells, so 1-cell budget
ticks never count as an effect).

Output: state/click_labels.npz with frames/px/py/eff/game_idx + game id list.
This is the training set for the per-colour click-heatmap proposer.
"""
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, 'F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages')
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

ENV_DIR = 'F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/environment_files'
OUT = Path('F:/pro2/state/click_labels.npz')
PROGRESS = Path('F:/pro2/state/click_labels_progress.json')
CHANGED_MIN = 5
MAX_PROBES_PER_GAME = 140

lg = logging.getLogger('collect')
lg.setLevel(logging.ERROR)
arc = Arcade(environments_dir=ENV_DIR, logger=lg, operation_mode=OperationMode.OFFLINE)
GAMES = sorted(
    f'{d.name}-{h.name}'
    for d in Path(ENV_DIR).iterdir() if d.is_dir()
    for h in d.iterdir() if h.is_dir() and (h / 'metadata.json').exists()
)


def grid_of(frame):
    return np.asarray(frame.frame)[-1]


def regions(mask):
    """connected components of a boolean mask -> list of cell lists."""
    seen = np.zeros_like(mask, bool)
    out = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    na, nb = a + da, b + db
                    if 0 <= na < 64 and 0 <= nb < 64 and mask[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
        out.append(cells)
    return out


def probe_cells(g):
    """generic candidate probe cells: region centroids of every non-dominant colour + 8px grid."""
    bg = Counter(g.ravel().tolist()).most_common(1)[0][0]
    cells = set()
    for colour in range(16):
        if colour == bg:
            continue
        for comp in regions(g == colour):
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            cells.add((int(np.mean(xs)), int(np.mean(ys))))
            if len(comp) > 4:                      # big regions: also sample the middle
                cells.add((min(int(round(np.mean(xs))) + 1, 63),
                       min(int(round(np.mean(ys))) + 1, 63)))
    for y in range(2, 64, 8):
        for x in range(2, 64, 8):
            cells.add((x, y))
    return sorted(cells)[:MAX_PROBES_PER_GAME]


frames, pxs, pys, effs, gidx, nch = [], [], [], [], [], []
summary = {}
t0 = time.time()
for gi, game in enumerate(GAMES):
    try:
        env = arc.make(game, seed=0, save_recording=False)
        fr = env.reset()
        base = grid_of(fr).copy()
        cells = probe_cells(base)
        eff_n = 0
        for (x, y) in cells:
            env = arc.make(game, seed=0, save_recording=False)
            fr = env.reset()
            before = grid_of(fr).copy()
            fr2 = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
            if fr2 is None or fr2.frame is None or len(fr2.frame) == 0:
                continue
            after = np.asarray(fr2.frame)[-1]
            diff = (after != before)
            diff[0, :] = False                     # 预算条行每动作必跳,不算效果
            diff[63, :] = False
            n_changed = int(diff.sum())
            eff = int(n_changed >= 3)              # 1-2 格=弱标签,训练时剔除
            frames.append(before)
            pxs.append(x)
            pys.append(y)
            effs.append(eff)
            nch.append(n_changed)
            gidx.append(gi)
            eff_n += eff
        summary[game] = {'probes': len(cells), 'effective': int(eff_n)}
        print(f'[{gi + 1}/{len(GAMES)}] {game}: {len(cells)} probes, {eff_n} effective '
              f'({time.time() - t0:.0f}s)', flush=True)
    except Exception as exc:                     # noqa: BLE001 — keep collecting other games
        summary[game] = {'error': str(exc)[:120]}
        print(f'[{gi + 1}/{len(GAMES)}] {game}: ERROR {exc}', flush=True)
    PROGRESS.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')

np.savez_compressed(
    OUT,
    frames=np.asarray(frames, dtype=np.int8),
    px=np.asarray(pxs, dtype=np.int8),
    py=np.asarray(pys, dtype=np.int8),
    eff=np.asarray(effs, dtype=np.int8),
    n_changed=np.asarray(nch, dtype=np.int16),
    game_idx=np.asarray(gidx, dtype=np.int16),
    games=np.asarray(GAMES),
)
print('DONE', json.dumps({'total': len(frames), 'games': len(GAMES),
                          'seconds': round(time.time() - t0, 1)}))
