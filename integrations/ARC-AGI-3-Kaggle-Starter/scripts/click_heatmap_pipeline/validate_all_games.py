"""Full 25-game validation of the click proposer (entry state, probe ground truth).

Per game: precision@8 / recall@8 of the proposer's top-8 cells vs known-effective
probe clicks (distance <=2). Games with zero effective clicks are reported and
excluded from the precision aggregate (they are keyboard/no-click games).
Also dumps the top-8 proposals for r11l to diagnose any regression.
"""
import sys

import numpy as np

sys.path.insert(0, 'F:/pro2')
from arc_adaptor import click_heatmap as CH

D = np.load('F:/pro2/state/click_labels.npz', allow_pickle=False)
GAMES = [g for g in D['games']]


def hits(cell, effective, dist=2):
    return any(abs(cell[0] - e[0]) <= dist and abs(cell[1] - e[1]) <= dist
               for e in effective)


rows = []
for gi, game in enumerate(GAMES):
    m = D['game_idx'] == gi
    idx = np.where(m)[0]
    frame = D['frames'][idx[0]]
    eff_mask = D['eff'][idx] == 1
    eff_cells = sorted(zip(D['py'][idx][eff_mask].tolist(),
                           D['px'][idx][eff_mask].tolist()))
    props = CH.propose_clicks(frame, topk=8)
    cells = [(p['data']['y'], p['data']['x']) for p in props]
    n_hit = sum(hits(c, eff_cells) for c in cells)
    covered = sum(any(hits(e, [c]) for c in cells) for e in eff_cells)
    rows.append((game, len(eff_cells), n_hit / 8.0,
                 covered / max(len(eff_cells), 1)))

print('%-16s %-8s %-12s %-8s' % ('game', 'effective', 'precision@8', 'recall@8'))
sm = hg = zg = 0
for game, ne, prec, rec in rows:
    flag = ''
    if ne == 0:
        zg += 1
        flag = ' (no effective clicks)'
    else:
        sm += 1
        if prec >= 0.5:
            hg += 1
            flag = '  <-- >=0.5'
    print('%-16s %-8d %-12.2f %-8.2f%s' % (game, ne, prec, rec, flag))

if sm:
    print(f'\nwith-effective games={sm}  zero-click games={zg}  '
          f'precision@8>=0.5 games={hg} ({hg / sm:.0%})')
    print('mean precision@8 = %.3f' % np.mean([r[2] for r in rows if r[1] > 0]))

print('\n=== r11l diagnostic (top-8 proposals vs 63 effective) ===')
gi = GAMES.index('r11l-495a7899')
m = D['game_idx'] == gi
idx = np.where(m)[0]
frame = D['frames'][idx[0]]
eff_cells = sorted(zip(D['py'][idx][D['eff'][idx] == 1].tolist(),
                       D['px'][idx][D['eff'][idx] == 1].tolist()))
for p in CH.propose_clicks(frame, topk=8):
    c = (p['data']['y'], p['data']['x'])
    print(f"  cell=({c[1]:2d},{c[0]:2d}) colour={p['color']:2d} score={p['score']:.2f} "
          f"near_effective={hits(c, eff_cells)}")