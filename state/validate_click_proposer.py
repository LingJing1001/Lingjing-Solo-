"""Validate the click-heatmap proposer on games it never saw (held-out).

Checkpoint identity is printed first: v3 ships two weight files whose numbers are
not interchangeable (version / in_channels live inside the npz).

Metrics per held-out game (entry state, probe labels = ground truth):
  precision@k   of the proposer's top-k cells, fraction within distance 2 of a
                known-effective click
  recall@k      of known-effective clicks, fraction covered by the top-k proposals
Games whose entry state has zero effective clicks (tu93-class keyboard levels) are
listed but EXCLUDED from the pooled means: with nothing reachable, their precision
and recall are undefined, and pooling used to turn that undefined value into a fake
0.00 that dragged every v3 number down.

The identity A/B block re-scores each frame with the colour-identity check disabled
and counts proposals that would fire on a cell whose actual colour differs from the
channel scored there -- the v3 (32,32) defect -- and reports held-out precision both
ways so the fix's effect is attributable rather than asserted.

Usage: python state/validate_click_proposer.py
       env CLICK_HEATMAP_WEIGHTS=<npz> scores the ablation checkpoint instead.
"""
import os
import sys

import numpy as np

sys.path.insert(0, 'F:/pro2')
from arc_adaptor import click_heatmap as CH      # default: click_heatmap_v3.npz

D = np.load('F:/pro2/state/click_labels.npz', allow_pickle=False)
GAMES = [g for g in D['games']]
HELD = ['r11l-495a7899', 's5i5-18d95033', 'sb26-7fbdac44', 'tu93-0768757b', 'ft09-0d8bbf25']
TOPK = 10
DIST = 2
K = os.environ.get('CLICK_HEATMAP_WEIGHTS', 'state/click_heatmap_v3.npz')


def hits(cell, effective, dist=DIST):
    return any(abs(cell[0] - e[0]) <= dist and abs(cell[1] - e[1]) <= dist
               for e in effective)


def score_run(frames, enforce):
    """Score every frame once under the requested colour-identity setting."""
    os.environ['CLICK_HEATMAP_IDENTITY'] = '1' if enforce else '0'
    try:
        return [CH.propose_clicks(f, topk=TOPK) for f in frames]
    finally:
        os.environ['CLICK_HEATMAP_IDENTITY'] = '1'


def cell_of(p):
    return (p['data']['y'], p['data']['x'])


def min_chebyshev(props):
    cells = [cell_of(p) for p in props]
    pairs = [(a, b) for i, a in enumerate(cells) for b in cells[i + 1:]]
    if not pairs:
        return float('inf')
    return min(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in pairs)


with np.load(K, allow_pickle=True) as w:
    ver = str(w['version'][0]) if 'version' in w.files else '?'
    inch = int(w['in_channels'][0]) if 'in_channels' in w.files else w['Wa'].shape[1]
print(f'=== checkpoint: {K}  version={ver}  in_channels={inch}  '
      f'identity_check={"on" if CH._identity_enforced() else "off"} ===')

cases = []
for game in HELD:
    gi = GAMES.index(game)
    m = D['game_idx'] == gi
    frame = D['frames'][m][0]
    eff = sorted(zip(D['py'][m][D['eff'][m] == 1].tolist(),
                     D['px'][m][D['eff'][m] == 1].tolist()))
    cases.append((game, frame, eff))

on_runs = score_run([c[1] for c in cases], True)
off_runs = score_run([c[1] for c in cases], False)

print()
print('=== held-out games: proposer vs probe ground truth (entry state) ===')
pool = {'click_games': 0, 'props': 0, 'hits': 0, 'eff': 0, 'covered': 0,
        'off_props': 0, 'off_hits': 0, 'offgrid': 0}
g_all = np.clip(np.stack([c[1] for c in cases]).astype(np.int64), 0, 15)
for idx, (game, frame, eff) in enumerate(cases):
    props, props_off = on_runs[idx], off_runs[idx]
    cells = [cell_of(p) for p in props]
    off_cells = [cell_of(p) for p in props_off]
    n_hit = sum(hits(c, eff) for c in cells)
    covered = sum(any(hits(e, [c]) for c in cells) for e in eff)
    n_hit_off = sum(hits(c, eff) for c in off_cells)
    offgrid = sum(1 for p in props_off
                  if int(g_all[idx][cell_of(p)]) != int(p['color']))
    zero_click = len(eff) == 0
    tag = 'excluded(zero-click)' if zero_click else 'pooled'
    prec = n_hit / max(len(cells), 1)
    rec = covered / max(len(eff), 1)
    prec_off = n_hit_off / max(len(off_cells), 1)
    print(f'{game}: effective={len(eff):3d}  precision@{TOPK}={prec:.2f}  '
          f'recall@{TOPK}={rec:.2f}  [{tag}]')
    print(f'    unique_cells={len(set(cells))}/{len(cells)}  '
          f'min_chebyshev={min_chebyshev(props)}  '
          f'identity_off={offgrid}/{len(props_off)}  '
          f'precision@{TOPK} w/o identity check={prec_off:.2f}')
    if zero_click:
        continue
    pool['click_games'] += 1
    pool['props'] += len(cells)
    pool['hits'] += n_hit
    pool['eff'] += len(eff)
    pool['covered'] += covered
    pool['off_props'] += len(off_cells)
    pool['off_hits'] += n_hit_off
    pool['offgrid'] += offgrid

print()
if pool['props']:
    print(f"=== pooled over {pool['click_games']} click games "
          f"(zero-click games dropped, {TOPK} proposals each) ===")
    print(f"precision@{TOPK} = {pool['hits']}/{pool['props']} = "
          f"{pool['hits'] / pool['props']:.3f}")
    print(f"recall@{TOPK}    = {pool['covered']}/{pool['eff']} = "
          f"{pool['covered'] / max(pool['eff'], 1):.3f}")
    print(f"identity check OFF would give precision@{TOPK} = "
          f"{pool['off_hits']}/{pool['off_props']} = "
          f"{pool['off_hits'] / pool['off_props']:.3f}  "
          f"(off-colour proposals: {pool['offgrid']})")
print('note: hits counted within distance <=%d cells; single-game numbers at '
      'n=%d proposals are noise-level, compare absolute hits.' % (DIST, pool['props']))

print()
print('=== held-out detail: proposals vs ground truth ===')
for game, frame, eff in cases:
    props = score_run([frame], True)[0]
    print(f'-- {game} (effective={len(eff)}) --')
    for p in props:
        c = cell_of(p)
        mark = '*' if hits(c, eff) else ' '
        print(f'   {mark} cell=({c[1]:2d},{c[0]:2d}) colour={p["color"]:2d} '
              f'actual={int(np.clip(frame, 0, 15)[c]):2d} score={p["score"]:.2f}')

print()
print('=== hand extractor comparison (vc33 entry, purple-only extractor) ===')
sys.path.insert(0, 'F:/pro2/arc_adaptor')
import r2r3r4_combo as combo                       # noqa: E402

gi = GAMES.index('vc33-5430563c')
m = D['game_idx'] == gi
frame = D['frames'][m][0]
eff_cells = sorted(zip(D['py'][m][D['eff'][m] == 1].tolist(),
                       D['px'][m][D['eff'][m] == 1].tolist()))
hand = combo.vc33_click_candidates(frame)
hand_cells = [(h['data']['y'], h['data']['x']) for h in hand]
hand_hits = sum(hits(c, eff_cells) for c in hand_cells)
print(f'hand   : {hand_hits}/{len(hand_cells)} effective at precision={hand_hits / max(len(hand_cells), 1):.2f}')
for h in hand:
    c = (h['data']['y'], h['data']['x'])
    print(f"  hand cell=({h['data']['x']:2d},{h['data']['y']:2d}) "
          f"effective={hits(c, eff_cells)}")
learned = score_run([frame], True)[0]
learn_cells = [cell_of(p) for p in learned]
learn_hits = sum(hits(c, eff_cells) for c in learn_cells)
print(f'learned: {learn_hits}/{len(learn_cells)} effective at '
      f'precision={learn_hits / max(len(learn_cells), 1):.2f}')
print(f'promotion bar: precision@8 >= 0.5 AND not behind the hand extractor; '
      f'measured here at k={TOPK} for both sides.')
