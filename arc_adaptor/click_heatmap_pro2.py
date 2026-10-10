"""Per-colour click-heatmap proposer — one learned model replaces per-game click extractors.

Problem this solves (2026-09-29): every game needed a hand-tuned candidate extractor
(`vc33_click_candidates` purple-only, bp35 colour-14 diamonds, cd82 swatches, ...) and
each one bred blind-spot bugs — vc33's saturation false-negative was born from one.
This module instead learns, from cheap probe data (click cells, observe if the picture
changed), a heatmap over "cells whose click does something", and proposes candidates
for ANY game from a single frame. Layout inspiration: tonghuikang/arc3's per-colour
click head; implementation is ours and numpy-only (no torch at inference).

Model (v3): 17x64x64 input planes = 16 colour one-hot + 1 structural plane
       (log size of the cell's same-colour connected component, see structural_plane)
       -> 1x1 conv 17->24 + ReLU   (per-feature embedding)          Wa (24,17)
       -> 3x3 conv 24->24 + ReLU   (local receptive field)          Wb (24,24,3,3)
       -> 1x1 conv 24->16          (per-colour click logit)        Wc (16,24)
Weights: state/click_heatmap_v3.npz (trained by state/train_click_heatmap.py;
override with env CLICK_HEATMAP_WEIGHTS to score an ablation model).

Public API:
    propose_clicks(frame, topk=8, threshold=0.5)
        -> [{'action': 6, 'data': {'x': x, 'y': y}, 'score': s, 'color': c}, ...]
        one proposal per cell (deduped, Chebyshev radius 1) — v2 wasted top-k slots
        on the same region firing on several colour channels. Every proposal also
        satisfies colour identity: the cell's actual pixel colour equals the channel
        it was scored on, which is the quantity training supervises (v3 fix).
    click_candidates(frame, **kw)
        drop-in for the hand extractors (same dict shape they return).
CLI: python arc_adaptor/click_heatmap.py <game_id>   — proposals for a fresh entry frame.
"""
import os
import sys
from pathlib import Path

import numpy as np

_WEIGHTS = Path(os.environ.get('CLICK_HEATMAP_WEIGHTS',
                               str(Path(__file__).resolve().parents[1] / 'data' / 'aop' / 'click_heatmap_v6.npz')))
_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        with np.load(_WEIGHTS) as d:
            _MODEL = {k: d[k] for k in ('Wa', 'Wb', 'Wc')}
    return _MODEL


def onehot(frame):
    """(64,64) int grid -> (16,64,64) float32 colour planes."""
    g = np.clip(np.asarray(frame).astype(np.int64), 0, 15)
    return (np.arange(16)[:, None, None] == g[None]).astype(np.float32)


def structural_plane(frame):
    """log(size of the same-colour connected component) per cell, normalised to [0,1].

    The strongest transferable clickability cue: buttons/nodes are SMALL isolated
    regions, walls/background are huge ones — and unlike colour identity this
    transfers across games (v2 finding: colour one-hot alone does not transfer).
    """
    g = np.clip(np.asarray(frame).astype(np.int64), 0, 15)
    plane = np.zeros((64, 64), np.float32)
    norm = np.log1p(4096)
    for colour in range(16):
        mask = g == colour
        if not mask.any():
            continue
        for comp in _components(mask):
            v = np.float32(np.log1p(len(comp)) / norm)
            for y, x in comp:
                plane[y, x] = v
    return plane


def build_input(frame):
    """(17,64,64) model input: 16 colour planes + 1 structural plane."""
    return np.concatenate([onehot(frame), structural_plane(frame)[None]], 0)


def heatmap(frame):
    """per-colour click logit maps, (16,64,64) float32."""
    w = _load()
    in_ch = w['Wa'].shape[1]
    if in_ch == 17:
        x = build_input(frame)                            # colour planes + structural
    elif in_ch == 16:
        x = onehot(frame)                                 # ablation checkpoint, no plane
    else:
        raise ValueError(f'unexpected Wa input width: {in_ch}')
    h = np.einsum('oc,chw->ohw', w['Wa'], x)            # 1x1 conv
    h = np.maximum(h, 0.0)
    p = np.pad(h, ((0, 0), (1, 1), (1, 1)))             # zero-pad for the 3x3
    win = np.lib.stride_tricks.sliding_window_view(p, (3, 3), axis=(1, 2))   # (24,64,64,3,3)
    h = np.tensordot(w['Wb'], win, axes=([1, 2, 3], [0, 3, 4]))
    h = np.maximum(h, 0.0)
    return np.einsum('oc,chw->ohw', w['Wc'], h)         # 1x1 conv -> (16,64,64)


def _components(mask):
    seen = np.zeros_like(mask, bool)
    out = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], [(y, x)]
        seen[y, x] = True
        while stack:
            a, b = stack.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    na, nb = a + da, b + db
                    if 0 <= na < 64 and 0 <= nb < 64 and mask[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
                        cells.append((na, nb))
        out.append(cells)
    return out


def _identity_enforced():
    """Proposals must land on a cell whose actual colour equals the scored channel.

    Training supervises the logit of the clicked cell's OWN colour channel, so an
    off-identity firing is unsupervised noise: v3 scored (32,32) as top-1 in 5/5
    held-out games at p~1.00 while that cell's real colour was 5/4/2, and 4 of the
    5 games had no effective click there — a fixed 10% budget tax. Set
    CLICK_HEATMAP_IDENTITY=0 only to A/B-measure the effect of this check.
    """
    return os.environ.get('CLICK_HEATMAP_IDENTITY', '1') != '0'


def propose_from_prob(prob, g, topk=8, threshold=0.5, enforce=True):
    """Rank and dedupe click proposals from a (16,64,64) probability map.

    This is the whole deployment decision, split out from `propose_clicks` so the
    trainer's checkpoint metric scores a prob map through the SAME code the agent
    runs — a copy of this logic would eventually differ from inference and the
    selected numbers would be fiction.

    `g` is the clipped colour grid; `enforce` requires a proposal's cell to carry the
    colour of the channel that scored it (see _identity_enforced).
    """
    proposals = []
    for colour in range(16):
        fire = prob[colour] >= threshold
        if enforce:
            fire &= (g == colour)
        for cells in _components(fire):
            y, x = max(cells, key=lambda p: prob[colour, p[0], p[1]])
            proposals.append({'action': 6, 'data': {'x': int(x), 'y': int(y)},
                              'score': float(prob[colour, y, x]), 'color': colour,
                              'cells': len(cells)})
    proposals.sort(key=lambda p: -p['score'])
    kept_props = []
    for p in proposals:
        c = (p['data']['y'], p['data']['x'])
        if any(max(abs(c[0] - q['data']['y']), abs(c[1] - q['data']['x'])) <= 1
               for q in kept_props):
            continue
        kept_props.append(p)
        if len(kept_props) >= topk:
            break
    if not kept_props:                   # never starve the search: best cell of each colour
        own = np.take_along_axis(prob, g[None], axis=0)[0]
        flat = np.dstack(np.unravel_index(np.argsort(-own, axis=None), own.shape))[0]
        for y, x in flat[:topk]:
            kept_props.append({'action': 6, 'data': {'x': int(x), 'y': int(y)},
                               'score': float(own[y, x]),
                               'color': int(g[y, x]), 'cells': 1})
    return kept_props


def propose_clicks(frame, topk=8, threshold=0.5):
    """ranked generic click candidates for one frame (public ACTION6 dict shape).

    One proposal per firing region, scored on that region's own colour channel, and
    placed on the region's best cell (not its geometric centroid, which can fall on
    another colour). Dedup by cell: the same region lights up on several colour
    channels, so only the highest-scoring proposal per cell (Chebyshev radius 1)
    survives and the top-k budget is not wasted on duplicates."""
    g = np.clip(np.asarray(frame).astype(np.int64), 0, 15)
    prob = 1.0 / (1.0 + np.exp(-heatmap(frame)))
    return propose_from_prob(prob, g, topk=topk, threshold=threshold,
                             enforce=_identity_enforced())


def click_candidates(frame, **kw):
    """drop-in shape compatible with the hand extractors (e.g. vc33_click_candidates)."""
    return [{'action': p['action'], 'data': p['data']} for p in propose_clicks(frame, **kw)]


if __name__ == '__main__':
    game = sys.argv[1] if len(sys.argv) > 1 else 'vc33-5430563c'
    logging = __import__('logging')
    logging.disable(logging.WARNING)
    from arc_agi import Arcade, OperationMode

    env_dir = 'F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/environment_files'
    arc = Arcade(environments_dir=env_dir, logger=logging.getLogger('ch'),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(game, seed=0, save_recording=False)
    frame = np.asarray(env.reset().frame)[-1]
    for p in propose_clicks(frame, topk=10):
        print(f"cell=({p['data']['x']:2d},{p['data']['y']:2d}) colour={p['color']:2d} "
              f"score={p['score']:.2f} region={p['cells']}")
