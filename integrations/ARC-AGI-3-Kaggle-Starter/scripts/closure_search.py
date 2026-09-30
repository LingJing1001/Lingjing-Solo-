"""su15: proposer-driven closure search — first full test of the
proposer -> search -> level-clear loop on a game with NO hand-written route.

Per level: BFS over click paths; candidates = learned proposer top-6 per state;
dedup by picture hash; goal = levels_completed increases. Honest stop reasons:
time / nodes / frontier_exhausted (only the last one means "no solution at this
depth cap"). After each level-up the search restarts from the new entry state.
"""
import logging
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, 'F:/pro2')
from arc_adaptor import click_heatmap as CH

sys.path.insert(0, 'F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/.venv/Lib/site-packages')
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

GAME = sys.argv[1] if len(sys.argv) > 1 else 'su15-1944f8ab'
NUM_LEVELS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
ENV_DIR = 'F:/pro2/integrations/ARC-AGI-3-Kaggle-Starter/environment_files'
PER_LEVEL_SECONDS = 1500
MAX_NODES_PER_LEVEL = 30000
MAX_DEPTH = 45
TOTAL_SECONDS = 5400

lg = logging.getLogger('su15')
lg.setLevel(logging.ERROR)
arc = Arcade(environments_dir=ENV_DIR, logger=lg, operation_mode=OperationMode.OFFLINE)


def reach(path):
    """fresh env, replay mixed path; steps are ('click',(x,y)) or ('ACTION5',)."""
    env = arc.make(GAME, seed=0, save_recording=False)
    fr = env.reset()
    for step in path:
        if step[0] == 'ACTION5':
            fr = env.step(GameAction.ACTION5)
        else:
            _, (x, y) = step
            fr = env.step(GameAction.ACTION6, data={'x': int(x), 'y': int(y)})
        if fr is None or fr.frame is None or len(fr.frame) == 0:
            return None
        if fr.state.name in ('WIN', 'GAME_OVER'):
            break
    return fr


def pic(fr):
    return np.asarray(fr.frame)[-1].tobytes()


def generic_candidates(g):
    """structural fallback: centroid of every non-dominant-colour region (the
    collector's probe set — covers what the learned proposer may rank low)."""
    g = np.clip(np.asarray(g).astype(np.int64), 0, 15)
    bg = np.bincount(g.ravel()).argmax()
    cells = []
    for colour in range(16):
        if colour == bg:
            continue
        for comp in CH._components(g == colour):
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            cells.append((int(np.mean(xs)), int(np.mean(ys))))
    return cells


def union_candidates(g, topk=6):
    """P4 union mode: learned proposer + structural region centroids."""
    out, seen = [], set()
    for p in CH.propose_clicks(g, topk=topk):
        c = (p['data']['x'], p['data']['y'])
        if c not in seen:
            seen.add(c); out.append(c)
    for c in generic_candidates(g):
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def search_level(base_path, base_level, t_deadline):
    """continuation BFS: root = state after base_path (previous levels' solution),
    children extend base_path; goal = one more level-up. Honest stop reasons."""
    root_fr = reach(base_path)
    if root_fr is None:
        return None, 'dead_prefix', (0, 0)
    root_pic = pic(root_fr)
    frontier = deque([()])
    seen = {root_pic}
    nodes = expanded = 0
    base_len = len(base_path)
    while frontier:
        if time.time() > t_deadline:
            return None, 'time', (expanded, len(seen))
        if nodes > MAX_NODES_PER_LEVEL:
            return None, 'nodes', (expanded, len(seen))
        rel = frontier.popleft()
        path = base_path + list(rel)
        fr = reach(path)
        if fr is None:
            continue
        if fr.state.name in ('WIN', 'GAME_OVER'):
            continue
        g = np.asarray(fr.frame)[-1]
        expanded += 1
        if len(rel) >= MAX_DEPTH:
            continue
        # P4 union mode: learned proposer + structural region centroids + ACTION5 settle
        cands = [('click', (p['data']['x'], p['data']['y']))
                 for p in CH.propose_clicks(g, topk=6)]
        cands += [('click', c) for c in generic_candidates(g)]
        cands.append(('ACTION5',))
        for spec in cands:
            child = path + [spec]
            cf = reach(child)
            if cf is None:
                continue
            if int(cf.levels_completed) > base_level or cf.state.name == 'WIN':
                return child, 'solved', (expanded, len(seen))
            if cf.state.name in ('WIN', 'GAME_OVER'):
                continue
            h = pic(cf)
            if h not in seen:
                seen.add(h)
                frontier.append(list(rel) + [spec])
            nodes += 1
    return None, 'frontier_exhausted', (expanded, len(seen))


total_deadline = time.time() + TOTAL_SECONDS
solutions = []
base_path = []
prev_len = 0
base = 0
for lv in range(base, NUM_LEVELS):
    if time.time() > total_deadline:
        print(f'L{lv + 1}: total time budget hit, stopping', flush=True)
        break
    t0 = time.time()
    path, reason, stats = search_level(base_path, lv,
                                       min(time.time() + PER_LEVEL_SECONDS, total_deadline))
    if path:
        fr = reach(path)
        base = int(fr.levels_completed)
        base_path = list(path)
        solutions.append((lv + 1, path))
        print(f'L{lv + 1}: SOLVED +{len(path) - prev_len} new clicks ({reason}) '
              f'nodes={stats[0]} states={stats[1]} {time.time()-t0:.0f}s -> levels={base}',
              flush=True)
        prev_len = len(path)
    else:
        print(f'L{lv + 1}: NOT CLEARED ({reason}) nodes={stats[0]} states={stats[1]} '
              f'{time.time()-t0:.0f}s — search ends here', flush=True)
        break

print('\n=== summary ===')
total_clicks = sum(len(p) for _, p in solutions)
print(f'levels cleared autonomously: {len(solutions)}; total clicks so far: {total_clicks}')
for lv, p in solutions:
    print(f'  L{lv}: {len(p)} clicks: {p}')
