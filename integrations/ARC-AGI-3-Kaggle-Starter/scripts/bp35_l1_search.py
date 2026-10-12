"""Search bp35 L1: replay L0 plan, then BFS ACTION3/4 + colour-14 clicks."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")]
os.chdir(ROOT)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from agent.my_agent import _BP35_PLANS, _BP35_NODE_COLOUR  # noqa: E402


def step(env, aid, x=0, y=0):
    if aid == 6:
        return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return env.step(GameAction.from_id(int(aid)), data={})


def frame_grid(fr):
    ff = fr.frame
    arr = np.asarray(ff[0] if isinstance(ff, list) else ff)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def c14_clicks(arr):
    ys, xs = np.where(arr == _BP35_NODE_COLOUR)
    out = []
    seen = set()
    for y, x in zip(ys.tolist(), xs.tolist()):
        # cluster to 3x3 centers
        cx, cy = (x // 3) * 3 + 1, (y // 3) * 3 + 1
        if (cx, cy) not in seen:
            seen.add((cx, cy))
            out.append((cx, cy))
    return out[:20]


def main() -> None:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = arc.make("bp35")
    fr = env.reset()
    g = env._game
    for aid, x, y in _BP35_PLANS[0]:
        fr = step(env, aid, x, y)
        if g.level_index > 0:
            break
    print("at L", g.level_index, "state", g._state)
    if g.level_index < 1:
        print("L0 plan failed to advance")
        return
    start_li = g.level_index
    arr = frame_grid(fr)
    clicks = c14_clicks(arr)
    print("c14 clicks", clicks)

    prefix = []
    env2 = arc.make("bp35")
    env2.reset()
    g2 = env2._game
    for aid, x, y in _BP35_PLANS[0]:
        step(env2, aid, x, y)
        prefix.append((aid, x, y))
        if g2.level_index > 0:
            break

    def run(path):
        env = arc.make("bp35")
        env.reset()
        gg = env._game
        last = None
        for aid, x, y in prefix + path:
            last = step(env, aid, x, y)
            if gg.level_index > start_li:
                return True, gg, last
        return False, gg, last

    pool = [(3, 0, 0), (4, 0, 0)] + [(6, x, y) for x, y in clicks]
    beam = [[]]
    t0 = time.time()
    for depth in range(1, 20):
        nxt = []
        for path in beam:
            for act in pool:
                cand = path + [act]
                ok, gg, last = run(cand)
                if ok:
                    print("FOUND", cand)
                    (ROOT / "scripts" / "_bp35_l1_plan.txt").write_text(
                        repr(cand), encoding="utf-8"
                    )
                    return
                # refresh clicks from last frame occasionally
                nxt.append(cand)
        # prune: keep unique by refreshing c14 from a sample
        beam = nxt[:150]
        # also expand click pool from a mid state
        if beam and depth % 2 == 0:
            _, _, last = run(beam[0])
            if last is not None:
                for xy in c14_clicks(frame_grid(last)):
                    act = (6, xy[0], xy[1])
                    if act not in pool:
                        pool.append(act)
        print(f"d={depth} beam={len(beam)} pool={len(pool)} t={time.time()-t0:.1f}")
        if time.time() - t0 > 120:
            break
    print("FAIL")


if __name__ == "__main__":
    main()
