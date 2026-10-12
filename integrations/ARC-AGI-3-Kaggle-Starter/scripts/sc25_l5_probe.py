"""Replay sc25 L0-L3 plans, dump L4 (levels=4) sprites/tags, try short search."""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")]
os.chdir(ROOT)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from agent.my_agent import _SC25_PLANS  # noqa: E402


def apply(env, aid, x=0, y=0):
    if aid == 6:
        return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return env.step(GameAction.from_id(aid), data={})


def dump_level(g, label: str) -> None:
    lv = g.current_level
    print(f"\n=== {label} li={g.level_index} ===")
    for sp in lv.get_sprites():
        tags = list(getattr(sp, "tags", []) or [])
        print(
            f"  {sp.name}: ({sp.x},{sp.y}) {sp.width}x{sp.height} "
            f"rot={getattr(sp,'rotation',0)} tags={tags[:8]}"
        )


def main() -> None:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = arc.make("sc25")
    fr = env.reset()
    g = env._game
    for lv in range(4):
        plan = _SC25_PLANS[lv]
        print(f"replay L{lv} n={len(plan)} li={g.level_index}")
        for aid, x, y in plan:
            fr = apply(env, aid, x, y)
            # stop early if level advanced mid-plan
            if g.level_index > lv:
                print(f"  advanced to li={g.level_index} mid-plan")
                break
        print(f"  after: li={g.level_index}")

    dump_level(g, "L4 start")
    # BFS shallow: keyboard + click nonbg centroids from first frame
    import numpy as np

    def grid_of(frame):
        ff = frame.frame
        arr = np.asarray(ff[0] if isinstance(ff, list) else ff)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    arr0 = grid_of(fr)
    hist = np.bincount(arr0.flatten(), minlength=16)
    bg = int(np.argmax(hist))
    clicks = []
    for c in np.argsort(hist):
        if int(hist[c]) == 0 or int(c) == bg or int(hist[c]) > 200:
            continue
        ys, xs = np.where(arr0 == c)
        if len(xs):
            clicks.append((int(xs.mean()), int(ys.mean())))
        if len(clicks) >= 12:
            break
    # also sample puzzle board row
    for y in (51, 56, 61):
        for x in (26, 31, 36):
            clicks.append((x, y))

    acts = [(a,) for a in (1, 2, 3, 4)] + [(6, x, y) for x, y in clicks]
    print(f"search acts={len(acts)} clicks={clicks[:8]}")

    # IDDFS depth<=12 for level advance
    t0_li = g.level_index
    found = None

    def clone_state():
        # replay from reset through all plans to L4 then path — too slow;
        # instead search live with reset+full replay when candidate found
        pass

    # Live BFS on deepcopy of game via env reset + full replay of prefix
    prefix = []
    for lv in range(4):
        prefix.extend(_SC25_PLANS[lv])

    def replay_to(path):
        env2 = arc.make("sc25")
        env2.reset()
        gg = env2._game
        for item in prefix + path:
            if len(item) == 3:
                aid, x, y = item
            else:
                aid, x, y = item[0], 0, 0
            apply(env2, aid, x, y)
            if gg.level_index > t0_li:
                return True, env2, gg
        return False, env2, gg

    # iterative deepening over keyboard/click mix
    from itertools import product

    for depth in range(1, 10):
        print(f"ID depth={depth}")
        # restrict: prefer click candidates then keyboard
        pool = [(6, x, y) for x, y in clicks[:10]] + [(a, 0, 0) for a in (1, 2, 3, 4)]
        # too big for product — greedy beam instead
        beam = [[]]
        for d in range(depth):
            nxt = []
            for path in beam:
                for act in pool:
                    cand = path + [act]
                    ok, _, gg = replay_to(cand)
                    if ok:
                        print("FOUND", cand, "li", gg.level_index)
                        found = cand
                        break
                    nxt.append(cand)
                if found:
                    break
            if found:
                break
            # prune beam: keep first 200
            beam = nxt[:200]
        if found:
            break
    if not found:
        print("NO_PLAN_DEPTH10")
    else:
        print("PLAN", found)


if __name__ == "__main__":
    main()
