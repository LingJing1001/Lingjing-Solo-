"""bp35 L2 (levels_completed=1): carve platforms via grid-mapped ACTION6 + L/R."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")]
os.chdir(ROOT)
logging.disable(logging.CRITICAL)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from agent.my_agent import _BP35_PLANS  # noqa: E402


def step(env, aid, x=0, y=0):
    if aid == 6:
        return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return env.step(GameAction.from_id(int(aid)), data={})


def to_l2(env):
    fr = env.reset()
    g = env._game
    for aid, x, y in _BP35_PLANS[0]:
        fr = step(env, aid, x, y)
        if g.level_index >= 1:
            break
    return fr


def screen_for(g, gx, gy):
    cam = g.oztjzzyqoek.camera.rczgvgfsfb[1]
    for px in range(0, 64):
        for py in range(0, 64):
            if g.oztjzzyqoek.hdnrlfmyrj.hyntnfvpgl(px, py + cam) == (gx, gy):
                return (px, py)
    return None


def visible_plats(g):
    out = []
    for s in g.oztjzzyqoek.hdnrlfmyrj.wwkbcxznzg("qclfkhjnaac"):
        cell = s.qumspquyus
        scr = screen_for(g, *cell)
        if scr is not None:
            out.append((cell, scr))
    return out


def player(g):
    return g.oztjzzyqoek.twdpowducb.qumspquyus


def gem(g):
    return g.oztjzzyqoek.hdnrlfmyrj.wwkbcxznzg("fjlzdjxhant")[0].qumspquyus


def dead(g):
    return bool(getattr(g.oztjzzyqoek, "jrhqdvdwpsb", False))


def won(g, start_li):
    return g.level_index > start_li or bool(getattr(g.oztjzzyqoek, "nkuphphdgrp", False))


def run(env, path):
    to_l2(env)
    g = env._game
    start_li = g.level_index
    for aid, x, y in path:
        step(env, aid, x, y)
        if won(g, start_li):
            return True, g
        if dead(g):
            return False, g
    return won(g, start_li), g


def heuristic(g):
    px, py = player(g)
    gx, gy = gem(g)
    # gravity up: lower y is 'below' feet toward gem
    plats = {s.qumspquyus for s in g.oztjzzyqoek.hdnrlfmyrj.wwkbcxznzg("qclfkhjnaac")}
    # blocking cells in column of player between py and gy
    block = sum(1 for y in range(gy + 1, py) if (px, y) in plats)
    return (abs(px - gx) + abs(py - gy), block, len(plats))


def main() -> None:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = arc.make("bp35")
    to_l2(env)
    g = env._game
    print("start", player(g), "gem", gem(g), "plats", heuristic(g), flush=True)

    # Scripted carve: clear column x=3 from near player up, then shift to gem x=5
    script = []
    # click all visible plats in column 3 with y < player.y, topmost first? bottom first safer
    # Prefer: remove (3,36),(3,35),(3,30),(3,29),... then move right toward gem column

    def plan_clicks_for_column(env, col, path):
        to_l2(env)
        g = env._game
        for aid, x, y in path:
            step(env, aid, x, y)
        g = env._game
        px, py = player(g)
        cells = sorted(
            [s.qumspquyus for s in g.oztjzzyqoek.hdnrlfmyrj.wwkbcxznzg("qclfkhjnaac") if s.qumspquyus[0] == col and s.qumspquyus[1] < py],
            key=lambda c: -c[1],  # highest y first (closest under player when grav up)
        )
        added = []
        for cell in cells:
            scr = screen_for(g, *cell)
            if scr is None:
                # move camera by walking? camera follows player
                continue
            act = (6, scr[0], scr[1])
            step(env, *act)
            added.append(act)
            g = env._game
            if won(g, 1) or dead(g):
                return added
            # refresh remaining after removal
        return added

    # Beam: prefer reducing heuristic
    beam = [[]]
    t0 = time.time()
    best = (heuristic(g), [])
    for depth in range(1, 40):
        nxt = []
        seen = set()
        for path in beam:
            ok, gg = run(env, path)
            if ok:
                print("FOUND", path, flush=True)
                (ROOT / "scripts" / "_bp35_l2_plan.txt").write_text(
                    repr(path), encoding="utf-8"
                )
                return
            if dead(gg):
                continue
            h = heuristic(gg)
            if h < best[0]:
                best = (h, path)
                print("best", h, "len", len(path), "p", player(gg), flush=True)
            acts = [(3, 0, 0), (4, 0, 0)]
            for cell, scr in visible_plats(gg):
                # prioritize plats in player column or adjacent, between player and gem
                px, py = player(gg)
                gx, gy = gem(gg)
                if not (gy <= cell[1] <= py and abs(cell[0] - px) <= 2):
                    continue
                acts.append((6, scr[0], scr[1]))
            for act in acts:
                cand = path + [act]
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                ok2, gg2 = run(env, cand)
                if ok2:
                    print("FOUND", cand, flush=True)
                    (ROOT / "scripts" / "_bp35_l2_plan.txt").write_text(
                        repr(cand), encoding="utf-8"
                    )
                    return
                if dead(gg2):
                    continue
                nxt.append((heuristic(gg2), cand))
        nxt.sort(key=lambda t: (t[0], len(t[1])))
        beam = [c for _, c in nxt[:50]]
        print(
            f"d={depth} beam={len(beam)} best={best[0]} t={time.time()-t0:.1f}",
            flush=True,
        )
        if time.time() - t0 > 120:
            break
    print("FAIL best", best, flush=True)


if __name__ == "__main__":
    main()
