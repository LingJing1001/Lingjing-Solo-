"""Interactive-style trial of sc25 L5 spell sequences; report what changes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")]
os.chdir(ROOT)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from agent.my_agent import _SC25_PLANS  # noqa: E402


def step(env, aid, x=0, y=0):
    if aid == 6:
        return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return env.step(GameAction.from_id(int(aid)), data={})


def to_l4(arc):
    env = arc.make("sc25")
    env.reset()
    g = env._game
    for lv in range(4):
        for aid, x, y in _SC25_PLANS[lv]:
            step(env, aid, x, y)
            if g.level_index > lv:
                break
    return env, g


def names(g):
    return sorted({sp.name for sp in g.current_level.get_sprites()})


def main() -> None:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    trials = {
        "sieesc_x": [
            (6, 31, 51), (6, 26, 56), (6, 36, 56), (6, 31, 61),
        ],
        "fibcey_col": [
            (4, 0, 0), (4, 0, 0),
            (6, 31, 51), (6, 31, 56), (6, 31, 61),
        ],
        "tevyeq_box": [
            (6, 26, 51), (6, 31, 51), (6, 31, 56),
        ],
        "scale_then_beam_left": [
            (6, 31, 51), (6, 26, 56), (6, 36, 56), (6, 31, 61),
            (2, 0, 0), (2, 0, 0), (2, 0, 0), (2, 0, 0), (2, 0, 0),
            (3, 0, 0), (3, 0, 0), (3, 0, 0),
            (6, 31, 51), (6, 31, 56), (6, 31, 61),
            (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0),
            (1, 0, 0), (1, 0, 0), (1, 0, 0),
            (4, 0, 0), (4, 0, 0), (4, 0, 0), (4, 0, 0),
        ],
        "l3_like_full": list(_SC25_PLANS[3]),
    }
    for name, seq in trials.items():
        env, g = to_l4(arc)
        before = names(g)
        pl = [sp for sp in g.current_level.get_sprites() if sp.name == "pluyoo"]
        p0 = (pl[0].x, pl[0].y, pl[0].rotation) if pl else None
        for aid, x, y in seq:
            step(env, aid, x, y)
            if g.level_index > 4:
                print(name, "WIN_ADVANCE", seq)
                (ROOT / "scripts" / "_sc25_l5_plan.txt").write_text(
                    repr(seq), encoding="utf-8"
                )
                return
        after = names(g)
        pl = [sp for sp in g.current_level.get_sprites() if sp.name == "pluyoo"]
        p1 = (pl[0].x, pl[0].y, pl[0].rotation) if pl else None
        gone = set(before) - set(after)
        added = set(after) - set(before)
        print(
            f"{name}: li={g.level_index} player {p0}->{p1} "
            f"gone={sorted(gone)[:8]} added={sorted(added)[:8]}"
        )
    print("no trial advanced")


if __name__ == "__main__":
    main()
