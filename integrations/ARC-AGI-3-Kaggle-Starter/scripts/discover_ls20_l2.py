"""Discover ls20 L1+L2 scripts with correct per-level boundaries."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
import lingjing_solo.planning.ls20_solver as s

OUT = ROOT / "lingjing_solo" / "planning" / "data" / "ls20_scripts.json"


def main() -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    obs = env.step(GameAction.RESET)
    game = env._game
    l1: list[str] = []
    l2: list[str] = []

    def do(a: str, bucket: list[str]) -> int:
        nonlocal obs
        obs = env.step(getattr(GameAction, a))
        bucket.append(a)
        return int(obs.levels_completed or 0)

    def pos():
        return (game.gudziatsk.x, game.gudziatsk.y)

    def nav(target: tuple[int, int], bucket: list[str], stop_level: int) -> bool:
        gg = s._as_grid(np.array(obs.frame[0]))
        p = pos()
        path = s._bfs(gg, p, target) or s._bfs_push(gg, p, target) or []
        print(
            f"nav {p}->{target} len={len(path)} steps={game._step_counter_ui.current_steps} "
            f"rot={game.dhksvilbb[game.cklxociuu]}"
        )
        if not path and p != target:
            print("  UNREACHABLE")
            return False
        for a in path:
            if do(a, bucket) >= stop_level:
                return True
        return False

    # L1
    p0 = pos()
    g0 = s._as_grid(np.array(obs.frame[0]))
    for a in (s._bfs(g0, p0, (19, 30)) or []):
        if do(a, l1) >= 1:
            break
    else:
        g1 = s._as_grid(np.array(obs.frame[0]))
        for a in (s._bfs(g1, pos(), (34, 10)) or []):
            if do(a, l1) >= 1:
                break
        if int(obs.levels_completed or 0) < 1:
            do("ACTION1", l1)

    print("L1", len(l1), "L", obs.levels_completed, "pos", pos(), "steps", game._step_counter_ui.current_steps)
    if int(obs.levels_completed or 0) < 1:
        sys.exit("L1 failed")

    # Align to the corridor row used by the known-good recipe (29,35)
    if pos() == (29, 40):
        do("ACTION1", l2)
        print("align", pos(), "steps", game._step_counter_ui.current_steps)

    # Known-good L2: pickup → pad → 1 toggle → leave W → refill → goal
    if nav((39, 50), l2, 2):
        pass
    print("pickup", pos(), game._step_counter_ui.current_steps, game.dhksvilbb[game.cklxociuu],
          "picks", len(game.current_level.get_sprites_by_tag("npxgalaybz")))
    if nav((49, 45), l2, 2):
        pass
    print("pad", pos(), "rot", game.dhksvilbb[game.cklxociuu], "m", game.bejndxqqzf(0),
          "steps", game._step_counter_ui.current_steps)

    if not game.bejndxqqzf(0) and int(obs.levels_completed or 0) < 2:
        do("ACTION3", l2)
        do("ACTION4", l2)
    print("toggle", pos(), "rot", game.dhksvilbb[game.cklxociuu], "m", game.bejndxqqzf(0),
          "steps", game._step_counter_ui.current_steps)

    if int(obs.levels_completed or 0) < 2 and pos() == (49, 45):
        do("ACTION3", l2)  # leave west
    print("left", pos(), "m", game.bejndxqqzf(0), "steps", game._step_counter_ui.current_steps)

    if int(obs.levels_completed or 0) < 2:
        nav((14, 15), l2, 2)
    print("refill", pos(), game._step_counter_ui.current_steps, "m", game.bejndxqqzf(0))
    if int(obs.levels_completed or 0) < 2:
        nav((14, 40), l2, 2)
    if int(obs.levels_completed or 0) < 2:
        for _ in range(8):
            if do("ACTION2", l2) >= 2:
                break

    print("final L", obs.levels_completed, "l1", len(l1), "l2", len(l2), "pos", pos())
    if int(obs.levels_completed or 0) < 2:
        sys.exit("L2 failed")

    env2 = arc.make("ls20")
    obs2 = env2.step(GameAction.RESET)
    for a in l1 + l2:
        obs2 = env2.step(getattr(GameAction, a))
        if int(obs2.levels_completed or 0) >= 2:
            break
    print("verify", obs2.levels_completed)
    if int(obs2.levels_completed or 0) < 2:
        sys.exit("verify failed")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "game_id": "ls20",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "levels": {"0": l1, "1": l2},
                "note": "L2: align N + pickup(39,50) + pad + toggle + refill + goal",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
