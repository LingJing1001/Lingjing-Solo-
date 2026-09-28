"""Probe all public games: can a simple sweep clear any level?"""
from __future__ import annotations

import random
from collections import Counter

from arc_agi import Arcade, OperationMode
from arcengine import GameAction


def as_grid(frame) -> list[list[int]]:
    layer = frame[-1]
    return [[int(c) for c in row] for row in layer]


def non_bg_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]
    h, w = len(grid), len(grid[0])
    return [(x, y) for y in range(h) for x in range(w) if grid[y][x] != bg]


def main() -> None:
    arc = Arcade(operation_mode=OperationMode.NORMAL)
    results = []
    for e in arc.get_environments():
        gid = e.game_id.split("-")[0]
        env = arc.make(gid)
        if env is None:
            continue
        obs = env.observation_space
        avail = list(obs.available_actions or [])
        acts = []
        for v in avail:
            m = next((a for a in GameAction if a.value == int(v)), None)
            if m is not None and m is not GameAction.RESET:
                acts.append(m)
        if not acts:
            acts = [a for a in GameAction if a is not GameAction.RESET]

        best = 0
        only_click = all(a.is_complex() for a in acts)
        grid = as_grid(obs.frame)
        click_list = non_bg_cells(grid)
        random.shuffle(click_list)
        click_list += [(x, y) for y in range(0, 64, 3) for x in range(0, 64, 3)]
        ci = 0

        for i in range(800):
            if only_click:
                a = acts[0]
                x, y = click_list[ci % len(click_list)]
                ci += 1
                a.set_data({"x": int(x), "y": int(y)})
            else:
                a = random.choice(acts)
                if a.is_complex():
                    x, y = click_list[ci % len(click_list)]
                    ci += 1
                    a.set_data({"x": int(x), "y": int(y)})
            r = env.step(a)
            if r is None:
                break
            if r.levels_completed > best:
                best = int(r.levels_completed)
                print(f"HIT {gid} level={best} at={i+1}", flush=True)
            st = r.state.name if hasattr(r.state, "name") else str(r.state)
            if st == "WIN":
                break
            if st == "GAME_OVER":
                env.step(GameAction.RESET)
                # refresh clicks
                grid = as_grid(env.observation_space.frame)
                click_list = non_bg_cells(grid) + click_list
                ci = 0

        results.append((gid, avail, best))
        print(f"DONE {gid} avail={avail} best={best}", flush=True)

    print("==== RANKED ====")
    for g, a, b in sorted(results, key=lambda t: -t[2]):
        print(f"{g:6} best={b} avail={a}")


if __name__ == "__main__":
    main()
