"""Discover ls20 level scripts (L1 via solver, L2 via guided recipe + short env BFS).

Writes lingjing_solo/planning/data/ls20_scripts.json for script_bank playback.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState
import lingjing_solo.planning.ls20_solver as s

OUT = ROOT / "lingjing_solo" / "planning" / "data" / "ls20_scripts.json"
NAME = {
    GameAction.ACTION1: "ACTION1",
    GameAction.ACTION2: "ACTION2",
    GameAction.ACTION3: "ACTION3",
    GameAction.ACTION4: "ACTION4",
}
TO_ACT = {v: k for k, v in NAME.items()}


def grid(obs) -> np.ndarray:
    return s._as_grid(np.asarray(obs.frame[0]))


def step(env, name: str):
    return env.step(TO_ACT[name]), name


def clear_l1(env) -> tuple[list[str], object]:
    obs = env.reset()
    solver = s.Ls20Solver()
    solver._try_load_script = lambda: False  # type: ignore
    g = grid(obs)
    solver.reset_level(g)
    prev = g
    acts: list[str] = []
    for _ in range(100):
        if int(obs.levels_completed or 0) >= 1:
            return acts, obs
        name = solver.plan(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]) or "ACTION1"
        obs2, _ = step(env, name)
        acts.append(name)
        if obs2 is None:
            break
        curr = grid(obs2)
        solver.observe(prev, curr, name, int(obs2.levels_completed or 0))
        solver._script_mode = False
        prev = curr
        obs = obs2
        if obs.state == GameState.GAME_OVER:
            break
    return acts, obs


def settle_layout(env, obs, n: int = 4) -> tuple[list[str], object]:
    acts = []
    for _ in range(n):
        obs, name = step(env, "ACTION1")
        acts.append(name)
        if obs is None or int(obs.levels_completed or 0) >= 2:
            break
    return acts, obs


def frame_key(obs) -> bytes:
    return np.asarray(obs.frame[0]).tobytes()


def short_env_bfs(env, prefix_names: list[str], target_level: int, max_depth: int = 24):
    """BFS suffix from current prefix to raise levels_completed."""

    def replay(names: list[str]):
        o = env.reset()
        for n in names:
            o = env.step(TO_ACT[n])
            if o is None or o.state == GameState.GAME_OVER:
                return o
        return o

    start = replay(prefix_names)
    if start is None:
        return None
    if int(start.levels_completed or 0) >= target_level:
        return []

    visited = {frame_key(start)}
    q: deque[list[str]] = deque([[]])
    tries = 0
    t0 = time.time()
    while q:
        suf = q.popleft()
        if len(suf) >= max_depth:
            continue
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            cand = suf + [name]
            obs = replay(prefix_names + cand)
            tries += 1
            if obs is None:
                continue
            lv = int(obs.levels_completed or 0)
            if lv >= target_level or obs.state == GameState.WIN:
                print(f"env-BFS hit L{target_level} +{len(cand)} tries={tries} {time.time()-t0:.1f}s")
                return cand
            if obs.state == GameState.GAME_OVER:
                continue
            key = frame_key(obs)
            if key in visited:
                continue
            visited.add(key)
            q.append(cand)
        if tries and tries % 300 == 0:
            print(f"  bfs tries={tries} q={len(q)} vis={len(visited)}")
    print(f"env-BFS miss tries={tries} {time.time()-t0:.1f}s")
    return None


def guided_l2(env, prefix: list[str]) -> list[str] | None:
    """Navigate pad → 2 toggles → approach cell, then short BFS for exit."""
    obs = env.reset()
    for n in prefix:
        obs = env.step(TO_ACT[n])
        if obs is None:
            return None

    l2_acts: list[str] = []
    g = grid(obs)
    player = s._find_player(g)
    pad = s._find_rot_pad(g, player) or (49, 45)
    goal = (14, 40)
    approach = (14, 35)
    avoid = {(44, 40)}

    def do(name: str):
        nonlocal obs
        obs2, _ = step(env, name)
        l2_acts.append(name)
        obs = obs2
        return obs

    # path to pad
    path = s._path_to(g, player, pad, avoid) if player else []
    for n in path or []:
        do(n)
        if int(obs.levels_completed or 0) >= 2:
            return l2_acts
    # 2 left-out right-in toggles
    for _ in range(2):
        do("ACTION3")
        do("ACTION4")
        if int(obs.levels_completed or 0) >= 2:
            return l2_acts

    # leave pad south then go to approach
    g = grid(obs)
    player = s._find_player(g)
    avoid = set(s._rot_pad_cells(g)) | {(44, 40)}
    if player and player == pad:
        do("ACTION2")  # leave south
        g = grid(obs)
        player = s._find_player(g)
    for _ in range(60):
        g = grid(obs)
        player = s._find_player(g)
        if player is None:
            do("ACTION1")
            continue
        if player == approach:
            break
        path = s._path_to(g, player, approach, avoid)
        do(path[0] if path else "ACTION3")
        if int(obs.levels_completed or 0) >= 2:
            return l2_acts

    # From approach, try short env BFS for goal entry (full reset replay)
    full_prefix = prefix + l2_acts
    extra = short_env_bfs(env, full_prefix, target_level=2, max_depth=20)
    if extra is not None:
        return l2_acts + extra
    return None


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")

    print("=== clear L1 ===")
    l1, obs = clear_l1(env)
    print(f"L1 acts={len(l1)} levels={getattr(obs, 'levels_completed', None)}")
    if int(getattr(obs, "levels_completed", 0) or 0) < 1:
        print("L1 failed")
        sys.exit(1)

    settle, obs = settle_layout(env, obs, 4)
    prefix = l1 + settle
    print(f"prefix after settle={len(prefix)}")

    print("=== guided L2 ===")
    l2 = guided_l2(env, prefix)
    levels = {"0": l1}
    if l2:
        # Store L2 script as actions after L1 clear, including settle in level-1 bank?
        # script_bank keys by levels_completed at level start.
        # After L1, levels_seen=1 and layout wait uses ACTION1 x4 in agent — then reset_level.
        # So level "1" script should start from post-layout L2 start, NOT include settle.
        # Re-discover L2 from prefix = l1 only + agent will settle separately.
        print(f"L2 raw (from prefix with settle) len={len(l2)}")
        # Re-run guided from l1 only; agent does layout wait itself
        l2b = guided_l2(env, l1)
        if l2b:
            # Drop leading ACTION1 settle-like if any? Keep as-is; solver loads after layout wait.
            # guided_l2 resets and replays l1 then immediately plans L2 — layout may need settle.
            # Better: prefix = l1 + 4x ACTION1
            l2c = guided_l2(env, l1 + ["ACTION1"] * 4)
            if l2c:
                levels["1"] = l2c
                print(f"L2 script len={len(l2c)}")
            else:
                levels["1"] = l2b
        else:
            levels["1"] = l2
    else:
        print("L2 guided failed; trying pure env BFS depth=28")
        extra = short_env_bfs(env, l1 + ["ACTION1"] * 4, target_level=2, max_depth=28)
        if extra:
            levels["1"] = extra

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_id": "ls20",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "levels": levels,
        "note": "L0=L1 clear via solver; L1=L2 guided+BFS after layout settle",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} levels={list(levels)}")

    # Verify: replay scripts with fresh env
    print("=== verify replay ===")
    env2 = arc.make("ls20")
    obs = env2.reset()
    for n in levels.get("0", []):
        obs = env2.step(TO_ACT[n])
    print(f"after L1 script: L={obs.levels_completed}")
    for _ in range(4):
        obs = env2.step(GameAction.ACTION1)
    if "1" in levels:
        for n in levels["1"]:
            obs = env2.step(TO_ACT[n])
            if int(obs.levels_completed or 0) >= 2:
                break
        print(f"after L2 script: L={obs.levels_completed} state={obs.state}")
    else:
        print("no L2 script")


if __name__ == "__main__":
    main()
