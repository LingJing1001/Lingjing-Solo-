"""Offline multi-level env BFS for ARC-AGI-3 games (discovery / oracle tool).

Improvements:
  - No hardcoded API key (Arcade / ARC_API_KEY env)
  - Cumulative prefix across levels (reset + replay prefix, then search suffix)
  - ACTION1–5 by default; --with-click enables coarse ACTION6 grid
  - --seed-prefix for known L1 (e.g. ar25)
  - Writes lingjing_solo/planning/data/<game>_scripts.json
  - Merges into existing JSON when --merge

Usage:
  python scripts/offline_level_bfs.py ls20 30
  python scripts/offline_level_bfs.py ar25 25 --max-levels 4 --seed-prefix ACTION3x5,ACTION2x10
  python scripts/offline_level_bfs.py ar25 18 --with-click --click-step 8
"""
from __future__ import annotations

import argparse
import json
import re
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

SIMPLE_ACTIONS = (
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
)
ACTION_NAMES = {
    GameAction.ACTION1: "ACTION1",
    GameAction.ACTION2: "ACTION2",
    GameAction.ACTION3: "ACTION3",
    GameAction.ACTION4: "ACTION4",
    GameAction.ACTION5: "ACTION5",
    GameAction.ACTION6: "ACTION6",
    GameAction.ACTION7: "ACTION7",
}


def parse_seed_prefix(spec: str) -> list[str]:
    """Parse 'ACTION3,ACTION3' or 'ACTION3x5,ACTION2x10' into action name list."""
    out: list[str] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(ACTION[1-7]|RESET)[xX×*](\d+)", part)
        if m:
            out.extend([m.group(1).upper()] * int(m.group(2)))
            continue
        out.append(part.upper())
    return out


def names_to_actions(names: list[str]) -> list[GameAction]:
    out: list[GameAction] = []
    for n in names:
        if n.startswith("ACTION6:"):
            # ACTION6:x,y
            _, xy = n.split(":", 1)
            x_s, y_s = xy.split(",")
            a = GameAction.ACTION6
            a.set_data({"x": int(x_s), "y": int(y_s)})
            out.append(a)
        else:
            out.append(getattr(GameAction, n))
    return out


def action_to_spec(action) -> str | dict:
    if isinstance(action, _Click):
        return {"action": "ACTION6", "x": action.x, "y": action.y}
    name = ACTION_NAMES.get(action, getattr(action, "name", str(action)))
    if name == "ACTION6":
        data = getattr(action, "data", None) or {}
        return {"action": "ACTION6", "x": int(data.get("x", 0)), "y": int(data.get("y", 0))}
    return name


def extract_state_key(obs) -> bytes | str | None:
    if obs is None:
        return None
    frame = getattr(obs, "frame", None)
    levels = int(getattr(obs, "levels_completed", 0) or 0)
    state = getattr(obs, "state", None)
    st = state.name if hasattr(state, "name") else str(state)
    if frame is not None and len(frame) > 0:
        # Include levels so post-clear frames don't collide with pre-clear.
        return np.asarray(frame[0]).tobytes() + f"|{levels}|{st}".encode()
    return f"{st}:{levels}"


def replay(env, actions: list):
    obs = env.reset()
    if obs is None:
        return None
    for action in actions:
        obs = _step_one(env, action)
        if obs is None:
            return None
        if getattr(obs, "state", None) == GameState.GAME_OVER:
            return obs
    return obs


def _step_one(env, action):
    if isinstance(action, _Click) or (
        getattr(action, "name", "") == "ACTION6" and isinstance(getattr(action, "data", None), dict)
    ):
        a = GameAction.ACTION6
        data = getattr(action, "data", {"x": 0, "y": 0})
        a.set_data({"x": int(data.get("x", 0)), "y": int(data.get("y", 0))})
        return env.step(a)
    return env.step(action)


def expand_actions(*, with_click: bool, click_step: int, grid: int = 64) -> list:
    acts: list = list(SIMPLE_ACTIONS)
    if with_click:
        for y in range(click_step // 2, grid, click_step):
            for x in range(click_step // 2, grid, click_step):
                acts.append(_Click(x, y))
    return acts


class _Click:
    """Stand-in so ACTION6 with different (x,y) are distinct BFS branches."""

    name = "ACTION6"

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.data = {"x": x, "y": y}

    def __eq__(self, other):  # pragma: no cover
        return other is GameAction.ACTION6 or getattr(other, "name", "") == "ACTION6"


def solve_level_bfs(
    env,
    *,
    prefix: list,
    level_num: int,
    max_depth: int,
    with_click: bool = False,
    click_step: int = 8,
) -> list | None:
    print(f"\n{'=' * 50}")
    print(f"  Level {level_num + 1}  max_depth={max_depth}  prefix={len(prefix)}")
    print(f"{'=' * 50}")

    start_obs = replay(env, prefix)
    if start_obs is None:
        print("ERROR: prefix replay failed")
        return None

    state0 = getattr(start_obs, "state", None)
    levels0 = int(getattr(start_obs, "levels_completed", 0) or 0)
    print(f"start: state={state0} levels={levels0}")
    if state0 == GameState.WIN:
        return []
    if levels0 > level_num:
        print("already past this level")
        return []

    action_space = expand_actions(with_click=with_click, click_step=click_step)
    visited: set = set()
    key0 = extract_state_key(start_obs)
    if key0 is not None:
        visited.add(key0)

    queue: deque[list] = deque([[]])
    total = 0
    t0 = time.time()

    while queue:
        suffix = queue.popleft()
        if len(suffix) >= max_depth:
            continue

        for action in action_space:
            cand = suffix + [action]
            obs = replay(env, prefix + cand)
            total += 1
            if obs is None:
                continue
            st = getattr(obs, "state", None)
            lv = int(getattr(obs, "levels_completed", 0) or 0)
            if st == GameState.WIN or lv > level_num:
                elapsed = time.time() - t0
                print(
                    f"SOLVED L{level_num + 1}: +{len(cand)} steps "
                    f"(tries={total}, {elapsed:.1f}s)"
                )
                print(" -> ".join(_label(a) for a in cand))
                return cand
            if st == GameState.GAME_OVER:
                continue
            key = extract_state_key(obs)
            if key is None or key in visited:
                continue
            visited.add(key)
            queue.append(cand)

        if total and total % 400 == 0:
            print(
                f"  progress tries={total} q={len(queue)} "
                f"vis={len(visited)} {time.time() - t0:.1f}s"
            )

    print(f"FAILED L{level_num + 1}: tries={total} {time.time() - t0:.1f}s")
    return None


def _label(action) -> str:
    if isinstance(action, _Click):
        return f"ACTION6({action.x},{action.y})"
    return ACTION_NAMES.get(action, getattr(action, "name", str(action)))


def seed_l1_via_solver(env) -> list[GameAction] | None:
    from lingjing_solo.planning import ls20_solver as s

    obs = env.reset()
    if obs is None:
        return None
    recorded: list[GameAction] = []
    solver = s.Ls20Solver()
    g = s._as_grid(np.asarray(obs.frame[0]))
    solver.reset_level(g)
    prev = g
    name_to = {v: k for k, v in ACTION_NAMES.items() if k in SIMPLE_ACTIONS[:4]}
    for _ in range(80):
        lv = int(getattr(obs, "levels_completed", 0) or 0)
        if lv >= 1 or getattr(obs, "state", None) == GameState.WIN:
            print(f"seed L1 ok: {len(recorded)} actions")
            return recorded
        act_name = solver.plan(["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        if not act_name:
            act_name = "ACTION1"
        action = name_to[act_name]
        new_obs = env.step(action)
        recorded.append(action)
        if new_obs is None:
            break
        curr = s._as_grid(np.asarray(new_obs.frame[0]))
        solver.observe(prev, curr, act_name, int(new_obs.levels_completed or 0))
        prev = curr
        obs = new_obs
        if getattr(obs, "state", None) == GameState.GAME_OVER:
            break
    print(f"seed L1 failed at {len(recorded)} actions L={getattr(obs, 'levels_completed', 0)}")
    return None


def save_scripts(
    game_id: str,
    level_scripts: dict[str, list],
    out: Path,
    *,
    merge: bool = False,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if merge and out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    levels = dict(existing.get("levels") or {})
    levels.update(level_scripts)
    payload = {
        "game_id": game_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "levels": levels,
        "note": existing.get("note")
        or "Per-level action lists from offline BFS; verify levels_completed before shipping.",
    }
    if "evidence" in existing:
        payload["evidence"] = existing["evidence"]
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline multi-level BFS discovery")
    ap.add_argument("game_id", nargs="?", default="ls20")
    ap.add_argument("max_depth", nargs="?", type=int, default=30)
    ap.add_argument("--max-levels", type=int, default=7)
    ap.add_argument("--seed-l1-solver", action="store_true")
    ap.add_argument(
        "--seed-prefix",
        default="",
        help="Known prefix e.g. ACTION3x5,ACTION2x10 (fills level 0 if empty)",
    )
    ap.add_argument("--with-click", action="store_true", help="Include ACTION6 grid clicks")
    ap.add_argument("--click-step", type=int, default=8)
    ap.add_argument("--merge", action="store_true", help="Merge into existing JSON")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (
        ROOT / "lingjing_solo" / "planning" / "data" / f"{args.game_id}_scripts.json"
    )

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game_id)
    if env is None:
        print(f"make({args.game_id}) failed")
        sys.exit(1)

    obs = env.reset()
    total_levels = int(getattr(obs, "win_levels", 7) or 7)
    target_levels = min(total_levels, args.max_levels)
    print(f"game={args.game_id} win_levels={total_levels} solve_up_to={target_levels}")

    prefix: list = []
    level_scripts: dict[str, list] = {}

    if args.seed_prefix:
        names = parse_seed_prefix(args.seed_prefix)
        prefix = names_to_actions(names)
        # Verify seed clears at least L1 or is a valid start
        check = replay(env, prefix)
        lv = int(getattr(check, "levels_completed", 0) or 0) if check else 0
        print(f"seed-prefix len={len(prefix)} → levels={lv}")
        if lv >= 1 and "0" not in level_scripts:
            # Split: attribute all seed to level 0 if it cleared L1 in one go
            level_scripts["0"] = [action_to_spec(a) for a in prefix]
            # continue search from level lv
            start_level = lv
        else:
            start_level = 0
            if lv == 0:
                print("WARNING: seed did not clear L1; searching from L1 with seed as prefix")
    else:
        start_level = 0

    for level in range(start_level if args.seed_prefix else 0, target_levels):
        if level == 0 and args.seed_l1_solver and args.game_id.startswith("ls20"):
            seeded = seed_l1_via_solver(env)
            if seeded is not None:
                level_scripts["0"] = [action_to_spec(a) for a in seeded]
                prefix = list(seeded)
                continue

        if str(level) in level_scripts and level < start_level:
            continue

        suffix = solve_level_bfs(
            env,
            prefix=prefix,
            level_num=level,
            max_depth=args.max_depth,
            with_click=args.with_click,
            click_step=args.click_step,
        )
        if suffix is None:
            print(f"\nStopped at level {level + 1}")
            break
        level_scripts[str(level)] = [action_to_spec(a) for a in suffix]
        prefix = prefix + suffix

    save_scripts(args.game_id, level_scripts, out, merge=args.merge or bool(args.seed_prefix))
    print(json.dumps({"levels_found": len(level_scripts), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
