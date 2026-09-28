"""Probe ls20 mechanics: learn movement vectors and test naive navigation."""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState


def load_agent():
    spec = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid_of(frame) -> np.ndarray:
    raw = frame.frame or []
    if not raw:
        return np.zeros((64, 64), dtype=np.int8)
    layer = raw[-1] if raw and isinstance(raw[0][0], list) else raw
    return np.array(layer, dtype=np.int8)


def centroid_delta(prev: np.ndarray, curr: np.ndarray):
    ys, xs = np.where(prev != curr)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), len(xs)


def run_manual(actions: list[str], max_steps: int = 500):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    mod = load_agent()
    agent = mod.MyAgent(
        card_id="probe", game_id="ls20", agent_name="probe",
        ROOT_URL="http://x", record=False, arc_env=env, tags=[],
    )
    agent.MAX_ACTIONS = max_steps

    move_vec: dict[str, list[tuple[float, float]]] = {}
    walls: set[tuple[str, str]] = set()

    for i, act_name in enumerate(actions):
        if agent.is_done(agent.frames, agent.frames[-1]):
            break
        prev = grid_of(agent.frames[-1])
        sh_before = prev.tobytes()
        raw = agent.brain.choose_action if False else None
        from arcengine import GameAction as GA
        action = getattr(GA, act_name)
        action.reasoning = "manual"
        # use Agent.step pattern
        latest = agent.frames[-1]
        agent.action_counter += 1
        resp = agent.arc_env.step(action)
        agent.frames.append(resp.frame)
        curr = grid_of(agent.frames[-1])
        cen = centroid_delta(prev, curr)
        levels = int(agent.frames[-1].levels_completed or 0)
        state = agent.frames[-1].state
        if cen:
            move_vec.setdefault(act_name, []).append((cen[0], cen[1]))
        if prev.shape == curr.shape and np.array_equal(prev, curr):
            walls.add((sh_before[:32].hex(), act_name))
        print(
            f"  {i:3d} {act_name} levels={levels} state={state} "
            f"delta_px={cen[2] if cen else 0} cen={cen[:2] if cen else None}"
        )
        if state == GameState.WIN or levels > 0:
            print("  *** LEVEL UP / WIN ***")
            break
    print("\nLearned vectors (avg dx,dy):")
    for a, pts in move_vec.items():
        if pts:
            print(f"  {a}: dx={np.mean([p[0] for p in pts]):.1f} dy={np.mean([p[1] for p in pts]):.1f} n={len(pts)}")
    return agent


if __name__ == "__main__":
    # Level 1 hint: start ~(34,45), goal ~(34,10), rotate pad ~(19,30)
    # Start: move up toward y=10; may need rotate at (19,30) first (270->0 = 1x rotate)
    seq = []
    # go to rotate pad: left and up from (34,45) -> (19,30) rough path
    seq += ["ACTION3"] * 15  # left
    seq += ["ACTION1"] * 15  # up
    seq += ["ACTION4"] * 5   # right adjust
    seq += ["ACTION2"] * 5   # down adjust
    # try rotations at pad
    seq += ["ACTION4", "ACTION3", "ACTION1", "ACTION2"] * 8  # wiggle on pad
    # go to goal column
    seq += ["ACTION4"] * 15
    seq += ["ACTION1"] * 40
    run_manual(seq, max_steps=len(seq) + 20)
