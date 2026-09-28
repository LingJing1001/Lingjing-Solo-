"""Probe ls20: learn moves, test path to rotation pad and goal."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState


def grid_of(frame) -> np.ndarray:
    raw = frame.frame or []
    if not raw:
        return np.zeros((64, 64), dtype=np.int8)
    layer = raw[-1] if raw and isinstance(raw[0], list) and isinstance(raw[0][0], list) else raw
    g = np.array(layer, dtype=np.int8)
    if g.ndim == 1:
        return g.reshape(64, 64)
    return g


def load_agent():
    spec = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def player_centroid(g: np.ndarray) -> tuple[float, float] | None:
    """Agent sprite ~ color 8/12 bright blob; use delta tracking instead for precision."""
    return None


def shape_signature(g: np.ndarray) -> tuple:
    """Bottom-left HUD shape area (htkmubhry display)."""
    patch = g[48:64, 0:24]
    return tuple(patch.flatten().tolist())


def run_sequence(actions: list[str], max_steps: int = 500):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    mod = load_agent()
    agent = mod.MyAgent(
        card_id="p", game_id="ls20", agent_name="probe",
        ROOT_URL="x", record=False, arc_env=env, tags=[],
    )
    agent.MAX_ACTIONS = 0
    agent.main()  # reset only - actually runs full? MAX_ACTIONS=0 might still run once

    # Fresh: manually step
    env2 = arc.make("ls20")
    agent2 = mod.MyAgent(
        card_id="p", game_id="ls20", agent_name="probe",
        ROOT_URL="x", record=False, arc_env=env2, tags=[],
    )

    pos = None
    sig0 = shape_signature(grid_of(agent2.frames[-1]))
    print("initial sig len", len(sig0), "levels", agent2.frames[-1].levels_completed)

    for i, name in enumerate(actions):
        prev = grid_of(agent2.frames[-1])
        action = getattr(GameAction, name)
        action.reasoning = "probe"
        frame = agent2.take_action(action)
        if not frame:
            break
        agent2.append_frame(frame)
        agent2.action_counter += 1
        curr = grid_of(frame)
        ys, xs = np.where(prev != curr)
        cen = (float(xs.mean()), float(ys.mean())) if len(xs) else None
        if cen and (pos is None or np.hypot(cen[0]-pos[0], cen[1]-pos[1]) > 0.5):
            if pos:
                print(f"  move {name}: ({pos[0]:.0f},{pos[1]:.0f}) -> ({cen[0]:.0f},{cen[1]:.0f})")
            pos = cen
        sig = shape_signature(curr)
        rot_changed = sig != shape_signature(prev)
        lv = frame.levels_completed
        print(f"{i:3d} {name} L={lv} state={frame.state} pos={cen} rot_delta={rot_changed}")
        if lv > 0 or frame.state == GameState.WIN:
            print("SUCCESS level up!")
            return True
    return False


if __name__ == "__main__":
    # Level1: start ~(34,45) rot pad (19,30) goal (34,10)
    # Learn: go left to x~19, up to y~30, rotate, go right to x~34, up to y~10
    seq = []
    seq += ["ACTION3"] * 3   # left - will calibrate count
    for n in [3, 5, 8, 12, 15]:
        test = ["ACTION3"] * n + ["ACTION1"] * 35
        print("=== try left", n, "then up ===")
        if run_sequence(test[:80]):
            break
