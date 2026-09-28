"""Benchmark ls20 levels with Lingjing EtherealRealm-Solo agent."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode


def run_ls20(max_steps: int = 400) -> dict:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    spec = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ag = mod.MyAgent(
        card_id="bench", game_id="ls20", agent_name="bench",
        ROOT_URL="x", record=False, arc_env=env, tags=[],
    )
    ag.MAX_ACTIONS = max_steps
    ag.timer = time.time()
    t0 = time.time()
    level_steps: dict[int, int] = {}
    last_level = 0
    while (
        not ag.is_done(ag.frames, ag.frames[-1])
        and ag.action_counter <= max_steps
    ):
        action = ag.choose_action(ag.frames, ag.frames[-1])
        action.reasoning = "bench"
        frame = ag.take_action(action)
        ag.append_frame(frame)
        ag.action_counter += 1
        lv = int(frame.levels_completed or 0)
        if lv > last_level:
            level_steps[lv] = ag.action_counter
            last_level = lv
    elapsed = time.time() - t0
    final = ag.frames[-1]
    return {
        "levels_completed": int(final.levels_completed or 0),
        "total_actions": ag.action_counter,
        "elapsed_sec": round(elapsed, 2),
        "level_step_milestones": level_steps,
        "state": str(final.state),
    }


if __name__ == "__main__":
    out = run_ls20(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    print(json.dumps(out, indent=2, ensure_ascii=False))
