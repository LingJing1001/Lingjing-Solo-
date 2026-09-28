"""Diagnose Solo+L6 without coverage: does TransferLayer accumulate skills?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
import arc_agi
from arc_agi import OperationMode
from lingjing_solo import SoloConfig
from lingjing_solo.agent import LingjingSoloAgent
from lingjing_solo.plugins import _norm_game


def _as_game_action(action):
    if isinstance(action, GameAction):
        return action
    name = str(action or "ACTION1").strip().upper()
    if hasattr(GameAction, name):
        return getattr(GameAction, name)
    return GameAction.from_id(1)


class SoloOnlyAgent(Agent):
    MAX_ACTIONS = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = LingjingSoloAgent(
            cfg=SoloConfig(
                use_llm_advisor=False,
                enable_transfer=True,
                enable_mouse=True,
                enable_click_sweep=True,
                return_game_action=False,
                human_baseline_estimate=80,
                hard_step_multiplier=4.0,
            )
        )

    def is_done(self, frames, latest_frame):
        if latest_frame.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.brain.reset(env=type("E", (), {"game_id": _norm_game(self.game_id)})())
            return GameAction.RESET
        raw = self.brain.choose_action(frames, latest_frame)
        action = _as_game_action(raw)
        if action.is_complex() and not getattr(action, "data", None):
            action.set_data({"x": 32, "y": 32})
        return action


def main():
    games = ["vc33", "sp80", "ls20", "ar25", "wa30", "ft09", "tn36", "sc25"]
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    rows = []
    for gid in games:
        env = arc.make(gid)
        agent = SoloOnlyAgent(
            card_id="solo-diag",
            game_id=gid,
            agent_name=f"solo.{gid}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["diag"],
        )
        agent.main()
        final = agent.frames[-1]
        m = agent.brain.transfer.metrics() if agent.brain and agent.brain.transfer else {}
        rationale = getattr(agent.brain, "last_rationale", "")
        row = {
            "game_id": gid,
            "levels": int(final.levels_completed or 0),
            "actions": int(agent.action_counter),
            "state": final.state.name if hasattr(final.state, "name") else str(final.state),
            "last_rationale": rationale,
            "transfer": m,
        }
        rows.append(row)
        print(row, flush=True)

    out = ROOT / "ui" / "static" / "solo_l6_diag_200.json"
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    nz = sum(1 for r in rows if r["levels"] > 0)
    skills = sum(1 for r in rows if (r.get("transfer") or {}).get("skills", 0) > 0)
    print(json.dumps({"nonzero": nz, "with_skills": skills, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
