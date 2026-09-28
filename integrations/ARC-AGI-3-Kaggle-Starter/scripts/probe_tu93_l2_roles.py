"""Offline tu93 frame dump: goal/player geometry under CEAX roles."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from arcengine import GameAction, GameState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from agents.agent import Agent
from lingjing_solo.core import extract_grid
from lingjing_solo.transfer.ceax_controller import CeaxController


class Dump(Agent):
    MAX_ACTIONS = 40

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ceax = CeaxController()
        self.prev = None

    def is_done(self, frames, latest):
        return latest.state is GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames, latest):
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest.state is GameState.NOT_PLAYED:
                self.ceax.reset_game()
            a = GameAction.RESET
            a.reasoning = "reset"
            return a

        grid = extract_grid(latest)
        levels = int(latest.levels_completed or 0)
        delta = 0
        if self.prev is not None and grid is not None:
            delta = int((self.prev != grid).sum())
            self.ceax.observe_outcome(
                delta_pixels=delta,
                progressed=False,
                levels=levels,
                prev_grid=self.prev,
                grid=grid,
            )

        valid = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        act, _, why = self.ceax.choose(valid_actions=valid, objects=[], grid=grid)
        if grid is not None and (
            self.action_counter < 5
            or self.action_counter in (10, 20, 30)
            or self.action_counter % 15 == 0
        ):
            play = grid[:58]
            hist = Counter(int(c) for c in play.flatten())
            g = self.ceax._kb_goal
            pc = self.ceax._kb_player_colors
            corr = self.ceax._kb_corridor
            def mean_xy(c):
                ys, xs = np.where(play == c)
                if len(xs) == 0:
                    return None
                return (float(xs.mean()), float(ys.mean()), int(len(xs)))
            print(
                f"s={self.action_counter} L={levels} act={act} delta={delta} "
                f"pc={pc}@{mean_xy(pc[0]) if pc else None} "
                f"g={g}@{mean_xy(g) if g is not None else None} "
                f"corr={corr} n={hist.get(corr,0) if corr is not None else None} "
                f"pos={self.ceax._kb_prev_pos} motion={self.ceax._kb_motion_pos} "
                f"blocked={len(self.ceax._kb_blocked)} step={self.ceax._kb_step}",
                flush=True,
            )
            if g is not None and pc:
                gyx = mean_xy(g)
                pyx = mean_xy(pc[0])
                if gyx and pyx:
                    print(
                        f"  dist≈({gyx[0]-pyx[0]:.1f},{gyx[1]-pyx[1]:.1f}) "
                        f"hist_top={hist.most_common(6)}",
                        flush=True,
                    )

        if grid is not None:
            self.prev = grid.copy()
        a = getattr(GameAction, act)
        a.reasoning = why
        return a


def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("tu93")
    agent = Dump(
        card_id="probe",
        game_id="tu93",
        agent_name="probe.tu93",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["probe"],
    )
    agent.main()
    final = agent.frames[-1]
    print(f"=== L={final.levels_completed} state={final.state} act={agent.action_counter}")


if __name__ == "__main__":
    main()
