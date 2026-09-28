"""CEAX-primary unknown agent — NO plugins, NO canned solvers.

Decision order:
  1) RESET when required
  2) CeaxController (hypothesis → experiment → exploit → transfer)
  3) Never ScriptBank / PluginRegistry / ls20 canned

Priority break-zero games (short human L1, click):
  lp85, vc33, sb26, tu93, s5i5, r11l, su15, bp35
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingjing_solo.core import SoloConfig, extract_grid, hash_grid
from lingjing_solo.perception import PerceptionEncoder
from lingjing_solo.transfer import CeaxController
from lingjing_solo.plugins import _norm_game

BUILD_TAG = "ceax-primary-unknown-v3.6"

# Short-L1 first (human baseline L1 steps)
PRIORITY_GAMES = (
    "lp85", "vc33", "sb26", "tu93", "s5i5", "r11l", "su15", "bp35",
    "tn36", "cd82", "ft09", "sc25",
)


def _canon(name: str) -> str:
    return str(name or "ACTION1").strip().upper()


def _as_game_action(action: Any) -> GameAction:
    if isinstance(action, GameAction):
        return action
    name = _canon(str(action or "ACTION1"))
    if hasattr(GameAction, name):
        return getattr(GameAction, name)
    mapping = {
        "RESET": 0,
        "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
        "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
    }
    return GameAction.from_id(mapping.get(name, 1))


def _valid_names(frame: FrameData) -> list[str]:
    raw = getattr(frame, "available_actions", None) or []
    out: list[str] = []
    for a in raw:
        if hasattr(a, "name"):
            name = str(a.name).upper()
        elif hasattr(a, "id"):
            name = f"ACTION{int(a.id)}"
        else:
            name = _canon(str(a))
        if name.isdigit():
            name = f"ACTION{int(name)}"
        out.append(name)
    return out


class MyAgent(Agent):
    """Pure CEAX unknown-capability agent."""

    MAX_ACTIONS: int = 800

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = SoloConfig(
            use_llm_advisor=False,
            enable_transfer=True,
            enable_mouse=True,
            enable_click_sweep=True,
            return_game_action=False,
        )
        self.encoder = PerceptionEncoder(self.cfg)
        self.ceax = CeaxController(self.cfg)
        self._prev_grid: Optional[np.ndarray] = None
        self._last_was_click = False
        self._levels_seen = 0
        gid = _norm_game(getattr(self, "game_id", ""))
        print(
            f"[{BUILD_TAG}] game={self.game_id} gid={gid} "
            f"priority={gid in PRIORITY_GAMES} max_actions={self.MAX_ACTIONS}",
            flush=True,
        )

    @property
    def name(self) -> str:
        return f"{super().name}.ceax-unknown.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        gid = _norm_game(
            getattr(self, "game_id", "") or getattr(latest_frame, "game_id", "")
        )

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.NOT_PLAYED:
                self.ceax.reset_game()
                self._prev_grid = None
                self._levels_seen = 0
            action = GameAction.RESET
            action.reasoning = f"{BUILD_TAG}:reset"
            return action

        grid = extract_grid(latest_frame)
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        valid = _valid_names(latest_frame)

        # Observe previous action outcome
        delta_px = 0
        if self._prev_grid is not None and grid is not None:
            delta_px = int((self._prev_grid != grid).sum())
        progressed = levels > self._levels_seen
        ghash = hash_grid(grid) if grid is not None else ""
        if self.action_counter > 0:
            self.ceax.observe_outcome(
                delta_pixels=delta_px,
                progressed=progressed,
                grid_hash=ghash,
                levels=levels,
                prev_grid=self._prev_grid,
                grid=grid,
            )
        if levels > self._levels_seen:
            self._levels_seen = levels

        # Full-frame objects for CEAX click experiments (not delta-only)
        objects = []
        if grid is not None:
            try:
                objects = self.encoder.segment(grid, delta_pixels=None)
            except Exception:
                objects = []

        act_name, xy, why = self.ceax.choose(
            valid_actions=valid or ["ACTION1"],
            objects=objects,
            grid=grid,
            grid_hash=ghash,
        )
        action = _as_game_action(act_name)
        self._last_was_click = act_name == "ACTION6"
        if action.is_complex():
            if xy is not None:
                action.set_data({"x": int(xy[0]), "y": int(xy[1])})
            else:
                action.set_data({"x": 32, "y": 32})

        if grid is not None:
            self._prev_grid = grid.copy()

        action.reasoning = f"{BUILD_TAG}:{why} L{levels}"
        if self.action_counter < 3 or self.action_counter % 25 == 0 or progressed:
            snap = self.ceax.snapshot()
            print(
                f"[{BUILD_TAG}] step={self.action_counter} L={levels} "
                f"{act_name} {why} pc={snap.get('kb_pc')} g={snap.get('kb_goal')} "
                f"corr={snap.get('kb_corr')} pos={snap.get('kb_pos')} "
                f"blocked={snap.get('kb_blocked')} failed={snap.get('kb_failed')} "
                f"skills={snap['skills']}",
                flush=True,
            )
        return action
