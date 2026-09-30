"""Benchmark agent: Coverage + Solo + L6 Transfer, NO canned plugins.

Used to measure true unknown-environment coverage (all 25 games treated as unknown).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingjing_solo import SoloConfig
from lingjing_solo.agent import LingjingSoloAgent
from lingjing_solo.plugins.coverage import CoverageBootstrap
from lingjing_solo.plugins import _norm_game

BUILD_TAG = "unknown-solo+l6-no-plugins"


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
            out.append(str(a.name).upper())
        else:
            out.append(_canon(str(a)))
    return out


class MyAgent(Agent):
    MAX_ACTIONS: int = 800

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coverage = CoverageBootstrap(max_probe_steps=120, click_grid=6)
        self.brain = LingjingSoloAgent(
            cfg=SoloConfig(
                use_llm_advisor=False,
                llm_calls_per_game=0,
                enable_undo=False,
                enable_mouse=True,
                enable_click_sweep=True,
                enable_transfer=True,
                return_game_action=False,
                human_baseline_estimate=120,
                hard_step_multiplier=6.0,
                lightweight_search_depth=10,
                loop_detect_window=5,
                click_stall_threshold=3,
                click_warmup=10,
            )
        )
        print(
            f"[{BUILD_TAG}] game={self.game_id} max_actions={self.MAX_ACTIONS}",
            flush=True,
        )

    @property
    def name(self) -> str:
        return f"{super().name}.unknown-l6.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        gid = _norm_game(getattr(self, "game_id", "") or getattr(latest_frame, "game_id", ""))

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.NOT_PLAYED:
                self.coverage.reset()
            try:
                self.brain.reset(env=type("E", (), {"game_id": gid})())
            except Exception:
                self.brain.reset()
            action = GameAction.RESET
            action.reasoning = f"{BUILD_TAG}:reset"
            return action

        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        self.coverage.note_progress(levels)
        valid = _valid_names(latest_frame)

        cov = self.coverage.suggest(valid, levels)
        if cov is not None:
            act_name, xy, why = cov
            action = _as_game_action(act_name)
            if xy is not None and action.is_complex():
                action.set_data({"x": int(xy[0]), "y": int(xy[1])})
            action.reasoning = f"{BUILD_TAG}:{why}"
            return action

        try:
            raw = self.brain.choose_action(frames, latest_frame, valid_actions=valid or None)
            action = _as_game_action(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"[{BUILD_TAG}] solo error: {exc}", flush=True)
            action = GameAction.ACTION1

        if action.is_complex() and not getattr(action, "data", None):
            action.set_data({"x": 32, "y": 32})
        action.reasoning = f"{BUILD_TAG}:solo L{levels}"
        return action
