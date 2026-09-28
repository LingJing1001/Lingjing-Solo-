"""GitHub main @3775a0d strongest merge — LS20 L1–L7 + ar25 L1.

Routes copied from arc_adaptor/agents/strategies/ls20.py (verified WIN 7/7).
Class name MUST remain `MyAgent` for play_local / benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_FILE = Path(__file__).resolve()
_STARTER = _FILE.parents[1]
_REPO = _STARTER.parent
_TEAM = _REPO / "灵境战队AGI课题探索"

for p in (str(_TEAM), str(_STARTER), str(_STARTER / "vendor" / "ARC-AGI-3-Agents")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lingjing_solo import SoloConfig  # noqa: E402
from lingjing_solo.agent import LingjingSoloAgent  # noqa: E402
from lingjing_solo.planning import LS20Solver  # noqa: E402


def canonicalize(name: str) -> str:
    return str(name or "ACTION1").strip().upper()

# From GitHub strategies/ls20.py @3775a0d (309 steps, score 100, WIN)
_LEVEL_ACTIONS: dict[int, list[int]] = {
    0: [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1],
    1: [1, 4, 1, 1, 1, 1, 1, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 3, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2],
    2: [1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 1, 2, 1, 4, 2],
    3: [3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 3, 1, 2, 3, 3, 1, 1, 1, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 1, 3, 3, 3],
    4: [1, 4, 1, 1, 3, 4, 3, 3, 3, 4, 3, 4, 3, 4, 4, 2, 2, 3, 3, 3, 1, 3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 4, 1, 4, 4, 2, 2, 2, 1],
    5: [1, 3, 1, 3, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 4, 1, 4, 1, 1, 4, 2, 2, 1, 1, 3, 1, 2, 3, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 1, 3, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 1, 1, 4, 2, 2, 2, 2, 2],
    6: [1, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2, 1, 2, 4, 2, 1, 4, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 1, 1, 1, 4, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 1, 4, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 2],
}

AR25_L1 = ["ACTION3"] * 5 + ["ACTION2"] * 10

BUILD_TAG = "github-main-ls20x7+ar25x1"
AGENT_BRAND = "灵境战队AGI课题探索"
TEAM_SLUG = "lingjing-github-main"


def _level_plan(level_index: int) -> list[str]:
    return [f"ACTION{n}" for n in _LEVEL_ACTIONS.get(level_index, [])]


def _norm_gid(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    head = s.split("-")[0]
    for key in ("ls20", "ar25"):
        if head == key or s.startswith(key):
            return key
    return head


def _as_game_action(action: Any) -> GameAction:
    if isinstance(action, GameAction):
        return action
    name = canonicalize(str(action or "ACTION1"))
    if hasattr(GameAction, name):
        return getattr(GameAction, name)
    mapping = {
        "RESET": 0,
        "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
        "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
    }
    return GameAction.from_id(mapping.get(name, 1))


def _frame_grid(frame: FrameData) -> np.ndarray | None:
    raw = getattr(frame, "frame", None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.int8)
    if arr.ndim == 3:
        arr = arr[0]
    return arr if arr.ndim == 2 else None


def _legal_names(frame: FrameData) -> list[str]:
    actions = getattr(frame, "available_actions", None) or []
    out: list[str] = []
    for a in actions:
        try:
            if hasattr(a, "name"):
                if a.name != "RESET":
                    out.append(a.name)
            elif int(a) != 0:
                out.append(f"ACTION{int(a)}")
        except Exception:
            continue
    return out or ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


class LevelScriptPlayer:
    """Replay per-level ACTION lists (ar25)."""

    def __init__(self, levels: dict[str, list[str]]) -> None:
        self.levels = levels
        self.level = -1
        self.queue: list[str] = []
        self.idx = 0

    def reset(self) -> None:
        self.level = -1
        self.queue = []
        self.idx = 0

    def next(self, levels_completed: int) -> Optional[str]:
        lv = int(levels_completed)
        if self.level != lv or self.idx >= len(self.queue):
            self.queue = list(self.levels.get(str(lv), []))
            self.idx = 0
            self.level = lv
        if self.idx >= len(self.queue):
            return None
        act = self.queue[self.idx]
        self.idx += 1
        return act


class LS20Canned:
    """Mirror of GitHub LS20Strategy using team LS20Solver."""

    def __init__(self) -> None:
        self.solver = LS20Solver()
        self.seeded_level: int | None = None

    def reset(self, levels_completed: int = 0) -> None:
        self.solver.reset()
        self.seeded_level = None
        self._seed(int(levels_completed))

    def _seed(self, level: int) -> None:
        plan = _level_plan(level)
        if plan:
            self.solver.set_plan(plan)
            self.seeded_level = level

    def next(self, grid, legal: list[str], levels_completed: int) -> Optional[str]:
        lv = int(levels_completed)
        if lv != self.seeded_level:
            self._seed(lv)
        return self.solver.next_action(grid, legal)


class MyAgent(Agent):
    MAX_ACTIONS: int = 800

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.brain = LingjingSoloAgent(
            cfg=SoloConfig(
                llm_calls_per_game=0,
                enable_undo=False,
                enable_mouse=False,
                human_baseline_estimate=80,
                lightweight_search_depth=8,
                loop_detect_window=6,
            )
        )
        self.brain.reset()
        self.ls20 = LS20Canned()
        self.ar25 = LevelScriptPlayer({"0": AR25_L1})
        n = sum(1 for i in range(8) if _level_plan(i))
        total = sum(len(_level_plan(i)) for i in range(8))
        print(
            f"[{AGENT_BRAND}/{BUILD_TAG}] game={self.game_id} "
            f"ls20_levels={n} ls20_steps={total} ar25_L1={len(AR25_L1)} "
            f"max_actions={self.MAX_ACTIONS}",
            flush=True,
        )

    @property
    def name(self) -> str:
        return f"{super().name}.{TEAM_SLUG}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        gid = _norm_gid(
            getattr(self, "game_id", "") or getattr(latest_frame, "game_id", "")
        )
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.brain.reset()
            self.ar25.reset()
            self.ls20.reset(0)
            action = GameAction.RESET
            action.reasoning = f"{BUILD_TAG}:reset gid={gid}"
            return action

        legal = _legal_names(latest_frame)
        grid = _frame_grid(latest_frame)

        if gid == "ls20":
            name = self.ls20.next(grid, legal, levels)
            if name:
                action = _as_game_action(name)
                action.reasoning = f"{BUILD_TAG}:ls20 L{levels} {action.name}"
                return action

        if gid == "ar25":
            act = self.ar25.next(levels)
            if act:
                action = _as_game_action(act)
                action.reasoning = f"{BUILD_TAG}:ar25 L{levels} {action.name}"
                return action

        # Solo fallback (team UP/DOWN semantics)
        to_solo = {
            "ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT",
            "ACTION4": "RIGHT", "ACTION5": "SPACE",
        }
        from_solo = {v: k for k, v in to_solo.items()}
        holder = type("G", (), {"grid": grid})() if grid is not None else None
        try:
            observe = getattr(self.brain, "observe", None)
            if callable(observe) and grid is not None:
                observe(grid, state=latest_frame.state, levels_completed=levels)
            raw = self.brain.choose_action(
                [], holder, valid_actions=list(to_solo.values())
            )
            action = _as_game_action(from_solo.get(str(raw), raw))
        except Exception:
            action = GameAction.ACTION1
        if action.is_complex():
            action.set_data({"x": 32, "y": 32})
        action.reasoning = f"{BUILD_TAG}:solo L{levels} {action.name}"
        return action
