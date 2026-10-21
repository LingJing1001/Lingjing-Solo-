"""The engine, wrapped so the sandbox can talk to it in one line.

Deliberately thin. Everything that decides anything lives elsewhere: the model
decides, `sandbox.py` enforces cost and refusal, this file only reports what the
engine did.

The local engine exposes exactly four things we need -- the frame layers, the
current level count, the game state, and which actions are available -- so
`StepResult` is those four and nothing else. In particular there is no
`gameplay_changed`: the engine does not tell us whether what moved was a timer
bar or the board, so the sandbox infers it by cropping the border.
"""
from __future__ import annotations

import numpy as np

from .sandbox import MODEL_TO_ENGINE


def _reverse_engine_names() -> dict[str, str]:
    """ACTION1 -> UP, ... , ACTION6 -> MOUSE, RESET -> RESET."""
    out: dict[str, str] = {}
    for model_name, engine_name in MODEL_TO_ENGINE.items():
        out.setdefault(engine_name, model_name if model_name != "CLICK" else "MOUSE")
    # keep the shortest, most conventional label for each engine action
    out["ACTION6"] = "MOUSE"
    out["ACTION3"] = "LEFT"
    out["ACTION4"] = "RIGHT"
    return out


ENGINE_TO_MODEL = _reverse_engine_names()


class StepResult:
    __slots__ = ("frame", "flags", "state", "level", "grid")

    def __init__(self, frame, grid, flags, state, level):
        self.frame = frame
        self.grid = grid
        self.flags = flags
        self.state = state
        self.level = level


class ArcEnv:
    """One game, stepped synchronously."""

    def __init__(self, game_id: str, *, seed: int = 0, mode: str = "NORMAL"):
        import arc_agi
        from arc_agi import OperationMode

        self.game_id = game_id
        self.arc = arc_agi.Arcade(
            operation_mode=getattr(OperationMode, mode, OperationMode.NORMAL))
        self.env = self.arc.make(game_id, render_mode=None)
        self.frame = None
        self.level = 0
        self.steps = 0

    # ------------------------------------------------------------------ reads

    @staticmethod
    def grid_of(frame) -> list[list[int]]:
        layers = getattr(frame, "frame", None)
        if not layers:
            return []
        return np.asarray(layers[-1]).astype(int).tolist()

    def available(self, frame) -> list[str]:
        raw = getattr(frame, "available_actions", None) or []
        names = []
        for a in raw:
            if isinstance(a, bool):
                continue
            if isinstance(a, int):
                key = f"ACTION{int(a)}"
            else:
                key = getattr(a, "name", None) or str(a)
                if key.isdigit():
                    key = f"ACTION{key}"
            names.append(ENGINE_TO_MODEL.get(key, key))
        return names

    # ------------------------------------------------------------------ steps

    def reset(self) -> StepResult:
        from arcengine import GameAction

        raw = self.env.step(GameAction.RESET, data={}, reasoning={"note": "level start"})
        self.steps = 0
        return self._wrap(raw, previous_level=self.level, explicit=None)

    def step(self, engine_name: str, spec: dict | None = None) -> StepResult:
        from arcengine import GameAction

        spec = spec or {}
        action = getattr(GameAction, engine_name, None)
        if action is None:
            raise ValueError(f"unknown engine action {engine_name!r}")
        data = {}
        if engine_name == "ACTION6":
            # the engine takes screen coordinates; the model gave row/col
            data = {"x": int(spec.get("col", 0)), "y": int(spec.get("row", 0))}
        raw = self.env.step(action, data=data, reasoning={"note": "model action"})
        self.steps += 1
        return self._wrap(raw, previous_level=self.level, explicit=engine_name)

    def _wrap(self, raw, *, previous_level: int, explicit: str | None) -> StepResult:
        from arcengine import GameState

        grid = self.grid_of(raw)
        level = int(getattr(raw, "levels_completed", 0) or 0)
        state = getattr(raw, "state", None)
        state_name = getattr(state, "name", str(state))

        flags: dict = {}
        if state_name == "GAME_OVER":
            flags["game_over"] = True
        if state_name == "WIN":
            flags["run_complete"] = True
        if level > previous_level:
            flags["level_completed"] = True
            flags["from_level"] = previous_level
        if explicit == "RESET":
            flags["reset"] = True

        self.frame = raw
        self.level = level
        return StepResult(raw, grid, flags, state_name, level)

    # ------------------------------------------------------------------ misc

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - best effort
            pass
