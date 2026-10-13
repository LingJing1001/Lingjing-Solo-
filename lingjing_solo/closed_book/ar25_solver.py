"""Closed-book AR25 planner: translate the selected shape until its mirror
image covers every target cell.

No baked answers, no engine access — the only input is the frame.  Each call
re-derives the plan from scratch, so it is a closed loop that tolerates walls
and clamping.

Actions (AR25): 1=UP 2=DOWN 3=LEFT 4=RIGHT 5=CYCLE 6=CLICK 7=UNDO.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from lingjing_solo.core.roles import RoleTracker

from .ar25_frame import Ar25FrameObs, Cell, parse_frame, parse_frame_with_roles

ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT, ACT_CYCLE = 1, 2, 3, 4, 5

# hard ceiling on how far we will chase a target (grid is 21 wide)
_MAX_SHIFT = 21

#: Off by default (see ``tu93_solver.ROLES_FALLBACK``): with it set, a frame whose
#: shape/cursor cannot be read on the hardcoded palette is retried with the
#: colours inferred from the trajectory.
ROLES_FALLBACK = os.environ.get("LINGJING_ROLES_FALLBACK", "") == "1"


def solve_shift(obs: Ar25FrameObs, shape: list[Cell]) -> Optional[tuple[int, int]]:
    """Smallest-cost (dr, dc) with mirror(shape + shift) superset of targets.

    Cost is Manhattan distance, matching how the level is actually played.
    Returns ``(0, 0)`` when the shape already covers everything.
    """
    if not obs.ok or obs.axis_coord is None or not obs.target_cells:
        return None
    n = obs.n
    best: Optional[tuple[int, int]] = None
    best_cost: Optional[int] = None
    for dr in range(-_MAX_SHIFT, _MAX_SHIFT + 1):
        for dc in range(-_MAX_SHIFT, _MAX_SHIFT + 1):
            cost = abs(dr) + abs(dc)
            if best_cost is not None and cost >= best_cost:
                continue
            shifted = [(r + dr, c + dc) for r, c in shape]
            if any(not (0 <= r < n and 0 <= c < n) for r, c in shifted):
                continue
            if obs.target_cells <= obs.mirror_cells(shifted):
                best, best_cost = (dr, dc), cost
    return best


@dataclass
class Ar25ClosedBookSolver:
    """Per-game state; call :meth:`next_action` once per tick."""

    _level: Optional[int] = None
    _sig: Optional[frozenset] = None
    _stall: int = 0
    _vertical_first: bool = False
    _roles: Any = None                             # RoleTracker, only under the fallback flag
    _last_act: Optional[int] = None                # action that produced the newest frame

    def reset(self) -> None:
        self._level = None
        self._sig = None
        self._stall = 0
        self._vertical_first = False
        self._roles = None
        self._last_act = None

    def _retry_with_roles(self, frame, obs, levels):
        """Re-read a frame the hardcoded palette could not resolve.

        Only reached under ``LINGJING_ROLES_FALLBACK=1``.  ``shape`` must be
        readable for anything to be planned, and ``selected`` (which needs the
        cursor) decides *which* shape moves -- so an inference that names either
        one differently is worth a second parse.

        The tracker is fed the action that produced this frame: without it the
        inference has no way to tell the commanded shape from its reflection
        (both translate, and on this game they translate together on one axis).
        """
        if self._roles is None:
            self._roles = RoleTracker()
        spec = self._roles.observe(frame, action=self._last_act, levels=levels)
        if spec.avatar is None:
            return obs
        if obs.ok and obs.selected is not None:
            return obs
        alt = parse_frame_with_roles(frame, spec)
        if alt.selected is not None or (not obs.shapes and alt.shapes):
            return alt
        return obs

    def next_action(self, frame: Any, levels: int) -> Optional[tuple[int, str]]:
        """Return ``(action_id, reason)`` or ``None`` to fall through to CEAX."""
        act = self._plan_action(frame, levels)
        # remember what we asked for: the newest frame is that action's render,
        # which is the anchor the fallback's agreement test needs.
        self._last_act = act[0] if act is not None else None
        return act

    def _plan_action(self, frame: Any, levels: int) -> Optional[tuple[int, str]]:
        if self._level != levels:
            self._level = levels
            self._sig = None
            self._stall = 0
            self._vertical_first = False

        obs = parse_frame(frame)
        if ROLES_FALLBACK:
            obs = self._retry_with_roles(frame, obs, levels)

        # The cursor must be sitting on a shape for arrow keys to move it.
        shape = obs.selected
        if shape is None and len(obs.shapes) == 1:
            shape = obs.shapes[0]
        if shape is None:
            return None

        sig = frozenset(shape)
        if self._sig is not None and sig == self._sig:
            # the last arrow produced no movement -> the other axis may be blocked
            self._stall += 1
            if self._stall >= 1:
                self._vertical_first = not self._vertical_first
        else:
            self._stall = 0
        self._sig = sig

        shift = solve_shift(obs, list(shape))
        if shift is None:
            return None
        dr, dc = shift
        if dr == 0 and dc == 0:
            return None  # already covering; nothing useful to do

        horiz = (ACT_LEFT if dc < 0 else ACT_RIGHT) if dc else None
        vert = (ACT_UP if dr < 0 else ACT_DOWN) if dr else None
        order = (vert, horiz) if self._vertical_first else (horiz, vert)
        for act in order:
            if act is not None:
                tag = "V" if act in (ACT_UP, ACT_DOWN) else "H"
                return act, f"ar25-cb shift=({dr},{dc}) step={tag}"
        return None
