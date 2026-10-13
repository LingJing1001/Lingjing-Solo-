"""Closed-book tu93 solver: parse the 64x64 frame, plan on the maze, emit arrows.

No baked answers, no engine access -- the only input is the frame.  The engine
returns the sprite mid-animation, so a raw blob read lags one action; we keep our
own position estimate and advance it by *simulating* the parsed maze, using the
frame only to (re)sync.

Two regimes:

* **Static levels (L0-L2).**  Only stationary hazards (colour-8 tiles) exist.  A
  hazard's line of sight is lethal until the hazard is stepped on from a side it
  is not facing; BFS over ``(node, cleared)`` finds the route.

* **Patrolled levels (L3+).**  Colour-12/13 tiles advance exactly one node per
  *valid* player move along a straight corridor, bouncing at the ends, and their
  heading is published in the frame itself (the offset of the odd pixel inside
  the 3x3 block).  Their node is therefore a deterministic function of the number
  of valid moves, so we can plan in space *and* time: the search refuses to stand
  where a patroller will be (see :func:`tu93_frame.plan_path`).  Waiting is not a
  lever -- A5 is a true no-op that does not advance the game clock, and a bump
  burns the level's arrow budget without moving the patrol -- so the plan is
  simply a walk whose length fits the budget.  On L3 the frame-only plan is
  ``[4,3,4,3,4,4,4,4,1,1,3,1,1,3,3,2,3]``: 17 arrows, identical to the reference
  solution, inside the 20-arrow timer.

Actions (tu93): 1=up 2=down 3=left 4=right 5=no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Optional

from lingjing_solo.core.roles import RoleTracker

from .tu93_frame import (
    DEADLY,
    DELTA,
    VOID,
    bfs_first_action,
    build_patrols,
    neighbours,
    observed_patrol_nodes,
    parse_frame,
    parse_frame_with_roles,
    patrol_next_cells,
    plan_path,
    transient_cells,
)

NOOP = 5
_BACK = {1: 2, 2: 1, 3: 4, 4: 3}

#: Off by default.  When set, an unreadable frame is retried with the player
#: colour *inferred* from the trajectory (``core.roles``) instead of the
#: hardcoded ``PLAYER``.  The default path does not construct a tracker at all,
#: so the shipped behaviour is bit-for-bit unchanged.
ROLES_FALLBACK = os.environ.get("LINGJING_ROLES_FALLBACK", "") == "1"


def _raw(frame: Any):
    """Unwrap an observation object to its raw array (accept both forms)."""
    return getattr(frame, "frame", frame)


def advance(pos: tuple[int, int], act: int, cells) -> tuple[int, int]:
    """Optimistic one-action step: legal iff the connector cell is not void."""
    dr, dc = DELTA[act]
    r, c = pos
    mid = (r + dr // 2, c + dc // 2)
    if (
        0 <= mid[0] < cells.shape[0]
        and 0 <= mid[1] < cells.shape[1]
        and int(cells[mid]) != VOID
    ):
        return (r + dr, c + dc)
    return pos


@dataclass
class Tu93ClosedBookSolver:
    """Per-game state; call :meth:`next_action` once per tick."""

    _level: Optional[int] = None
    _pos: Optional[tuple[int, int]] = None
    _prev_pos: Optional[tuple[int, int]] = None
    _last_act: Optional[int] = None
    _resyncs: int = 0
    _holds: int = 0
    _k: int = 0                                    # valid moves made this level
    _cleared: frozenset = frozenset()              # hazard tiles already cleared
    _patrols: Any = None                           # cached model for this level
    _broken: bool = False                          # patrol model failed -> reactive
    _reactive: bool = False                        # using the reactive patrol rule
    _plan: list = field(default_factory=list)      # committed space-time plan
    _goals: list = field(default_factory=list)     # last seen target nodes
    _roles: Any = None                             # RoleTracker, only under the fallback flag

    def reset(self) -> None:
        self._level = None
        self._pos = None
        self._prev_pos = None
        self._last_act = None
        self._resyncs = 0
        self._holds = 0
        self._k = 0
        self._cleared = frozenset()
        self._patrols = None
        self._broken = False
        self._reactive = False
        self._plan = []
        self._goals = []
        self._roles = None

    def _retry_with_roles(self, raw, obs, levels):
        """Re-read an unreadable frame using the inferred player colour.

        Only reached under ``LINGJING_ROLES_FALLBACK=1``.  The tracker sees the
        same frames the solver does, so the roles come from the live trajectory
        rather than from a table; if the inferred colour does not improve the
        read, the original observation is returned untouched.
        """
        if self._roles is None:
            self._roles = RoleTracker()
        spec = self._roles.observe(raw, action=self._last_act, levels=levels)
        if obs.ok or spec.avatar is None:
            return obs
        alt = parse_frame_with_roles(raw, spec)
        if alt.ok or (obs.player is None and alt.player is not None):
            return alt
        return obs

    # -- patrol bookkeeping -------------------------------------------------
    def _reconcile(self, patrols, seen) -> None:
        """Re-anchor the valid-move counter against what the frame currently shows.

        The frame lags one action, so once we have moved it displays node
        ``p.at(k-1)``; the very first frame of a level (``k = 0``) shows
        ``p.at(0)``.  A mismatch re-derives *k* from the observation; a total miss
        means the model is wrong, so we stand down and let CEAX take over.

        The comparison is over **node sets**, not positional lists.  Two things
        break a positional zip on L4 (and beyond): ``np.where`` returns patrol
        cells in row-major order, which reshuffles as the tiles slide past each
        other, and when two tiles land on the *same* node the frame collapses
        them to one cell (4 tiles -> 3 observed cells).  Both are invisible to a
        set comparison -- see ``state/_dbg_l4reconcile.py``: the zip finds *no*
        candidate phase from step 2 on, while the set match always contains the
        true phase.
        """
        if not patrols:
            return
        P = 1
        for p in patrols:
            P = P * p.period // gcd(P, p.period)
        seen_set = set(seen)
        if not seen_set:
            return                                  # nothing readable -- keep the count
        # Subset, not equality: a patroller sliding under a colour-8 tile (L5's
        # row-14 corridor passes over (14,10)) vanishes from the read, so the
        # observed set is a *subset* of the predicted one.  Equality would look
        # like a broken model and wedge the solver forever.
        cands = [j for j in range(P) if seen_set <= {p.at(j) for p in patrols}]
        if not cands:
            self._broken = True
            return
        expected = max(0, self._k - 1)
        if expected % P in cands:
            return                                  # consistent -- keep the count
        target = expected % P
        best = min(cands, key=lambda j: min((j - target) % P, (target - j) % P))
        self._k = best + 1

    def next_action(self, frame: Any, levels: int) -> Optional[tuple[int, str]]:
        """Return ``(action_id, reason)`` or ``None`` to fall through to CEAX."""
        if self._level is None:
            # first observation of the game (reset frame) -- no settle needed
            self._level = levels
        elif levels != self._level:
            # level transition: the frame we hold still shows the previous level
            # (one-action render lag), so push a no-op to pull the new frame in.
            self._level = levels
            self._pos = None
            self._prev_pos = None
            self._last_act = NOOP
            self._holds = 0
            self._k = 0
            self._cleared = frozenset()
            self._patrols = None
            self._broken = False
            self._reactive = False
            self._plan = []
            self._goals = []
            return NOOP, f"tu93-cb settle L{levels}"

        raw = _raw(frame)
        obs = parse_frame(raw)
        if ROLES_FALLBACK:
            obs = self._retry_with_roles(raw, obs, levels)
        # A patrol tile passing over the target's node hides the marker, which
        # would otherwise look like "no target"; the maze is static within a
        # level, so keep the last target set we ever saw.
        goals = list(obs.targets) or list(self._goals)
        if obs.targets:
            self._goals = list(obs.targets)
        if not obs.ok:
            if obs.player is None or not goals or "target" not in obs.note:
                # Unreadable frame -> let the rest of the chain (CEAX) take the tick.
                return None

        # resync: the read must equal our predicted node or the node we left
        if self._pos is None or obs.player not in (self._pos, self._prev_pos):
            if self._pos is not None:
                self._resyncs += 1
                self._plan = []                     # we are off-plan -> re-solve
            self._pos = obs.player

        # Build the patrol model on the level's first frame (v = 0), then reuse it.
        if self._patrols is None:
            built = build_patrols(obs.cells, obs.offset, raw)
            if built is None:
                return None                         # unreadable heading
            self._patrols, seen0 = built
            if self._patrols and not all(p.at(0) == n for p, n in zip(self._patrols, seen0)):
                self._broken = True
        patrols = self._patrols

        if patrols and not self._broken:
            seen = observed_patrol_nodes(raw, obs.cells, obs.offset)
            self._reconcile(patrols, seen)
            if self._broken:
                self._plan = []
            if self._plan:
                act = self._plan[0]                 # committed plan: replay it
                self._holds = 0
                why = f"tu93-cb L{levels} st-plan tgt={len(goals)}"
            else:
                acts = plan_path(
                    obs.cells, self._pos, goals, obs.facing, patrols, self._cleared
                )
                if acts:
                    self._plan = list(acts)
                    act = self._plan[0]
                    self._holds = 0
                    why = f"tu93-cb L{levels} st-plan tgt={len(goals)}"
                else:
                    act = NOOP                      # no space-time route: do not risk it
                    self._holds += 1
                    why = f"tu93-cb hold L{levels}"
        else:
            # No space-time model: either a static level (no patrols at all), or
            # a patrol we cannot model -- its path turns corners (L6), so no
            # ``k -> node`` function exists and the committed-plan approach
            # cannot be used.  Fall back to a tight reactive rule that is safe by
            # construction: a patrol advances exactly one node per valid player
            # move and publishes its heading, so the player is only ever killed by
            # *landing on* the node the patrol is about to enter -- standing on
            # its current node is a harmless swap, since the patrol has left.
            block: set = set()
            stray_hold = False
            if patrols:
                block |= patrol_next_cells(raw, obs.cells, obs.offset)
                block.discard(self._pos)
                self._reactive = True
            else:
                stray = transient_cells(raw, obs)
                stray.discard(self._pos)
                # A patrol tile on a level we believe has none -- do not move.
                stray_hold = bool(stray)

            if stray_hold:
                act = NOOP
            else:
                act = bfs_first_action(
                    obs.cells, self._pos, goals, obs.facing, obs.danger, block
                )
                if act is None or advance(self._pos, act, obs.cells) in block:
                    # Direct route blocked by the patrol's next node: sidestep.
                    # Prefer reversing the last move (deterministic, and it cannot
                    # be a no-op on the spot), else any safe legal move.
                    back = _BACK.get(self._last_act)
                    legal = [a for a, nxt2 in neighbours(obs.cells, self._pos, obs.facing)
                             if nxt2 not in block]
                    if back in legal:
                        act = back
                    elif legal:
                        act = legal[0]
                    else:
                        act = NOOP
            if act == NOOP:
                self._holds += 1
                why = f"tu93-cb hold L{levels}"
            else:
                self._holds = 0
                why = (f"tu93-cb L{levels} reactive tgt={len(goals)}" if patrols
                       else f"tu93-cb L{levels} tgt={len(goals)}")

        nxt = advance(self._pos, act, obs.cells) if act in DELTA else self._pos
        if nxt != self._pos:                        # a real move: clock + patrol tick
            self._k += 1
            if int(obs.cells[nxt]) in DEADLY:
                self._cleared = self._cleared | {nxt}
            if self._plan and self._plan[0] == act:
                self._plan = self._plan[1:]
        elif self._plan:
            self._plan = []                         # action was a bump -> plan is stale
        self._prev_pos = self._pos
        self._pos = nxt
        self._last_act = act
        return act, why
