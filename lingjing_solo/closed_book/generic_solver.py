"""Generic closed-book solver -- today's generic primitives actually steering play.

Nothing here is game-specific.  The board geometry comes from
:mod:`lingjing_solo.core.lattice` (cell size *and* phase recovered from the pixels)
and the actors come from :mod:`lingjing_solo.core.roles` (roles **measured** from
the frame trajectory, never read off a palette table).

Why this module exists
----------------------
Steps 4 and 5 of the M1 remediation produced ``core/lattice.py`` and
``core/roles.py``, but both were only reachable from the two hand-written
solvers' *fallback* branch -- which never fires, because tu93/ar25 parse fine
with their hard-coded palettes.  Measured consequence: 1 612 lines of new
production code that changed exactly zero decisions.

This solver puts them on the decision path.  It is wired into the closed-book
branch of ``agent/my_agent.py`` for every game that has no dedicated solver, so
it runs on real frames and its effect is measurable.

Decision ladder (first rung that yields a step wins)
---------------------------------------------------
1. ``goal`` known          -> walk to the nearest goal cell.
2. something to touch      -> walk to the nearest non-background object cell.
3. nothing left to touch   -> walk to the nearest cell never visited this level.
4. cannot read an avatar   -> ``None`` (fall through to CEAX).

Across all rungs: lethal cells are never entered; terrain is avoided on the first
attempt and only given up if no path exists without crossing it.  Ties are broken
towards the destination with the most free neighbours, so the agent does not walk
itself into a corner -- these games end the episode on the first death, so a step
that cannot be undone is the most expensive kind of mistake.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..core.lattice import as_grid, detect_cell_lattice, to_cell_grid
from ..core.roles import ACTION_DIR, RoleTracker

__all__ = ["GenericClosedBookSolver", "SolverStatus"]

#: ARC-AGI-3 direction keys, as (action id, (drow, dcol)).
_DIRS = ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1)))

#: How many distinct object colours to consider before giving up on rung 2.
MAX_OBJECT_COLOURS = 8

#: Cell-grid shapes vary when the lattice is re-detected; anything larger than
#: this is not a board we can search sensibly.
MAX_CELLS = 96 * 96


class SolverStatus:
    """Small mutable counter block, kept out of the solver's ``__slots__``."""

    __slots__ = ("steps", "picks", "skipped", "reasons")

    def __init__(self):
        self.steps = 0
        self.picks = 0
        self.skipped = 0
        self.reasons = {}

    def note(self, reason):
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def to_dict(self):
        return {
            "steps": self.steps,
            "picks": self.picks,
            "skipped": self.skipped,
            "reasons": dict(sorted(self.reasons.items())),
        }


# --------------------------------------------------------------------------- #
# cell-grid geometry (pure numpy, no game knowledge)
# --------------------------------------------------------------------------- #
def _direction_anchor(frame: Any, last: Optional[int]) -> Optional[int]:
    """The direction key that *produced* ``frame``, or ``None`` if there was none.

    Read straight off ``FrameData.action_input`` rather than remembering our own
    last move: the anchor has to stay correct while another layer (CEAX) is the
    one choosing, and a solver that only remembers its own picks never learns
    anything -- it never picks, so it never has an anchor, so it never picks.
    Measured: that chicken-and-egg deadlock left the first version at a 0% pick
    rate on every game.

    A click (or RESET) carries no direction, so it yields ``None``; the roles
    search a short lag window and simply find no agreement there.
    """
    raw = getattr(frame, "action_input", None)
    if raw is not None:
        value = getattr(raw, "id", raw)
        value = getattr(value, "value", value)
        try:
            return int(value) if int(value) in ACTION_DIR else None
        except (TypeError, ValueError):
            pass
    return last if last in ACTION_DIR else None


def _largest_blob(cells: np.ndarray, colour: int) -> Optional[list]:
    """Cells of the largest 4-connected component of ``colour`` (``None`` if absent).

    A colour can hold several disjoint components (ar25's avatar colour also
    paints a panel); the commanded piece is the big one.
    """
    if cells is None or cells.size == 0:
        return None
    mask = cells == colour
    if not mask.any():
        return None
    h, w = cells.shape
    seen = np.zeros(mask.shape, bool)
    best: list = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        comp = []
        stack = [(y0, x0)]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    return best or None


def _centroid(blob: list) -> tuple:
    ys = sum(c[0] for c in blob) / len(blob)
    xs = sum(c[1] for c in blob) / len(blob)
    # snap to the blob's own cells so the start is always a real cell
    return min(blob, key=lambda c: (c[0] - ys) ** 2 + (c[1] - xs) ** 2)


def _distance_field(blocked: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Multi-source BFS: steps from each free cell to the nearest ``sources`` cell.

    Blocked cells and unreachable free cells stay at ``-1``.  Sources are free by
    construction (callers mask out blocked targets).
    """
    dist = np.full(blocked.shape, -1, np.int32)
    h, w = blocked.shape
    ys, xs = np.nonzero(sources & ~blocked)
    if ys.size == 0:
        return dist
    frontier = list(zip(ys.tolist(), xs.tolist()))
    for y, x in frontier:
        dist[y, x] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for y, x in frontier:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] < 0 and not blocked[ny, nx]:
                    dist[ny, nx] = d
                    nxt.append((ny, nx))
        frontier = nxt
    return dist


def _free_degree(blocked: np.ndarray, cell: tuple) -> int:
    """Free 4-neighbours of ``cell`` -- the anti-corner tie-breaker."""
    h, w = blocked.shape
    y, x = cell
    return sum(
        1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= y + dy < h and 0 <= x + dx < w and not blocked[y + dy, x + dx]
    )


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """``mask`` plus every cell orthogonally touching it."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _step_towards(blocked: np.ndarray, start: tuple, targets: np.ndarray,
                  inert: Optional[set] = None):
    """First key of a shortest path from ``start`` to any ``targets`` cell.

    Returns ``(action, destination)`` or ``None``.  Among equally near successors
    the one with the most free neighbours wins, then the smallest action id, so
    the choice is deterministic.

    ``inert`` holds ``(cell, action)`` pairs already observed to produce no
    movement -- pressing a wall.  Without it the agent keeps re-picking the same
    direction and mashes one key: measured 28 consecutive "up" presses on sp80 and
    127 consecutive "down" on ls20, which is not path-following, it is a wedged
    key.
    """
    if not targets.any():
        return None
    if targets[start]:
        return None                       # already standing on it
    dist = _distance_field(blocked, targets)
    h, w = blocked.shape
    y, x = start
    best = None
    for action, (dy, dx) in _DIRS:
        if inert and (start, action) in inert:
            continue
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w):
            continue
        if blocked[ny, nx] or dist[ny, nx] <= 0:
            continue
        # Nearest first; among equals the roomier cell, then the smaller action id.
        # Leaving the distance out of the sort is a real bug, not a style choice:
        # the neighbour *filter* alone then lets any reachable cell win, so the
        # agent wanders instead of following the path it just computed.
        score = (-int(dist[ny, nx]), _free_degree(blocked, (ny, nx)), -action)
        if best is None or score > best[0]:
            best = (score, action, (ny, nx))
    if best is None:
        return None
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# the solver
# --------------------------------------------------------------------------- #
class GenericClosedBookSolver:
    """Frame-only, palette-free solver.  ``next_action`` -> ``(action, reason)`` | None."""

    __slots__ = (
        "_roles", "_level", "_last", "_seen", "_shape", "_status", "_max_cell",
        "_use_lethal", "_terrain_hard", "_prev_cell", "_prev_action", "_inert",
    )

    def __init__(self, warmup: int = 4, max_cell: int = 8, use_lethal: bool = True,
                 terrain_hard: bool = False):
        self._roles = RoleTracker(warmup=warmup)
        self._level = None
        self._last = None
        self._seen: set = set()
        self._shape = None
        self._max_cell = max_cell
        self._use_lethal = use_lethal
        #: When set, terrain is impassable rather than merely avoided, so the
        #: agent walks only on background.  Both variants are worth measuring:
        #: whether the board's non-background colours are floor or furniture is
        #: not something the roles can decide for us.
        self._terrain_hard = terrain_hard
        self._prev_cell = None
        self._prev_action = None
        self._inert: set = set()
        self._status = SolverStatus()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        self._roles = RoleTracker()
        self._level = None
        self._last = None
        self._seen = set()
        self._shape = None
        self._prev_cell = None
        self._prev_action = None
        self._inert = set()
        self._status = SolverStatus()

    @property
    def status(self) -> dict:
        return self._status.to_dict()

    # -- main entry --------------------------------------------------------- #
    def next_action(self, frame: Any, levels: int, state: Any = None):
        """Return ``(action_id, reason)`` or ``None`` to fall through to CEAX."""
        if self._level != levels:
            self._level = levels
            self._seen = set()
            self._shape = None
            self._inert = set()
            self._prev_cell = None
            self._prev_action = None

        grid = as_grid(frame)
        if grid is None:
            self._last = None
            return None

        # The anchor is the action that produced this frame, read from the frame
        # itself -- see `_direction_anchor` for why it must not be our own pick.
        anchor = _direction_anchor(frame, self._last)
        spec = self._roles.observe(grid, action=anchor, levels=levels, state=state)
        self._status.steps += 1

        avatar = spec.avatar
        if avatar is None:
            self._status.skipped += 1
            self._status.note("no-avatar")
            self._last = None
            return None

        lattice = detect_cell_lattice(grid, max_cell=self._max_cell, background=spec.background)
        cells = to_cell_grid(grid, lattice, background=spec.background) if lattice.active else None
        if cells is None or cells.size == 0 or cells.size > MAX_CELLS:
            self._status.skipped += 1
            self._status.note("no-cell-grid")
            self._last = None
            return None

        if self._shape != cells.shape:
            self._shape = cells.shape
            self._seen = set()

        blob = _largest_blob(cells, avatar)
        if blob is None:
            self._status.skipped += 1
            self._status.note("avatar-not-on-board")
            self._last = None
            return None
        start = _centroid(blob)
        # "Pressed a wall": the last key was a direction and the piece did not budge.
        # Remember cell+key so the search stops re-picking it -- otherwise the agent
        # re-selects the same direction forever (measured: 28 consecutive "up" on
        # sp80, 127 consecutive "down" on ls20).
        if anchor in ACTION_DIR and self._prev_cell is not None and start == self._prev_cell:
            self._inert.add((start, anchor))
        self._prev_cell = start
        self._prev_action = anchor
        self._seen.add(start)

        hard, soft = self._blocked(cells, spec, blob)
        pick = self._choose(cells, start, hard, soft, spec, levels)
        if pick is None:
            self._status.skipped += 1
            self._status.note("no-move")
            self._last = None
            return None

        action, reason = pick
        self._status.picks += 1
        self._status.note(reason)
        self._last = action
        return action, reason

    # -- internals ---------------------------------------------------------- #
    def _blocked(self, cells, spec, blob):
        """``(hard, soft)`` blocked masks; hard is lethal, soft adds terrain."""
        hard = np.zeros(cells.shape, bool)
        if self._use_lethal:
            for colour in spec.lethal:
                hard |= cells == colour
        soft = hard.copy()
        for colour in spec.terrain:
            soft |= cells == colour
        if self._terrain_hard:
            hard = soft.copy()
        own = np.zeros(cells.shape, bool)
        for y, x in blob:
            own[y, x] = True
        # never let a guard colour that happens to sit under us block the escape
        hard &= ~own
        soft &= ~own
        return hard, soft

    def _object_mask(self, cells, spec, own_mask):
        """Cells holding something that is *not* chrome and *not* us."""
        chrome = set()
        for name in ("hud", "attached", "cursor"):
            value = getattr(spec, name)
            if value is None:
                continue
            chrome.update(value if hasattr(value, "__iter__") else (value,))
        if spec.mirror is not None:
            chrome.add(spec.mirror)
        chrome.update(spec.mobile or ())
        chrome.add(spec.avatar)
        if spec.background is not None:
            chrome.add(spec.background)

        mask = np.zeros(cells.shape, bool)
        for colour in np.unique(cells).tolist():
            colour = int(colour)
            if colour in chrome:
                continue
            mask |= cells == colour
        mask &= ~own_mask
        for colour in spec.lethal:
            mask &= cells != colour
        return mask

    def _choose(self, cells, start, hard, soft, spec, levels):
        own_mask = cells == spec.avatar
        inert = self._inert

        # rung 1 -- a goal we actually have evidence for
        if spec.goal:
            goal = np.zeros(cells.shape, bool)
            for colour in spec.goal:
                goal |= cells == colour
            goal &= ~own_mask
            got = (_step_towards(hard, start, goal, inert)
                   or _step_towards(soft, start, goal, inert))
            if got:
                return got[0], "generic:goal"

        # rung 2 -- go stand next to something.  Deliberately *not* "step on it":
        # whether a board's non-background colours are rewards or hazards is not
        # something the roles can decide, and on ls20 walking onto them is fatal
        # (measured: a held direction survives 400 steps, random dies at 130, and
        # a policy that walks onto objects died at 129 -- i.e. it stepped on the
        # hazard every time).  Standing adjacent still makes progress, and keeps
        # the option of a final step open to whoever does know the rule.
        objects = self._object_mask(cells, spec, own_mask)
        if objects.any():
            approach = _dilate4(objects) & ~objects & ~hard & ~own_mask
            if approach.any():
                got = _step_towards(hard, start, approach, inert)
                if got:
                    return got[0], "generic:approach"

        # rung 3 -- nothing left to touch: systematic frontier sweep
        h, w = cells.shape
        frontier = np.zeros(cells.shape, bool)
        for y in range(h):
            for x in range(w):
                if (y, x) not in self._seen and not soft[y, x]:
                    frontier[y, x] = True
        got = (_step_towards(hard, start, frontier, inert)
               or _step_towards(soft, start, frontier, inert))
        if got:
            return got[0], "generic:frontier"

        # rung 4 -- every route we know is wedged.  Forget the wedges so the next
        # frame can re-probe, and hand this step to CEAX.
        inert.clear()
        return None


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def search_only_action(cells: np.ndarray, start: tuple, blocked: np.ndarray,
                       targets: np.ndarray, inert: Optional[set] = None):
    """Expose the raw search for tests: ``(action, destination)`` or ``None``."""
    return _step_towards(blocked, start, targets, inert)


def enabled(default: bool = True) -> bool:
    """``LINGJING_GENERIC_SOLVER=0`` switches the solver off (for A/B runs)."""
    raw = str(os.environ.get("LINGJING_GENERIC_SOLVER", "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "off", "false", "no"}
