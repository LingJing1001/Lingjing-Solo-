"""The generic closed-book solver: today's primitives actually steering play.

The three claims worth testing here.

1. **The anchor comes from the frame, not from our own picks.**  The first
   version remembered only the actions *it* had chosen.  That is a deadlock: it
   never picks, so it has no anchor, so the roles cannot find an avatar, so it
   never picks.  Measured on the real engine: a 0.0% pick rate on every game,
   byte-identical to picking at random.  ``test_without_an_anchor...`` keeps the
   bug from coming back.
2. **The search is honest about walls.**  Blocked cells are never entered, a
   walled-in start declines instead of guessing, and equal-length paths break
   towards the roomier cell so the agent does not corner itself in a game that
   ends the episode on the first death.
3. **It declines rather than guesses.**  With no readable avatar it returns
   ``None`` -- which is what lets the closed-book branch fall through to CEAX
   with every other game's behaviour untouched.

Boards come from ``test_roles`` so this suite measures the solver against a
scene whose lattice and roles are already pinned down elsewhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import test_roles as T
from lingjing_solo.closed_book.generic_solver import (
    GenericClosedBookSolver,
    _centroid,
    _direction_anchor,
    _distance_field,
    _largest_blob,
    enabled,
    search_only_action,
)


# --------------------------------------------------------------------------- #
# frame stubs (the shape the real FrameData has)
# --------------------------------------------------------------------------- #
class _Enum:
    def __init__(self, value):
        self.value = value


class _ActionInput:
    """``FrameData.action_input``: ``.id`` is a ``GameAction`` enum with ``.value``."""

    def __init__(self, value, data=None):
        self.id = _Enum(value)
        self.data = data or {}


def _frame(grid, action=None, state="NOT_FINISHED", levels=0):
    return SimpleNamespace(
        frame=[grid],
        action_input=None if action is None else _ActionInput(action),
        state=SimpleNamespace(name=state),
        levels_completed=levels,
    )


def _replay(frames, acts, **kw):
    """Drive the solver over a trajectory, pairing frame i with the action that made it."""
    solver = GenericClosedBookSolver(**kw)
    picks = [solver.next_action(_frame(g, a), 0) for g, a in zip(frames, acts)]
    return solver, picks


def _tu93_trajectory():
    return [T.tu93_like(p) for p in T.WALK], list(T.WALK_ACTS)


# --------------------------------------------------------------------------- #
# the anchor (the bug this module was rewritten for)
# --------------------------------------------------------------------------- #
def test_the_anchor_is_read_off_the_frame():
    assert _direction_anchor(_frame(np.zeros((4, 4)), action=3), None) == 3


def test_a_click_or_reset_carries_no_direction():
    """A click is not a direction, and the frame wins over our own memory."""
    assert _direction_anchor(_frame(np.zeros((4, 4)), action=6), 2) is None
    assert _direction_anchor(_frame(np.zeros((4, 4)), action=0), 2) is None


def test_the_anchor_falls_back_only_when_the_frame_is_silent():
    assert _direction_anchor(SimpleNamespace(), 4) == 4
    assert _direction_anchor(SimpleNamespace(), 6) is None
    assert _direction_anchor(SimpleNamespace(), None) is None


def test_without_an_anchor_the_solver_cannot_bootstrap():
    """The deadlock, pinned: no anchor -> no avatar -> no pick, forever."""
    frames, acts = _tu93_trajectory()

    blind = GenericClosedBookSolver()
    blind_picks = [blind.next_action(_frame(g), 0) for g in frames]
    assert all(p is None for p in blind_picks)
    assert blind.status["reasons"].get("no-avatar", 0) == len(frames)
    assert blind.status["picks"] == 0

    # ...while the very same frames, carrying the frame's own action, do steer.
    sharp = GenericClosedBookSolver()
    sharp_picks = [sharp.next_action(_frame(g, a), 0) for g, a in zip(frames, acts)]
    assert any(p is not None for p in sharp_picks)
    assert sharp.status["picks"] > 0


# --------------------------------------------------------------------------- #
# geometry primitives
# --------------------------------------------------------------------------- #
def test_the_largest_component_of_a_colour_wins():
    cells = np.zeros((6, 6), np.int64)
    cells[0, 0] = 9                                     # stray speck
    cells[3, 3] = cells[3, 4] = cells[4, 3] = 9         # the piece
    blob = _largest_blob(cells, 9)
    assert len(blob) == 3
    assert (0, 0) not in blob
    assert _largest_blob(cells, 7) is None


def test_the_start_cell_belongs_to_the_blob():
    cells = np.zeros((8, 8), np.int64)
    cells[1, 1] = cells[1, 2] = cells[2, 1] = 9
    start = _centroid(_largest_blob(cells, 9))
    assert cells[start] == 9


def test_distance_field_counts_steps_around_a_wall():
    blocked = np.zeros((5, 5), bool)
    blocked[0:4, 2] = True                              # wall, but the bottom row is open
    src = np.zeros((5, 5), bool)
    src[4, 4] = True
    dist = _distance_field(blocked, src)
    assert dist[4, 4] == 0
    assert dist[4, 0] == 4                              # straight along the bottom
    assert dist[0, 3] == 5                              # the near side of the wall
    assert dist[0, 1] == 7                              # the far side
    assert dist[0, 0] == 8                              # all the way round
    assert dist[0, 2] == -1, "the wall itself is never reached"
    assert (dist[blocked] == -1).all()


def test_a_blocked_cell_is_never_entered():
    blocked = np.zeros((3, 3), bool)
    blocked[0, 1] = True
    targets = np.zeros((3, 3), bool)
    targets[0, 2] = True
    got = search_only_action(np.zeros((3, 3), np.int64), (1, 1), blocked, targets)
    assert got is not None
    assert got[1] != (0, 1), "stepped into a wall"
    assert got == (4, (1, 2)), "should go right and then up, not through the wall"


def test_it_declines_when_walled_in():
    blocked = np.zeros((3, 3), bool)
    blocked[0, 1] = blocked[2, 1] = blocked[1, 0] = blocked[1, 2] = True
    targets = np.zeros((3, 3), bool)
    targets[0, 0] = True                                # exists, but unreachable
    assert search_only_action(np.zeros((3, 3), np.int64), (1, 1), blocked, targets) is None


def test_ties_break_towards_the_roomier_cell():
    """Two equally short first steps: take the one with more ways out."""
    blocked = np.zeros((3, 3), bool)
    # targets are reachable both up and right; the up cell is a dead end
    blocked[0, 0] = True
    targets = np.zeros((3, 3), bool)
    targets[0, 1] = True
    targets[1, 2] = True
    got = search_only_action(np.zeros((3, 3), np.int64), (1, 1), blocked, targets)
    assert got is not None
    assert got[0] in (1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# end to end on a board whose lattice and roles are already pinned
# --------------------------------------------------------------------------- #
def test_it_steers_once_the_avatar_is_readable():
    frames, acts = _tu93_trajectory()
    solver, picks = _replay(frames, acts)
    live = [p for p in picks if p is not None]
    assert live, "the solver should steer once an avatar has been measured"
    assert all(p[0] in (1, 2, 3, 4) for p in live)
    assert all(p[1].startswith("generic:") for p in live)
    assert solver.status["reasons"], "every pick should be attributed a rung"


def test_it_stands_beside_an_object_rather_than_on_it():
    """Stepping *onto* an unknown colour is how ls20 kills you; approach, don't touch."""
    import numpy as np

    from lingjing_solo.closed_book.generic_solver import _dilate4

    objects = np.zeros((5, 5), bool)
    objects[2, 2] = True
    near = _dilate4(objects)
    assert near[2, 2] and near[1, 2] and near[2, 1] and near[2, 3] and near[3, 2]
    assert not near[0, 0]


def test_it_declines_when_no_avatar_can_be_read():
    """A frozen board has no mover -- the solver must not invent one."""
    frames = [T.tu93_like((6, 10)) for _ in T.WALK]
    solver, picks = _replay(frames, [None] * len(frames))
    assert all(p is None for p in picks)
    assert solver.status["skipped"] == len(frames)


def test_the_pick_is_deterministic():
    frames, acts = _tu93_trajectory()
    runs = [
        [s.next_action(_frame(g, a), 0) for g, a in zip(frames, acts)]
        for s in (GenericClosedBookSolver(), GenericClosedBookSolver())
    ]
    assert runs[0] == runs[1]


def test_status_accounts_for_every_step():
    frames, acts = _tu93_trajectory()
    solver, _ = _replay(frames, acts)
    st = solver.status
    assert st["steps"] == len(frames)
    assert st["picks"] + st["skipped"] == st["steps"]
    assert sum(st["reasons"].values()) == st["steps"]


def test_a_new_level_forgets_the_visited_cells():
    frames, acts = _tu93_trajectory()
    solver = GenericClosedBookSolver()
    for g, a in zip(frames, acts):
        solver.next_action(_frame(g, a), 0)
    assert solver._seen, "a level's worth of walking should leave a trail"
    solver.next_action(_frame(frames[-1], acts[-1], levels=1), 1)
    assert not solver._seen, "cells visited on the previous board mean nothing here"


# --------------------------------------------------------------------------- #
# terrain policy + kill switch
# --------------------------------------------------------------------------- #
def test_terrain_is_soft_by_default_and_hard_on_request():
    cells = np.zeros((3, 3), np.int64)
    cells[0, 0] = 3                                     # a terrain colour
    spec = SimpleNamespace(lethal=frozenset({7}), terrain=frozenset({3}))
    blob = [(1, 1)]

    hard_soft, soft = GenericClosedBookSolver()._blocked(cells, spec, blob)
    assert not hard_soft[0, 0] and soft[0, 0]

    hard_hard, _ = GenericClosedBookSolver(terrain_hard=True)._blocked(cells, spec, blob)
    assert hard_hard[0, 0]


def test_the_kill_switch_is_honoured(monkeypatch):
    monkeypatch.delenv("LINGJING_GENERIC_SOLVER", raising=False)
    assert enabled() is True
    monkeypatch.setenv("LINGJING_GENERIC_SOLVER", "0")
    assert enabled() is False
    monkeypatch.setenv("LINGJING_GENERIC_SOLVER", "off")
    assert enabled() is False
    monkeypatch.setenv("LINGJING_GENERIC_SOLVER", "1")
    assert enabled() is True
