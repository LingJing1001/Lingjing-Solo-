"""Closed-book AR25: frame-only perception + translation solver.

These tests build **synthetic frames** (no engine, no arcengine import) from the
geometry observed on the real board, then assert that the perception recovers
the object ontology and that the solver derives the correct translation.

The reference geometry is AR25 level 1:

    axis    vertical line on column 10
    shape   L-polyomino at {(5,6),(5,7),(5,8),(6,8),(7,8)}
    target  {(15,17),(15,18),(15,19),(16,17),(17,17)}
    solution shift the shape by (dr=+10, dc=-5); its mirror then covers the target

A regression here means the closed-book path can no longer solve AR25 L1,
which is the only thing keeping the headline closed-book score above zero.
"""
from __future__ import annotations

from lingjing_solo.closed_book.ar25_frame import (
    COL_AXIS,
    COL_BG,
    COL_CURSOR,
    COL_SHAPE,
    COL_TARGET,
    GRID_N,
    parse_frame,
)
from lingjing_solo.closed_book.ar25_solver import Ar25ClosedBookSolver, solve_shift

CELL = 3
L1_AXIS_COL = 10
L1_SHAPE = [(5, 6), (5, 7), (5, 8), (6, 8), (7, 8)]
L1_TARGET = [(15, 17), (15, 18), (15, 19), (16, 17), (17, 17)]
L1_SHIFT = (10, -5)


class FakeFrame:
    """Minimal stand-in for arcengine FrameData (only .frame is used)."""

    def __init__(self, layer):
        self.frame = [layer]


def make_layer(shape_cells, axis_col=L1_AXIS_COL, targets=L1_TARGET, cursor=True):
    n = GRID_N * CELL
    layer = [[COL_BG for _ in range(n)] for _ in range(n)]

    def paint(cells, colour, inset=None):
        for r, c in cells:
            for y in range(r * CELL, r * CELL + CELL):
                for x in range(c * CELL, c * CELL + CELL):
                    if inset is not None and not (y == r * CELL and x == c * CELL):
                        continue
                    layer[y][x] = colour

    if axis_col is not None:
        paint([(r, axis_col) for r in range(GRID_N)], COL_AXIS)
    paint(list(targets), COL_TARGET)
    paint(list(shape_cells), COL_SHAPE)
    if cursor and shape_cells:
        rs = [r for r, _ in shape_cells]
        cs = [c for _, c in shape_cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        outline = [(r0, c) for c in range(c0, c1 + 1)] + [(r, c1) for r in range(r0, r1 + 1)]
        paint(outline, COL_CURSOR, inset="first-pixel")
    return layer


def test_perception_recovers_ar25_level1():
    obs = parse_frame(FakeFrame(make_layer(L1_SHAPE)))

    assert obs.ok, obs.note
    assert obs.axis_kind == "V"
    assert obs.axis_coord == L1_AXIS_COL
    assert len(obs.shapes) == 1
    assert sorted(obs.shapes[0]) == sorted(L1_SHAPE)
    assert obs.target_cells == set(L1_TARGET)
    assert obs.selected is not None, "cursor must attach to the shape"
    assert sorted(obs.selected) == sorted(L1_SHAPE)


def test_solver_finds_level1_shift():
    obs = parse_frame(FakeFrame(make_layer(L1_SHAPE)))
    assert solve_shift(obs, list(obs.shapes[0])) == L1_SHIFT

    # the derived shift really does cover every target through the mirror
    shifted = [(r + L1_SHIFT[0], c + L1_SHIFT[1]) for r, c in L1_SHAPE]
    assert obs.target_cells <= obs.mirror_cells(shifted)


def test_closed_loop_reproduces_the_reference_plan():
    """Driving the solver against a toy simulator must yield LEFT x5, DOWN x10."""
    layer = make_layer(L1_SHAPE)
    shape = list(L1_SHAPE)
    solver = Ar25ClosedBookSolver()

    deltas = {
        1: (-1, 0),  # UP
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
        4: (0, 1),   # RIGHT
    }
    actions = []
    for _ in range(40):
        pick = solver.next_action(FakeFrame(make_layer(shape)), levels=0)
        if pick is None:
            break
        act, _why = pick
        actions.append(act)
        dr, dc = deltas[act]
        shape = [(r + dr, c + dc) for r, c in shape]
        layer = make_layer(shape)

    assert actions == [3] * 5 + [2] * 10, actions


def test_unparsable_frame_yields_no_action():
    """No axis -> perception is not confident -> fall through to CEAX."""
    solver = Ar25ClosedBookSolver()
    obs = parse_frame(FakeFrame(make_layer(L1_SHAPE, axis_col=None)))
    assert not obs.ok
    assert solver.next_action(FakeFrame(make_layer(L1_SHAPE, axis_col=None)), levels=0) is None


def test_empty_frame_is_safe():
    obs = parse_frame(FakeFrame([]))
    assert not obs.ok
    assert isinstance(obs.note, str)
