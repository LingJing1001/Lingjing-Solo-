"""Unit tests for the closed-book tu93 frame model.

Everything here is built from **synthetic** 64x64 frames, so the suite runs
without the engine.  The point is to pin the world model that the online solver
depends on:

  * lattice-phase recovery (the maze sits at a per-level pixel offset),
  * reading the player / target out of the frame,
  * arrow -> node deltas and the "connector must not be void" rule,
  * the colour-8 hazard: which entry sides are fatal, its line of sight, and
    that flanking it clears it.
"""
from __future__ import annotations

import numpy as np
import pytest

from lingjing_solo.closed_book.tu93_frame import (
    CONNECTOR,
    CORRIDOR,
    DEADLY,
    DELTA,
    PLAYER,
    TARGET,
    VOID,
    Patrol,
    _hazard_owner,
    _steps,
    bfs_first_action,
    bfs_path,
    build_patrols,
    detect_lattice,
    neighbours,
    observed_patrol_nodes,
    parse_frame,
    patrol_next_cells,
    plan_path,
    transient_cells,
)

GEM = 15          # patroller / hazard-marker colour
PATROL = 13       # colour of a roaming 3x3 tile
_UP, _DOWN, _LEFT, _RIGHT = 1, 2, 3, 4
_NOOP = 5

# pixel index of the "odd pixel" inside a 3x3 colour-8 tile, per facing
_MISS = {"E": (1, 2), "W": (1, 0), "N": (0, 1), "S": (2, 1)}


def _blank() -> np.ndarray:
    return np.full((64, 64), VOID, dtype=np.int8)


def _fill(arr: np.ndarray, r: int, c: int, colour: int, offset=(0, 0)) -> None:
    orow, ocol = offset
    r0, c0 = orow + 3 * r, ocol + 3 * c
    arr[r0:r0 + 3, c0:c0 + 3] = colour


def _patrol_tile(arr: np.ndarray, r: int, c: int, d: str, offset=(0, 0)) -> None:
    """Draw a roaming tile: a 3x3 colour-13 block with a gem pixel at *d*."""
    _fill(arr, r, c, PATROL, offset)
    orow, ocol = offset
    dr, dc = _MISS[d]
    arr[orow + 3 * r + dr, ocol + 3 * c + dc] = GEM


def build(
    offset=(0, 0),
    nodes=(),
    edges=(),
    player=None,
    target=None,
    hazard=None,      # ((r, c), side)
    patroller=None,   # (r, c)
) -> np.ndarray:
    """Assemble a frame: odd-cell nodes, colour-2 connectors, one of each actor."""
    arr = _blank()
    for r, c in nodes:
        _fill(arr, r, c, CORRIDOR, offset)
    for (r1, c1), (r2, c2) in edges:
        assert abs(r1 - r2) + abs(c1 - c2) == 2, "edges connect lattice-adjacent nodes"
        _fill(arr, (r1 + r2) // 2, (c1 + c2) // 2, CONNECTOR, offset)
    if player is not None:
        _fill(arr, *player, PLAYER, offset)
    if target is not None:
        _fill(arr, *target, TARGET, offset)
    if hazard is not None:
        (hr, hc), side = hazard
        _fill(arr, hr, hc, DEADLY[0], offset)
        dr, dc = _MISS[side]
        orow, ocol = offset
        arr[orow + 3 * hr + dr, ocol + 3 * hc + dc] = GEM
    if patroller is not None:
        _fill(arr, *patroller, GEM, offset)
    return arr


# --- lattice & basic parse ---------------------------------------------------

def test_parse_straight_corridor():
    arr = build(
        nodes=[(5, 5), (5, 7), (5, 9)],
        edges=[((5, 5), (5, 7)), ((5, 7), (5, 9))],
        player=(5, 5),
        target=(5, 9),
    )
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert obs.offset == (0, 0)
    assert obs.player == (5, 5)
    assert obs.targets == [(5, 9)]
    # one node to the right is the shortest first move
    assert bfs_first_action(obs.cells, obs.player, obs.targets) == _RIGHT
    # and the reverse leg heads left
    assert bfs_first_action(obs.cells, (5, 9), [(5, 5)]) == _LEFT
    assert bfs_path(obs.cells, (5, 5), obs.targets) == [_RIGHT, _RIGHT]


@pytest.mark.parametrize("offset", [(0, 0), (1, 2), (2, 1), (1, 0), (0, 2)])
def test_lattice_phase_recovery(offset):
    """The maze can sit at any sub-3px phase; the parse must find it."""
    arr = build(
        offset=offset,
        nodes=[(5, 5), (5, 7), (5, 9)],
        edges=[((5, 5), (5, 7)), ((5, 7), (5, 9))],
        player=(5, 5),
        target=(5, 9),
    )
    det_offset, _ = detect_lattice(arr)
    assert det_offset == offset
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert obs.player == (5, 5)
    assert obs.targets == [(5, 9)]
    assert bfs_first_action(obs.cells, obs.player, obs.targets) == _RIGHT


def test_void_connector_breaks_the_edge():
    """Without the connecting cell the two nodes are not adjacent."""
    arr = build(nodes=[(5, 5), (5, 7)], edges=[], player=(5, 5), target=(5, 7))
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert bfs_first_action(obs.cells, (5, 5), obs.targets) is None


def test_route_around_a_wall():
    """BFS must prefer the open detour over the missing edge."""
    arr = build(
        nodes=[(5, 5), (5, 7), (7, 5), (7, 7), (7, 9), (5, 9)],
        edges=[
            ((5, 5), (7, 5)),
            ((7, 5), (7, 7)),
            ((7, 7), (7, 9)),
            ((7, 9), (5, 9)),
        ],
        player=(5, 5),
        target=(5, 9),
    )
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    path = bfs_path(obs.cells, (5, 5), obs.targets)
    assert path is not None
    assert path == [_DOWN, _RIGHT, _RIGHT, _UP]


# --- colour-8 hazard semantics ----------------------------------------------

def test_hazard_entry_side_is_fatal():
    """Stepping onto the tile from the side it faces must be refused."""
    arr = build(
        nodes=[(5, 5), (5, 7), (5, 9)],
        edges=[((5, 5), (5, 7)), ((5, 7), (5, 9))],
        player=(5, 5),
        target=(5, 9),
        hazard=((5, 7), "W"),          # faces west -> its face is entered from the west
    )
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert obs.facing == {(5, 7): "W"}

    acts_from_west = {a for a, _ in neighbours(obs.cells, (5, 5), obs.facing)}
    assert _RIGHT not in acts_from_west       # that would walk into its face

    acts_from_east = {a for a, _ in neighbours(obs.cells, (5, 9), obs.facing)}
    assert _LEFT in acts_from_east            # same tile, entered from behind: fine


def test_hazard_line_of_sight_is_lethal_until_cleared():
    """The cell two ahead of a live hazard cannot be entered."""
    arr = build(
        nodes=[(5, 5), (5, 7), (5, 9)],
        edges=[((5, 5), (5, 7)), ((5, 7), (5, 9))],
        player=(5, 5),
        target=(5, 9),
        hazard=((5, 7), "E"),
    )
    obs = parse_frame(arr)
    assert obs.danger == {(5, 9)}
    # reachable only by first clearing the hazard (walking over it from behind)
    assert bfs_first_action(obs.cells, (5, 5), obs.targets, obs.facing, obs.danger) == _RIGHT


def test_live_hazard_can_seal_the_only_route():
    """If the sight line covers the corridor and no flank exists -> no path."""
    arr = build(
        nodes=[(1, 7), (3, 7), (5, 7), (7, 7)],
        edges=[((1, 7), (3, 7)), ((3, 7), (5, 7)), ((5, 7), (7, 7))],
        player=(7, 7),
        target=(1, 7),
        hazard=((3, 7), "S"),          # looks down the corridor we must walk up
    )
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert obs.danger == {(5, 7)}
    assert bfs_first_action(obs.cells, obs.player, obs.targets, obs.facing, obs.danger) is None


# --- patroller detection -----------------------------------------------------

def test_transient_cell_flags_a_patroller():
    arr = build(
        nodes=[(5, 5), (5, 7), (5, 9)],
        edges=[((5, 5), (5, 7)), ((5, 7), (5, 9))],
        player=(5, 5),
        target=(5, 9),
        patroller=(5, 7),
    )
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert transient_cells(arr, obs) == {(5, 7)}


def test_transient_cell_ignores_the_hazard_marker():
    """The marker pixel lives inside the hazard tile -- not a patroller."""
    arr = build(
        nodes=[(5, 5), (5, 7), (5, 9)],
        edges=[((5, 5), (5, 7)), ((5, 7), (5, 9))],
        player=(5, 5),
        target=(5, 9),
        hazard=((5, 7), "E"),
    )
    obs = parse_frame(arr)
    assert obs.ok, obs.note
    assert transient_cells(arr, obs) == set()


# --- parse rejects junk ------------------------------------------------------

def test_parse_rejects_wrong_shape():
    obs = parse_frame(np.zeros((10, 10), dtype=np.int8))
    assert not obs.ok
    assert "shape" in obs.note


def test_parse_accepts_batched_frame():
    """The engine hands back a (1, 64, 64) array; parse must squeeze it."""
    arr = build(nodes=[(5, 5), (5, 7)], edges=[((5, 5), (5, 7))], player=(5, 5), target=(5, 7))
    obs = parse_frame(arr[None, ...])
    assert obs.ok, obs.note
    assert obs.player == (5, 5)


def test_solver_unwraps_observation_object():
    """The solver takes the observation object and unwraps .frame itself."""
    from lingjing_solo.closed_book.tu93_solver import _raw

    class _ObsLike:
        def __init__(self, frame):
            self.frame = frame

    arr = build(nodes=[(5, 5), (5, 7)], edges=[((5, 5), (5, 7))], player=(5, 5), target=(5, 7))
    obj = _ObsLike(arr)
    assert _raw(obj) is arr          # unwrapped
    assert _raw(arr) is arr          # raw array passes through
    assert parse_frame(_raw(obj)).player == (5, 5)


# --- line of sight: every watcher must be cleared ----------------------------

def test_hazard_owner_lists_every_watcher():
    """Two hazards can share a sight line; both must be recorded (L5 bug).

    ``(3,7)`` facing S and ``(5,9)`` facing W both look at ``(5,7)``.  An earlier
    version stored a single owner per cell, so the second silently erased the
    first and the solver walked into a still-watched cell.
    """
    owner = _hazard_owner({(3, 7): "S", (5, 9): "W"})
    assert owner[(5, 7)] == {(3, 7), (5, 9)}


def test_line_of_sight_needs_every_watcher_cleared():
    """A doubly-watched cell only opens once *both* watchers are cleared."""
    nodes = [(5, 1), (5, 3), (5, 5), (5, 7), (3, 5), (7, 5)]
    edges = [((5, 1), (5, 3)), ((5, 3), (5, 5)), ((5, 5), (5, 7)),
             ((3, 5), (5, 5)), ((7, 5), (5, 5))]
    arr = build(nodes=nodes, edges=edges, player=(5, 1), target=(5, 7),
                hazard=((3, 5), "S"))          # looks at (5,5)
    # a second watcher: (7,5) drawn as colour-8 but facing N (also at (5,5))
    _fill(arr, 7, 5, DEADLY[0])
    arr[3 * 7 + _MISS["N"][0], 3 * 5 + _MISS["N"][1]] = GEM

    obs = parse_frame(arr)
    assert obs.facing.get((3, 5)) == "S"
    assert obs.facing.get((7, 5)) == "N"
    owner = _hazard_owner(obs.facing)

    def acts(cleared):
        return {a for a, _, _ in _steps(obs.cells, (5, 3), obs.facing, cleared, owner)}

    assert _RIGHT not in acts(frozenset({(3, 5)}))            # only one watcher down
    assert _RIGHT in acts(frozenset({(3, 5), (7, 5)}))        # both down


# --- the roaming patroller model ---------------------------------------------

def test_patrol_bounces_along_its_segment():
    p = Patrol(seg=[(5, 9), (5, 7), (5, 5), (5, 3)], start=2)
    assert p.period == 6
    assert [p.at(k) for k in range(7)] == [
        (5, 5), (5, 3), (5, 5), (5, 7), (5, 9), (5, 7), (5, 5),
    ]


def test_build_patrols_reads_the_corridor_and_heading():
    arr = build(
        nodes=[(5, 1), (5, 3), (5, 5), (5, 7)],
        edges=[((5, 1), (5, 3)), ((5, 3), (5, 5)), ((5, 5), (5, 7))],
        player=(5, 1),
        target=(5, 7),
    )
    _patrol_tile(arr, 5, 3, "W")       # at (5,3), about to move west
    obs = parse_frame(arr)
    pats, seen0 = build_patrols(obs.cells, obs.offset, arr)
    assert len(pats) == 1
    assert seen0 == [(5, 3)]
    assert pats[0].at(0) == (5, 3)
    assert set(pats[0].seg) == {(5, 1), (5, 3), (5, 5), (5, 7)}
    # observation + one-step lookahead
    assert observed_patrol_nodes(arr, obs.cells, obs.offset) == [(5, 3)]
    assert patrol_next_cells(arr, obs.cells, obs.offset) == {(5, 1)}


def _sim(acts, start, patrol):
    node, k = start, 0
    for a in acts:
        dr, dc = DELTA[int(a)]
        node, k = (node[0] + dr, node[1] + dc), k + 1
        assert node != patrol.at(k), f"plan lands on the patroller at step {k}"
    return node, k


def test_plan_path_never_lands_on_the_patroller():
    arr = build(
        nodes=[(5, 1), (5, 3), (5, 5), (5, 7)],
        edges=[((5, 1), (5, 3)), ((5, 3), (5, 5)), ((5, 5), (5, 7))],
        player=(5, 1),
        target=(5, 7),
    )
    obs = parse_frame(arr)
    patrol = Patrol(seg=[(5, 3), (5, 5), (5, 7)], start=2)
    acts = plan_path(obs.cells, obs.player, obs.targets, obs.facing, [patrol], frozenset())
    assert acts is not None
    _sim(acts, obs.player, patrol)


def test_plan_path_gives_up_when_the_patroller_seals_the_route():
    arr = build(
        nodes=[(5, 1), (5, 3), (5, 5), (5, 7)],
        edges=[((5, 1), (5, 3)), ((5, 3), (5, 5)), ((5, 5), (5, 7))],
        player=(5, 1),
        target=(5, 7),
    )
    obs = parse_frame(arr)
    # this phase is always standing on (5,5) exactly when we must pass it
    patrol = Patrol(seg=[(5, 3), (5, 5), (5, 7)], start=1)
    assert plan_path(obs.cells, obs.player, obs.targets, obs.facing, [patrol], frozenset()) is None


# --- solver bookkeeping ------------------------------------------------------

def test_reconcile_tolerates_overlapping_patrols():
    """Two tiles sharing a node collapse to one observed cell -- not a failure."""
    from lingjing_solo.closed_book.tu93_solver import Tu93ClosedBookSolver

    p1 = Patrol(seg=[(5, 3), (5, 5)], start=0)
    p2 = Patrol(seg=[(5, 5), (5, 7)], start=0)
    s = Tu93ClosedBookSolver()
    s._patrols = [p1, p2]
    s._reconcile([p1, p2], [(5, 5)])           # both tiles on (5,5)
    assert not s._broken


def test_reconcile_tolerates_an_occluded_patrol():
    """A tile sliding under a hazard vanishes from the read -- not a failure."""
    from lingjing_solo.closed_book.tu93_solver import Tu93ClosedBookSolver

    p1 = Patrol(seg=[(5, 3), (5, 5)], start=0)
    p2 = Patrol(seg=[(9, 3), (9, 5)], start=0)
    s = Tu93ClosedBookSolver()
    s._patrols = [p1, p2]
    s._reconcile([p1, p2], [p1.at(0)])         # second tile hidden
    assert not s._broken


def test_reconcile_stands_down_when_a_patrol_is_somewhere_impossible():
    from lingjing_solo.closed_book.tu93_solver import Tu93ClosedBookSolver

    p1 = Patrol(seg=[(5, 3), (5, 5)], start=0)
    s = Tu93ClosedBookSolver()
    s._patrols = [p1]
    s._reconcile([p1], [(9, 9)])               # nowhere near the model
    assert s._broken


def test_reactive_solver_never_enters_the_patrols_next_cell():
    """When the model is unusable the solver sidesteps instead of gambling.

    The corridor is ``(5,1)-(5,3)-(5,5)-(5,7)``; the only forward move from
    ``(5,1)`` is onto ``(5,3)``, which is exactly where the patroller is about to
    move.  A solver that committed to the shortest path would walk in and die.
    """
    from lingjing_solo.closed_book.tu93_solver import Tu93ClosedBookSolver

    arr = build(
        nodes=[(5, 1), (5, 3), (5, 5), (5, 7)],
        edges=[((5, 1), (5, 3)), ((5, 3), (5, 5)), ((5, 5), (5, 7))],
        player=(5, 1),
        target=(5, 7),
    )
    _patrol_tile(arr, 5, 5, "W")               # at (5,5), about to enter (5,3)
    obs = parse_frame(arr)
    pats, _ = build_patrols(obs.cells, obs.offset, arr)
    assert patrol_next_cells(arr, obs.cells, obs.offset) == {(5, 3)}

    s = Tu93ClosedBookSolver()
    s._level = 0
    s._pos = (5, 1)
    s._prev_pos = (5, 1)
    s._patrols = pats
    s._broken = True                            # force the reactive rule
    act, _ = s.next_action(arr, 0)
    assert act == _NOOP                         # the only step forward is fatal

