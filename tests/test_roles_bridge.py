"""The closed-book role bridge: generic inference feeding the hand-written parsers.

Three claims are checked here.

1. **The bridge is a no-op on the shipped palette.**  ``parse_frame_with_roles``
   with the inferred colour equal to the module constant must produce *exactly*
   the same observation as ``parse_frame`` -- otherwise the fallback could
   silently change behaviour that every other suite depends on.
2. **The probe reaches the palette the module hardcodes.**  ``roles_probe`` sees
   only pixels and the action stream, yet independently lands on
   ``PLAYER``/``HUD`` (tu93) and ``COL_SHAPE``/``COL_REFLECT``/``COL_CURSOR``
   (ar25).  That is the evidence that prior #3 is really generic.
3. **The bridge covers the constant's blind spot.**  Permute the palette and the
   hardcoded read fails while the inferred one still names the piece.

The runner-level flag (``LINGJING_ROLES_FALLBACK``) is off by default, so the
permitted path is exercised by calling the retry helper directly instead of
flipping a module global -- every other suite therefore still sees the shipped
behaviour.
"""
from __future__ import annotations

from types import SimpleNamespace

import test_roles as T


def _as_frame(grid):
    """A ``FrameData``-shaped object: ar25's parser reads ``.frame`` (layers)."""
    return SimpleNamespace(frame=[grid])


# --------------------------------------------------------------------------- #
# 1. a no-op when the inferred colour is the constant
# --------------------------------------------------------------------------- #
def test_the_bridge_matches_the_hardcoded_read_on_tu93():
    from lingjing_solo.closed_book import tu93_frame as TF

    frame = T.tu93_like((6, 10), extra={(10, 5): 14})          # give it a target
    plain = TF.parse_frame(frame)
    bridged = TF.parse_frame_with_roles(frame, {"avatar": TF.PLAYER})
    assert plain.ok and bridged.ok
    assert bridged.player == plain.player
    assert bridged.offset == plain.offset
    assert bridged.targets == plain.targets and bridged.doors == plain.doors
    assert bridged.gems == plain.gems and bridged.facing == plain.facing


def test_the_bridge_matches_the_hardcoded_read_on_ar25():
    from lingjing_solo.closed_book import ar25_frame as AF

    frame = _as_frame(T.ar25_like((4, 4)))
    plain = AF.parse_frame(frame)
    bridged = AF.parse_frame_with_roles(
        frame,
        {"avatar": AF.COL_SHAPE, "cursor": AF.COL_CURSOR, "mirror": AF.COL_REFLECT},
    )
    assert plain.ok and bridged.ok
    assert bridged.shapes == plain.shapes
    assert bridged.target_cells == plain.target_cells
    assert bridged.axis_kind == plain.axis_kind
    assert bridged.axis_coord == plain.axis_coord


def test_an_uninferred_role_falls_back_to_the_constant():
    """``avatar=None`` (nothing settled yet) must not blank the player colour."""
    from lingjing_solo.closed_book import tu93_frame as TF

    frame = T.tu93_like((6, 10), extra={(10, 5): 14})
    reference = TF.parse_frame(frame).player
    assert TF.parse_frame_with_roles(frame, {"avatar": None}).player == reference
    assert TF.parse_frame_with_roles(frame, {}).player == reference
    assert TF.parse_frame_with_roles(frame, None).player == reference


# --------------------------------------------------------------------------- #
# 2. the probe independently reaches the hand-written palette
# --------------------------------------------------------------------------- #
def test_roles_probe_agrees_with_the_hand_written_tu93_palette():
    from lingjing_solo.closed_book import tu93_frame as TF

    spec = TF.roles_probe([T.tu93_like(p) for p in T.WALK], T.WALK_ACTS,
                          levels=[0] * len(T.WALK))
    assert spec.background == TF.VOID
    assert spec.avatar == TF.PLAYER
    assert set(spec.hud) == {TF.HUD}
    assert spec.mirror is None and spec.cursor is None


def test_roles_probe_agrees_with_the_hand_written_ar25_palette():
    from lingjing_solo.closed_book import ar25_frame as AF

    spec = AF.roles_probe([_as_frame(T.ar25_like(p)) for p in T.AV_PATH], T.AV_ACTS,
                          levels=[0] * len(T.AV_PATH))
    assert spec.background == AF.COL_BG
    assert spec.avatar == AF.COL_SHAPE
    assert spec.mirror == AF.COL_REFLECT
    assert spec.cursor == AF.COL_CURSOR


# --------------------------------------------------------------------------- #
# 3. the bridge covers what the constant cannot
# --------------------------------------------------------------------------- #
def test_the_bridge_reads_a_repaletted_tu93_board():
    from lingjing_solo.closed_book import tu93_frame as TF

    plain = [T.tu93_like(p) for p in T.WALK]
    moved = [T._remap(f, {TF.PLAYER: 3}) for f in plain]      # the player is 3 now

    broken = TF.parse_frame(moved[-1])
    assert broken.ok is False and broken.note == "no player"

    spec = TF.roles_probe(moved, T.WALK_ACTS, levels=[0] * len(moved))
    assert spec.avatar == 3, "the colour moved; the inference must follow it"
    assert TF.parse_frame_with_roles(moved[-1], spec).player == TF.parse_frame(plain[-1]).player


def test_the_bridge_reads_a_repaletted_ar25_board():
    from lingjing_solo.closed_book import ar25_frame as AF

    plain = [_as_frame(T.ar25_like(p)) for p in T.AV_PATH]
    moved = [_as_frame(T._remap(T.ar25_like(p), {AF.COL_SHAPE: 12})) for p in T.AV_PATH]
    assert AF.parse_frame(plain[-1]).ok
    assert not AF.parse_frame(moved[-1]).shapes

    spec = AF.roles_probe(moved, T.AV_ACTS, levels=[0] * len(moved))
    assert spec.avatar == 12
    assert AF.parse_frame_with_roles(moved[-1], spec).shapes


# --------------------------------------------------------------------------- #
# 4. the solvers: shipped path untouched, retry reachable when asked
# --------------------------------------------------------------------------- #
def test_the_default_solver_never_builds_a_role_tracker():
    from lingjing_solo.closed_book.tu93_solver import Tu93ClosedBookSolver

    solver = Tu93ClosedBookSolver()
    solver.next_action(T.tu93_like((6, 10)), 0)
    assert solver._roles is None, "the shipped path must not pay for the fallback"


def test_the_tu93_retry_recovers_what_the_constant_cannot():
    from lingjing_solo.closed_book import tu93_solver as S

    solver = S.Tu93ClosedBookSolver()
    frames = [T._remap(T.tu93_like(p), {9: 3}) for p in T.WALK]
    obs = None
    for frame, act in zip(frames, T.WALK_ACTS):
        # ``WALK_ACTS[i]`` is the action that *produced* ``frames[i]`` -- exactly
        # what the solver holds in ``_last_act`` when it reads a frame, since the
        # frame it sees is always the previous action's render.
        solver._last_act = act
        obs = solver._retry_with_roles(frame, S.parse_frame(frame), 0)
    assert obs is not None and obs.player is not None


def test_the_ar25_retry_recovers_what_the_constant_cannot():
    from lingjing_solo.closed_book import ar25_frame as AF
    from lingjing_solo.closed_book import ar25_solver as S

    solver = S.Ar25ClosedBookSolver()
    obs = None
    for p, act in zip(T.AV_PATH, T.AV_ACTS):
        frame = _as_frame(T._remap(T.ar25_like(p), {AF.COL_SHAPE: 12}))
        solver._last_act = act
        obs = solver._retry_with_roles(frame, S.parse_frame(frame), 0)
    assert obs is not None and obs.shapes


def test_the_ar25_solver_still_plans_with_the_constant_palette():
    """The wrapper must not have moved the planning body behind a no-op.

    The target sits on the mirror image of the shape's start, so an arrow is
    genuinely required and a ``None`` answer would mean planning never ran.
    """
    from lingjing_solo.closed_book import ar25_solver as S

    solver = S.Ar25ClosedBookSolver()
    planned = [solver.next_action(_as_frame(T.ar25_like(p, targets=((4, 14),))), 0)
               for p in T.AV_PATH]
    assert any(pick is not None and pick[0] in (1, 2, 3, 4) for pick in planned)
    assert solver._last_act in (1, 2, 3, 4)
