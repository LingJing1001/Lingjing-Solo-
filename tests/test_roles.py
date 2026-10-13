"""Unit tests for the generic palette-role inference (``core/roles.py``).

Every scene here is *synthetic but faithful*: the board is quantised into real
cells, the avatar occupies whole cells, and the palette matches the one the
hand-written solvers use (tu93: VOID 5 / CORRIDOR 0 / CONNECTOR 2 / PLAYER 9 /
HEADING 4 / GEMS 11,15 / HUD 6; ar25: CURSOR 0 / REFLECT 4 / SHAPE 5 / BG 9 /
AXIS 10 / TARGET 11).  The point of the file is that **no test names a colour to
the module** -- every role is asserted from evidence the tracker measured, so
the palette-permutation test can prove the inference is colour-agnostic.
"""
from __future__ import annotations

import numpy as np
import pytest

from lingjing_solo.core.roles import (
    ACTION_DIR,
    RoleSpec,
    RoleTracker,
    infer_roles,
    rigid_shift,
)

CELL, N = 3, 21
SIZE = 64


# --------------------------------------------------------------------------- #
# scenes
# --------------------------------------------------------------------------- #
def _cells_to_frame(cells):
    """Quantise a 21x21 cell board into a 64x64 frame (VOID 5 outside the board)."""
    g = np.full((SIZE, SIZE), 5, dtype=np.int64)
    for r in range(N):
        for c in range(N):
            g[CELL * r:CELL * r + CELL, CELL * c:CELL * c + CELL] = cells[r, c]
    return g


def tu93_like(player, *, hud_len=None, heading=(0, 1), gem=(7, 7), extra=None):
    """A tu93-shaped board: sparse maze on VOID + a HUD bar in the frame margin.

    ``extra`` paints additional cells (e.g. a hazard) *before* the player, so the
    player is never erased by the decoration it stands on.

    ``hud_len`` defaults to a position-dependent length, mirroring the real timer
    bar (measured: 20 px on a late frame, and *repainting*).  A frozen bar would
    not be found at all: the module's rule is that chrome **repaints in place**,
    and the real HUD does -- see ``_animated``.  Pass an int to pin it.
    """
    cells = np.full((N, N), 5, dtype=np.int64)          # VOID
    cells[10, :] = 0                                    # CORRIDOR
    cells[:, 10] = 0
    cells[3:5, 3] = 2                                   # CONNECTOR
    cells[3, 3:6] = 2
    cells[gem] = 11                                     # GEMS
    cells[13, 13] = 15
    for (r, c), colour in (extra or {}).items():
        cells[r, c] = colour
    g = _cells_to_frame(cells)
    pr, pc = player
    g[CELL * pr:CELL * pr + CELL, CELL * pc:CELL * pc + CELL] = 9     # PLAYER
    if heading is not None:
        g[CELL * pr + heading[0], CELL * pc + heading[1]] = 4         # HEADING dot
    if hud_len is None:
        hud_len = 6 + 2 * ((pr + pc) % 4)                             # repaints each step
    g[63, 0:hud_len] = 6                                             # HUD bar
    return g


def ar25_like(av, *, targets=((3, 4), (15, 17)), cursor=True, axis_c=10):
    """ar25-shaped board: avatar + its reflection + a sparse selection star.

    Two details are load-bearing and both were lifted from the real frame:

    * the mirror axis is drawn **one cell wide** (3 px).  A 1 px line breaks the
      3 px period, the generic detector then falls back to ``cell=1``, and every
      measurement degrades to the pixel grid the module exists to avoid;
    * the cursor is a **5 px star in a 7x7 box** (fill 0.10), not a solid ring.
      The sparsity is what keeps it from competing for the avatar.
    """
    g = np.full((SIZE, SIZE), 9, dtype=np.int64)         # COL_BG
    g[:, axis_c * CELL:axis_c * CELL + CELL] = 10        # COL_AXIS (one cell wide)
    for (tr, tc) in targets:
        g[CELL * tr:CELL * tr + CELL, CELL * tc:CELL * tc + CELL] = 11   # COL_TARGET
    ar, ac = av
    mc = 2 * axis_c - ac
    g[CELL * ar:CELL * ar + CELL, CELL * mc:CELL * mc + CELL] = 4        # COL_REFLECT
    g[CELL * ar:CELL * ar + CELL, CELL * ac:CELL * ac + CELL] = 5        # COL_SHAPE
    if cursor:
        r0, c0 = CELL * ar, CELL * ac
        for dr, dc in ((1, 1), (-2, -2), (-2, 4), (4, -2), (4, 4)):
            g[r0 + dr, c0 + dc] = 0                                      # COL_CURSOR star
    return g


def replay(frames, actions, levels=None, states=None, **kw):
    rt = RoleTracker(**kw)
    snap = None
    n = len(frames)
    levels = levels if levels is not None else [0] * n
    states = states if states is not None else ["PLAYING"] * n
    for i, f in enumerate(frames):
        snap = rt.observe(f, action=actions[i], levels=levels[i], state=states[i])
    return snap


# the consecutive actions all differ, so a lag-0 and a lag-1 explanation cannot
# both score -- that is what makes the render-lag assertions below meaningful
WALK = [(6, 10), (7, 10), (7, 11), (6, 11), (6, 12), (7, 12), (7, 11)]
WALK_ACTS = [None, 2, 4, 1, 4, 2, 3]      # down right up right down left


# --------------------------------------------------------------------------- #
# tu93-shaped scene
# --------------------------------------------------------------------------- #
def test_tu93_scene_roles_without_render_lag():
    snap = replay([tu93_like(p) for p in WALK], WALK_ACTS)
    assert snap.background == 5
    assert snap.avatar == 9
    assert snap.hud == {6}
    assert snap.mirror is None and snap.cursor is None
    assert set(snap.terrain) == {0, 2, 11, 15}
    assert snap.role_of(9) == "avatar"
    assert snap.role_of(5) == "background"
    assert snap.role_of(6) == "hud"


def test_an_always_moving_avatar_is_not_swallowed_by_the_animation_rule():
    """The avatar moved on every transition; "changes every step" must not
    brand it chrome -- only *in-place* change does."""
    snap = replay([tu93_like(p) for p in WALK], WALK_ACTS)
    assert snap.steps == len(WALK)
    assert snap.evidence["transitions"] >= 4, "the animation rule needs its warmup"
    assert 9 not in snap.hud
    assert 9 not in snap.evidence["self_animating"], "a mover is never chrome"


def test_the_one_pixel_heading_dot_does_not_steal_the_avatar():
    """The dot rides along with the player and agrees with the action 100% of
    the time; only the "at least half a cell" floor keeps it from winning.

    It is reported in ``attached`` rather than ``mobile`` because that is what it
    is on the real frame: a decoration *inside* the avatar's body.  ``role_of``
    still names it an honest mover.
    """
    snap = replay([tu93_like(p, heading=(0, 1)) for p in WALK], WALK_ACTS)
    assert snap.avatar == 9
    assert 4 in snap.attached, "the dot rides inside the player's own cell"
    assert snap.role_of(4) == "mobile", "an attached colour is still an honest mover"


def test_a_flickering_hud_bar_is_found_as_self_animating():
    """The bar repaints every step -- which is the rule that finds it.

    Measured on the real frame: tu93's timer bar repaints in place and never
    translates, and the "outside the lattice" clause does *not* fire there (a
    21x21 board in a 64x64 frame leaves only a 1px margin, below
    ``HUD_OUTSIDE_MIN_FRAC``).  So the load-bearing rule is self-animation.
    """
    frames = [tu93_like(p, hud_len=6 + (i % 3) * 4) for i, p in enumerate(WALK)]
    snap = replay(frames, WALK_ACTS)
    assert snap.hud == {6}
    assert snap.evidence["self_animating"] == [6]
    assert snap.avatar == 9, "chrome must not disturb the avatar call"


def test_a_hud_panel_outside_a_coarse_lattice_is_found():
    """When the board does *not* tile the frame, what lies outside it is a panel.

    A 3 px board leaves a 1 px margin in a 64 px frame -- 3.1%, below
    ``HUD_OUTSIDE_MIN_FRAC`` -- so this clause only fires on a *coarser* lattice:
    5 px cells over 12 rows leave a 4 px margin (12%).  Unlike tu93's bar this one
    is static, which is the point: the clause finds chrome without needing it to
    repaint.
    """
    n, cell = 12, 5

    def board(player, hud_len=12):
        cells = np.full((n, n), 5, dtype=np.int64)       # VOID
        cells[6, :] = 0                                  # CORRIDOR
        cells[:, 6] = 0
        g = np.full((SIZE, SIZE), 5, dtype=np.int64)
        for r in range(n):
            for c in range(n):
                g[cell * r:cell * r + cell, cell * c:cell * c + cell] = cells[r, c]
        pr, pc = player
        g[cell * pr:cell * pr + cell, cell * pc:cell * pc + cell] = 9   # PLAYER
        g[63, 0:hud_len] = 6                                            # HUD bar
        return g

    path = [(5, 6), (6, 6), (6, 7), (5, 7), (5, 8)]
    snap = replay([board(p) for p in path], [None, 2, 4, 1, 4])
    assert snap.hud == {6}
    assert snap.evidence["hud_outside_lattice"] == [6]
    assert snap.avatar == 9


def test_render_lag_is_absorbed_by_the_lag_window():
    """Frames that trail the action by one step must still agree at lag 1."""
    lagged = [WALK[0]] + WALK[:-1]              # frame i shows position i-1
    snap = replay([tu93_like(p) for p in lagged], WALK_ACTS)
    assert snap.avatar == 9
    assert snap.evidence["colours"]["9"]["lag"] == 1
    assert snap.evidence["colours"]["9"]["agree_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# ar25-shaped scene
# --------------------------------------------------------------------------- #
AV_PATH = [(4, 4), (4, 5), (5, 5), (5, 6), (4, 6), (4, 7)]
AV_ACTS = [None, 4, 2, 4, 1, 4]


def test_ar25_scene_roles():
    snap = replay([ar25_like(p) for p in AV_PATH], AV_ACTS)
    assert snap.background == 9
    assert snap.avatar == 5
    assert snap.mirror == 4, "the reflection moves opposite the avatar"
    assert snap.cursor == 0, "the 1px ring hugs the avatar and follows it"
    assert set(snap.terrain) == {10, 11}
    assert snap.role_of(0) == "cursor" and snap.role_of(5) == "avatar"


def test_two_far_apart_targets_are_not_mistaken_for_a_cursor():
    """Two 3x3 targets over one bounding box look 'thin' but do not hug the
    avatar, so the hugging rule is what rejects them."""
    snap = replay([ar25_like(p) for p in AV_PATH], AV_ACTS)
    assert 11 not in (snap.cursor,)
    assert 11 in snap.terrain


def test_the_reflection_alone_is_not_called_a_cursor():
    snap = replay([ar25_like(p, cursor=False) for p in AV_PATH], AV_ACTS)
    assert snap.avatar == 5
    assert snap.mirror == 4
    assert snap.cursor is None


# --------------------------------------------------------------------------- #
# nothing may depend on the colour *values*
# --------------------------------------------------------------------------- #
PERMUTATION = {0: 7, 2: 3, 4: 6, 5: 12, 6: 1, 8: 2, 9: 15, 11: 0, 15: 4, 10: 13}


def _remap(frame, mapping):
    out = frame.copy()
    for src, dst in mapping.items():
        out[frame == src] = dst
    return out


def test_roles_survive_a_whole_palette_permutation():
    plain = replay([tu93_like(p) for p in WALK], WALK_ACTS)
    moved = replay([_remap(tu93_like(p), PERMUTATION) for p in WALK], WALK_ACTS)
    assert moved.background == PERMUTATION[plain.background]
    assert moved.avatar == PERMUTATION[plain.avatar]
    assert moved.hud == {PERMUTATION[c] for c in plain.hud}
    assert set(moved.terrain) == {PERMUTATION[c] for c in plain.terrain}


def test_ar25_roles_survive_a_palette_permutation():
    plain = replay([ar25_like(p) for p in AV_PATH], AV_ACTS)
    moved = replay([_remap(ar25_like(p), PERMUTATION) for p in AV_PATH], AV_ACTS)
    assert moved.background == PERMUTATION[plain.background]
    assert moved.avatar == PERMUTATION[plain.avatar]
    assert moved.mirror == PERMUTATION[plain.mirror]
    assert moved.cursor == PERMUTATION[plain.cursor]


# --------------------------------------------------------------------------- #
# goal / lethal are evidence-gated, never guessed
# --------------------------------------------------------------------------- #
def test_goal_needs_an_observed_level_increment():
    frames = [tu93_like((7, 10), gem=(6, 10)),      # the gem sits where the player steps
              tu93_like((6, 10), gem=(6, 10))]
    snap = replay(frames, [None, 2], levels=[0, 1])
    assert snap.evidence["level_ups"] == 1
    assert 11 in snap.goal, "the gem that vanished on the level-up is the goal"
    assert "goal needs a level increment" not in snap.note


def test_goal_stays_empty_without_a_level_increment():
    snap = replay([tu93_like(p) for p in WALK], WALK_ACTS)
    assert snap.goal == frozenset()
    assert "goal needs a level increment" in snap.note


def test_lethal_needs_an_observed_death():
    """Walk one step into the hazard cell and lose; the colour at the attempted
    destination is the lethal one."""
    hazard = {(8, 12): 8}

    path = [(8, 10), (8, 11), (8, 12), (8, 11)]
    acts = [None, 4, 4, 3]
    frames = [tu93_like(p, extra=hazard) for p in path]
    snap = replay(frames, acts, states=["PLAYING", "PLAYING", "LOSE", "PLAYING"])
    assert snap.avatar == 9, "the avatar call must survive the death"
    assert snap.evidence["deaths"] >= 1
    assert snap.lethal == {8}


def test_lethal_is_not_guessed_when_nothing_died():
    snap = replay([tu93_like(p) for p in WALK], WALK_ACTS)
    assert snap.lethal == frozenset()
    assert "lethal needs an observed death" in snap.note


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def test_a_level_change_resets_the_accumulators():
    first = replay([tu93_like(p) for p in WALK], WALK_ACTS, levels=[0] * len(WALK))
    assert first.avatar == 9
    crossing = replay([tu93_like(p) for p in WALK], WALK_ACTS,
                      levels=[0, 0, 0, 1, 1, 1, 1])
    assert crossing.evidence["level_ups"] == 1
    assert crossing.steps == 4, "three transitions were kept, the rest dropped"


def test_snapshot_does_not_retain_frames():
    """Only the newest frame and its cell-space twin may be held.

    ``_cells`` / ``_cells_prev`` were added when motion moved to the cell grid
    (``cell_shift`` needs two consecutive boards); they are *current-state*
    caches, not a growing history -- nothing accumulates over the run.
    """
    rt = RoleTracker(background=5)
    for p, a in zip(WALK, WALK_ACTS):
        rt.observe(tu93_like(p), action=a, levels=0)
    state = {k: v for k, v in vars(rt).items()}
    arrays = sorted(k for k, v in state.items() if isinstance(v, np.ndarray))
    assert arrays == ["_cells", "_cells_prev", "_prev"], f"unexpected retained frames: {arrays}"


# --------------------------------------------------------------------------- #
# rigid_shift
# --------------------------------------------------------------------------- #
def _sprite(shape, r, c, colour, size=16):
    g = np.zeros((size, size), dtype=np.int64)
    g[r:r + shape[0], c:c + shape[1]] = colour
    return g


def test_rigid_shift_reads_an_integer_translation():
    a = _sprite((3, 3), 4, 4, 7)
    b = _sprite((3, 3), 5, 6, 7)
    assert rigid_shift(a, b, 7) == (1, 2)


def test_rigid_shift_is_none_when_nothing_moved_or_the_colour_is_absent():
    a = _sprite((3, 3), 4, 4, 7)
    assert rigid_shift(a, a.copy(), 7) is None
    assert rigid_shift(a, np.zeros((16, 16), dtype=np.int64), 7) is None
    assert rigid_shift(a, a.copy(), 3) is None


def test_rigid_shift_refuses_a_wraparound():
    """A sprite vanishing off the left edge and a same-count sprite appearing on
    the right must not be read as one sprite teleporting across the frame."""
    a = _sprite((1, 1), 4, 0, 7)
    b = _sprite((1, 1), 4, 63, 7)
    assert rigid_shift(a, b, 7) is None


def test_rigid_shift_tolerates_a_one_pixel_decoration_punch():
    """tu93's player carries a heading pixel of another colour: 9 px becomes 8."""
    a = _sprite((3, 3), 4, 4, 7)
    b = _sprite((3, 3), 5, 5, 7)
    b[5, 5] = 4
    assert rigid_shift(a, b, 7) == (1, 1)


# --------------------------------------------------------------------------- #
# degenerate input and plumbing
# --------------------------------------------------------------------------- #
def test_degenerate_input_is_safe():
    assert RoleTracker().observe(None).avatar is None
    # an unusable frame is not counted as a step at all
    assert RoleTracker().observe(np.zeros((0, 0), dtype=np.int64)).steps == 0
    assert RoleSpec(note="x").role_of(9) is None
    assert infer_roles([]).note == "empty trajectory"


def test_replay_helper_and_unknown_kwarg():
    spec = infer_roles([tu93_like(p) for p in WALK], WALK_ACTS)
    assert spec.avatar == 9
    with pytest.raises(TypeError):
        RoleSpec(nonsense=1)


def test_role_spec_serialises():
    snap = replay([tu93_like(p) for p in WALK], WALK_ACTS)
    d = snap.to_dict()
    assert d["avatar"] == 9 and d["background"] == 5 and d["hud"] == [6]
    assert d["terrain"] == [0, 2, 11, 15]
    assert isinstance(d["evidence"]["colours"], dict)


def test_action_direction_table_matches_arc_agi_3():
    assert ACTION_DIR == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def test_an_explicit_spec_skips_detection():
    from lingjing_solo.core.lattice import detect_cell_lattice

    spec = detect_cell_lattice(tu93_like((6, 10)), max_cell=8, background=5)
    assert spec.cell == (CELL, CELL) and spec.shape == (N, N)
    snap = replay([tu93_like(p) for p in WALK], WALK_ACTS, spec=spec)
    assert snap.avatar == 9 and snap.hud == {6}
