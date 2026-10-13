"""Generic palette-role inference from a frame trajectory (pure numpy).

Why this module exists
----------------------
The two hand-written closed-book solvers name their colours in module constants::

    tu93_frame.py : VOID=5 CONNECTOR=2 CORRIDOR=0 PLAYER=9 HEADING=4 TARGET=14
                    DOORS=(12,13) GEMS=(11,15) HUD=6 DEADLY=(8,)
    ar25_frame.py : COL_CURSOR=0 COL_REFLECT=4 COL_SHAPE=5 COL_BG=9
                    COL_AXIS=10 COL_TARGET=11

That is the M1 list's prior **#3** ("who is the player / the wall / the target /
the chrome") -- the one prior with no generic implementation, and the one every
other prior consumes.  This module derives those roles by *measurement* over the
trajectory the agent already collects.  Nothing here reads the engine, an
``environment_files`` path, or a level table.

Two spaces, because the two questions are different
---------------------------------------------------
**Motion is measured on the cell grid, not on pixels.**  tu93's player is a 3x3
cell whose "heading" pixel is punched out of a *different* position on every
step: the set of pixels of the player's colour is therefore *not* a translate of
the previous set (measured: the pixel bounding box moves by (-5, +1) on a step
where the true cell move is (-2, 0)).  Pixel-space rigid matching picks exactly
that bogus offset and the action test then scores 48% instead of 95%.  On the
majority-colour cell grid the decoration disappears and the block translates by
an integer number of cells -- measured 95% action agreement at lag 2 over 138
real moves.  So:

* **cell space** (majority colour per lattice cell) answers *what moves*:
  ``mobile`` / ``avatar`` / ``mirror``.
* **pixel space** answers *what is drawn*: the chrome outside the lattice, the
  1px selection ring around the avatar, and the decorations that ride inside the
  avatar's glyph.  Those never survive quantisation, which is why they need
  their own pass.

The measurable signals
----------------------
1. **background**  the modal colour of the frame (``lattice.infer_background``).
2. **hud**         pixels (a) outside the lattice region, or (b) self-animating
                   *in place* on >= ``ANIM_RATIO`` of steps while never
                   translating.
3. **mobile**      a cell-space colour whose cell mask rigidly translates by an
                   integer number of cells: the operational meaning of "a moving
                   block" with no object, sprite or tracker.
4. **avatar**      the mobile colour whose translation *direction agrees with the
                   action taken*.  ARC-AGI-3's arrow actions are 1=up 2=down
                   3=left 4=right, so a cell shift of ``(-2, 0)`` after action 1
                   is agreement.
5. **mirror**      a mobile colour whose shift is the *negation* of the avatar's
                   on the steps where both moved (ar25's reflected image).
6. **cursor**      a small, thin pixel-space colour that wraps the avatar's bbox
                   and changes on the steps the avatar moves (ar25's 1px ring).
7. **attached**    small pixel-space colours that change in lockstep but ride
                   *inside* the avatar's bbox (tu93's heading pixel).
8. **terrain**     every other colour seen: walls, corridors, axes, targets.
                   Which terrain is *passable* is deliberately **not** decided
                   here -- that is a move-model question (priors #4/#6).
9. **goal**        only from an observed level increment: the colours the avatar
                   stood on, or that vanished, as the level completed.
10. **lethal**     only from an observed death (``state`` LOSE / GAME_OVER): the
                   colour at the avatar's attempted destination.  Never guessed.

Three measurement details that are load-bearing (each was a bug first)
---------------------------------------------------------------------
* **The rendered frame trails the action.**  Measured on tu93: the true lag is
  **2** (95% agreement at lag 2, 36% at lag 0, 33% at lag 1) and a player step is
  **2 cells**.  The tracker therefore keeps the best-scoring lag in
  ``range(MAX_LAG + 1)`` rather than assuming the action and the frame line up.
* **Level runs must not be pooled**: a palette index means different things on
  different levels (ar25 re-draws its axis), so every accumulator restarts when
  ``levels`` changes, and the transition across a level change is not a move.
  A level *decrease* (a RESET) is not a level-up and must not vote for ``goal``.
* **An always-moving avatar is not chrome.**  "Changes on every step" also
  describes a block the agent commands, so the animation rule only fires for
  colours that never translate.

Cost: no frames are retained.  Per transition there is one boolean reduction per
colour for the change/animation signals, and one rigid-match per colour on the
small cell grid; the pixel pass runs only for colours small enough to be chrome.
"""
from __future__ import annotations

import numpy as np

from .lattice import as_grid, detect_cell_lattice, infer_background, to_cell_grid

__all__ = [
    "ACTION_DIR",
    "ANIM_RATIO",
    "MAX_LAG",
    "RoleSpec",
    "RoleTracker",
    "cell_shift",
    "infer_roles",
    "rigid_shift",
]

#: A colour that changes on >= this fraction of steps *without ever translating*
#: is self-animating (HUD) -- subject to the guards in ``RoleTracker._animated``.
ANIM_RATIO = 0.9

#: ...and whose pixels stay inside a panel of at most this share of the frame.
#: This is the clause that separates chrome from a repainting board feature:
#: tu93's timer bar spans 1x34 px (0.8% of the frame) while its doors and gems,
#: which also repaint on 97% of steps and are equally small, wander over ~1000 px
#: because each one sits in a different cell.
ANIM_SPAN_FRAC = 0.08

#: Frame transitions to gather before history-based signals are trusted.
WARMUP = 4

#: How many actions back the rendered frame may lag (measured: 2 on tu93).
MAX_LAG = 3

#: Largest rigid translation, in pixels, considered a sprite move (pixel space).
MAX_SHIFT = 24

#: Largest rigid translation, in cells, considered a sprite move (cell space).
MAX_SHIFT_CELLS = 12

#: Overlap required, relative to the larger of the two sets, for "rigid".
OVERLAP_MIN = 0.75

#: A mobile block occupies at most this share of the frame / cell board.
MOBILE_MAX_FRAC = 0.06

#: Minimum rigid-translation events before a colour is called mobile.
MIN_MOVES = 2

#: Action-direction agreement before a mobile colour is called the avatar.
AGREE_MIN = 0.6

#: npix / bbox-area above which a colour reads as a solid blob, not an outline.
#: ar25's 1px selection ring around a 3x3 block scores 16/25 = 0.64; a full 3x3
#: block scores 1.0, so the threshold sits clear of both.
OUTLINE_FILL = 0.75

#: How much of the frame must lie outside the lattice before "outside the
#: lattice" means anything.  A 64x64 frame with a 63x63 board leaves a 1px canvas
#: edge -- 3% of the frame -- and pixels there are the board's own content
#: clipped by the canvas, not a panel.  Measured: ar25's shape and target colours
#: both touch that edge, which made them look like chrome.
HUD_OUTSIDE_MIN_FRAC = 0.05

#: A cursor occupies at most this share of the frame.
CURSOR_MAX_FRAC = 0.03

#: ARC-AGI-3 arrow semantics: 1=up 2=down 3=left 4=right, as (drow, dcol).
ACTION_DIR = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


class RoleSpec:
    """A snapshot of the roles supported by the evidence so far.

    A plain class on purpose: ``benchmark_all_games._load_agent`` loads agent
    modules with ``spec_from_file_location`` **without** registering
    ``sys.modules``, and a module-level ``@dataclass`` then raises
    ``AttributeError`` while resolving annotations -- the same trap that keeps
    ``lattice.LatticeSpec`` a plain class.
    """

    __slots__ = (
        "background", "hud", "mobile", "avatar", "mirror", "cursor", "attached",
        "terrain", "goal", "lethal", "evidence", "steps", "settled", "note",
    )

    def __init__(self, **kw):
        self.background = None
        self.hud = frozenset()
        self.mobile = frozenset()
        self.avatar = None
        self.mirror = None
        self.cursor = None
        self.attached = frozenset()
        self.terrain = frozenset()
        self.goal = frozenset()
        self.lethal = frozenset()
        self.evidence = {}
        self.steps = 0
        self.settled = False
        self.note = ""
        for key, value in kw.items():
            if key not in self.__slots__:
                raise TypeError(f"RoleSpec has no field {key!r}")
            setattr(self, key, value)

    def role_of(self, colour):
        """Name the role a colour currently plays (``None`` if never seen).

        Order matters: the avatar is by construction *also* in ``mobile``, so the
        specific labels win over the general ones.
        """
        colour = int(colour)
        if colour == self.background:
            return "background"
        for name in ("hud", "goal", "lethal"):
            if colour in getattr(self, name):
                return name
        if colour == self.avatar:
            return "avatar"
        if colour == self.mirror:
            return "mirror"
        if colour == self.cursor:
            return "cursor"
        # `attached` colours ride the avatar without being it; calling them
        # "mobile" is the honest label -- they do move with the commanded piece.
        if colour in self.mobile or colour in self.attached:
            return "mobile"
        if colour in self.terrain:
            return "terrain"
        return None

    def to_dict(self):
        return {
            "background": self.background,
            "hud": sorted(self.hud),
            "avatar": self.avatar,
            "mirror": self.mirror,
            "cursor": self.cursor,
            "mobile": sorted(self.mobile),
            "attached": sorted(self.attached),
            "terrain": sorted(self.terrain),
            "goal": sorted(self.goal),
            "lethal": sorted(self.lethal),
            "steps": self.steps,
            "settled": self.settled,
            "evidence": self.evidence,
            "note": self.note,
        }

    def __repr__(self):
        return (f"RoleSpec(bg={self.background}, avatar={self.avatar}, "
                f"mirror={self.mirror}, cursor={self.cursor}, hud={sorted(self.hud)}, "
                f"attached={sorted(self.attached)}, goal={sorted(self.goal)}, "
                f"lethal={sorted(self.lethal)}, terrain={sorted(self.terrain)}, "
                f"steps={self.steps})")


# --------------------------------------------------------------------------- #
# measurement primitives
# --------------------------------------------------------------------------- #
def _overlap(before_mask, after_mask, dr, dc):
    """Pixels of ``after`` covered by ``before`` translated by ``(dr, dc)``.

    Slicing rather than ``np.roll`` so nothing wraps around an edge -- a wrap
    would let a sprite leaving the frame "arrive" on the far side.
    """
    if before_mask.shape != after_mask.shape:
        return 0
    h, w = before_mask.shape
    r_lo, r_hi = max(0, dr), min(h, h + dr)
    c_lo, c_hi = max(0, dc), min(w, w + dc)
    if r_lo >= r_hi or c_lo >= c_hi:
        return 0
    return int(np.count_nonzero(
        after_mask[r_lo:r_hi, c_lo:c_hi]
        & before_mask[r_lo - dr:r_hi - dr, c_lo - dc:c_hi - dc]
    ))


def _rigid_shift_masks(b, a, *, max_shift=MAX_SHIFT, overlap_min=OVERLAP_MIN):
    """The integer ``(dr, dc)`` translating bool mask ``b`` onto ``a``.

    ``None`` when the mask did not move rigidly: empty in either frame, count
    changed beyond tolerance (a 3x3 sprite carrying one "heading" pixel of
    another colour legitimately varies by one), no candidate shift explains
    ``overlap_min`` of the mask, or the best explanation is "it stayed put".

    Candidates are the rounded centroid offset +-2, which covers any whole-cell
    move and costs ~9 boolean reductions instead of an ``HxW`` search.
    """
    if b.shape != a.shape:
        return None                       # different boards: not comparable
    nb, na = int(b.sum()), int(a.sum())
    if nb == 0 or na == 0:
        return None
    big = max(nb, na)
    if abs(nb - na) > max(2, 0.2 * big):
        return None                       # no longer the same block
    ys, xs = np.where(b)
    cy, cx = float(ys.mean()), float(xs.mean())
    ys, xs = np.where(a)
    dy = int(round(float(ys.mean()) - cy))
    dx = int(round(float(xs.mean()) - cx))
    best = None
    for dr in range(dy - 2, dy + 3):
        for dc in range(dx - 2, dx + 3):
            if abs(dr) > max_shift or abs(dc) > max_shift:
                continue
            ov = _overlap(b, a, dr, dc)
            if best is None or ov > best[0]:
                best = (ov, (dr, dc))
    if best is None or best[1] == (0, 0):
        return None                       # nothing moved
    if best[0] < overlap_min * big:
        return None
    return best[1]


def rigid_shift(before, after, colour, *, max_shift=MAX_SHIFT, overlap_min=OVERLAP_MIN):
    """The integer ``(dr, dc)`` translating *colour*'s mask from before to after.

    ``before`` / ``after`` are int grids -- pixels, or the majority-colour *cell*
    grid from :func:`cell_shift`; pass :func:`_rigid_shift_masks` directly when
    the masks are already cropped.
    """
    return _rigid_shift_masks(before == colour, after == colour,
                              max_shift=max_shift, overlap_min=overlap_min)


def _cropped_rigid(before, after, colour, pad=3, exact=False):
    """Rigid shift of *colour*, matched only inside its own bounding box.

    The pixel-space pass runs for small colours on every step, and matching on
    the full 64x64 frame would spend 4096 boolean ops per candidate shift where a
    few hundred suffice -- these masks are decoration-sized by construction.

    ``exact`` demands the pixel count be unchanged.  It is what tells a *moving*
    marker (ar25's cursor: 5 px before, 5 px after) from a *growing* one (tu93's
    timer bar: 26 px of row 63 becoming 32 px reads as a 6 px translation under
    the loose test).
    """
    b = before == colour
    a = after == colour
    if not b.any() or not a.any():
        return None
    if exact and int(b.sum()) != int(a.sum()):
        return None
    ys, xs = np.where(b | a)
    r0 = max(0, int(ys.min()) - pad)
    c0 = max(0, int(xs.min()) - pad)
    r1 = min(before.shape[0], int(ys.max()) + 1 + pad)
    c1 = min(before.shape[1], int(xs.max()) + 1 + pad)
    return _rigid_shift_masks(b[r0:r1, c0:c1], a[r0:r1, c0:c1])


def cell_shift(cells_before, cells_after, colour):
    """Rigid translation of *colour* in **cell** units.

    The measure that actually works on ARC-AGI-3 frames: tu93's player punches a
    "heading" pixel out of a new position inside its 3x3 cell every step, so its
    *pixel* mask is not a translate of the previous one, while its *cell* mask is.
    """
    return rigid_shift(cells_before, cells_after, colour, max_shift=MAX_SHIFT_CELLS)


def _bbox_of(mask):
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _fill_ratio(mask):
    box = _bbox_of(mask)
    if box is None:
        return 0.0
    r0, c0, r1, c1 = box
    return float(mask.sum()) / max(1, (r1 - r0 + 1) * (c1 - c0 + 1))


def _sign_dir(shift):
    """Direction of an axis-aligned shift, else ``None``."""
    dr, dc = shift
    if dr and dc:
        return None
    if dr:
        return (-1 if dr < 0 else 1, 0)
    if dc:
        return (0, -1 if dc < 0 else 1)
    return (0, 0)


def _contains(outer, inner, margin):
    """Is bbox *inner* inside bbox *outer*, expanded by ``margin``?"""
    if outer is None or inner is None:
        return False
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (or0 - margin <= ir0 and ir1 <= or1 + margin
            and oc0 - margin <= ic0 and ic1 <= oc1 + margin)


def _hugs(outer, inner, margin):
    """Is bbox *outer* within ``margin`` of bbox *inner* on every side?

    Stricter than :func:`_contains`: a selection cursor is drawn *around* the
    avatar, so its box may be a pixel or two larger but cannot be a different
    part of the board.  Without this rule a board carrying two 3x3 targets looks
    "thin" over their combined bounding box (18 px in a 39x42 box) and gets
    mistaken for a selection outline.
    """
    if outer is None or inner is None:
        return False
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (or0 >= ir0 - margin and oc0 >= ic0 - margin
            and or1 <= ir1 + margin and oc1 <= ic1 + margin)


def _largest_component(mask, connectivity=8):
    """Coordinates of the largest blob in a bool mask, as ``[(r, c), ...]``.

    Needed because one palette index can do double duty: ar25 paints both its
    board shape *and* a panel with the same colour, so "the bbox of colour 5" is
    the whole frame.  The board object is the biggest connected piece.
    """
    if not mask.any():
        return []
    seen = np.zeros_like(mask, dtype=bool)
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr, dc) != (0, 0)]
    if connectivity == 4:
        offsets = [(dr, dc) for dr, dc in offsets if dr == 0 or dc == 0]
    h, w = mask.shape
    best: list = []
    for y0, x0 in zip(*np.where(mask)):
        y0, x0 = int(y0), int(x0)
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dr, dc in offsets:
                ny, nx = y + dr, x + dc
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    return best


# --------------------------------------------------------------------------- #
# the tracker
# --------------------------------------------------------------------------- #
class RoleTracker:
    """Fold frames in one at a time; read the roles off the accumulated evidence.

    Typical agent-loop use::

        rt = RoleTracker()
        ...
        if level_changed:
            rt.reset()
        roles = rt.observe(frame, action=act, levels=levels, state=state)
        if roles.avatar is not None:
            # the cell-space colour the agent commands, found without naming it

    Each call returns a fresh :class:`RoleSpec`; only counters are retained, never
    frames, so a caller can inspect the evidence mid-level for free.
    """

    def __init__(self, *, background=None, spec=None, warmup=WARMUP,
                 anim_ratio=ANIM_RATIO):
        self.fixed_background = background
        self.fixed_spec = spec
        self.warmup = int(warmup)
        self.anim_ratio = float(anim_ratio)
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self):
        """Forget the current level (call on a level change or a RESET)."""
        self._prev = None
        self._cells = None            # cell grid of the newest frame
        self._cells_key = None        # lattice that produced it
        self._cells_prev = None       # cell grid of the frame before it
        self._cells_key_prev = None
        self._actions = []            # plain list: lag indexing needs stable indices
        self._shift_log = []          # per transition: {colour: cell shift}
        self._changed_log = []        # per transition: small colours whose mask changed
        self._stats = {}              # colour -> {"moves": int, "agree": [int]*n}
        self._tick = {}               # colour -> steps on which it changed
        self._diffs = 0
        self._seen = set()            # colours seen on this level
        self._npix_px = {}            # colour -> pixel count in the newest frame
        self._absent = {}             # colour -> frames it was missing from
        self._span = {}               # colour -> union bbox over the level
        self._moved_px = {}           # colour -> steps it translated rigidly in pixels
        self._level = None
        self._levelups = []           # (before_grid, after_grid) at a level increase
        self._deaths = []             # (before_grid, after_grid, action)
        self._steps = 0
        self._spec = self.fixed_spec

    # -- helpers -------------------------------------------------------------
    def _spec_for(self, grid, background=None):
        """The lattice, re-detected while the evidence is still thin.

        A level often opens on a blank or half-drawn frame, and caching that
        degenerate reading would defeat the ``outside the lattice`` HUD test for
        the whole level -- so the detection is retried until ``warmup``
        transitions have been seen.

        ``background`` is forwarded: on a board dominated by an *interior* colour
        a freshly inferred background is the interior, not the void, and the
        lattice then lands on the wrong phase entirely.
        """
        if self.fixed_spec is not None:
            return self.fixed_spec
        if (self._spec is None or self._diffs < self.warmup
                or getattr(self._spec, "source", "") == "degenerate"):
            try:
                spec = detect_cell_lattice(grid, max_cell=8, background=background)
                if getattr(spec, "source", "") != "degenerate":
                    self._spec = spec
            except Exception:                                      # noqa: BLE001
                pass
        return self._spec

    def _cell_grid(self, grid):
        """``(cell grid, lattice key)`` -- the key lets two frames be compared.

        The lattice is re-detected while the evidence is thin, so the cell grid of
        one frame may come from a different lattice than the next; comparing two
        such grids would be meaningless even when their shapes happen to match.
        """
        spec = self._spec_for(grid, self.fixed_background)
        if spec is None or getattr(spec, "cell", None) is None:
            return None, None
        try:
            cells = to_cell_grid(grid, spec)
        except Exception:                                          # noqa: BLE001
            return None, None
        key = (spec.cell_r, spec.cell_c, spec.origin_r, spec.origin_c,
               spec.n_rows, spec.n_cols)
        return cells, key

    def _inside_mask(self, grid, background=None):
        """Bool mask of the lattice region, or ``None`` when there is no inside/outside.

        ``None`` also when the region *outside* the board is a sliver: with a
        1px canvas edge the test finds only the board's own clipped pixels.
        """
        spec = self._spec_for(grid, background)
        if spec is None or getattr(spec, "cell", None) is None:
            return None
        ce, cr = spec.cell
        mask = np.zeros(grid.shape, dtype=bool)
        mask[spec.origin_r:spec.origin_r + spec.n_rows * ce,
             spec.origin_c:spec.origin_c + spec.n_cols * cr] = True
        outside = int((~mask).sum())
        if not mask.any() or mask.all() or outside < HUD_OUTSIDE_MIN_FRAC * grid.size:
            return None
        return mask

    def _animated(self, rigid):
        """Colours that repaint *in place*, inside a fixed panel, on every step.

        Four guards, each one earned on real frames:

        * a colour that *translates* changes on every step too, so rigid movers
          are excluded -- an avatar moves, chrome stays put.  Movement is checked
          in both spaces: ar25's cursor is a 5px marker with no cell of its own,
          so only the pixel pass catches it;
        * chrome is *small*: tu93 repaints the corridor under the player, so the
          corridor colour changes on 100% of transitions while being most of the
          board;
        * chrome is *always there* -- a board feature can blink in and out;
        * chrome lives in a *fixed panel*: the union of the colour's bounding box
          over the run must stay small.  Without this, tu93's doors and gems
          (small, present every frame, repainting on 97% of steps) are called
          chrome.
        """
        if self._diffs < self.warmup or self._prev is None:
            return set()
        frame = self._prev.size
        span_bound = ANIM_SPAN_FRAC * frame
        out = set()
        for c, n in self._tick.items():
            if n / self._diffs < self.anim_ratio or c in rigid:
                continue
            if self._moved_px.get(c, 0) >= MIN_MOVES:
                continue
            if self._npix_px.get(c, 0) > MOBILE_MAX_FRAC * frame:
                continue
            if self._absent.get(c, 0):
                continue
            span = self._span.get(c)
            if span is None:
                continue
            area = (span[2] - span[0] + 1) * (span[3] - span[1] + 1)
            if area <= span_bound:
                out.add(c)
        return out

    # -- the fold ------------------------------------------------------------
    def observe(self, frame, action=None, levels=None, state=None):
        """Fold one frame in; return the roles currently supported by evidence."""
        grid = as_grid(frame)
        if grid is None:
            return self._snapshot()
        lv = None if levels is None else int(levels)

        crossed = self._level is not None and lv is not None and lv != self._level
        if crossed:
            before = self._prev
            rose = lv > self._level          # a RESET lowers the count -- not a level-up
            self.reset()
            self._level = lv
            if before is not None and rose:
                self._levelups.append((before, grid))
        elif self._level is None:
            self._level = lv

        self._steps += 1
        seen_now = set()
        counts, spans = {}, {}
        bound = MOBILE_MAX_FRAC * grid.size
        for c in np.unique(grid):
            c = int(c)
            mask = grid == c
            counts[c] = int(mask.sum())
            seen_now.add(c)
            if counts[c] <= bound:
                spans[c] = _bbox_of(mask)
        for c in self._seen - seen_now:              # present earlier, gone now
            self._absent[c] = self._absent.get(c, 0) + 1
        self._seen |= seen_now
        self._npix_px = counts
        for c, box in spans.items():
            was = self._span.get(c)
            self._span[c] = box if was is None else (
                min(was[0], box[0]), min(was[1], box[1]),
                max(was[2], box[2]), max(was[3], box[3]))
        # Record the action *before* folding so ``len(_actions) - 1`` inside the
        # fold is the index of the action that produced ``grid`` -- the anchor
        # every lag test is measured from.
        self._actions.append(None if action is None else int(action))
        self._cells, self._cells_key = self._cell_grid(grid)
        if self._prev is not None:
            self._fold_transition(self._prev, grid, action)
            name = str(state or "").upper()
            if name in ("LOSE", "GAME_OVER"):
                self._deaths.append((self._prev, grid, action))

        self._prev = grid
        self._cells_prev = self._cells
        self._cells_key_prev = self._cells_key
        return self._snapshot()

    def _fold_transition(self, before, after, action):
        """Accumulate one frame-to-frame transition.

        Three signals are gathered:

        * ``_tick`` -- a colour ever touching a changed pixel.  Self-animation.
        * ``_changed_log`` -- a *small* colour whose pixel mask changed.  This is
          what "the cursor follows the avatar" is tested against; restricted to
          small colours because chrome is what the test looks for, and the log
          would otherwise cost one reduction per colour per step.
        * ``_shift_log`` -- the rigid translation of every colour **on the cell
          grid** (see the module docstring for why not on pixels).
        """
        changed = after != before
        self._diffs += 1
        size = after.size
        bound = MOBILE_MAX_FRAC * size

        palette = np.unique(np.concatenate([after.ravel(), before.ravel()]))
        touches, small_changed, small_moved = set(), set(), set()
        for c in palette:
            c = int(c)
            m_a = after == c
            m_b = before == c
            n_a, n_b = int(m_a.sum()), int(m_b.sum())
            if changed[m_a].any() or changed[m_b].any():
                touches.add(c)
            if n_a and n_b and min(n_a, n_b) <= bound:
                if not np.array_equal(m_a, m_b):
                    small_changed.add(c)
                if _cropped_rigid(before, after, c, exact=True) is not None:
                    small_moved.add(c)
        for c in touches:
            self._tick[c] = self._tick.get(c, 0) + 1
        for c in small_moved:
            self._moved_px[c] = self._moved_px.get(c, 0) + 1

        # Rigid translation, in cell space when a lattice was found.
        act_idx = len(self._actions) - 1        # action that produced ``after``
        shifts = {}
        cells_b, cells_a = self._cells_prev, self._cells
        same_lattice = (self._cells_key is not None
                        and self._cells_key == self._cells_key_prev)
        if cells_b is not None and cells_a is not None and same_lattice:
            cell_palette = np.unique(np.concatenate([cells_a.ravel(), cells_b.ravel()]))
            for c in cell_palette:
                c = int(c)
                shift = cell_shift(cells_b, cells_a, c)
                if shift is None:
                    continue
                shifts[c] = shift
                rec = self._stats.setdefault(c, {"moves": 0, "agree": [0] * (MAX_LAG + 1)})
                rec["moves"] += 1
                direction = _sign_dir(shift)
                if direction is None:
                    continue
                for lag in range(MAX_LAG + 1):
                    j = act_idx - lag
                    if 0 <= j < len(self._actions):
                        acted = self._actions[j]
                        if acted is not None and ACTION_DIR.get(acted) == direction:
                            rec["agree"][lag] += 1
        self._shift_log.append(shifts)
        self._changed_log.append(small_changed)

    # -- scoring -------------------------------------------------------------
    def _score_stats(self):
        """Finalise per-colour motion evidence into a dict of records (cells)."""
        cells = self._cells
        if cells is None:
            return {}
        out = {}
        for c, rec in self._stats.items():
            moves = rec["moves"]
            lag, agree = max(((k, rec["agree"][k]) for k in range(MAX_LAG + 1)),
                             key=lambda t: (t[1], -t[0]))
            mask = cells == c
            out[c] = {
                "moves": moves,
                "coverage": moves / self._diffs if self._diffs else 0.0,
                "agree": agree,
                "agree_rate": agree / moves if moves else 0.0,
                "lag": lag,
                "npix": int(mask.sum()),
                "fill": _fill_ratio(mask),
                "bbox": _bbox_of(mask),
            }
        return out

    def _avatar_shifts(self, avatar):
        return {i: s for i, shifts in enumerate(self._shift_log)
                for s in (shifts.get(avatar),) if s is not None}

    def _find_mirror(self, mobile, avatar):
        """A mobile colour that is the avatar's *reflection*.

        ar25 mirrors across a **vertical** axis: a horizontal move of the avatar
        ``(0, +1)`` shows up as ``(0, -1)`` on the reflection, while a vertical
        move ``(-1, 0)`` is shared by both.  So the test is not "the exact
        negation" (that would only ever match half the moves) but: **opposite on
        one axis, identical on the other**, with the flip actually exercised a few
        times -- otherwise a colour that simply travels with the avatar would
        score as a mirror on the two zero components.
        """
        av = self._avatar_shifts(avatar)
        best = None
        for c in sorted(mobile):
            if c == avatar:
                continue
            both = vertical = horiz = flip_v = flip_h = 0
            for i, (br, bc) in self._avatar_shifts(c).items():
                if i not in av:
                    continue
                ar_, ac_ = av[i]
                both += 1
                if bc == -ac_ and br == ar_:          # cols mirrored, rows shared
                    vertical += 1
                    flip_v += 1 if ac_ else 0
                if br == -ar_ and bc == ac_:          # rows mirrored, cols shared
                    horiz += 1
                    flip_h += 1 if ar_ else 0
            if both < MIN_MOVES:
                continue
            for hits, flips in ((vertical, flip_v), (horiz, flip_h)):
                if flips >= MIN_MOVES and hits / both >= 0.6:
                    if best is None or hits > best[0]:
                        best = (hits, c)
        return best[1] if best else None

    def _avatar_pixel_bbox(self, colour):
        """Pixel bbox of the *body* of a cell-space colour.

        Not the bbox of every pixel of that colour: ar25 paints its panel with the
        same index as the board shape, so the naive bbox is the whole frame and
        every small colour then looks like it rides the avatar.  The body is the
        largest connected component of the colour's cell mask.
        """
        cells, grid = self._cells, self._prev
        spec = self._spec_for(grid, self.fixed_background) if grid is not None else None
        if cells is None or grid is None or spec is None:
            return None
        comp = _largest_component(cells == colour)
        if not comp:
            return None
        ce, cr = spec.cell
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        h, w = grid.shape
        return (max(0, spec.origin_r + min(rows) * ce),
                max(0, spec.origin_c + min(cols) * cr),
                min(h - 1, spec.origin_r + (max(rows) + 1) * ce - 1),
                min(w - 1, spec.origin_c + (max(cols) + 1) * cr - 1))

    def _small_followers(self, avatar, av_bbox, hud, background):
        """Pixel-space small colours that ride the avatar.

        Returns ``(cursor, attached)``.  Both must change on the steps the avatar
        moves, so a static decoration is not mistaken for either.  They are split
        by *shape*, which is the only thing that tells them apart on real frames:

        * ``cursor`` -- a thin marker (fill well below solid).  ar25 draws 5 px
          inside a 7x7 box (fill 0.10), and a 1px selection ring reads the same
          way: hollow, sparse, wrapping or inside the avatar;
        * ``attached`` -- a solid small blob riding inside the avatar's body.
          tu93's heading pixel is a filled 1x1 (fill 1.0).
        """
        if self._prev is None or avatar is None or av_bbox is None:
            return None, set()
        grid = self._prev
        av_moves = [i for i, shifts in enumerate(self._shift_log) if avatar in shifts]
        if not av_moves:
            return None, set()
        size = grid.size
        cursor, attached = None, set()
        best = None
        for c in sorted(self._seen):
            if c == avatar or c == background or c in hud:
                continue
            mask = grid == c
            npix = int(mask.sum())
            if npix == 0 or npix > MOBILE_MAX_FRAC * size:
                continue
            follows = sum(1 for i in av_moves if c in self._changed_log[i])
            if follows < MIN_MOVES or follows < 0.6 * len(av_moves):
                continue
            box = _bbox_of(mask)
            if not (_contains(av_bbox, box, 2) or _hugs(box, av_bbox, 2)):
                continue
            if _fill_ratio(mask) <= OUTLINE_FILL and npix <= CURSOR_MAX_FRAC * size:
                if best is None or npix > best[0]:
                    best = (npix, c)
            elif _contains(av_bbox, box, 1):
                attached.add(c)
        return (best[1] if best else None), attached

    def _snapshot(self):
        grid = self._prev
        if grid is None:
            return RoleSpec(steps=self._steps, note="no frames yet")
        spec = self._spec_for(grid, self.fixed_background)

        background = self.fixed_background
        if background is None:
            try:
                background = infer_background(grid)
            except Exception:                                      # noqa: BLE001
                background = None

        records = self._score_stats()
        # A colour that ever translated rigidly.  Blocks the "changes often"
        # HUD rule from swallowing an always-moving avatar (see `_animated`).
        rigid = {c for c, r in records.items() if r["moves"] >= MIN_MOVES}

        inside = self._inside_mask(grid, background)
        hud = set()
        if inside is not None:
            for c in self._seen:
                if c in rigid:
                    continue          # chrome stays put; a mover is not chrome
                m = grid == c
                if m.any() and float((~inside[m]).mean()) >= 0.6:
                    hud.add(c)
        animated = self._animated(rigid)
        hud |= animated
        hud.discard(background)

        # A mobile colour is one that rigidly translates.  Action agreement is a
        # *separate* test: it promotes one mover to `avatar`, and its absence
        # (ar25's reflected image) leaves the colour in `mobile` as a mirror
        # candidate -- exactly the distinction prior #3 needs.
        cells = self._cells
        board = cells.size if cells is not None else grid.size
        mobile = {c for c, r in records.items()
                  if c not in hud
                  and r["moves"] >= MIN_MOVES
                  and r["npix"] <= MOBILE_MAX_FRAC * board}
        ranked = sorted((c for c in mobile if records[c]["agree_rate"] >= AGREE_MIN),
                        key=lambda c: (-records[c]["agree_rate"],
                                       -records[c]["coverage"],
                                       -records[c]["fill"],
                                       records[c]["npix"], c))
        avatar = ranked[0] if ranked else None
        cursor, attached = (self._small_followers(avatar, self._avatar_pixel_bbox(avatar),
                                                  hud, background)
                            if avatar is not None else (None, set()))
        # ``cursor is None`` rather than ``not cursor``: palette index 0 is a
        # perfectly legal cursor (ar25's is exactly 0) and ``not 0`` is True, so
        # the falsy test would fire on a correct reading and hand the avatar to
        # the runner-up.
        if avatar is not None and cursor is None and not attached and len(ranked) > 1:
            # An exact tie between a block and its reflection happens whenever the
            # action moves both the same way (ar25 mirrors one axis only, so every
            # vertical move is a tie).  Break it on a real signal: the UI marks the
            # piece the agent commands, so prefer the candidate a marker rides.
            head = records[ranked[0]]
            for c in ranked[1:4]:
                rec = records[c]
                if (rec["agree_rate"], rec["coverage"]) != (head["agree_rate"],
                                                            head["coverage"]):
                    break
                cur, att = self._small_followers(c, self._avatar_pixel_bbox(c),
                                                 hud, background)
                if cur is not None or att:
                    avatar, cursor, attached = c, cur, att
                    break

        mirror = self._find_mirror(mobile, avatar) if avatar is not None else None

        # --- goal: only what a level increment showed ------------------------
        # Colours that travel with the avatar (its own heading pixel, the selection
        # ring) are part of the commanded piece, never a goal or a hazard.
        own = set(mobile) | set(attached) | {avatar, mirror, cursor}
        goal_votes = {}
        for before, after in self._levelups:
            for c in np.unique(np.concatenate([before.ravel(), after.ravel()])):
                c = int(c)
                if c == background or c in hud or c in own:
                    continue
                was, now = int((before == c).sum()), int((after == c).sum())
                if was > 0 and now < was:
                    goal_votes[c] = goal_votes.get(c, 0) + 1
            if avatar is not None:
                for r, cc in np.argwhere(after == avatar):
                    c = int(before[r, cc])
                    if c != avatar and c != background and c not in hud and c not in own:
                        goal_votes[c] = goal_votes.get(c, 0) + 1

        # --- lethal: only what an observed death showed ----------------------
        lethal_votes = {}
        for before, after, act in self._deaths:
            if avatar is None or act is None:
                continue
            step = ACTION_DIR.get(int(act))
            if step is None:
                continue
            for r, cc in np.argwhere(before == avatar):
                nr, nc = r + step[0], cc + step[1]
                if 0 <= nr < before.shape[0] and 0 <= nc < before.shape[1]:
                    c = int(before[nr, nc])
                    if c != avatar and c != background and c not in hud and c not in own:
                        lethal_votes[c] = lethal_votes.get(c, 0) + 1
        lethal = {c for c, v in lethal_votes.items() if v >= 1}

        named = ({background, avatar, mirror, cursor} | set(hud)
                 | set(mobile) | set(attached) | set(lethal))
        terrain = sorted(c for c in self._seen
                         if c not in named and c not in goal_votes)

        notes = []
        if self._diffs < self.warmup:
            notes.append(f"warming up ({self._diffs}/{self.warmup})")
        if avatar is None and mobile:
            notes.append("mobile colours but no action agreement")
        elif avatar is None:
            notes.append("no mobile block seen")
        if not self._levelups:
            notes.append("goal needs a level increment")
        if not self._deaths:
            notes.append("lethal needs an observed death")

        evidence = {
            "colours": {str(c): r for c, r in sorted(records.items())},
            "hud_outside_lattice": sorted(c for c in hud if inside is not None
                                          and c not in rigid and (grid == c).any()
                                          and float((~inside[grid == c]).mean()) >= 0.6),
            "self_animating": sorted(animated),
            "goal_votes": {str(k): v for k, v in sorted(goal_votes.items())},
            "lethal_votes": {str(k): v for k, v in sorted(lethal_votes.items())},
            "level_ups": len(self._levelups),
            "deaths": len(self._deaths),
            "transitions": self._diffs,
            "cell": None if spec is None else list(spec.cell),
        }

        return RoleSpec(
            background=background,
            hud=frozenset(hud),
            mobile=frozenset(mobile),
            avatar=avatar,
            mirror=mirror,
            cursor=cursor,
            attached=frozenset(attached),
            terrain=frozenset(terrain),
            goal=frozenset(goal_votes),
            lethal=frozenset(lethal),
            evidence=evidence,
            steps=self._steps,
            settled=self._diffs >= self.warmup,
            note="; ".join(notes),
        )


def infer_roles(frames, actions=None, levels=None, states=None, *,
                background=None, spec=None, warmup=WARMUP, trackers=None):
    """Replay a whole trajectory and return the final :class:`RoleSpec`.

    ``frames`` is an iterable of frames (``FrameData``, an ``HxW`` array or a
    nested list); ``actions`` / ``levels`` / ``states`` are parallel sequences and
    may be ``None``.  ``trackers`` (a list) collects the per-step snapshots, which
    shows *when* each role became known.
    """
    frames = list(frames)
    n = len(frames)
    actions = list(actions) if actions is not None else [None] * n
    levels = list(levels) if levels is not None else [None] * n
    states = list(states) if states is not None else [None] * n

    rt = RoleTracker(background=background, spec=spec, warmup=warmup)
    snapshot = None
    for i, frame in enumerate(frames):
        snapshot = rt.observe(
            frame,
            action=actions[i] if i < len(actions) else None,
            levels=levels[i] if i < len(levels) else None,
            state=states[i] if i < len(states) else None,
        )
        if trackers is not None:
            trackers.append(snapshot)
    return snapshot if snapshot is not None else RoleSpec(note="empty trajectory")
