"""HUD / playfield separation + playfield-only change detection.

Why this module exists
----------------------
ARC-AGI-3 frames carry a persistent HUD (timer, step counter, score bar, animated
trim).  That has two downstream costs, both measured on the real episodes:

1. **The whole-screen delta lies.**  ``my_agent.py`` feeds CEAX
   ``delta_px = (prev_grid != grid).sum()``.  ``ceax_controller.py`` decides "this
   action did nothing" with thresholds like ``delta_pixels <= 2`` / ``<= 4`` /
   ``<= 0``.  If a HUD pixel ticks on its own, the raw delta never reaches those
   thresholds, so the *no-effect* branch never fires and action credit is
   inflated (``effect = min(1.0, delta_pixels / 64.0)``, ceax_controller.py:673).

2. **The click proposer's first pick is HUD.**  ``_perc_click_candidates`` sorts
   components by pixel count descending and takes the bbox centre, so the largest
   component -- the background/echo border -- is proposed first.  That is exactly
   the failure the Duck harness writes into its prompt: *"a common failure mode is
   to mistake a segmented edge bar for clickable puzzle pieces ... classify it as
   HUD/timer state ... DON'T DO THIS!"*

Duck's answer was to hand the judgement to a model.  This module's answer is a
measurement, so it works with no model at all:

    freq[r, c] = (steps since level start where pixel (r,c) changed) / steps

A pixel that changes on essentially **every** step is changing independently of
what the agent chose -- it is HUD/animated.  A pixel that changes on only *some*
steps is being driven by the agent.  So:

    animated_mask = freq >= ANIM_RATIO          (after WARMUP steps)
    board_changed = ((prev != cur) & ~animated_mask).any()

That is causal and online: it needs one bool array and one counter per level, no
lookahead, no model, no extra dependency beyond numpy.

Geometry is the cold-start complement: before any history exists (frame 0, or a
fresh level) an object is still HUD-suspect if it is an edge-hugging strip, a
full-span band, or the giant background blob.  ``object_hud_score`` blends both
and returns a single [0, 1] number, so a caller can demote rather than delete.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ANIM_RATIO",
    "WARMUP",
    "LevelStability",
    "interior_point",
    "object_hud_score",
    "rerank_click_candidates",
]

#: A pixel changing on >= this fraction of steps is treated as self-animating (HUD).
ANIM_RATIO = 0.9

#: Steps of history to gather before the animated mask is trusted.
WARMUP = 6

#: bbox/cell distance from the grid border that still counts as "hugging the edge".
EDGE_MARGIN = 1

#: Aspect ratio above which a rectangle reads as a bar rather than a puzzle piece.
STRIP_ELONGATION = 4.0


def interior_point(pixels, bbox=None):
    """A point that is guaranteed to lie *inside* the object.

    ``_perc_click_candidates`` currently uses the bounding-box centre, which for an
    L-shape, a ring, or any concave blob can land on a cell the object does not
    occupy -- the click then goes to whatever is there instead.  This returns the
    object's own cell closest to its centroid, which is always a real cell.

    ``pixels`` is an iterable of ``(row, col)`` (the layout ``GameObject.pixels``
    uses).  Returns ``(row, col)`` as ints, or ``None`` for an empty object.
    """
    pts = list(pixels or [])
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    rc = arr[:, :2]
    centre = rc.mean(axis=0)
    d = ((rc - centre) ** 2).sum(axis=1)
    r, c = rc[int(np.argmin(d))]
    return int(r), int(c)


def _bbox_geometry(bbox, shape):
    """Normalise an ``(x0, y0, x1, y1)`` bbox to ``(h, w, touch, elongation, span)``.

    ``shape`` is ``(height, width)`` of the grid the bbox lives on.  ``touch`` is the
    number of grid sides (0-4) the bbox comes within ``EDGE_MARGIN`` of.  ``span`` is
    True when the bbox covers >= 90% of a full row or a full column.
    """
    if not bbox or len(bbox) < 4 or shape is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in bbox[:4])
    height, width = int(shape[0]), int(shape[1])
    # ARC bboxes are inclusive; GameObjects store (min_x, min_y, max_x, max_y).
    bh = max(1, abs(y1 - y0) + 1)
    bw = max(1, abs(x1 - x0) + 1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    touch = (
        int(lo_y <= EDGE_MARGIN)
        + int(hi_y >= height - 1 - EDGE_MARGIN)
        + int(lo_x <= EDGE_MARGIN)
        + int(hi_x >= width - 1 - EDGE_MARGIN)
    )
    elongation = max(bh, bw) / float(max(1, min(bh, bw)))
    span = (bw >= 0.9 * width and bh <= 3) or (bh >= 0.9 * height and bw <= 3)
    return bh, bw, touch, elongation, span


def object_hud_score(bbox, shape, *, pixels=None, npix=None,
                     animated=None, changed=None, settled=False):
    """How much does this object look like chrome rather than playfield?

    Returns a score in ``[0, 1]``; ``>= 0.5`` means "treat as HUD".  Evidence is
    additive so a caller can inspect partial agreement rather than a bare bool.

    - geometry (always available, works at frame 0)
    - ``animated``  : bool HxW mask of self-animating (HUD) pixels
    - ``changed``   : bool HxW mask of pixels that ever changed this level
    - ``settled``   : True once enough history exists for ``changed`` to mean
      something.  Before that, "never changed" is not evidence of anything, so the
      frozen term is skipped -- that matters because at frame 0 *nothing* has changed.
    """
    score = 0.0

    n = int(npix) if npix is not None else (len(pixels) if pixels is not None else 0)
    if shape is not None and n >= 0.40 * int(shape[0]) * int(shape[1]):
        score += 0.45  # the background / echo field is never a puzzle piece

    geo = _bbox_geometry(bbox, shape)
    if geo is not None:
        _bh, _bw, touch, elongation, span = geo
        if touch >= 1:
            score += 0.30
        if touch >= 3:
            score += 0.10
        if elongation >= STRIP_ELONGATION:
            score += 0.25
        if elongation >= 8.0:
            score += 0.15
        if span:
            score += 0.40

    if animated is not None and pixels:
        try:
            idx = np.asarray(list(pixels), dtype=np.int64)[:, :2]
            inside = np.clip(idx[:, 0], 0, animated.shape[0] - 1)
            incol = np.clip(idx[:, 1], 0, animated.shape[1] - 1)
            frac = float(animated[inside, incol].mean()) if len(idx) else 0.0
            if frac >= 0.5:
                # an object whose cells self-animate on ~every step IS chrome, so this
                # alone has to clear the 0.5 bar even with no geometric evidence
                score += 0.50
            elif frac >= 0.25:
                score += 0.20
        except Exception:
            pass

    if settled and changed is not None and pixels:
        try:
            idx = np.asarray(list(pixels), dtype=np.int64)[:, :2]
            inside = np.clip(idx[:, 0], 0, changed.shape[0] - 1)
            incol = np.clip(idx[:, 1], 0, changed.shape[1] - 1)
            frozen = float(1.0 - changed[inside, incol].mean()) if len(idx) else 0.0
            if frozen >= 0.8:
                score += 0.25
        except Exception:
            pass

    return float(min(1.0, score))


class LevelStability:
    """Online HUD/animated separator for one level of one game.

    Feed it every frame you see via :meth:`observe`; call :meth:`reset` on a level
    change.  It keeps two HxW arrays (a change counter and the last frame), which is
    all that is needed -- no frame buffer, no lookahead.

    Typical use inside the agent loop::

        st = LevelStability((64, 64))
        ...
        if level_changed:
            st.reset()
        info = st.observe(grid)
        if info["board_changed"]:
            # something the agent controls moved
        ...

    ``observe`` returns a dict:

      ``raw_delta``    pixels that differ from the previous frame (what the agent
                       feeds CEAX today -- HUD included)
      ``board_delta``  differing pixels excluding the animated mask
      ``board_changed````board_delta > 0``
      ``hud_delta``    ``raw_delta - board_delta``: how much of the change was chrome
      ``settled``      whether the animated mask is warm yet
      ``animated_pixels`` size of the current animated mask
    """

    def __init__(self, shape, warmup=WARMUP, anim_ratio=ANIM_RATIO):
        self.shape = (int(shape[0]), int(shape[1]))
        self.warmup = int(warmup)
        self.anim_ratio = float(anim_ratio)
        self.reset()

    def reset(self):
        """Forget everything; call this on a level transition or a RESET."""
        self._count = np.zeros(self.shape, dtype=np.int32)
        self._prev = None
        self._steps = 0
        self._diffs = 0

    # -- state exposed for callers that want the masks themselves ----------------
    @property
    def steps(self):
        return self._steps

    @property
    def settled(self):
        return self._diffs >= self.warmup

    def changed_ever(self):
        """HxW bool mask of pixels that have changed at least once this level."""
        return self._count > 0

    def animated_mask(self):
        """HxW bool mask of pixels that change on >= ANIM_RATIO of steps.

        The denominator is the number of *diffs* observed, not the number of frames,
        so the first frame (which has no predecessor) does not dilute the rate.
        Empty until :attr:`settled`, because a rate is meaningless before warmup.
        """
        if not self.settled or self._diffs <= 0:
            return np.zeros(self.shape, dtype=bool)
        return (self._count / float(self._diffs)) >= self.anim_ratio

    def observe(self, grid):
        """Fold in one frame and report what happened since the previous one."""
        info = {
            "raw_delta": 0,
            "board_delta": 0,
            "hud_delta": 0,
            "board_changed": False,
            "settled": False,
            "animated_pixels": 0,
        }
        if grid is None:
            return info
        cur = np.asarray(grid)
        if cur.ndim != 2 or cur.shape[:2] != self.shape:
            return info

        if self._prev is not None:
            d = cur != self._prev
            aligned = self._count.shape == d.shape
            if aligned:
                self._count += d
            self._diffs += 1
            # mask is computed after folding in this diff, so a pixel that really does
            # change on every step reaches the ratio on the step it is evaluated
            animated = self.animated_mask() if aligned else np.zeros(self.shape, dtype=bool)
            raw = int(d.sum())
            board = int((d & ~animated).sum())
            info["raw_delta"] = raw
            info["board_delta"] = board
            info["hud_delta"] = raw - board
            info["board_changed"] = board > 0
            info["animated_pixels"] = int(animated.sum())
        self._prev = cur.copy()
        self._steps += 1
        info["settled"] = self.settled
        return info


def rerank_click_candidates(objects, shape, *, k=8, animated=None, changed=None,
                            settled=False, chrome_floor=0.5):
    """Order segmentation objects as click targets, chrome last.

    Replaces the "sort by pixel count descending, take the bbox centre" rule in
    ``_perc_click_candidates``.  Two independent defects are fixed:

    1. **Chrome first.** The largest component is usually the background field, the
       full-frame border, or an edge bar -- measured at 19/23 episodes.  Objects
       scoring ``>= chrome_floor`` are demoted to the back (never silently dropped:
       on a game we have never seen, the border might genuinely be the target).
    2. **The point is not on the object.** The bbox centre of a ring, frame or
       concave blob can fall on a cell the object does not occupy -- measured at
       10/23 episodes.  An interior cell is used instead.

    Within each group objects stay ordered by pixel count descending, so the output
    is directly comparable to the old rule when nothing is chrome.  ``objects`` is
    any iterable of items exposing ``bbox`` and ``pixels`` (e.g. ``GameObject``, whose
    ``pixels`` are ``(row, col)`` pairs).  Returns a list of ``(row, col)`` points,
    at most ``k``.
    """
    scored = []
    for obj in objects:
        bbox = getattr(obj, "bbox", None)
        pixels = getattr(obj, "pixels", None)
        if pixels is None and isinstance(obj, dict):
            bbox = obj.get("bbox")
            pixels = obj.get("pixels")
        npix = len(pixels) if pixels else 0
        if npix <= 0:
            continue
        pt = interior_point(pixels, bbox)
        if pt is None:
            continue
        s = object_hud_score(
            bbox, shape, pixels=pixels, npix=npix,
            animated=animated, changed=changed, settled=settled,
        )
        scored.append((0 if s < chrome_floor else 1, -npix, pt))

    scored.sort(key=lambda t: (t[0], t[1]))
    return [(int(r), int(c)) for _, _, (r, c) in scored[:k]]
