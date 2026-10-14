"""Trajectory-aware status-bar / HUD masker for state hashing.

Why this exists
---------------
ARC-AGI-3 frames carry UI chrome -- progress bars, score/level readouts, timers,
per-step counters -- that repaint on (almost) every frame.  Hashing the whole
frame makes the agent believe it has entered a *new* state every step, which
wrecks graph exploration.  This module detects the chrome by *measurement* over
the trajectory the agent already collects and returns a boolean mask of pixels to
neutralise before hashing.

This is the fix recommended in ``CORRECTION_HUD_DETECTOR_20261014.md``.  The old
``core/roles.py`` detector output nothing usable on our data (3/22 episodes, 0.0%
state reduction) because of four faults:

* it worked at the **colour** level (one colour = one HUD) instead of the
  **row/segment** level;
* ``ANIM_RATIO = 0.9`` was far above the real ceiling (measured 55-63%), so
  almost nothing qualified;
* a colour that both self-animates *and* "translates" (re86's growth-bar) was
  thrown out by the ``if c in rigid: continue`` guard -- the self-animation and
  the translation were wrongly coupled;
* position ("outside the lattice") was effectively the only working cue, and it
  misses chrome fused into the scene.

The detector here uses two independent, OR-combined signals:

1. **Edge bands** (just-explore "crude" mode).  The playfield sits in the centre;
   the outermost ``edge_px`` rows/columns are chrome.  Cheap and safe -- this is
   what SWARM's 5.7x ft09 fix amounts to (hash rows 0-62, i.e. drop the last
   row).  Position is *a* cue, not the only one.

2. **Self-animating rows / columns** ("in place", **decoupled from translation**).
   A row whose pixel content changes on >= ``anim_ratio`` of transitions *while
   keeping the same row index* is a status bar, not a moving object: a moving
   object visits many row indices, so no single fixed row reaches the threshold.
   Crucially we never look at whether the content *within* the row translates -- an
   animating, growth-style bar that also shifts its filled pixels is still masked.
   This is exactly the re86 failure mode from the old code.

Detection is cumulative over the trajectory.  Level changes (including RESETs)
restart the statistics so a bar that moves between levels is re-learned; a few
frames of warm-up after each reset lose masking but that is negligible.

The mask value is a sentinel outside the palette (``-1``) so hashing sees a
constant, content-free pixel where chrome used to be.
"""
from __future__ import annotations

import numpy as np

from ..core.lattice import as_grid


class StatusMasker:
    """Fold frames in one at a time; ask for the HUD mask of the latest frame.

    Typical use inside :class:`lingjing_solo.world_model.field.WorldModelField`::

        self._masker = StatusMasker(edge_px=cfg.status_bar_edge_px,
                                     anim_ratio=cfg.status_bar_anim_ratio)
        ...
        def _hash(self, grid, levels):
            g = self._masker.masked(grid) if self._masker is not None else grid
            return hash_grid(g, levels)

    Cost: one boolean reduction per row and per column per transition.  No frames
    are retained.
    """

    def __init__(self, h=None, w=None, *, edge_px=2, anim_ratio=0.6, warmup=4,
                 mask_value=-1, per_level_reset=True):
        self.edge_px = int(edge_px)
        self.anim_ratio = float(anim_ratio)
        self.warmup = int(warmup)
        self.mask_value = int(mask_value)
        self.per_level_reset = bool(per_level_reset)
        self.h = h
        self.w = w
        self._prev = None
        self._trans = 0
        self._level = None
        self._row_change = None   # int64[h]: transitions where row r changed
        self._col_change = None   # int64[w]: transitions where col c changed
        self._mask = None         # cached mask for the latest observed frame

    # -- config knobs ---------------------------------------------------------
    def _edge_rows(self):
        if self.h is None:
            return ()
        return tuple(range(self.edge_px)) + tuple(range(self.h - self.edge_px, self.h))

    def _edge_cols(self):
        if self.w is None:
            return ()
        return tuple(range(self.edge_px)) + tuple(range(self.w - self.edge_px, self.w))

    # -- lifecycle ------------------------------------------------------------
    def reset(self, h=None, w=None):
        if h is not None:
            self.h = h
        if w is not None:
            self.w = w
        self._prev = None
        self._trans = 0
        self._row_change = (np.zeros(self.h, dtype=np.int64) if self.h else None)
        self._col_change = (np.zeros(self.w, dtype=np.int64) if self.w else None)
        self._mask = None

    # -- fold ----------------------------------------------------------------
    def observe(self, grid, level=None):
        """Fold one frame in.  ``level`` (int) restarts stats on change."""
        g = as_grid(grid)
        if g is None:
            return
        h, w = g.shape
        if self.h != h or self.w != w or self._row_change is None:
            self.reset(h, w)
        if (self.per_level_reset and level is not None
                and self._level is not None and level != self._level):
            self.reset(h, w)
        self._level = level

        if self._prev is not None:
            changed = g != self._prev
            self._row_change += changed.any(axis=1).astype(np.int64)
            self._col_change += changed.any(axis=0).astype(np.int64)
            self._trans += 1
        self._prev = g
        self._mask = None   # the latest mask is now stale

    # -- query ---------------------------------------------------------------
    def mask_for(self, grid=None):
        """Boolean mask (H, W) of HUD pixels for the latest observed frame.

        Returns ``None`` until a frame with a known shape has been observed.
        """
        if self._mask is None:
            self._build()
        return self._mask

    def _build(self):
        if self.h is None or self._row_change is None:
            self._mask = None
            return
        mask = np.zeros((self.h, self.w), dtype=bool)
        # 1. edge bands -- the safe, position-based cue
        for r in self._edge_rows():
            mask[r, :] = True
        for c in self._edge_cols():
            mask[:, c] = True
        # 2. self-animating rows / columns -- decoupled from translation
        if self._trans >= self.warmup:
            rr = self._row_change / self._trans
            mask[rr >= self.anim_ratio, :] = True
            cr = self._col_change / self._trans
            mask[:, cr >= self.anim_ratio] = True
        self._mask = mask

    def masked(self, grid):
        """Return ``grid`` with HUD pixels replaced by ``mask_value``.

        Leaves the input untouched (returns it unchanged) when there is no mask
        or the mask is empty -- so the hash never changes unless chrome is found.
        """
        g = as_grid(grid)
        if g is None:
            return None
        mask = self.mask_for(g)
        if mask is None or not mask.any():
            return g
        out = g.copy()
        out[mask] = self.mask_value
        return out

    # -- introspection (for tests / diagnostics) ------------------------------
    def hot_rows(self):
        if self._row_change is None or self._trans < self.warmup:
            return np.array([], dtype=int)
        return np.where(self._row_change / self._trans >= self.anim_ratio)[0]

    def hot_cols(self):
        if self._col_change is None or self._trans < self.warmup:
            return np.array([], dtype=int)
        return np.where(self._col_change / self._trans >= self.anim_ratio)[0]
