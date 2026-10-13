"""Generic lattice (cell-grid) recovery from a raw ARC-AGI-3 frame.

Why this module exists
----------------------
Both hand-written closed-book solvers recover a *cell lattice* from the picture
before they can reason (``closed_book/tu93_frame.py``, ``closed_book/ar25_frame.py``).
Historically each hardcoded ``cell = 3`` and ran its own phase scan, so the
"next game" would need the same code written again by hand.  This module turns
the geometry inference into one game-agnostic primitive:

    board pixels -> (cell size, lattice origin, cell grid)

The only input is pixels: no engine access, no per-level tables, no baked
answers.

The measure
-----------
For a candidate ``(cell size, origin)`` the frame is cut into cells and every
cell is scored by **solidity**: is it a single colour, allowing a small
decoration allowance -- a 1px cursor outline, a hazard's direction pixel, an
axis line drawn inside the cell?  The allowance is scale-consistent,
``(area - 1) // 8`` pixels (~12%), so a cell is solid when its majority colour
covers at least 7/8 of the block:

  * tu93's hazard tile = 8px of colour 8 + 1px marker -> solid;
  * ar25's reflection axis = a 1px line living inside a cell -> solid;
  * a *wrong* phase straddles cell/background boundaries, so its blocks are
    ~50% one colour -> not solid.

The reported ``purity`` is the fraction of informative cells that are solid.
Cells that are entirely background carry no information and are excluded (a
mostly empty frame would otherwise score 100% for free).  With
``tolerance=0`` solidity degrades to exact single-colour uniformity, which is
byte-for-byte the legacy tu93 scan -- that identity is what the regression test
pins.

Choosing the cell size
----------------------
Solidity alone never picks a size: ``cell = 1`` is solid for every image.  The
size is chosen by the **solidity cliff**: among the sizes that are demonstrably
blocky (purity >= ``floor`` and enough informative cells), take the coarse end
of the sharpest drop ``purity(s) - purity(s+1)`` -- the coarsest scale at which
the picture is still block-constant.  On a native pixel-art frame no coarse size
is blocky and the detector answers ``cell = 1`` ("this is just a pixel grid"),
which is the honest answer and what a caller needs in order to treat the frame
as a plain grid.

Determinism and cost
--------------------
Ties break towards the smaller cell size and the smaller origin (matching the
legacy tu93 scan).  Every phase of every size is scored from integral images
built once per colour, so a full scan of a 64x64 frame costs ~10 ms instead of
the ``O(s^2 * H * W)`` of a reshape per phase; ``_measure`` keeps that obvious
version as the reference the fast path is tested against.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MAX_CELL = 8
"""Largest cell size considered by default (a 64px frame with cell=8 is an 8x8 board)."""

_DEFAULT_FLOOR = 0.75
"""A coarse size must be at least this solid over its informative cells to compete."""

_DEFAULT_MIN_SUPPORT = 4
"""A coarse size must have at least this many informative cells (guards tiny flukes).

Deliberately low: a sparse board (six occupied lattice cells on a void 64x64
frame) is still a real lattice, and missing it costs far more than the rare
"three stray clusters happen to align to an 8-grid" false positive -- a wrong
lattice only makes a solver fall through, never emit a wrong action."""

_MIN_CELLS_PER_AXIS = 4
"""Each axis must yield at least this many cells, else the size is not a lattice."""


def tolerance_for(area):
    """Decoration allowance (pixels) for a cell of ``area`` pixels (~1/8 of it)."""
    return max(0, (int(area) - 1) // 8)


class LatticeSpec:
    """A recovered lattice: cell size, origin, and the cell grid itself.

    Deliberately a plain class (not a ``@dataclass``): closed-book modules are
    sometimes loaded through ``spec_from_file_location`` without being
    registered in ``sys.modules``, which breaks dataclass annotation resolution.
    """

    __slots__ = (
        "cell_r", "cell_c", "origin_r", "origin_c", "n_rows", "n_cols",
        "purity", "purity_exact", "active", "cells", "source", "sweep",
    )

    def __init__(self, cell_r, cell_c, origin_r, origin_c, n_rows, n_cols,
                 purity, purity_exact, active, cells, source="drop", sweep=None):
        self.cell_r = int(cell_r)
        self.cell_c = int(cell_c)
        self.origin_r = int(origin_r)
        self.origin_c = int(origin_c)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.purity = float(purity)                 # solid within tolerance
        self.purity_exact = float(purity_exact)     # exactly single-colour
        self.active = int(active)
        self.cells = cells                          # (n_rows, n_cols) int array
        self.source = source                        # "drop" | "native" | "forced" | "degenerate"
        self.sweep = sweep or {}                    # size -> measured purity (report)

    # --- convenience ------------------------------------------------------
    @property
    def cell(self):
        """``(cell_r, cell_c)``."""
        return (self.cell_r, self.cell_c)

    @property
    def origin(self):
        """``(origin_r, origin_c)`` pixel offset of the first cell."""
        return (self.origin_r, self.origin_c)

    @property
    def shape(self):
        """``(n_rows, n_cols)``."""
        return (self.n_rows, self.n_cols)

    @property
    def is_native(self):
        """True when the frame is a plain pixel grid (no coarse lattice found)."""
        return self.cell_r == 1 and self.cell_c == 1

    def pixel_of(self, r, c):
        """Top-left pixel of cell ``(r, c)``."""
        return (self.origin_r + r * self.cell_r, self.origin_c + c * self.cell_c)

    def center_pixel(self, r, c):
        """Pixel centre of cell ``(r, c)``."""
        return (
            self.origin_r + r * self.cell_r + self.cell_r // 2,
            self.origin_c + c * self.cell_c + self.cell_c // 2,
        )

    def cell_of_pixel(self, y, x):
        """Cell containing pixel ``(y, x)``, or ``None`` when outside the lattice."""
        r = (int(y) - self.origin_r) // self.cell_r
        c = (int(x) - self.origin_c) // self.cell_c
        if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
            return (int(r), int(c))
        return None

    def to_dict(self):
        return {
            "cell": [self.cell_r, self.cell_c],
            "origin": [self.origin_r, self.origin_c],
            "shape": [self.n_rows, self.n_cols],
            "purity": round(self.purity, 4),
            "purity_exact": round(self.purity_exact, 4),
            "active": self.active,
            "source": self.source,
            "sweep": {int(k): round(float(v), 4) for k, v in self.sweep.items()},
        }

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"LatticeSpec(cell={self.cell}, origin={self.origin}, "
                f"shape={self.shape}, purity={self.purity:.3f}, source={self.source})")


# ---------------------------------------------------------------------------
# grid normalisation
# ---------------------------------------------------------------------------

def as_grid(frame):
    """Normalise any frame-ish input to a 2D int ndarray.

    Accepts a 2D array, a ``(1, H, W)`` batch, a ``FrameData`` (``.frame`` is a
    list of layers) or a nested list.  Returns ``None`` when there is no grid.
    """
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        arr = frame
    else:
        raw = getattr(frame, "frame", None)
        if raw is None and isinstance(frame, dict):
            raw = frame.get("frame")
        if raw is None:
            raw = frame
        arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr.astype(np.int64, copy=False)


def infer_background(arr):
    """Most frequent colour in the frame: the colour that carries no information."""
    flat = arr.reshape(-1)
    nc = max(16, int(flat.max()) + 1) if flat.size else 16
    return int(np.bincount(flat, minlength=nc).argmax())


def _integral(mask):
    """2D prefix sums of a boolean mask, with one row/col of padding."""
    cum = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    out[1:, 1:] = cum
    return out


def _block_counts(intg, size):
    """Sum over every ``size x size`` block: shape ``(H-size+1, W-size+1)``."""
    return intg[size:, size:] - intg[:-size, size:] - intg[size:, :-size] + intg[:-size, :-size]


def _majority(blocks, n_colours):
    """Per-block majority colour; ties break towards the smaller colour index."""
    n, k = blocks.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    key = rows * n_colours + blocks.reshape(-1)
    counts = np.bincount(key, minlength=n * n_colours).reshape(n, n_colours)
    return counts.argmax(axis=1)


# ---------------------------------------------------------------------------
# the reference measurement (reshape per phase; square or rectangular cells)
# ---------------------------------------------------------------------------

def _measure(arr, cell_r, cell_c, orow, ocol, background, want_cells=False, tolerance=None):
    """Readable definition of the score for one ``(size, origin)`` candidate.

    Returns ``purity`` (solid within tolerance over informative cells),
    ``purity_exact`` (exactly uniform), the informative-cell count and, on
    request, the majority-colour cell grid.
    """
    h, w = arr.shape
    n_r = (h - orow) // cell_r
    n_c = (w - ocol) // cell_c
    if n_r < 1 or n_c < 1:
        return None
    sub = arr[orow:orow + n_r * cell_r, ocol:ocol + n_c * cell_c]
    blocks = sub.reshape(n_r, cell_r, n_c, cell_c).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(n_r * n_c, cell_r * cell_c)
    area = cell_r * cell_c
    tol = tolerance_for(area) if tolerance is None else int(tolerance)

    colours = np.unique(blocks)
    counts = np.stack([(blocks == c).sum(axis=1) for c in colours], axis=1)
    top = counts.max(axis=1)
    solid = (area - top) <= tol
    exact = top == area
    if background is None:
        active = np.ones_like(solid)
    else:
        active = ~((blocks == background).all(axis=1))
    n_active = int(active.sum())
    if n_active == 0:
        return None
    n_bg = int(active.size - n_active)
    purity = float((int(solid.sum()) - n_bg) / n_active)
    purity_exact = float((int(exact.sum()) - n_bg) / n_active)
    cells = None
    if want_cells:
        n_colours = max(16, int(arr.max()) + 1)
        cells = _majority(blocks, n_colours).reshape(n_r, n_c).astype(np.int16)
    return {
        "purity": purity,
        "purity_exact": purity_exact,
        "active": n_active,
        "n_rows": int(n_r),
        "n_cols": int(n_c),
        "orow": int(orow),
        "ocol": int(ocol),
        "cells": cells,
    }


def _scan_size(arr, size, background, integrals, tolerance=None):
    """Best origin for one cell size, scoring every phase without a Python loop.

    ``integrals`` maps colour -> prefix sums, so ``top[y, x]`` (majority count of
    the block starting at ``(y, x)``) is a max over a handful of block-count maps.
    Every block belongs to exactly one phase, so padding the maps up to a
    multiple of ``size`` turns the whole phase grid into a single reshape +
    reduction: work per size is ``O(#colours * H * W)``, independent of the
    number of phases.

    Tie-break: higher purity, then more informative cells, then the smallest
    origin -- the ordering the legacy tu93 scan used.
    """
    area = size * size
    if size > min(arr.shape):
        return None
    tol = tolerance_for(area) if tolerance is None else int(tolerance)
    h, w = arr.shape
    rows, cols = h - size + 1, w - size + 1
    if rows < 1 or cols < 1:
        return None
    pad_r, pad_c = -(-rows // size) * size, -(-cols // size) * size
    shape = (pad_r, pad_c)

    def _padded(m):
        if m.shape == shape:
            return m
        out = np.full(shape, -1, dtype=m.dtype)
        out[:rows, :cols] = m
        return out

    top = None
    for intg in integrals.values():
        m = _padded(_block_counts(intg, size))
        top = m if top is None else np.maximum(top, m)
    if top is None:
        return None

    valid = np.zeros(shape, dtype=np.int64)
    valid[:rows, :cols] = 1
    solid = (area - top) <= tol                     # padded cells: top == -1 -> False
    exact = top == area
    if background is not None and background in integrals:
        bg = _padded(_block_counts(integrals[background], size)) == area
    else:
        bg = np.zeros(shape, dtype=bool)

    def per_phase(m):
        """Group block starts by their phase: ``(y mod size, x mod size)``."""
        return m.reshape(pad_r // size, size, pad_c // size, size).sum(axis=(0, 2))

    n_valid = per_phase(valid)
    n_solid = per_phase(solid)
    n_exact = per_phase(exact)
    n_bg = per_phase(bg)
    n_active = n_valid - n_bg
    ok = n_active > 0
    if not ok.any():
        return None
    purity = np.where(ok, (n_solid - n_bg) / np.maximum(n_active, 1), -1.0)
    phase = np.argwhere(ok)                          # (k, 2) -> (orow, ocol)
    flat = phase[:, 0] * (pad_c // size) + phase[:, 1]   # row-major phase index
    pick = np.lexsort((flat, -n_active[ok], -purity[ok]))[0]
    orow, ocol = (int(v) for v in phase[pick])
    return {
        "purity": float(purity[orow, ocol]),
        "purity_exact": float((n_exact[orow, ocol] - n_bg[orow, ocol]) / n_active[orow, ocol]),
        "active": int(n_active[orow, ocol]),
        "n_rows": (rows - orow + size - 1) // size,
        "n_cols": (cols - ocol + size - 1) // size,
        "orow": orow,
        "ocol": ocol,
        "cells": None,
    }


def _best_for_size(arr, size, background, integrals, want_cells=False,
                   cell_shape=None, tolerance=None):
    """Best phase for one size (fast scan, plus the cell grid when asked)."""
    got = _scan_size(arr, size, background, integrals, tolerance=tolerance)
    if got is None:
        return None
    if not want_cells:
        return got
    hull = cell_shape or (size, size)
    return _measure(arr, hull[0], hull[1], got["orow"], got["ocol"], background,
                    want_cells=True, tolerance=tolerance)


def _best_for_size_bruteforce(arr, size, background, integrals, tolerance=None):
    """Reference twin of :func:`_best_for_size` -- one reshape per phase.

    Kept so ``tests/test_lattice.py`` can assert the integral-image scan returns
    exactly the same phase and score as the obvious implementation.
    """
    best = None
    for orow in range(size):
        for ocol in range(size):
            got = _measure(arr, size, size, orow, ocol, background, tolerance=tolerance)
            if got is None or got["active"] == 0:
                continue
            key = (got["purity"], got["active"])
            if best is None or key > best[0]:
                best = (key, got)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# diagnostics: do the candidate boundaries sit on colour changes?
# ---------------------------------------------------------------------------

def _boundary_profile(arr, axis):
    """Change counts along one axis: ``prof[x]`` = #rows where col x != col x-1."""
    if axis == 0:
        diff = (arr[:, 1:] != arr[:, :-1]).sum(axis=0)
        prof = np.zeros(arr.shape[1], dtype=np.int64)
        prof[1:] = diff
    else:
        diff = (arr[1:, :] != arr[:-1, :]).sum(axis=1)
        prof = np.zeros(arr.shape[0], dtype=np.int64)
        prof[1:] = diff
    return prof


def _alignment(prof, size):
    """How much the lattice lines of ``size`` stand out from a random position.

    ``1.0`` means boundaries are indistinguishable from random columns; a real
    lattice concentrates colour changes on its grid lines, so its phase scores
    well above 1.  Every phase is evaluated in one O(n) pass.  Note that a
    multiple of the true cell size also aligns (every other line is still a
    line), so this is a *diagnostic*, not the size decision -- solidity does that.
    """
    n = prof.size
    base = float(prof[1:].mean()) if n > 1 else 0.0
    if base <= 0:
        return np.ones(size), np.zeros(size, dtype=np.int64)
    out = np.ones(size)
    counts = np.zeros(size, dtype=np.int64)
    for o in range(size):
        idx = np.arange(o if o > 0 else size, n, size)     # skip x == 0 (no boundary)
        if idx.size == 0:
            continue
        out[o] = float(prof[idx].mean()) / base
        counts[o] = int(idx.size)
    return out, counts


def _integrals_of(arr):
    """Prefix sums for every colour that actually appears in the frame."""
    return {int(c): _integral(arr == c) for c in np.unique(arr)}


def _candidate_sizes(arr, sizes, max_cell):
    h, w = arr.shape
    cap = max(1, min(h, w) // _MIN_CELLS_PER_AXIS)
    if sizes is not None:
        cand = sorted({int(s) for s in sizes if int(s) >= 1})
        if not cand:
            cand = [1]
        top = max(cand)
    else:
        top = min(max_cell or DEFAULT_MAX_CELL, cap)
        cand = list(range(1, top + 1))
    return cand, top


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def lattice_sweep(grid, sizes=None, background=None, max_cell=None, tolerance=None):
    """Solidity of every candidate cell size -- the evidence behind a detection.

    Returns ``{size: {purity, purity_exact, active, origin, shape, align}}``.
    ``align`` is the diagnostic of :func:`_alignment`.  Use this to audit a
    detection or to profile an unknown game's geometry.
    """
    arr = as_grid(grid)
    if arr is None:
        return {}
    if background is None:
        background = infer_background(arr)
    integrals = _integrals_of(arr)
    cand, _ = _candidate_sizes(arr, sizes, max_cell)
    row_prof = _boundary_profile(arr, 1)
    col_prof = _boundary_profile(arr, 0)
    out = {}
    for s in cand:
        got = _scan_size(arr, s, background, integrals, tolerance=tolerance)
        if got is None:
            continue
        ar, _ = _alignment(row_prof, s)
        ac, _ = _alignment(col_prof, s)
        out[int(s)] = {
            "purity": round(float(got["purity"]), 4),
            "purity_exact": round(float(got["purity_exact"]), 4),
            "active": got["active"],
            "origin": (got["orow"], got["ocol"]),
            "shape": (got["n_rows"], got["n_cols"]),
            "align": round(float(min(ar.max(), ac.max())), 2),
        }
    return out


def detect_cell_lattice(grid, *, max_cell=None, background=None, sizes=None,
                        floor=_DEFAULT_FLOOR, min_support=_DEFAULT_MIN_SUPPORT,
                        tolerance=None, force_cell=None):
    """Recover ``(cell size, origin, cell grid)`` from a frame.

    Parameters
    ----------
    grid
        Raw frame: 2D array, ``(1,H,W)`` batch, ``FrameData`` or nested list.
    max_cell
        Largest cell size allowed as an answer (default 8).  Sizes above
        ``min(h, w) // 4`` are never considered.
    background
        Colour meaning "empty"; excluded from the purity score.  ``None`` uses
        the frame's most frequent colour.
    sizes
        Explicit size candidates (``(3,)`` restricts the scan to 3px cells).
        ``None`` scans every size up to the cap.
    floor, min_support
        Guards for the coarse sizes: purity at least ``floor`` over at least
        ``min_support`` informative cells.
    tolerance
        Decoration allowance in pixels inside a cell (default ``(area-1)//8``).
        ``0`` makes solidity mean exact single-colour uniformity.
    force_cell
        Skip size selection and use this cell size (the origin is still chosen
        by purity).

    Returns
    -------
    LatticeSpec
        ``cells`` is the majority-colour cell grid; ``source`` reports how the
        size was chosen: ``"drop"`` (real lattice), ``"native"`` (plain pixel
        grid), ``"forced"``, or ``"degenerate"`` when the frame is unusable.
    """
    arr = as_grid(grid)
    if arr is None:
        return LatticeSpec(1, 1, 0, 0, 0, 0, -1.0, -1.0, 0,
                           np.zeros((0, 0), dtype=np.int16), source="degenerate")
    if background is None:
        background = infer_background(arr)
    integrals = _integrals_of(arr)

    if force_cell:
        s = int(force_cell)
        got = _best_for_size(arr, s, background, integrals, want_cells=True,
                             tolerance=tolerance)
        if got is None:
            return LatticeSpec(s, s, 0, 0, 0, 0, -1.0, -1.0, 0,
                               np.zeros((0, 0), dtype=np.int16), source="degenerate")
        return LatticeSpec(s, s, got["orow"], got["ocol"], got["n_rows"], got["n_cols"],
                           got["purity"], got["purity_exact"], got["active"],
                           got["cells"], source="forced", sweep={s: got["purity"]})

    cand, top = _candidate_sizes(arr, sizes, max_cell)
    # measure one size past the cap so the drop at `top` is real, not a boundary artefact
    h, w = arr.shape
    probe = sorted(set(cand) | {min(min(h, w) // 2, top + 1)})

    measured = {}
    for s in probe:
        got = _scan_size(arr, s, background, integrals, tolerance=tolerance)
        if got is not None and got["active"] > 0:
            measured[s] = got
    if not measured:
        # nothing informative anywhere: a blank board is still a pixel grid
        cells = np.full(arr.shape, int(background), dtype=np.int16)
        return LatticeSpec(1, 1, 0, 0, arr.shape[0], arr.shape[1], -1.0, -1.0, 0,
                           cells, source="native")

    def score(s):
        got = measured.get(s)
        return float(got["purity"]) if got is not None else 0.0

    sweep = {s: round(score(s), 4) for s in sorted(measured)}

    coarse = [s for s in cand
              if s >= 2 and s in measured
              and score(s) >= floor
              and measured[s]["active"] >= min_support]
    if coarse:
        def drop(s):
            return score(s) - score(min(min(h, w) // 2, s + 1))

        s_star = max(coarse, key=lambda s: (round(drop(s), 6), -s))
        source = "drop"
    else:
        s_star, source = 1, "native"

    got = measured.get(s_star) or measured[min(measured)]
    got = _best_for_size(arr, s_star, background, integrals, want_cells=True,
                         tolerance=tolerance) or got
    return LatticeSpec(s_star, s_star, got["orow"], got["ocol"], got["n_rows"],
                       got["n_cols"], got["purity"], got["purity_exact"],
                       got["active"], got["cells"], source=source, sweep=sweep)


def to_cell_grid(grid, spec, background=None, tolerance=None):
    """Majority-colour cell grid for an explicit lattice, without re-detecting."""
    arr = as_grid(grid)
    if arr is None or spec is None:
        return None
    got = _measure(arr, spec.cell_r, spec.cell_c, spec.origin_r, spec.origin_c,
                   background, want_cells=True, tolerance=tolerance)
    return None if got is None else got["cells"]
