"""Unit tests for the generic lattice detector (``lingjing_solo.core.lattice``).

Everything here is synthetic: the point is to pin the *geometry inference* that
both closed-book solvers now depend on, and to pin the identity between the fast
integral-image scan and the obvious reshape-per-phase implementation.

Coverage:
  * the cell size is discovered, not hardcoded (2/3/4/5/6/8px boards),
  * the lattice phase is discovered (every sub-cell offset),
  * a native pixel-art frame stays native (cell = 1),
  * decorations inside a cell (hazard marker, 1px axis, cursor outline) do not
    break a cell -- that is what the tolerance is for,
  * ``tolerance=0`` reproduces the legacy tu93 scan byte for byte,
  * the fast scan equals the reference scan, on every frame and both modes.
"""
from __future__ import annotations

import numpy as np
import pytest

from lingjing_solo.closed_book.tu93_frame import VOID
from lingjing_solo.closed_book.tu93_frame import detect_lattice as tu93_detect_lattice
from lingjing_solo.closed_book.tu93_frame import _legacy_detect_lattice
from lingjing_solo.core.lattice import (
    LatticeSpec,
    _best_for_size_bruteforce,
    _integrals_of,
    _scan_size,
    as_grid,
    detect_cell_lattice,
    infer_background,
    lattice_sweep,
    to_cell_grid,
    tolerance_for,
)

RNG = np.random.default_rng(20261013)


class FakeFrame:
    """Minimal stand-in for arcengine FrameData (only .frame is read)."""

    def __init__(self, layer):
        self.frame = [layer]


def blocky(cell, n_cells=21, phase=(0, 0), size=64, colours=9, seed=0):
    """A full-bleed board: ``cell`` px per cell, all cells informative."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.int64)
    vals = rng.integers(0, colours, size=(n_cells, n_cells))
    for r in range(n_cells):
        for c in range(n_cells):
            y, x = phase[0] + cell * r, phase[1] + cell * c
            if y + cell > size or x + cell > size:
                continue
            img[y:y + cell, x:x + cell] = vals[r, c]
    return img, vals


def sparse_maze(phase=(0, 0), size=64, hazard_pixel=True):
    """A tu93-shaped board: void everywhere, a few 3px nodes and one hazard."""
    img = np.full((size, size), VOID, dtype=np.int64)
    o_r, o_c = phase
    for (r, c) in [(5, 5), (5, 7), (5, 9), (7, 5), (7, 9), (9, 7)]:
        img[o_r + 3 * r:o_r + 3 * r + 3, o_c + 3 * c:o_c + 3 * c + 3] = 0
    img[o_r + 15:o_r + 18, o_c + 15:o_c + 18] = 9      # player
    img[o_r + 15:o_r + 18, o_c + 21:o_c + 24] = 14     # target
    img[o_r + 21:o_r + 24, o_c + 15:o_c + 18] = 8      # hazard tile ...
    if hazard_pixel:
        img[o_r + 22, o_c + 16] = 15                   # ... with its direction pixel
    return img


# --- grid normalisation ------------------------------------------------------

def test_as_grid_accepts_every_frame_shape():
    layer = sparse_maze()
    assert as_grid(layer).shape == (64, 64)
    assert as_grid(layer[None, ...]).shape == (64, 64)            # (1, H, W) batch
    assert as_grid(FakeFrame(layer)).shape == (64, 64)            # FrameData
    assert as_grid({"frame": [layer]}).shape == (64, 64)          # dict payload
    assert as_grid(layer.tolist()).shape == (64, 64)              # nested list
    assert as_grid(None) is None
    assert as_grid(np.zeros((0, 0))) is None


# --- the cell size is discovered --------------------------------------------

@pytest.mark.parametrize("cell", [2, 3, 4, 5, 6, 8])
def test_cell_size_is_discovered(cell):
    img, vals = blocky(cell, n_cells=64 // cell, seed=cell)
    spec = detect_cell_lattice(img)
    assert spec.cell == (cell, cell), spec
    assert spec.source == "drop"
    assert spec.purity == pytest.approx(1.0)
    assert spec.n_rows == 64 // cell and spec.n_cols == 64 // cell
    assert np.array_equal(spec.cells, vals), "the cell grid must be the board that was drawn"


@pytest.mark.parametrize("phase", [(0, 0), (2, 1), (1, 2), (2, 2)])
def test_lattice_phase_is_discovered(phase):
    img, vals = blocky(3, n_cells=21, phase=phase, seed=11)
    spec = detect_cell_lattice(img)
    assert spec.cell == (3, 3)
    assert spec.origin == phase, spec
    assert np.array_equal(spec.cells, vals[:spec.n_rows, :spec.n_cols])


def test_native_pixel_art_stays_native():
    """A frame with no lattice must answer cell = 1, not a lucky multiple."""
    img = RNG.integers(0, 9, size=(64, 64))
    spec = detect_cell_lattice(img)
    assert spec.is_native and spec.source == "native"
    assert spec.cells.shape == (64, 64)


def test_sparse_board_is_still_a_lattice():
    """Six occupied cells on a void frame: too few for a purity *floor* fallacy."""
    spec = detect_cell_lattice(sparse_maze(), background=VOID)
    assert spec.cell == (3, 3) and spec.origin == (0, 0)


@pytest.mark.parametrize("phase", [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)])
def test_sparse_board_phase_is_discovered(phase):
    img = sparse_maze(phase=phase)
    spec = detect_cell_lattice(img, background=VOID)
    assert spec.cell == (3, 3) and spec.origin == phase, spec


# --- decorations inside a cell ----------------------------------------------

def test_a_one_pixel_decoration_does_not_break_a_cell():
    """tu93's hazard publishes its facing with one odd pixel inside the tile."""
    img = sparse_maze(hazard_pixel=True)
    spec = detect_cell_lattice(img, background=VOID)
    assert spec.cell == (3, 3)
    assert spec.purity == pytest.approx(1.0)                # the decoration is tolerated
    assert 0.0 < spec.purity_exact < 1.0                    # ... and is still reported
    assert spec.purity_exact == pytest.approx(5 / 6, abs=1e-6)


def test_tolerance_zero_means_exact_uniformity():
    img = sparse_maze(hazard_pixel=True)
    spec = detect_cell_lattice(img, background=VOID, tolerance=0)
    assert spec.cell == (3, 3)
    assert spec.purity == pytest.approx(spec.purity_exact)
    assert spec.purity == pytest.approx(5 / 6, abs=1e-6)


def test_ar25_shaped_board_with_a_one_pixel_cursor_is_readable():
    """ar25: 21x21 cells of 3px, a full-cell axis column, a 1px cursor outline."""
    cell, n = 3, 21
    img = np.full((n * cell, n * cell), 9, dtype=np.int64)      # background
    img[:, 10 * cell:11 * cell] = 10                            # axis: a full cell column
    for (r, c) in [(5, 6), (5, 7), (6, 8)]:
        img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = 5
    img[15 * cell:18 * cell, 17 * cell:20 * cell] = 11          # targets
    img[5 * cell, 6 * cell] = 0                                 # cursor: 1px inside a shape cell
    spec = detect_cell_lattice(img, background=9)
    assert spec.cell == (3, 3) and spec.origin == (0, 0)
    assert spec.shape == (21, 21)
    assert spec.purity == pytest.approx(1.0)
    assert spec.purity_exact < 1.0, "the 1px cursor makes a cell inexact -- the tolerance covers it"


# --- equivalence with the reference implementations --------------------------

def _frames():
    out = {"random": RNG.integers(0, 9, size=(64, 64)),
           "sparse": sparse_maze(),
           "blocky3": blocky(3, n_cells=21, phase=(2, 1), seed=3)[0],
           "blocky5": blocky(5, n_cells=12, seed=5)[0],
           "small": RNG.integers(0, 4, size=(16, 16)),
           "wide": RNG.integers(0, 5, size=(30, 40)),
           "image": RNG.integers(0, 3, size=(64, 64))}
    out["image"][10:20, 10:20] = 7                              # a flat patch
    return out


@pytest.mark.parametrize("tolerance", [None, 0])
def test_fast_scan_equals_the_reshape_reference(tolerance):
    """The integral-image scan must return exactly the reshape-per-phase answer."""
    for name, arr in _frames().items():
        bg = infer_background(arr)
        integrals = _integrals_of(arr)
        for size in range(1, 10):
            fast = _scan_size(arr, size, bg, integrals, tolerance=tolerance)
            slow = _best_for_size_bruteforce(arr, size, bg, integrals, tolerance=tolerance)
            assert (fast is None) == (slow is None), (name, size, fast, slow)
            if fast is None:
                continue
            assert fast["purity"] == slow["purity"], (name, size, fast, slow)
            assert fast["purity_exact"] == slow["purity_exact"], (name, size)
            assert fast["active"] == slow["active"], (name, size)
            assert (fast["orow"], fast["ocol"]) == (slow["orow"], slow["ocol"]), (name, size)
            assert (fast["n_rows"], fast["n_cols"]) == (slow["n_rows"], slow["n_cols"])


def test_tolerance_zero_reproduces_the_legacy_tu93_scan():
    """``tolerance=0`` restricted to 3px is the old hand-written scan."""
    for phase in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        arr = sparse_maze(phase=phase)
        old_offset, old_cells = _legacy_detect_lattice(arr)
        spec = detect_cell_lattice(arr, background=VOID, sizes=(3,), tolerance=0)
        assert spec.cell == (3, 3)
        assert spec.origin == old_offset, phase
        assert np.array_equal(spec.cells, old_cells), phase
        # and the public entry point the solver uses agrees with both
        assert tu93_detect_lattice(arr)[0] == old_offset


def test_detection_is_deterministic():
    img = blocky(3, n_cells=21, phase=(1, 2), seed=7)[0]
    first = detect_cell_lattice(img)
    second = detect_cell_lattice(img)
    assert first.to_dict() == second.to_dict()


# --- explicit overrides ------------------------------------------------------

def test_force_cell_pins_the_size():
    img = sparse_maze()
    spec = detect_cell_lattice(img, background=VOID, force_cell=3)
    assert spec.cell == (3, 3) and spec.source == "forced"
    assert spec.origin == (0, 0)


def test_explicit_sizes_restrict_the_scan():
    """Offering fewer sizes can only make the answer coarser-or-native."""
    img, _ = blocky(4, n_cells=16, seed=4)
    assert detect_cell_lattice(img).cell == (4, 4)            # coarsest wins when offered
    assert detect_cell_lattice(img, sizes=(2, 3, 4)).cell == (4, 4)
    # a 4px board is also exactly 2px-blocky, so 2 is a valid sub-lattice read
    assert detect_cell_lattice(img, sizes=(1, 2, 3)).cell == (2, 2)
    # ... and with neither 2 nor 4 on offer nothing explains the board
    assert detect_cell_lattice(img, sizes=(1, 3)).is_native


# --- reporting / helpers -----------------------------------------------------

def test_sweep_reports_the_evidence():
    sweep = lattice_sweep(sparse_maze(), background=VOID)
    assert sweep[3]["purity"] == pytest.approx(1.0)
    assert sweep[3]["purity_exact"] == pytest.approx(5 / 6, abs=1e-4)   # sweep is rounded
    assert sweep[2]["purity"] < 0.5 and sweep[4]["purity"] < 0.5
    assert sweep[3]["align"] > 2.0 > sweep[1]["align"]
    assert sweep[3]["origin"] == (0, 0)
    assert sweep[3]["shape"] == (21, 21)


def test_to_cell_grid_matches_the_detected_grid():
    img = blocky(3, n_cells=21, phase=(2, 1), seed=3)[0]
    spec = detect_cell_lattice(img)
    again = to_cell_grid(img, spec)
    assert np.array_equal(again, spec.cells)


def test_pixel_and_cell_coordinates_round_trip():
    spec = detect_cell_lattice(sparse_maze(phase=(1, 2)), background=VOID)
    n_r, n_c = spec.shape
    for (r, c) in [(0, 0), (5, 7), (n_r - 1, n_c - 1)]:
        y, x = spec.pixel_of(r, c)
        assert spec.cell_of_pixel(y, x) == (r, c)
        cy, cx = spec.center_pixel(r, c)
        assert spec.cell_of_pixel(cy, cx) == (r, c)
    assert spec.cell_of_pixel(-3, 0) is None     # above the lattice
    assert spec.cell_of_pixel(0, 10 ** 4) is None
    assert spec.cell_of_pixel(n_r * spec.cell_r, 0) is None


def test_tolerance_scales_with_the_cell():
    assert [tolerance_for(a) for a in (1, 4, 9, 16, 25, 64)] == [0, 0, 1, 1, 3, 7]


# --- degenerate inputs -------------------------------------------------------

def test_degenerate_inputs_are_safe():
    assert detect_cell_lattice(None).source == "degenerate"
    assert detect_cell_lattice(np.zeros((0, 0))).source == "degenerate"


def test_uniform_frame_has_no_information_but_is_still_a_pixel_grid():
    flat = np.full((64, 64), 3, dtype=np.int64)
    spec = detect_cell_lattice(flat)
    assert spec.is_native, spec
    assert spec.cells.shape == (64, 64)
    assert spec.purity == -1.0, "no informative cells: the score is undefined, not perfect"
    assert np.array_equal(spec.cells, flat.astype(np.int16))
