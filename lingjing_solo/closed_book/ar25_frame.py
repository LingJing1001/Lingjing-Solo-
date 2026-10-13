"""Frame-only perception for AR25 (kaleidoscope mirror puzzle).

Contract — this module **must never read the engine**:

    * no ``environment_files/ar25/*/ar25.py``
    * no obfuscated engine tags
    * no baked per-level answers

Everything is decoded from the ``FrameData`` the agent already receives.  The
render conventions below were established by *observation* (see
``state/_probe_ar25_*.py``) and are properties of the picture, not of the
engine's internals:

    playfield  21 x 21 cells, 3 px per cell, origin at pixel (0, 0);
               the 1 px right/bottom edges are the step-counter ring
    colour  0  selection cursor (a 1 px L outline on the selected object's bbox)
    colour  4  reflected image of a shape
    colour  5  movable shape
    colour  9  background
    colour 10  reflection axis (a 1-cell-wide full column or row)
    colour 11  target cells that must end up covered

Mirror geometry: a cell mirrors across the axis cell coordinate ``a`` as
``c -> 2a - c`` (vertical axis) / ``r -> 2a - r`` (horizontal axis).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from lingjing_solo.core.lattice import detect_cell_lattice

CELL = 3
GRID_N = 21

COL_CURSOR = 0
COL_REFLECT = 4
COL_SHAPE = 5
COL_BG = 9
COL_AXIS = 10
COL_TARGET = 11

Cell = tuple[int, int]


@dataclass
class Ar25FrameObs:
    """Everything recoverable from one AR25 frame."""

    n: int
    axis_kind: Optional[str] = None          # "V" (vertical line) | "H" (horizontal)
    axis_coord: Optional[int] = None         # column index for V, row index for H
    shapes: list[list[Cell]] = field(default_factory=list)
    target_cells: set[Cell] = field(default_factory=set)
    cursor_cells: set[Cell] = field(default_factory=set)
    reflect_cells: set[Cell] = field(default_factory=set)
    selected: Optional[list[Cell]] = None
    ok: bool = False                         # parse is unambiguous enough to act on
    note: str = ""

    def mirror(self, cell: Cell) -> Optional[Cell]:
        if self.axis_kind is None or self.axis_coord is None:
            return None
        r, c = cell
        if self.axis_kind == "V":
            return (r, 2 * self.axis_coord - c)
        return (2 * self.axis_coord - r, c)

    def mirror_cells(self, cells) -> set[Cell]:
        out = set()
        for cell in cells:
            m = self.mirror(cell)
            if m is not None:
                out.add(m)
        return out


def _geometry(layer) -> tuple[int, int, int, int]:
    """``(n_rows, n_cols, origin_r, origin_c)`` for the board.

    The lattice is *read from the pixels* (``core.lattice``): 21 x 21 cells of
    3px at origin (0, 0), with the 1px step-counter ring on the right/bottom
    edge falling outside the lattice.  The hardcoded 21/3 geometry is only the
    fallback for frames the generic read cannot confirm.
    """
    arr = np.asarray(layer)
    if arr.ndim == 2 and arr.size:
        spec = detect_cell_lattice(arr, background=COL_BG, max_cell=8)
        if spec.cell == (CELL, CELL) and spec.n_rows >= 2 and spec.n_cols >= 2:
            return spec.n_rows, spec.n_cols, spec.origin_r, spec.origin_c
    h = len(layer)
    w = len(layer[0]) if h else 0
    n = min(GRID_N, h // CELL, w // CELL)
    return n, n, 0, 0


def _to_cells(layer) -> list[list[Counter]]:
    n_rows, n_cols, o_r, o_c = _geometry(layer)
    cells: list[list[Counter]] = []
    for r in range(n_rows):
        row: list[Counter] = []
        for c in range(n_cols):
            cnt: Counter = Counter()
            for y in range(o_r + r * CELL, o_r + r * CELL + CELL):
                for x in range(o_c + c * CELL, o_c + c * CELL + CELL):
                    cnt[int(layer[y][x])] += 1
            row.append(cnt)
        cells.append(row)
    return cells


def _components(cells: set[Cell]) -> list[list[Cell]]:
    """4-connected components of a cell set."""
    remaining = set(cells)
    comps: list[list[Cell]] = []
    while remaining:
        seed = remaining.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    comp.append((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _role_colour(roles, name, default=None):
    """Read one colour off a ``RoleSpec`` (or a plain mapping); ``None`` -> default."""
    if roles is None:
        return default
    value = roles.get(name) if isinstance(roles, dict) else getattr(roles, name, None)
    return default if value is None else int(value)


def roles_probe(frames, actions=None, levels=None, background=None):
    """Infer this game's palette roles from a trajectory -- *side evidence only*.

    Planning never calls it.  On ar25 the generic inference independently reports
    ``background=9 avatar=5 mirror=4 cursor=0``, i.e. exactly
    ``COL_BG``/``COL_SHAPE``/``COL_REFLECT``/``COL_CURSOR``.
    """
    from lingjing_solo.core.roles import infer_roles

    return infer_roles(frames, actions=actions, levels=levels, background=background)


def parse_frame(frame: Any) -> Ar25FrameObs:
    """Decode one frame into an :class:`Ar25FrameObs`, with this module's palette."""
    return _parse(frame, shape=COL_SHAPE, cursor=COL_CURSOR, reflect=COL_REFLECT)


def parse_frame_with_roles(frame: Any, roles) -> Ar25FrameObs:
    """``parse_frame`` with shape / cursor / reflection taken from role inference.

    The axis and the targets keep this module's constants: the axis is already
    recovered geometrically (a full row or column) and a wrong target cell would
    make the planner chase the wrong thing.  Used only under
    ``LINGJING_ROLES_FALLBACK=1``.
    """
    return _parse(frame,
                  shape=_role_colour(roles, "avatar", COL_SHAPE),
                  cursor=_role_colour(roles, "cursor", COL_CURSOR),
                  reflect=_role_colour(roles, "mirror", COL_REFLECT))


def _parse(frame: Any, *, shape: int = COL_SHAPE, cursor: int = COL_CURSOR,
           reflect: int = COL_REFLECT) -> Ar25FrameObs:
    try:
        layers = getattr(frame, "frame", None) or []
        if not layers:
            return Ar25FrameObs(GRID_N, note="empty frame")
        grid = _to_cells(list(layers[0]))
        n = len(grid)
        axis_cells: set[Cell] = set()
        shape_cells: set[Cell] = set()
        target_cells: set[Cell] = set()
        cursor_cells: set[Cell] = set()
        reflect_cells: set[Cell] = set()

        for r in range(n):
            for c in range(n):
                cnt = grid[r][c]
                if COL_AXIS in cnt:
                    axis_cells.add((r, c))
                if shape in cnt:
                    shape_cells.add((r, c))
                if COL_TARGET in cnt:
                    target_cells.add((r, c))
                if cursor in cnt:
                    cursor_cells.add((r, c))
                if reflect in cnt:
                    reflect_cells.add((r, c))

        obs = Ar25FrameObs(
            n=n,
            target_cells=target_cells,
            cursor_cells=cursor_cells,
            reflect_cells=reflect_cells,
        )

        # --- axis: all axis cells share one row or one column -----------------
        if axis_cells:
            rows = {r for r, _ in axis_cells}
            cols = {c for _, c in axis_cells}
            if len(cols) == 1:
                obs.axis_kind, obs.axis_coord = "V", next(iter(cols))
            elif len(rows) == 1:
                obs.axis_kind, obs.axis_coord = "H", next(iter(rows))

        obs.shapes = _components(shape_cells)

        # --- selected shape: the component whose bbox contains the cursor -----
        if cursor_cells and obs.shapes:
            hits = []
            for comp in obs.shapes:
                r0, c0, r1, c1 = _bbox(comp)
                if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cursor_cells):
                    hits.append(comp)
            if len(hits) == 1:
                obs.selected = hits[0]
            elif len(hits) > 1:
                obs.selected = min(hits, key=len)

        obs.ok = (
            obs.axis_kind is not None
            and bool(obs.shapes)
            and bool(obs.target_cells)
        )
        if not obs.ok:
            missing = []
            if obs.axis_kind is None:
                missing.append("axis")
            if not obs.shapes:
                missing.append("shape")
            if not obs.target_cells:
                missing.append("target")
            obs.note = "unparsed: " + ",".join(missing)
        return obs
    except Exception as exc:  # noqa: BLE001 — perception must never kill the run
        return Ar25FrameObs(GRID_N, note=f"parse error: {type(exc).__name__}: {exc}")
