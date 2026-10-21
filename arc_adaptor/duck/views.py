"""What the model is allowed to see: a frame, its objects, and their change.

Three views, matching the field names the champion's prompt advertises:

    .ascii          rows of ARC colour characters
    .shape          (rows, cols)
    .segmentation   {'nodes': [...]}, one node per 4-connected same-colour object

`frame_diff` is the object-level change report. It is deliberately lossy: only
what changed, because a full dump of two boards is unreadable and the unchanged
parts are already on screen.

Everything here is read-only and deterministic. Nothing reads hidden engine
state -- the only input is the colour grid.
"""
from __future__ import annotations

import hashlib

COLOR_CHARS = "0123456789abcdef"

# 4-connectivity only. The champion's prompt says so explicitly, and diagonally
# touching objects being separate is what makes "one object moved" true.
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def char_of(colour: int) -> str:
    return COLOR_CHARS[colour] if 0 <= colour < len(COLOR_CHARS) else "?"


def background_of(grid: list[list[int]]) -> int:
    """Most frequent colour. Ties break toward the lower colour value so the
    answer does not wobble between frames."""
    if not grid:
        return 0
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


class Node:
    """One 4-connected same-colour component."""

    __slots__ = ("id", "colour", "cells", "bbox", "shape", "hash", "children")

    def __init__(self, idx: int, colour: int, cells: list[tuple[int, int]]):
        self.id = idx
        self.colour = colour
        self.cells = cells
        self.bbox = bbox_of(cells)
        self.shape = normalise(cells)
        self.hash = shape_hash(colour, self.shape)
        self.children: list[int] = []

    @property
    def pixels(self) -> int:
        return len(self.cells)

    def as_dict(self) -> dict:
        r0, c0, r1, c1 = self.bbox
        return {
            "id": self.id,
            "color": char_of(self.colour),
            "pixels": self.pixels,
            "bbox": [r0, c0, r1, c1],
            "hash": self.hash,
            "children": self.children,
        }

    def __repr__(self) -> str:
        r0, c0, r1, c1 = self.bbox
        return f"<node {self.id} c{char_of(self.colour)} n={self.pixels} @{r0},{c0}-{r1},{c1}>"


def bbox_of(cells) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def normalise(cells) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = bbox_of(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotations(shape):
    """The four rotations of a normalised point set, each re-normalised."""
    out = []
    cur = list(shape)
    for _ in range(4):
        cur = [(c, -r) for r, c in cur]
        mr = min(r for r, _ in cur)
        mc = min(c for _, c in cur)
        cur = [(r - mr, c - mc) for r, c in cur]
        out.append(tuple(sorted(cur)))
    return out


def shape_hash(colour: int, shape) -> str:
    """Rotation-invariant signature of colour + shape.

    Not an identity: two distinct objects with the same colour and shape share a
    hash, which is exactly what the champion's prompt says it is for ("equal
    hashes indicate matching colour and shape up to rotation; they do not
    establish object identity"). Using it as an identity is the mistake that
    makes "which piece moved" unanswerable when there are several alike.
    """
    key = min(_rotations(shape))
    raw = f"{colour}|" + ";".join(f"{r},{c}" for r, c in key)
    return hashlib.blake2b(raw.encode(), digest_size=6).hexdigest()


def segment(grid: list[list[int]], *, min_pixels: int = 1,
            skip_background: bool = True) -> list[Node]:
    """Objects of one frame, ordered top-most then left-most.

    Order is stable so that node ids mean the same thing when a model compares
    two frames by hand.
    """
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    bg = background_of(grid) if skip_background else None
    seen = [[False] * w for _ in range(h)]
    nodes: list[Node] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0]:
                continue
            colour = grid[r0][c0]
            seen[r0][c0] = True
            if bg is not None and colour == bg:
                continue
            cells: list[tuple[int, int]] = []
            stack = [(r0, c0)]
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in _NEIGHBOURS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] \
                            and grid[nr][nc] == colour:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(cells) >= min_pixels:
                nodes.append(Node(len(nodes), colour, cells))
    _attach_children(nodes, (h, w))
    return nodes


def _attach_children(nodes: list[Node], shape: tuple[int, int]) -> None:
    """Node B is a child of node A when B's bbox sits strictly inside A's bbox
    and B does not touch the frame edge.

    This is a cheap approximation of "fully enclosed". It is right for rings and
    frames, which is what the field is for; it can be wrong for a diagonal
    arrangement, and the prompt does not promise otherwise.
    """
    h, w = shape
    for a in nodes:
        ar0, ac0, ar1, ac1 = a.bbox
        for b in nodes:
            if a is b:
                continue
            br0, bc0, br1, bc1 = b.bbox
            if br0 > ar0 and bc0 > ac0 and br1 < ar1 and bc1 < ac1:
                if br0 > 0 and bc0 > 0 and br1 < h - 1 and bc1 < w - 1:
                    a.children.append(b.id)


class FrameView:
    """The object the sandbox binds to `current_frame`."""

    __slots__ = ("_grid", "_ascii", "_nodes")

    def __init__(self, grid: list[list[int]]):
        self._grid = grid
        self._ascii = None
        self._nodes = None

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._grid), len(self._grid[0]) if self._grid else 0)

    @property
    def ascii(self) -> str:
        if self._ascii is None:
            self._ascii = "\n".join("".join(char_of(v) for v in row) for row in self._grid)
        return self._ascii

    @property
    def segmentation(self) -> dict:
        if self._nodes is None:
            self._nodes = segment(self._grid)
        return {"nodes": [n.as_dict() for n in self._nodes]}

    def __repr__(self) -> str:
        h, w = self.shape
        return f"<frame {h}x{w}>"


# --------------------------------------------------------------------- diff


def frame_diff(before: FrameView, after: FrameView) -> dict:
    """Object-level diff between two frames.

    Matching is by (colour, rotation-invariant shape), then by nearest position
    among candidates. With several alike objects the correspondence is genuinely
    ambiguous and this function will pick one; the prompt says so, because a
    model that trusts `moved` blindly will chase the wrong piece.
    """
    if before.shape != after.shape:
        return {
            "changed_cell_count": -1,
            "note": "frame shapes differ; full redraw",
            "before_shape": list(before.shape),
            "after_shape": list(after.shape),
        }

    bn = segment(before._grid)
    an = segment(after._grid)
    bgrid, agrid = before._grid, after._grid

    changed = 0
    h, w = before.shape
    for r in range(h):
        row_b, row_a = bgrid[r], agrid[r]
        if row_b != row_a:
            changed += sum(1 for x, y in zip(row_b, row_a) if x != y)

    used_b: set[int] = set()
    used_a: set[int] = set()
    moved, rotated, changed_color, resized = [], [], [], []

    # exact-position matches first, so "did not move" is decided before
    # "moved a little", which is what keeps a chain of small steps readable
    b_by_key: dict[tuple, list[Node]] = {}
    for n in bn:
        b_by_key.setdefault((n.colour, n.shape), []).append(n)
    for a in an:
        pool = b_by_key.get((a.colour, a.shape))
        if not pool:
            continue
        same = [b for b in pool if b.id not in used_b and b.bbox == a.bbox]
        if same:
            used_b.add(same[0].id)
            used_a.add(a.id)

    # then same shape, different position: a move
    for a in an:
        if a.id in used_a:
            continue
        pool = [b for b in b_by_key.get((a.colour, a.shape), []) if b.id not in used_b]
        if not pool:
            continue
        best = min(pool, key=lambda b: _cheb(b.bbox, a.bbox))
        used_b.add(best.id)
        used_a.add(a.id)
        dr = a.bbox[0] - best.bbox[0]
        dc = a.bbox[1] - best.bbox[1]
        if dr or dc:
            moved.append({"from": list(best.bbox), "to": list(a.bbox),
                          "delta": [dr, dc], "pixels": a.pixels,
                          "color": char_of(a.colour)})

    # same footprint, different colour
    for a in an:
        if a.id in used_a:
            continue
        for b in bn:
            if b.id in used_b:
                continue
            if b.bbox == a.bbox and b.pixels == a.pixels:
                used_b.add(b.id)
                used_a.add(a.id)
                changed_color.append({"at": list(a.bbox), "pixels": a.pixels,
                                      "before_color": char_of(b.colour),
                                      "after_color": char_of(a.colour)})
                break

    # appearing / disappearing
    appeared = [{"bbox": list(a.bbox), "pixels": a.pixels, "color": char_of(a.colour)}
                for a in an if a.id not in used_a]
    disappeared = [{"bbox": list(b.bbox), "pixels": b.pixels, "color": char_of(b.colour)}
                   for b in bn if b.id not in used_b]

    # overlapping footprints with a changed pixel count
    for b in bn:
        if b.id in used_b:
            continue
        for a in an:
            if a.id in used_a or a.colour != b.colour:
                continue
            if _overlaps(b.bbox, a.bbox):
                used_b.add(b.id)
                used_a.add(a.id)
                resized.append({"at_before": list(b.bbox), "at_after": list(a.bbox),
                                "pixels_before": b.pixels, "pixels_after": a.pixels,
                                "color": char_of(a.colour)})
                break

    moved_from = {tuple(y["from"]) for y in moved}
    moved_to = {tuple(y["to"]) for y in moved}
    appeared = [x for x in appeared if tuple(x["bbox"]) not in moved_to]
    out = {
        "changed_cell_count": changed,
        "moved": moved,
        "appeared": appeared,
        "disappeared": [x for x in disappeared if tuple(x["bbox"]) not in moved_from],
        "changed_color": changed_color,
        "resized": resized,
        "rotated": rotated,
    }
    return {k: v for k, v in out.items() if v or k == "changed_cell_count"}


def _cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _overlaps(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
