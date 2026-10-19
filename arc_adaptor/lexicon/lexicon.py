"""ARCAID3 lexicon: turn an ARC-AGI-3 screen into a short "code" listing.

What it does, in plain words
----------------------------
An ARC-AGI-3 screen is a 64x64 grid of colour numbers (0-15). The official games
are not drawn pixel by pixel: each level is a small world of named "sprites"
(little multi-colour pictures) that the engine places on a grid and then scales
up to fill 64x64. This module tries to undo that, using the screen alone:

1. Native grid ("unzoom"): find the block size k the screen was scaled by, and
   read the world back at its real resolution (e.g. 16x16 cells instead of
   64x64 pixels).
2. Objects: cut the world into pieces of touching non-background cells. Unlike
   the Duck harness's `segmentation`, a piece may hold several colours, because
   the games' own building blocks usually do.
3. Lexicon: give every distinct picture a short symbol (A, B, C ...) that stays
   the same across all frames of a game, and note when one symbol is just a
   rotated or mirrored copy of another.
4. Code: print the frame as a legend (symbol -> small picture) plus a list of
   placements ("A at row 2, col 3"), and compare two frames as object moves
   ("A moved right by 1") instead of cell flips.

Rules this file keeps
---------------------
- Standard library only, no imports outside it, so its source can be pasted
  into the Duck harness's Python sandbox (which runs `python -I -S`).
- It reads only what an agent may read: the frame's colour grid. It never
  touches game files or hidden engine state.
- Deterministic: the same frames in the same order give the same symbols.
"""

DEFAULT_COLOR_CHARS = "WwgGcBMPRbSYOrNp"  # the Duck harness's ARC_COLOR_CHARS, index = colour value
COLOR_NAMES = (
    "white", "off-white", "light-grey", "grey", "dark-grey", "black", "magenta", "pink",
    "red", "blue", "light-blue", "yellow", "orange", "maroon", "green", "purple",
)


# --------------------------------------------------------------------------- native grid

def _boundaries(grid, axis):
    """Positions where a row (axis 0) or column (axis 1) differs from the previous one."""
    h, w = len(grid), len(grid[0])
    out = []
    if axis == 0:
        for y in range(1, h):
            if grid[y] != grid[y - 1]:
                out.append(y)
    else:
        for x in range(1, w):
            for y in range(h):
                if grid[y][x] != grid[y][x - 1]:
                    out.append(x)
                    break
    return out


def _best_offset(bounds, k):
    if not bounds:
        return 0, 1.0
    votes = [0] * k
    for b in bounds:
        votes[b % k] += 1
    best = max(range(k), key=lambda r: (votes[r], -r))
    return best, votes[best] / len(bounds)


def _uniform_fraction(grid, k, oy, ox):
    h, w = len(grid), len(grid[0])
    total = uniform = 0
    for by in range(oy, h - k + 1, k):
        for bx in range(ox, w - k + 1, k):
            v = grid[by][bx]
            ok = True
            for y in range(by, by + k):
                row = grid[y]
                for x in range(bx, bx + k):
                    if row[x] != v:
                        ok = False
                        break
                if not ok:
                    break
            total += 1
            uniform += ok
    return (uniform / total) if total else 0.0


def _interior_runs(grid):
    """Lengths of equal-colour runs along rows and columns that touch neither screen edge."""
    h, w = len(grid), len(grid[0])
    runs = []
    for line in [grid[y] for y in range(h)] + [[grid[y][x] for y in range(h)] for x in range(w)]:
        start = 0
        for i in range(1, len(line) + 1):
            if i == len(line) or line[i] != line[start]:
                if start > 0 and i < len(line):
                    runs.append(i - start)
                start = i
    return runs


def _fits(grid, k, oy, ox, runs, min_runs, min_uniform):
    """True when block size k at offset (oy, ox) passes both per-frame tests on this grid."""
    if not runs or sum(1 for n in runs if n % k == 0) / len(runs) < min_runs:
        return False
    return _uniform_fraction(grid, k, oy, ox) >= min_uniform


def _detect(grid, scales=(8, 7, 6, 5, 4, 3, 2), min_runs=0.85, min_uniform=0.75):
    """Per-frame block size: (k, oy, ox, uniform share). (1, 0, 0, 1.0) when no scaling is found."""
    h, w = len(grid), len(grid[0]) if grid else 0
    rows_b, cols_b = _boundaries(grid, 0), _boundaries(grid, 1)
    runs = _interior_runs(grid) if h and w else []
    for k in scales:
        if k * 2 > min(h, w) or not runs:
            continue
        if sum(1 for n in runs if n % k == 0) / len(runs) < min_runs:
            continue
        oy, _ = _best_offset(rows_b, k)
        ox, _ = _best_offset(cols_b, k)
        frac = _uniform_fraction(grid, k, oy, ox)
        if frac >= min_uniform:
            return (k, oy, ox, frac)
    return (1, 0, 0, 1.0)


def _mixed_blocks(grid, k, oy, ox, margin=4, limit=2):
    """How many k-blocks away from the screen edge hold more than one colour (counted up to `limit`).

    Blocks within `margin` pixels of the edge are skipped, because interface bars are drawn
    there at screen resolution.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    n = 0
    for by in range(oy, h - k + 1, k):
        if by < margin or by + k > h - margin:
            continue
        for bx in range(ox, w - k + 1, k):
            if bx < margin or bx + k > w - margin:
                continue
            v = grid[by][bx]
            if any(grid[y][x] != v for y in range(by, by + k) for x in range(bx, bx + k)):
                n += 1
                if n >= limit:
                    return n
    return n


def scale_lock(grids, scales=(8, 7, 6, 5, 4, 3, 2), min_runs=0.85, min_uniform=0.75, tolerance=0.0,
               max_frames=256, finer_check=True):
    """One block size for a whole level, agreed across its frames.

    A single sparse frame can look scaled by a coarser block than the game uses
    (everything on it happens to line up at that size), and then every name on
    that frame changes. The true block size passes the per-frame tests on every
    frame of the level, while a spurious coarse one fails on a frame with finer
    detail. So: take each frame's own detection as a candidate, and return the
    largest candidate that passes both tests on every frame given, as (k, oy, ox);
    `tolerance` is the share of frames allowed to fail (0 = strict).

    A coarse candidate is also rejected (`finer_check`) when some frame has two or more mixed
    blocks away from the screen edge at k while a smaller candidate reads that frame's interior
    with none: sprites are drawn at the camera's scale, so such detail means k is too coarse
    (typically an object that moved off the coarse grid). The divisors of every candidate (2 or
    more) join the candidates. Raw pixels (k = 1) are never used for the comparison, so a
    one-pixel interface overlay, mixed at every size, cannot force a level down to pixels.

    All frames are used, not a sample, so the lock is monotone: once a frame has
    rejected a block size, the level keeps it rejected and the lock cannot flip
    back. Only the level's first frame and its last `max_frames - 1` frames are
    read, to bound the cost. Pass frames of one level only: the camera scale can
    change between levels (4 of the 25 public games do).
    """
    grids = [g for g in grids if g]
    if not grids:
        return None
    grids = [g for i, g in enumerate(grids) if i == 0 or g != grids[i - 1]]
    if len(grids) > max_frames:
        keep_head = min(32, max_frames // 4)
        grids = grids[:keep_head] + grids[-(max_frames - keep_head):]
    grids = [[list(r) for r in g] for g in grids]
    runs = [_interior_runs(g) for g in grids]
    cands = {}
    for g in grids:
        k, oy, ox, _ = _detect(g, scales, min_runs, min_uniform)
        if k > 1:
            cands[(k, oy, ox)] = cands.get((k, oy, ox), 0) + 1
    allowed = int(tolerance * len(grids))
    if finer_check:
        for k, oy, ox in list(cands):
            for d in range(2, k):
                if k % d == 0:
                    cands.setdefault((d, oy % d, ox % d), 0)
    order = sorted(cands, key=lambda c: (-c[0], -cands[c], c[1], c[2]))
    for i, (k, oy, ox) in enumerate(order):
        failed = 0
        for g, r in zip(grids, runs):
            if not _fits(g, k, oy, ox, r, min_runs, min_uniform):
                failed += 1
                if failed > allowed:
                    break
        if failed > allowed:
            continue
        smaller = [c for c in order[i + 1:] if c[0] < k]
        if finer_check and smaller and any(
                _mixed_blocks(g, k, oy, ox) >= 2
                and any(_mixed_blocks(g, *c, limit=1) == 0 and _fits(g, *c, r, min_runs, min_uniform) for c in smaller)
                for g, r in zip(grids, runs)):
            continue                                    # too coarse: a smaller grid reads this level cleanly
        return (k, oy, ox)
    return (1, 0, 0)


def native_grid(grid, scales=(8, 7, 6, 5, 4, 3, 2), min_runs=0.85, min_uniform=0.75, lock=None):
    """Detect the scale-up block size and return the world at its native resolution.

    A screen scaled up by k has colour runs whose lengths are multiples of k,
    except where the game paints a HUD bar or counter over the picture. So k is
    the largest block size that (a) divides at least `min_runs` of the interior
    runs and (b) leaves at least `min_uniform` of the k x k blocks single-coloured.

    Returns a dict:
      scale       block size k (1 = no scaling found)
      offset      (row, col) of the first block in screen pixels
      cells       native grid, list of rows (majority colour of each k x k block)
      uniform     share of blocks that are a single colour (1.0 = lossless)
      residue     screen cells the upscaled native grid does not reproduce
                  (HUD bars drawn over the scaled picture, odd margins)
      locked      True when `lock` = (k, oy, ox) from `scale_lock` was used instead
                  of this frame's own detection
    """
    grid = [list(r) for r in grid]
    h, w = len(grid), len(grid[0]) if grid else 0
    if lock and 1 <= lock[0] and lock[0] * 2 <= min(h, w):
        k, oy, ox = lock
        frac = _uniform_fraction(grid, k, oy, ox) if k > 1 else 1.0
    else:
        lock = None
        k, oy, ox, frac = _detect(grid, scales, min_runs, min_uniform)
    cells = []
    for by in range(oy, h - k + 1, k):
        row = []
        for bx in range(ox, w - k + 1, k):
            counts = {}
            for y in range(by, by + k):
                for x in range(bx, bx + k):
                    counts[grid[y][x]] = counts.get(grid[y][x], 0) + 1
            row.append(max(sorted(counts), key=lambda c: counts[c]))
        cells.append(row)
    residue = 0
    nh, nw = len(cells), len(cells[0]) if cells else 0
    for y in range(h):
        for x in range(w):
            ny, nx = (y - oy) // k, (x - ox) // k
            inside = 0 <= y - oy and 0 <= x - ox and ny < nh and nx < nw
            if not inside or cells[ny][nx] != grid[y][x]:
                residue += 1
    return {"scale": k, "offset": (oy, ox), "cells": cells, "uniform": round(frac, 4),
            "residue": residue, "locked": lock is not None}


# --------------------------------------------------------------------------- objects

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def background_colour(cells):
    counts = {}
    for row in cells:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda c: counts[c])


def objects(cells, background=None, connectivity=8, multicolour=True):
    """Connected pieces of non-background cells.

    multicolour=True joins touching cells of any colours (the lexicon's view);
    multicolour=False joins only equal colours (like the Duck harness's view).
    Each object: {"cells": [(r, c), ...] sorted, "bbox": (r0, c0, r1, c1), "colours": [...]}.
    """
    if background is None:
        background = background_colour(cells)
    h, w = len(cells), len(cells[0]) if cells else 0
    nbrs = _N8 if connectivity == 8 else _N4
    seen = [[False] * w for _ in range(h)]
    out = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or cells[sr][sc] == background:
                continue
            colour = cells[sr][sc]
            stack, members = [(sr, sc)], []
            seen[sr][sc] = True
            while stack:
                r, c = stack.pop()
                members.append((r, c))
                for dr, dc in nbrs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc]:
                        v = cells[nr][nc]
                        if v != background and (multicolour or v == colour):
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            members.sort()
            rs = [p[0] for p in members]
            cs = [p[1] for p in members]
            out.append({
                "cells": members,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "colours": sorted({cells[r][c] for r, c in members}),
            })
    return out


def layered_objects(cells, background=None, connectivity=8, min_container=24, depth=3):
    """Objects with floors peeled off: the lexicon's default view.

    Games draw small sprites on top of large ones (a floor, a panel, a board).
    A flat cut glues everything on a floor into one piece. Here, a large
    multi-colour piece is treated as a container: its most common colour is its
    local floor, the floor cells become one object, and the cells that sit on it
    are cut again (up to `depth` levels). Each object gets a "depth" and, below
    the top, the index of its container ("parent").
    """
    if background is None:
        background = background_colour(cells)
    h, w = len(cells), len(cells[0]) if cells else 0
    out = []

    def cut(pieces, level, parent):
        for obj in pieces:
            members = obj["cells"]
            if level >= depth or len(members) < min_container or len(obj["colours"]) < 2:
                out.append(dict(obj, depth=level, parent=parent))
                continue
            counts = {}
            for r, c in members:
                counts[cells[r][c]] = counts.get(cells[r][c], 0) + 1
            floor = max(sorted(counts), key=lambda v: counts[v])
            floor_cells = sorted(p for p in members if cells[p[0]][p[1]] == floor)
            rest = set(members) - set(floor_cells)
            if not rest:
                out.append(dict(obj, depth=level, parent=parent))
                continue
            masked = [[-99] * w for _ in range(h)]
            for r, c in rest:
                masked[r][c] = cells[r][c]
            rs = [p[0] for p in floor_cells]
            cs = [p[1] for p in floor_cells]
            me = len(out)
            out.append({"cells": floor_cells, "bbox": (min(rs), min(cs), max(rs), max(cs)),
                        "colours": [floor], "depth": level, "parent": parent, "container": True})
            cut(objects(masked, -99, connectivity=connectivity, multicolour=True), level + 1, me)

    cut(objects(cells, background, connectivity=connectivity, multicolour=True), 0, None)
    return out


# --------------------------------------------------------------------------- lexicon

def _patch(cells, obj):
    r0, c0, r1, c1 = obj["bbox"]
    member = set(obj["cells"])
    return tuple(
        tuple(cells[r][c] if (r, c) in member else -1 for c in range(c0, c1 + 1))
        for r in range(r0, r1 + 1)
    )


def _rot90(p):  # clockwise
    return tuple(tuple(p[len(p) - 1 - r][c] for r in range(len(p))) for c in range(len(p[0])))


def _mirror(p):
    return tuple(tuple(reversed(row)) for row in p)


def _variants(p):
    """The 8 rotations/mirrors of a patch, with a readable name for each."""
    out, cur = [], p
    for k in range(4):
        out.append((f"rot{90 * k}", cur))
        out.append((f"mirror+rot{90 * k}", _mirror(cur)))
        cur = _rot90(cur)
    return out


def _symbol_name(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _stable_name(patch, color_chars):
    """Content-derived name: dominant colour letter + 4 hex digits of the picture's hash.

    Python hashes tuples of ints the same way in every process (only str/bytes
    hashing is randomised), so a fresh sandbox process names a picture the same
    way every time without any saved state. 4 digits, not 3: a level can hold 200
    pictures, 70 of one colour, and on 45 recorded public plays 3 digits gave two
    pictures one name on 34 of 161 levels (4 digits: none).
    """
    counts = {}
    for row in patch:
        for v in row:
            if v >= 0:
                counts[v] = counts.get(v, 0) + 1
    dom = max(sorted(counts), key=lambda c: counts[c]) if counts else 0
    return f"{color_chars[dom]}{hash(patch) & 0xFFFF:04x}"


class Lexicon:
    """Persistent symbol table for one game: picture -> symbol.

    naming="sequential" gives A, B, C ... in order of first sight (readable; needs
    one Lexicon kept across frames). naming="stable" gives content-derived names
    such as "R3fa" that come out the same in any process (for the Duck sandbox,
    which starts a fresh process for every tool call).
    """

    def __init__(self, color_chars=DEFAULT_COLOR_CHARS, naming="sequential"):
        self.color_chars = color_chars
        self.naming = naming
        self.by_key = {}      # exact patch -> symbol
        self.entries = {}     # symbol -> {"patch", "count", "first_seen", "family", "pose"}
        self.by_canon = {}    # rotation/mirror-canonical patch -> first symbol with that shape

    def symbol_for(self, patch, seen_at=None):
        sym = self.by_key.get(patch)
        if sym is None:
            if self.naming == "stable":
                sym = base = _stable_name(patch, self.color_chars)
                n = 0
                while sym in self.entries:        # rare: two pictures share a fingerprint
                    n += 1                        # the suffix comes from the picture itself, not from
                    sym = f"{base}_{hash((n, patch)) & 0xFFFFFF:06x}"   # which picture was seen first
            else:
                sym = _symbol_name(len(self.entries))
            variants = _variants(patch)
            canon_name, canon = min(variants, key=lambda v: v[1])
            family = self.by_canon.setdefault(canon, sym)
            pose = None
            if family != sym:
                fam_patch = self.entries[family]["patch"]
                for name, var in _variants(fam_patch):
                    if var == patch:
                        pose = name
                        break
            self.by_key[patch] = sym
            self.entries[sym] = {"patch": patch, "count": 0, "first_seen": seen_at,
                                 "family": family, "pose": pose}
        self.entries[sym]["count"] += 1
        return sym

    def picture(self, sym):
        cc = self.color_chars
        return ["".join("." if v < 0 else cc[v] for v in row) for row in self.entries[sym]["patch"]]


# --------------------------------------------------------------------------- encode / diff

def _piece(cells, members, like):
    members = sorted(members)
    rs, cs = [p[0] for p in members], [p[1] for p in members]
    return {"cells": members, "bbox": (min(rs), min(cs), max(rs), max(cs)),
            "colours": sorted({cells[r][c] for r, c in members}), "depth": like.get("depth", 0),
            "parent": like.get("parent")}


def _components(members):
    """Connected pieces (8-neighbour) of a set of cells."""
    members, out = set(members), []
    while members:
        stack = [members.pop()]
        comp = set(stack)
        while stack:
            r, c = stack.pop()
            for dr, dc in _N8:
                q = (r + dr, c + dc)
                if q in members:
                    members.discard(q)
                    comp.add(q)
                    stack.append(q)
        out.append(comp)
    return out


def carve(cells, pieces, prev, lexicon, radius=6):
    """Object permanence: cut a piece back into objects that were apart on the previous frame.

    Two objects that touch become one piece (pieces may hold several colours). A piece is cut
    only when it is new (its picture was not seen before), holds two or more colours, and splits
    exactly into pictures present on the previous frame: each found whole (same cells, same
    colours) within `radius` cells of where it was, the remainder also a picture from that frame.
    Otherwise the piece is kept as it is.
    """
    if not prev:
        return pieces
    before = [p for p in prev["placed"] if not p["container"]]
    prev_syms = {p["sym"] for p in before}
    pics = []
    for p in before:
        patch = lexicon.entries[p["sym"]]["patch"]
        pcs = [(i, j, v) for i, row in enumerate(patch) for j, v in enumerate(row) if v >= 0]
        pics.append((len(pcs), p["sym"], p["at"], pcs))
    pics.sort(key=lambda t: (-t[0], t[1], t[2]))
    offsets = sorted(((dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)),
                     key=lambda d: (abs(d[0]) + abs(d[1]), d))
    out = []
    for obj in pieces:
        members = set(obj["cells"])
        if (obj.get("container") or len(obj["colours"]) < 2 or len(members) < 2
                or _patch(cells, obj) in lexicon.by_key):
            out.append(obj)
            continue
        r0, c0, r1, c1 = obj["bbox"]
        left, cut = set(members), []
        for n, sym, (pr, pc), pcs in pics:
            if n >= len(left) or pr > r1 + radius or pc > c1 + radius or pr + radius < r0 - n or pc + radius < c0 - n:
                continue
            for dr, dc in offsets:
                r, c = pr + dr, pc + dc
                spot = [(r + i, c + j) for i, j, _ in pcs]
                if all(q in left and cells[q[0]][q[1]] == v for q, (_, _, v) in zip(spot, pcs)):
                    cut.append(set(spot))
                    left -= set(spot)
                    break
        if not cut:
            out.append(obj)
            continue
        rest = _components(left)
        ok = all(lexicon.by_key.get(_patch(cells, {"cells": sorted(comp), "bbox": (
            min(q[0] for q in comp), min(q[1] for q in comp), max(q[0] for q in comp), max(q[1] for q in comp))}))
            in prev_syms for comp in rest)
        if not ok:
            out.append(obj)
            continue
        out += [_piece(cells, comp, obj) for comp in cut + rest]
    return out


def encode(grid, lexicon=None, connectivity=8, layered=True, seen_at=None, lock=None, prev=None):
    """Encode one screen. Returns a dict that `to_text` renders as the code listing.

    `lock` = (k, oy, ox) from `scale_lock` reads the screen at the level's agreed
    block size instead of this frame's own detection. `prev` = the previous frame's
    encoding (same level, same Lexicon): pieces that merged objects seen apart there
    are cut back into them (`carve`).
    """
    lexicon = lexicon if lexicon is not None else Lexicon()
    nat = native_grid(grid, lock=lock)
    cells = nat["cells"]
    bg = background_colour(cells)
    pieces = (layered_objects(cells, bg, connectivity=connectivity) if layered
              else objects(cells, bg, connectivity=connectivity, multicolour=True))
    if prev is not None and (prev["scale"], prev["offset"]) == (nat["scale"], nat["offset"]):
        pieces = carve(cells, pieces, prev, lexicon)
    placed = []
    for obj in pieces:
        sym = lexicon.symbol_for(_patch(cells, obj), seen_at=seen_at)
        r0, c0 = obj["bbox"][0], obj["bbox"][1]
        placed.append({"sym": sym, "at": (r0, c0), "size": len(obj["cells"]), "cells": obj["cells"],
                       "depth": obj.get("depth", 0), "container": bool(obj.get("container"))})
    placed.sort(key=lambda p: (p["sym"], p["at"]))
    return {"scale": nat["scale"], "offset": nat["offset"], "shape": (len(cells), len(cells[0]) if cells else 0),
            "background": bg, "residue": nat["residue"], "placed": placed, "lexicon": lexicon,
            "native": cells, "locked": nat["locked"]}


def display_xy(code, r, c):
    """Screen pixel (x, y) at the centre of native cell (r, c): what ACTION6 expects."""
    k = code["scale"]
    oy, ox = code["offset"]
    return ox + c * k + k // 2, oy + r * k + k // 2


def to_text(code, legend=True, max_list=12, max_kinds=40):
    lex, cc = code["lexicon"], code["lexicon"].color_chars
    k, (oy, ox), (h, w) = code["scale"], code["offset"], code["shape"]
    bg = code["background"]
    groups = {}
    for p in code["placed"]:
        groups.setdefault(p["sym"], []).append(p["at"])
    if lex.naming == "stable":
        order = sorted(groups, key=lambda s: min(groups[s]))      # reading order of first instance
    else:
        order = sorted(groups, key=lambda s: (len(s), s))          # A, B, C ...
    hidden = len(order) - max_kinds
    order = order[:max_kinds]
    lines = [
        f"SCREEN 64x64 = world {h}x{w} cells, each cell {k}x{k} px"
        f"{' (locked for this level)' if code.get('locked') else ''}, first cell at px (row {oy}, col {ox}); "
        f"background {cc[bg]} ({COLOR_NAMES[bg]}); {code['residue']} px drawn outside the cell grid (HUD).",
        f"OBJECTS {len(code['placed'])} in {len(groups)} kinds (symbol xcount @ (row, col) of top-left cell):",
    ]
    floors = {p["sym"] for p in code["placed"] if p.get("container")}
    for sym in order:
        ats = groups[sym]
        shown = " ".join(f"({r},{c})" for r, c in ats[:max_list])
        more = f" ... +{len(ats) - max_list}" if len(ats) > max_list else ""
        tag = " (floor: other objects sit on it)" if sym in floors else ""
        lines.append(f"  {sym} x{len(ats)} @ {shown}{more}{tag}")
    if legend:
        lines.append("LEGEND (. = not part of the object):")
        for sym in order:
            e = lex.entries[sym]
            pic = lex.picture(sym)
            rel = f" = {e['family']} {e['pose']}" if e["pose"] else ""
            cols = ",".join(cc[v] for v in sorted({v for row in e["patch"] for v in row if v >= 0}))
            if len(pic) <= 6 and len(pic[0]) <= 12:
                lines.append(f"  {sym} {len(pic)}x{len(pic[0])} [{cols}]{rel}: " + "/".join(pic))
            else:
                lines.append(f"  {sym} {len(pic)}x{len(pic[0])} [{cols}]{rel}: (large; see .native)")
    if hidden > 0:
        lines.append(f"... {hidden} more kinds not listed")
    return "\n".join(lines)


def native_ascii(code):
    cc = code["lexicon"].color_chars
    return "\n".join("".join(cc[v] for v in row) for row in code["native"])


def diff(before, after, place_first=False):
    """Object-level change between two encodings that share one Lexicon.

    Unchanged objects are matched first; then (`place_first`) an object that keeps its place
    and shape but changes colour is read as recoloured; then instances of the same symbol are
    paired greedily by distance as moves. The place rule is off by default: graded against the
    engine on 45 recorded plays it read real moves as recolours (tiles moving along a ring, an
    avatar crossing a tiled floor) far more often than it fixed a recolour read as moves
    (kaggle/lexicon-dfranzen/eval_perception_v4_*.out). place_first="shared" applies the place rule only
    between pictures that have several copies on screen: a unique object that leaves a place
    is still read as moving (an avatar crossing a tiled floor). Returns a dict with moved /
    became / appeared / disappeared lists and a one-line-per-change text.
    """
    def by_sym(code):
        d = {}
        for p in code["placed"]:
            d.setdefault(p["sym"], []).append(p["at"])
        return d

    a, b = by_sym(before), by_sym(after)
    for sym in set(a) | set(b):                       # unchanged objects first
        for p in [p for p in a.get(sym, []) if p in b.get(sym, [])]:
            a[sym].remove(p)
            b[sym].remove(p)
    # identity by place: an object whose place and shape stay while its colours change is the same
    # object recoloured, not one object leaving and a look-alike arriving (checked before moves)
    lex = before.get("lexicon")
    became = []
    if lex is not None and place_first:
        def mask(sym):
            return tuple(tuple(v >= 0 for v in row) for row in lex.entries[sym]["patch"])
        copies0, copies1 = {}, {}
        for q in before["placed"]:
            copies0[q["sym"]] = copies0.get(q["sym"], 0) + 1
        for q in after["placed"]:
            copies1[q["sym"]] = copies1.get(q["sym"], 0) + 1
        at_after = {}
        for sym, spots in b.items():
            for p in spots:
                at_after.setdefault(p, []).append(sym)
        for s0 in sorted(a, key=lambda s: (len(s), s)):
            for p in list(a[s0]):
                for s1 in at_after.get(p, []):
                    if place_first == "shared" and (copies0.get(s0, 0) < 2 or copies1.get(s1, 0) < 2):
                        continue
                    if s1 != s0 and p in b.get(s1, []) and mask(s1) == mask(s0):
                        became.append((s0, s1, p))
                        a[s0].remove(p)
                        b[s1].remove(p)
                        break
    moved, appeared, disappeared = [], [], []
    for sym in sorted(set(a) | set(b), key=lambda s: (len(s), s)):
        left, right = list(a.get(sym, [])), list(b.get(sym, []))
        pairs = sorted(((abs(l[0] - r[0]) + abs(l[1] - r[1]), l, r) for l in left for r in right))
        used_l, used_r = set(), set()
        for _, l, r in pairs:
            if l in used_l or r in used_r:
                continue
            used_l.add(l)
            used_r.add(r)
            moved.append((sym, l, r))
        disappeared += [(sym, l) for l in left if l not in used_l]
        appeared += [(sym, r) for r in right if r not in used_r]
    # one symbol replaced by another at the same place: a recolour or a state change, not two events
    for s0, at in list(disappeared):
        for s1, at1 in appeared:
            if at1 == at:
                became.append((s0, s1, at))
                disappeared.remove((s0, at))
                appeared.remove((s1, at1))
                break
    text = [f"{s} moved {l}->{r} (d_row {r[0] - l[0]:+d}, d_col {r[1] - l[1]:+d})" for s, l, r in moved]
    text += [f"{s0} became {s1} @ {at}" for s0, s1, at in became]
    text += [f"{s} appeared @ {r}" for s, r in appeared]
    text += [f"{s} disappeared from {l}" for s, l in disappeared]
    return {"moved": moved, "became": became, "appeared": appeared, "disappeared": disappeared,
            "text": "\n".join(text) if text else "no object changed"}


def read_roles(effects, min_count=3, min_share=0.7):
    """Plain-language reading of a symbol's effect table.

    effects: {action: Counter({effect: n})}, where effect is a (d_row, d_col) move
    or a string such as "became B". Returns short sentences, e.g.
    "moves (-5, 0) on ACTION1 (22 of 24 times)".
    """
    out = []
    for act in sorted(effects):
        c = effects[act]
        n = sum(c.values())
        if n < min_count:
            continue
        eff, k = c.most_common(1)[0]
        if k / n < min_share:
            out.append(f"reacts to {act} in varied ways ({n} times)")
        elif isinstance(eff, tuple):
            out.append(f"moves {eff} on {act} ({k} of {n} times)")
        else:
            out.append(f"{eff} on {act} ({k} of {n} times)")
    return out
