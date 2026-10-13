"""Frame-only parser for ``tu93`` (a 3px-cell maze).

Nothing here touches ``environment_files`` / engine internals: the whole model is
recovered from the 64x64 observation.

Geometry recovered from observation (see docs/M1整改_Step2B_*.md):
  * 64x64 frame, ``int8`` palette indices, 3 px per lattice cell.
  * The maze sits at a *per-level* phase inside the frame, so the lattice offset
    is re-derived every frame (the reset frame and post-transition frames can
    differ).  We pick the offset that maximises "cell purity" over non-void
    cells -- a wrong phase straddles corridor/connector boundaries and shows up
    as impure cells.
  * Terrain, by lattice-cell colour:
        corridor / node      : 0      (the player, target, gems and doors sit on
                                       these nodes too, drawn in their own colour)
        open connector       : 2      (a passable gap between two adjacent nodes)
        blocked / outside    : 5      (void -- also fills everything off-maze)
  * Movement is one *node* (= 2 lattice cells = 6 px) per action:
        A1 = up (row-2), A2 = down (row+2), A3 = left (col-2), A4 = right (col+2)
    and a move is legal iff the single connector cell between the two nodes is
    not void.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lingjing_solo.core.lattice import detect_cell_lattice

# --- palette roles ---------------------------------------------------------
VOID = 5
CONNECTOR = 2
CORRIDOR = 0
PLAYER = 9
HEADING = 4
TARGET = 14
DOORS = (12, 13)
GEMS = (11, 15)
HUD = 6
DEADLY = (8,)  # lethal tile -- stepping on it ends the run (observed on L1/L2)

OPEN_COLOURS = frozenset({CORRIDOR, PLAYER, HEADING, TARGET, *DOORS, *GEMS})

# action id <-> node delta (row, col); one node == 2 lattice cells
DELTA = {1: (-2, 0), 2: (2, 0), 3: (0, -2), 4: (0, 2)}
ACTION_OF = {v: k for k, v in DELTA.items()}
ARROWS = ("up", "down", "left", "right")


@dataclass
class Tu93Obs:
    ok: bool
    offset: tuple[int, int]
    cells: np.ndarray = field(repr=False)
    player: tuple[int, int] | None = None
    targets: list[tuple[int, int]] = field(default_factory=list)
    doors: list[tuple[int, int]] = field(default_factory=list)
    gems: list[tuple[int, int]] = field(default_factory=list)
    facing: dict = field(default_factory=dict)
    danger: set = field(default_factory=set)
    note: str = ""

    def open_nodes(self) -> set[tuple[int, int]]:
        rr, cc = np.where(np.isin(self.cells, list(OPEN_COLOURS)))
        return {(int(r), int(c)) for r, c in zip(rr, cc)}

    def render(self) -> str:
        glyph = {
            VOID: " ", CONNECTOR: "#", CORRIDOR: ".", PLAYER: "@", TARGET: "Z",
            12: "D", 13: "D", 11: "g", 15: "g", 8: "8", HUD: "H",
        }
        out = []
        for r in range(self.cells.shape[0]):
            row = []
            for c in range(self.cells.shape[1]):
                row.append(glyph.get(int(self.cells[r, c]), "?"))
            out.append(f"{r:3d} " + "".join(row))
        return "\n".join(out)


def _purity_and_cells(arr: np.ndarray, orow: int, ocol: int):
    n_r = (arr.shape[0] - orow) // 3
    n_c = (arr.shape[1] - ocol) // 3
    sub = arr[orow:orow + 3 * n_r, ocol:ocol + 3 * n_c]
    flat = sub.reshape(n_r, 3, n_c, 3).transpose(0, 2, 1, 3).reshape(n_r * n_c, 9)
    pure = (flat == flat[:, :1]).all(axis=1)
    nonvoid = ~(flat == VOID).all(axis=1)
    active = int(nonvoid.sum())
    score = float(pure[nonvoid].mean()) if active else -1.0
    # majority colour per cell via bincount (palette indices are small ints)
    cells = np.empty(flat.shape[0], dtype=np.int64)
    for i in range(flat.shape[0]):
        cells[i] = int(np.bincount(flat[i], minlength=16).argmax())
    return score, active, cells.reshape(n_r, n_c)


def _legacy_detect_lattice(arr: np.ndarray):
    """The original 3px-only phase scan (offset 0..2, purity over non-void cells).

    Superseded by :func:`lingjing_solo.core.lattice.detect_cell_lattice`, which
    also searches the cell size; kept as the fallback for frames the generic
    detector cannot read as a 3px lattice, and as the reference the generic
    result is regression-tested against.
    """
    best = None
    for orow in (0, 1, 2):
        for ocol in (0, 1, 2):
            score, active, cells = _purity_and_cells(arr, orow, ocol)
            if active == 0:
                continue
            key = (score, active)
            if best is None or key > best[0]:
                best = (key, (orow, ocol), cells)
    if best is None:
        return (0, 0), np.zeros((21, 21), dtype=np.int64)
    return best[1], best[2]


CELL_HINT = 3


def detect_lattice(arr: np.ndarray):
    """Return ``(offset, cells)`` for the frame's lattice.

    The cell size is no longer hardcoded: the geometry comes from
    ``core.lattice.detect_cell_lattice``, which searches cell sizes and phases
    from the pixels alone (tu93's 3px lattice is recovered as cell=3 at the
    per-level phase).  If the generic read does not land on a 3px lattice the
    legacy scan still runs, so this module's behaviour on a tu93 frame is
    unchanged -- verified frame-by-frame in ``state/_probe_lattice_regress.py``.
    """
    spec = detect_cell_lattice(arr, background=VOID, max_cell=8)
    if spec.cell == (CELL_HINT, CELL_HINT):
        return spec.origin, spec.cells
    return _legacy_detect_lattice(arr)


def _nearest_node(cells: np.ndarray, cr: float, cc: float, limit: int = 2,
                  open_colours=None):
    """Nearest open (node) cell to a pixel-centre, searching nearby cells.

    ``open_colours`` defaults to the module's constant; the fallback path passes
    a set with an *inferred* player colour swapped in.
    """
    opens = OPEN_COLOURS if open_colours is None else open_colours
    r0 = int(round((cr - 1) / 3.0))
    c0 = int(round((cc - 1) / 3.0))
    best = None
    for dr in range(-limit, limit + 1):
        for dc in range(-limit, limit + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < cells.shape[0] and 0 <= c < cells.shape[1] and int(cells[r, c]) in opens:
                d = (dr * dr + dc * dc)
                if best is None or d < best[0]:
                    best = (d, (r, c))
    return best[1] if best else None


def _blobs(arr: np.ndarray, colours):
    ys, xs = np.where(np.isin(arr, list(colours)))
    if len(ys) == 0:
        return []
    return [(float(ys.mean()), float(xs.mean()))]


def _label_centroids(arr: np.ndarray, colours, min_px: int = 3):
    """Rough connected components (4-neighbour) centroids for a colour set."""
    mask = np.isin(arr, list(colours))
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) >= min_px:
                out.append((float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))))
    return out


def _role_colour(roles, name, default=None):
    """Read one colour off a ``RoleSpec`` (or a plain mapping); ``None`` -> default."""
    if roles is None:
        return default
    value = roles.get(name) if isinstance(roles, dict) else getattr(roles, name, None)
    return default if value is None else int(value)


def roles_probe(frames, actions=None, levels=None, background=None):
    """Infer this game's palette roles from a trajectory -- *side evidence only*.

    Nothing in the planning path calls this: it exists so the generic inference
    (``core.roles``) can be checked against the constants above.  On tu93 it
    independently reports ``background=5 avatar=9 hud={6}``, i.e. exactly
    ``VOID``/``PLAYER``/``HUD``.
    """
    from lingjing_solo.core.roles import infer_roles

    return infer_roles(frames, actions=actions, levels=levels, background=background)


def parse_frame(frame, want_doors: bool = True) -> Tu93Obs:
    """Decode one frame with this module's own palette constants."""
    return _parse(frame, want_doors=want_doors, player_colour=PLAYER)


def parse_frame_with_roles(frame, roles, want_doors: bool = True) -> Tu93Obs:
    """``parse_frame`` with the **player** colour taken from role inference.

    Only the player is overridden.  It is the one role the generic inference
    recovers from a positive, checkable signal (a block that rigidly translates
    on the cell grid *and* answers the actions); the target/door/gem constants
    stay put, because a wrong guess there would aim the planner at a lethal cell.
    The fallback exists to *unblock*, not to speculate.

    Used only when ``LINGJING_ROLES_FALLBACK=1``; the default path never calls it.
    """
    return _parse(frame, want_doors=want_doors,
                  player_colour=_role_colour(roles, "avatar", PLAYER))


def _parse(frame, *, want_doors: bool = True, player_colour: int = PLAYER) -> Tu93Obs:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != (64, 64):
        return Tu93Obs(ok=False, offset=(0, 0), cells=np.zeros((1, 1), np.int64), note=f"shape {arr.shape}")
    arr = arr.astype(np.int64)

    offset, cells = detect_lattice(arr)

    # ``open`` is the module's traversable set with the player colour swapped --
    # identical to OPEN_COLOURS on the default path (player_colour == PLAYER).
    opens = (OPEN_COLOURS if player_colour == PLAYER
             else (OPEN_COLOURS - {PLAYER}) | {int(player_colour)})

    # player: the player colour's blob centre
    pb = _blobs(arr, (player_colour,))
    if not pb:
        return Tu93Obs(ok=False, offset=offset, cells=cells, note="no player")
    player = _nearest_node(cells, pb[0][0], pb[0][1], open_colours=opens)

    tgts = []
    for (cr, cc) in _label_centroids(arr, (TARGET,), min_px=4):
        n = _nearest_node(cells, cr, cc, open_colours=opens)
        if n:
            tgts.append(n)
    doors = []
    gems = []
    if want_doors:
        for (cr, cc) in _label_centroids(arr, DOORS, min_px=4):
            n = _nearest_node(cells, cr, cc, open_colours=opens)
            if n:
                doors.append(n)
        for (cr, cc) in _label_centroids(arr, GEMS, min_px=1):
            n = _nearest_node(cells, cr, cc, open_colours=opens)
            if n:
                gems.append(n)

    if player is None:
        return Tu93Obs(ok=False, offset=offset, cells=cells, note="player off-node")
    if not tgts:
        return Tu93Obs(ok=False, offset=offset, cells=cells, player=player, note="no target")

    facing = _facing_map(arr, cells, offset)
    danger = _danger_cells(facing)
    return Tu93Obs(ok=True, offset=offset, cells=cells, player=player,
                   targets=sorted(set(tgts)), doors=sorted(set(doors)), gems=sorted(set(gems)),
                   facing=facing, danger=danger)


DIR_OF = {"N": (-2, 0), "S": (2, 0), "W": (0, -2), "E": (0, 2)}


def transient_cells(frame, obs: "Tu93Obs") -> set:
    """Cells currently occupied by a patroller (gem-coloured, not a hazard tile).

    L3 has a colour-15 object patrolling a corridor: stepping into its cell is
    fatal, and the reference solution simply waits until it has moved on.  We
    expose its current cell so the planner avoids it and, when that leaves no
    route, stands still for a beat.
    """
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[0]
    orow, ocol = obs.offset
    tile_cells = set(obs.facing.keys())
    out = set()
    for col in GEMS:
        ys, xs = np.where(arr == col)
        for y, x in zip(ys, xs):
            cell = (int(y - orow) // 3, int(x - ocol) // 3)
            if any(abs(cell[0] - tr) <= 1 and abs(cell[1] - tc) <= 1 for tr, tc in tile_cells):
                continue
            out.add(cell)
    return out


def _danger_cells(facing: dict) -> set:
    """Node a live hazard is looking at (2 cells ahead) -- fatal to enter.

    Observed on L1 (hazard at (9,12) facing W makes (9,10) fatal) and L2
    (hazard at (8,10) facing S makes (10,10) fatal): a live hazard's line of
    sight is lethal, which is why the reference solutions walk *through* the
    hazard from a side it is not facing and thereby clear it.
    """
    out = set()
    for (er, ec), side in facing.items():
        dr, dc = DIR_OF.get(side, (0, 0))
        out.add((er + dr, ec + dc))
    return out


# side the player comes FROM, per action
ENTRY_SIDE = {1: "S", 2: "N", 3: "E", 4: "W"}


def _facing_from_missing(dr: int, dc: int) -> str:
    if dr == 1 and dc == 0:
        return "W"
    if dr == 1 and dc == 2:
        return "E"
    if dc == 1 and dr == 0:
        return "N"
    if dc == 1 and dr == 2:
        return "S"
    if dr == 0:
        return "N"
    if dr == 2:
        return "S"
    if dc == 0:
        return "W"
    return "E"


def _facing_map(arr: np.ndarray, cells: np.ndarray, offset) -> dict:
    """For every hazard tile (colour 8): which side it faces.

    The tile is a 3x3 block of colour 8 with one pixel of a different colour; the
    position of that pixel gives the facing.  Entering the tile from the side it
    faces is fatal (observed on L1/L3), so the solver only crosses from another
    side.
    """
    orow, ocol = offset
    out = {}
    rr, cc = np.where(cells == DEADLY[0])
    for r, c in zip(rr, cc):
        r0, c0 = orow + 3 * int(r), ocol + 3 * int(c)
        blk = arr[r0:r0 + 3, c0:c0 + 3]
        if blk.shape != (3, 3):
            continue
        miss = None
        for dr in range(3):
            for dc in range(3):
                if int(blk[dr, dc]) != DEADLY[0]:
                    miss = (dr, dc)
        out[(int(r), int(c))] = _facing_from_missing(*miss) if miss else "?"
    return out


# --- graph -----------------------------------------------------------------
def _hazard_owner(facing: dict) -> dict:
    """Map each lethal line-of-sight cell to *every* hazard that looks at it.

    Several hazards can share a sight-line -- on L5 both ``(14,10)`` (facing W)
    and ``(16,8)`` (facing N) look at ``(14,8)`` -- and the line is only safe
    once *all* of them have been cleared.  An earlier version stored a single
    owner per cell, so the last hazard processed silently overwrote the others
    and the solver walked into a still-watched cell (see
    ``state/_dbg_l5diff.py``).
    """
    owner: dict = {}
    for (er, ec), side in (facing or {}).items():
        dr, dc = DIR_OF.get(side, (0, 0))
        owner.setdefault((er + dr, ec + dc), set()).add((er, ec))
    return owner


def _steps(cells, node, facing, cleared, owner, blocked=None):
    """Yield (act, next_node, new_cleared) for legal one-action moves."""
    blocked = blocked or set()
    r, c = node
    for (dr, dc), act in ((( -2, 0), 1), ((2, 0), 2), ((0, -2), 3), ((0, 2), 4)):
        nr, nc = r + dr, c + dc
        mid_r, mid_c = r + dr // 2, c + dc // 2
        if not (0 <= nr < cells.shape[0] and 0 <= nc < cells.shape[1]):
            continue
        if not (0 <= mid_r < cells.shape[0] and 0 <= mid_c < cells.shape[1]):
            continue
        if int(cells[mid_r, mid_c]) == VOID:
            continue
        if (nr, nc) in blocked:
            continue                       # transiently occupied by a patroller
        watchers = owner.get((nr, nc))
        if watchers and not watchers <= cleared:
            continue                       # still in a *live* hazard's line of sight
        dest = int(cells[nr, nc])
        new_cleared = cleared
        if dest in DEADLY:
            if facing.get((nr, nc), "?") == ENTRY_SIDE[act]:
                continue                   # would walk into its face
            new_cleared = cleared | {(nr, nc)}   # stepping on it clears it
        elif dest not in OPEN_COLOURS:
            continue
        yield act, (nr, nc), new_cleared


def neighbours(cells: np.ndarray, node: tuple[int, int], facing: dict | None = None, danger: set | None = None):
    """Stateless view of the legal moves (kept for debugging/rendering)."""
    owner = _hazard_owner(facing or {})
    out = []
    for act, nxt, _ in _steps(cells, node, facing or {}, frozenset(), owner):
        out.append((act, nxt))
    return out


def bfs_first_action(cells, start, goals, facing: dict | None = None, danger: set | None = None,
                     blocked: set | None = None) -> int | None:
    path = bfs_path(cells, start, goals, facing, danger, blocked)
    return path[0] if path else None


def bfs_path(cells, start, goals, facing: dict | None = None, danger: set | None = None,
             blocked: set | None = None):
    """Action list from *start* to the nearest goal, modelling hazard clearing.

    State is (node, cleared_hazards): a hazard's line-of-sight cell only opens
    once that hazard has been stepped on from a side it is not facing.  Cells in
    *blocked* (e.g. currently occupied by a patrolling hazard) are avoided, so an
    empty result means "wait here" rather than "give up".
    """
    from collections import deque

    facing = facing or {}
    owner = _hazard_owner(facing)
    goals = set(goals)
    if start in goals:
        return []
    start_state = (start, frozenset())
    seen = {start_state}
    q = deque([(start_state, [])])
    while q:
        (node, cleared), path = q.popleft()
        for act, nxt, nc_ in _steps(cells, node, facing, cleared, owner, blocked):
            st = (nxt, nc_)
            if st in seen:
                continue
            p = path + [act]
            if nxt in goals:
                return p
            seen.add(st)
            q.append((st, p))
    return []


neighbour_path = bfs_path


# --- patrolling hazards (the colour-12/13 tiles) ----------------------------
# A patrolling tile is a 3x3 block drawn in colour 12 or 13 with one pixel of a
# different colour inside it.  That odd pixel is *not* centred: its offset inside
# the block gives the direction the tile is about to move (the same trick the
# static colour-8 hazard uses to publish its facing).  Measured on L3:
#   offset (2,1) = bottom middle -> moves down (S);  (0,1) = top middle -> up (N)
# and the tile advances exactly one node per *valid* player move (bumps and A5
# do not move it), bouncing off the ends of its straight corridor.
PATROL = (12, 13)
GRAPH_COLOURS = frozenset(set(OPEN_COLOURS) | set(DEADLY) | set(PATROL))

_OFFSET_DIR = {(0, 1): "N", (2, 1): "S", (1, 0): "W", (1, 2): "E"}


def _as_grid(arr) -> np.ndarray:
    """Normalise a raw frame (possibly ``(1, 64, 64)`` or a nested list)."""
    grid = np.asarray(arr)
    if grid.ndim == 3:
        grid = grid[0]
    return grid


def _patrol_tiles(arr: np.ndarray, cells: np.ndarray, offset):
    """``[(node, direction)]`` for every patrolling tile; direction may be ``?``."""
    orow, ocol = offset
    out = []
    rr, cc = np.where(np.isin(cells, list(PATROL)))
    for r, c in zip(rr, cc):
        r0, c0 = orow + 3 * int(r), ocol + 3 * int(c)
        blk = arr[r0:r0 + 3, c0:c0 + 3]
        if blk.shape != (3, 3):
            continue
        off = None
        for dr in range(3):
            for dc in range(3):
                if int(blk[dr, dc]) != int(cells[r, c]):
                    off = (dr, dc)
        out.append(((int(r), int(c)), _OFFSET_DIR.get(off, "?")))
    return out


def _straight(cells: np.ndarray, node, dr: int, dc: int):
    """Nodes reachable from *node* keeping a straight heading (excluding it)."""
    seq, cur = [], node
    while True:
        nr, nc = cur[0] + dr, cur[1] + dc
        mr, mc = cur[0] + dr // 2, cur[1] + dc // 2
        if not (0 <= nr < cells.shape[0] and 0 <= nc < cells.shape[1]):
            break
        if not (0 <= mr < cells.shape[0] and 0 <= mc < cells.shape[1]):
            break
        if int(cells[mr, mc]) == VOID:
            break
        if int(cells[nr, nc]) not in GRAPH_COLOURS:
            break
        seq.append((nr, nc))
        cur = (nr, nc)
    return seq


@dataclass
class Patrol:
    """A bouncing patrol along a straight corridor, one node per valid move."""

    seg: list = field(default_factory=list)
    start: int = 0

    @property
    def period(self) -> int:
        return max(1, 2 * (len(self.seg) - 1))

    def at(self, k: int):
        n = len(self.seg)
        if n == 1:
            return self.seg[0]
        span = 2 * (n - 1)
        off = (self.start + k) % span
        return self.seg[off if off < n else span - off]


def build_patrols(cells: np.ndarray, offset, arr: np.ndarray):
    """Return ``(patrols, observed_nodes)``, or ``None`` if a heading is unreadable.

    Build this on the *first* frame of a level (which is ``v = 0``: no valid
    move has been made yet), so ``patrol.at(k)`` is the tile's node after exactly
    ``k`` valid moves.  ``observed_nodes`` is what that frame currently shows.
    """
    tiles = _patrol_tiles(_as_grid(arr), cells, offset)
    patrols, seen = [], []
    for node, d in tiles:
        if d == "?":
            return None
        dr, dc = DIR_OF[d]
        seg = list(reversed(_straight(cells, node, -dr, -dc))) + [node] + _straight(cells, node, dr, dc)
        patrols.append(Patrol(seg, seg.index(node)))
        seen.append(node)
    return patrols, seen


def observed_patrol_nodes(frame, cells: np.ndarray, offset):
    """Cells currently showing a patrol tile, in :func:`build_patrols` order."""
    return [n for n, _ in _patrol_tiles(_as_grid(frame), cells, offset)]


def patrol_next_cells(frame, cells: np.ndarray, offset) -> set:
    """Nodes the patrollers are *about to enter*, per the heading they publish.

    A patrol advances exactly one node per valid player move and its next
    direction is drawn in the frame (the odd pixel's offset), so we can always
    see where it will be after our move even when its longer path -- which may
    turn corners (L6) -- is not a simple ``k -> node`` function.
    """
    out = set()
    for node, d in _patrol_tiles(_as_grid(frame), cells, offset):
        dr, dc = DIR_OF.get(d, (0, 0))
        out.add((node[0] + dr, node[1] + dc))
    return out


def plan_path(cells: np.ndarray, start, goals, facing: dict | None = None,
              patrols=(), cleared: frozenset = frozenset(), hazard_cost: int = 100):
    """Shortest arrow sequence that reaches a goal without meeting a patroller.

    State is ``(node, k mod P, cleared)`` where *k* counts valid moves: the
    patroller's node is a deterministic function of *k*, so the search simply
    refuses to stand on it at step ``k``.  ``P`` is the joint patrol period.
    Stepping *onto* a hazard tile costs ``hazard_cost`` extra: it is the one
    genuinely uncertain move (it is only survivable from a side the tile is not
    facing), so a route that avoids the tiles wins whenever one exists.  Returns
    ``None`` when no plan exists.
    """
    import heapq
    from math import gcd

    facing = facing or {}
    owner = _hazard_owner(facing)
    goals = set(goals)
    patrols = list(patrols or ())
    P = 1
    for p in patrols:
        P = P * p.period // gcd(P, p.period)
    if start in goals:
        return []
    start_state = (start, 0, cleared)
    parent = {start_state: None}
    best = {start_state: 0}
    heap = [(0, 0, start_state)]
    n = 0
    while heap:
        cost, _ord, st = heapq.heappop(heap)
        if best.get(st) != cost:
            continue
        node, k, cl = st
        if node in goals and cost > 0:
            path, cur = [], st
            while parent[cur] is not None:
                pst, a = parent[cur]
                path.append(a)
                cur = pst
            return list(reversed(path))
        nk = k + 1
        blocked = {p.at(nk) for p in patrols}
        for act, nxt, nc_ in _steps(cells, node, facing, cl, owner, blocked):
            extra = hazard_cost if int(cells[nxt]) in DEADLY else 0
            ncost = cost + 1 + extra
            ns = (nxt, nk % P, nc_)
            if ncost < best.get(ns, 1 << 60):
                best[ns] = ncost
                parent[ns] = (st, act)
                n += 1
                heapq.heappush(heap, (ncost, n, ns))
    return None

