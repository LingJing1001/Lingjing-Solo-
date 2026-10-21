"""Find the goal from the picture, before touching the planner.

Why this module exists
----------------------
`graphsim` and `worldsim` both optimise "a board I have not stood on". That is the
honest objective when nothing is known about the goal, and it is also why the agent
can drain 76 boards and 273 edges on ka59 and still be at zero levels: *being
somewhere new* is not *winning*. Winning is reaching a state the game drew for you
and you ignored.

The games do draw it. On ar25 the target is five cells of colour 11 shaped like the
player's own pentomino, sitting at (15,17)..(19,17); the engine's `fswikrcrdmx`
agrees cell for cell. Colour 11 is otherwise used only for the one-pixel board
border. So the goal *is* visible -- you just have to look at connected components
rather than at colours, and you have to know what a target looks like:

    rare         a marker, not terrain        (< 20% of the frame)
    static       it does not move as you play (< 25% of its cells ever change)
    interior     it does not touch the frame  (border lines, walls)
    not a divider it does not span a full row or column (axes, separators)
    not the floor not the modal colour (that is the background)
    shape-linked best evidence: it repeats the shape of something that moves --
                 the "ghost you must stand on" pattern, which is common in this
                 family of games and is exactly what ar25 L0 is

Everything here is measured on the pixel frame, so it works before any perception
naming, any rule learning, and without a solver.

The second half of the module is what makes the goal *useful*: coverage.

    score(grid) = |goal cells now covered by something non-background| / |goal cells|

Because the goal mask is frozen from the level's first frame -- the one frame where
nothing is covered -- coverage is monotone-ish in the right direction and does not
care what colour the coverer is. A rise is progress; a fall means the last move was
wrong. That is a reward signal the agent can hill-climb on, and it exists on every
game that marks its target.
"""
from __future__ import annotations

import collections


def _components(mask, h, w):
    """4-connected components of a boolean matrix -> list of cell sets."""
    seen = [[False] * w for _ in range(h)]
    out = []
    for y0 in range(h):
        row = mask[y0]
        for x0 in range(w):
            if not row[x0] or seen[y0][x0]:
                continue
            stack = [(y0, x0)]
            seen[y0][x0] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            out.append(frozenset(cells))
    return out


def _bbox(cells):
    ys = [c[0] for c in cells]
    xs = [c[1] for c in cells]
    return min(ys), min(xs), max(ys), max(xs)


def _shape_key(cells):
    """Translation-invariant pattern of a component, as a set of offsets."""
    y0, x0, _y1, _x1 = _bbox(cells)
    return frozenset((y - y0, x - x0) for y, x in cells)


class Goal:
    """One hypothesis about what the game wants, plus the frozen mask."""

    __slots__ = ("colour", "cells", "shape", "why", "ref")

    def __init__(self, colour, cells, why, ref=None):
        self.colour = colour
        self.cells = frozenset(cells)
        self.shape = _shape_key(self.cells) if self.cells else frozenset()
        self.why = why
        # cell -> the colour it had at level start. Coverage is "this cell is no
        # longer what it was", which is the one test that works both for a target
        # you must paint over (ar25's ghost) and for a hole you must fill (ka59's
        # rings, whose holes are the same colour as the floor).
        self.ref = dict(ref) if ref else {}

    @property
    def bbox(self):
        return _bbox(self.cells) if self.cells else (0, 0, 0, 0)

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        return f"Goal(colour={self.colour}, cells={len(self.cells)}, {self.why})"


class GoalModel:
    """Watch frames, name the target, score coverage.

    Usage:
        g = GoalModel()
        g.reset_level(first_grid)          # level start: the mask is taken here
        g.observe(grid, action="ACTION1")  # every later frame
        g.score(grid)                      # 0..1, fraction of target covered
        g.goal                             # Goal or None
    """

    def __init__(self, max_share: float = 0.20, static_max: float = 0.25,
                 min_cells: int = 2, warmup: int = 0):
        self.max_share = max_share
        self.static_max = static_max
        self.min_cells = min_cells
        self.warmup = warmup
        self.reset_level(None)

    # ------------------------------------------------------------------ lifecycle

    def reset_level(self, grid):
        self.grid0 = None
        self.prev = None
        self.changed = None          # cells that changed at least once
        self.movable = set()         # colours seen in a diff
        self.goal = None
        self.goal_mask = None
        self.frames = 0
        self.trace = []              # coverage per observe(), oldest first
        self.best = 0.0
        self.last_gain = 0.0
        if grid is not None:
            self.observe(grid)

    # ------------------------------------------------------------------ observation

    def observe(self, grid, action=None):
        if grid is None:
            return None
        h = len(grid)
        w = len(grid[0]) if h else 0
        if not h or not w:
            return None
        if self.grid0 is None:
            self.grid0 = grid
            self.changed = [[False] * w for _ in range(h)]
            self.prev = grid
            self.frames = 1
            self._choose(h, w)
            return self.score(grid)
        if self.prev is not None and len(self.prev) == h and len(self.prev[0]) == w:
            ch = self.changed
            for y in range(h):
                rp, rc = self.prev[y], grid[y]
                cr = ch[y]
                for x in range(w):
                    if rp[x] != rc[x]:
                        cr[x] = True
                        self.movable.add(int(rp[x]))
                        self.movable.add(int(rc[x]))
        self.prev = grid
        self.frames += 1
        # Re-pick for the first handful of frames: staticness is evidence, not a guess,
        # and one frame is not evidence. Re-picking is safe because the mask is always
        # rebuilt from grid0 -- which never changes -- so it cannot drift. Stopping at
        # frame 3 was too early: a piece that happens to sit still for two moves looked
        # like a target and was then frozen in place.
        if self.goal is None or self.frames <= 8:
            self._choose(h, w)
        s = self.score(grid)
        self.trace.append(s)
        self.last_gain = s - (self.trace[-2] if len(self.trace) > 1 else 0.0)
        self.best = max(self.best, s)
        return s

    # ------------------------------------------------------------------ choosing

    def _choose(self, h, w):
        """Rebuild the candidate list from grid0 and pick the best target."""
        g0 = self.grid0
        bg = collections.Counter()
        for row in g0:
            for v in row:
                bg[int(v)] += 1
        background = bg.most_common(1)[0][0]
        self.background = background
        total = h * w

        by_colour = collections.defaultdict(set)
        for y in range(h):
            for x in range(w):
                by_colour[int(g0[y][x])].add((y, x))

        cands = []
        for colour, cells_all in by_colour.items():
            if colour == background:
                continue
            if len(cells_all) > self.max_share * total:
                continue                      # terrain, not a marker
            mask = [[False] * w for _ in range(h)]
            for y, x in cells_all:
                mask[y][x] = True
            for comp in _components(mask, h, w):
                if len(comp) < self.min_cells:
                    continue
                y0, x0, y1, x1 = _bbox(comp)
                if y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1:
                    continue                  # touches the frame: border / HUD
                if (y1 - y0 + 1) == h or (x1 - x0 + 1) == w:
                    continue                  # full-span: axis / divider
                stat = sum(1 for y, x in comp if not self.changed[y][x]) / len(comp)
                if 1.0 - stat > self.static_max:
                    continue                  # it moves: a piece, not a target
                cands.append((colour, comp, stat))

        if not cands:
            self.candidates = []
            self.goal = None
            self.goal_mask = None
            return

        # Prefer components whose shape repeats the shape of something that moves:
        # the ghost/outline pattern. Then prefer small, then very static.
        mover_shapes = set()
        for colour in self.movable:
            if colour == background:
                continue
            cells_all = by_colour.get(colour, set())
            if len(cells_all) > self.max_share * total:
                continue
            mask = [[False] * w for _ in range(h)]
            for y, x in cells_all:
                mask[y][x] = True
            for comp in _components(mask, h, w):
                if len(comp) >= self.min_cells:
                    mover_shapes.add(_shape_key(comp))

        def rank(item):
            colour, comp, stat = item
            same = 1 if _shape_key(comp) in mover_shapes else 0
            return (same, stat, -len(comp))

        cands.sort(key=rank, reverse=True)
        self.candidates = [Goal(c, cells, "same-shape-as-mover" if
                                _shape_key(cells) in mover_shapes else "static-interior")
                           for c, cells, _s in cands]
        # All components of the winning colour join the goal: a game that asks you to
        # collect three gems draws three, and all of them must be covered.
        win = self.candidates[0]
        same_colour = [cells for c, cells, _s in cands if c == win.colour]
        merged = frozenset().union(*same_colour)

        # A hollow shape marks its *hole*, not its outline. On ka59 the target is a
        # 3x3 ring of colour 14 with a single floor cell inside; painting the ring
        # would be a very different game from filling it. Holes are the interior
        # cells of the bbox that cannot be reached from the bbox border without
        # crossing the shape -- the standard "enclosed region" reading of a picture.
        holes = set()
        for comp in same_colour:
            holes |= self._holes(comp, h, w)
        if holes:
            merged = frozenset(holes)
            why = "hollow-target-hole"
        else:
            why = win.why

        ref = {}
        for y, x in merged:
            if 0 <= y < h and 0 <= x < w:
                ref[(y, x)] = int(g0[y][x])
        self.goal = Goal(win.colour, merged, why, ref)
        self.goal_mask = merged

    @staticmethod
    def _holes(comp, h, w):
        """Interior cells of `comp`'s bbox that the shape encloses."""
        y0, x0, y1, x1 = _bbox(comp)
        if (y1 - y0) < 2 or (x1 - x0) < 2:
            return set()
        inner = {(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                 if (y, x) not in comp}
        if not inner:
            return set()
        # flood the "outside" part of the bbox: anything the flood cannot reach is
        # enclosed by the shape
        border = {(y, x) for (y, x) in inner
                  if y in (y0, y1) or x in (x0, x1)}
        stack = list(border)
        out = set(border)
        while stack:
            cy, cx = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (cy + dy, cx + dx)
                if p in inner and p not in out:
                    out.add(p)
                    stack.append(p)
        return inner - out

    # ------------------------------------------------------------------ scoring

    def score(self, grid):
        """Fraction of the target cells that are no longer what they were at start."""
        if not self.goal_mask or grid is None:
            return 0.0
        h = len(grid)
        w = len(grid[0]) if h else 0
        ref = self.goal.ref
        goal_col = self.goal.colour
        hit = 0
        for y, x in self.goal_mask:
            if y >= h or x >= w:
                continue
            v = int(grid[y][x])
            want = ref.get((y, x), goal_col)
            if v != want:
                hit += 1
        return hit / len(self.goal_mask)

    def uncovered(self, grid):
        """Target cells that still look like they did at level start."""
        if not self.goal_mask or grid is None:
            return set()
        h = len(grid)
        w = len(grid[0]) if h else 0
        ref = self.goal.ref
        goal_col = self.goal.colour
        out = set()
        for y, x in self.goal_mask:
            if y >= h or x >= w:
                continue
            want = ref.get((y, x), goal_col)
            if int(grid[y][x]) == want:
                out.add((y, x))
        return out

    # ------------------------------------------------------------------ reporting

    def text(self):
        if self.goal is None:
            return "goal: (none found: no rare static interior group in the first frame)"
        y0, x0, y1, x1 = self.goal.bbox
        return (f"goal: colour {self.goal.colour}, {len(self.goal)} cells, "
                f"bbox y{y0}..{y1} x{x0}..{x1}, why={self.goal.why}, "
                f"covered {self.best:.0%}")

    def __bool__(self):
        return self.goal is not None


# ---------------------------------------------------------------------- policy


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _mirror_h(shape):
    return frozenset((r, -c) for r, c in shape)


def _mirror_v(shape):
    return frozenset((-r, c) for r, c in shape)


def _norm(shape):
    mr = min(r for r, _ in shape)
    mc = min(c for _, c in shape)
    return frozenset((r - mr, c - mc) for r, c in shape)


class Piece:
    """A rigid, small, non-floor component: the kind of thing a key can move."""

    __slots__ = ("colour", "cells", "shape", "at", "size")

    def __init__(self, colour, cells):
        self.colour = int(colour)
        self.cells = frozenset(cells)
        self.at = _bbox(cells)
        self.shape = _shape_key(self.cells)
        self.size = len(self.cells)

    def shifted(self, d):
        return frozenset((y + d[0], x + d[1]) for y, x in self.cells)

    def __repr__(self):
        return f"<p col={self.colour} n={self.size} at={self.at[:2]}>"


class GoalPolicy:
    """Move the piece that the target is shaped like, using learned key semantics.

    Coverage alone is a flat signal: on ar25 the score stays 0 for the eleven moves
    that walk the piece toward the ghost and only jumps at the end, so a hill-climb
    on coverage cannot start. What *is* informative from step one is the distance
    between the piece you can move and the mark you must reach.

    Three things, none of which assume a particular game:

    1. **Which piece matters.** The target is usually drawn as a copy of the thing
       that must reach it -- a ghost of the same shape, possibly mirrored. So pieces
       are matched to `goal.shape` (and its mirror images); that match is the
       cursor. If nothing matches, every small non-floor piece is a cursor.
    2. **What a key does.** After pressing `a`, match each piece to its partner in
       the new frame by (colour, shape) and read off the translation. A key's
       translation is a global fact -- "ACTION2 is down one cell" -- so one
       observation on one board transfers to every board. A key that leaves the
       piece where it is is recorded as *blocked*, not as a zero displacement;
       treating "nothing happened" as "distance unchanged" is what made the first
       version press one key for sixty steps.
    3. **Which key to press.** For each key with known semantics, where does the
       piece land, and how far is that from the nearest uncovered target cell.
       Shorter is better, and a landing outside the frame is blocked.

    Everything works in frame pixels, so it is scale-agnostic: on ar25 one board
    cell is 3x3 pixels and the distance still falls by 3 per good move.
    """

    def __init__(self, model: GoalModel, max_piece: int = 400, min_piece: int = 2):
        self.m = model
        self.max_piece = max_piece
        self.min_piece = min_piece
        self.disp = {}            # action -> (dy, dx)
        self.blocked = set()      # actions that did nothing on the last board
        self.votes = {}           # action -> {delta: count}
        self._last = []           # pieces of the previous frame
        self.cursors = []         # pieces of the current frame
        self.linked_now = False
        self.linked_colour = None  # colour of the piece the target is shaped like
        self.last_delta = None
        self._goal_sig = None      # identifies the current goal hypothesis

    # ------------------------------------------------------------------ pieces

    def pieces(self, grid):
        if grid is None or not grid:
            return []
        h, w = len(grid), len(grid[0])
        bg = getattr(self.m, "background", None)
        seen = [[False] * w for _ in range(h)]
        out = []
        for y0 in range(h):
            row = grid[y0]
            for x0 in range(w):
                if seen[y0][x0]:
                    continue
                col = int(row[x0])
                if col == bg:
                    continue
                stack = [(y0, x0)]
                seen[y0][x0] = True
                cells = []
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and not seen[ny][nx]
                                and int(grid[ny][nx]) == col):
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                if self.min_piece <= len(cells) <= self.max_piece:
                    out.append(Piece(col, cells))
        return out

    def cursors_of(self, grid):
        """Pieces shaped like the target; all movable pieces if none match.

        The colour that matched once is remembered. On ar25 the piece is *absorbed*
        one pixel at a time as it lands on the ghost, so its size goes 45 -> 44 -> 43
        and its shape stops matching the target exactly; re-deriving the cursor from
        shape every frame loses the piece at exactly the moment it matters.
        """
        ps = self.pieces(grid)
        goal = self.m.goal
        if goal is not None and goal.cells:
            # the target itself is not a piece: on ar25 the ghost is drawn with the
            # very shape the piece has, and picking it as the cursor means tracking
            # something that never moves.
            gc = goal.cells
            ps = [p for p in ps if not p.cells <= gc]
        if goal is not None and goal.shape:
            want = _norm(goal.shape)
            alt = {want, _norm(_mirror_h(want)), _norm(_mirror_v(want)),
                   _norm(_mirror_h(_mirror_v(want)))}
            hit = [p for p in ps if _norm(p.shape) in alt]
            if hit:
                self.linked_now = True
                self.linked_colour = hit[0].colour
                return hit
        if self.linked_colour is not None:
            hit = [p for p in ps if p.colour == self.linked_colour]
            if hit:
                self.linked_now = True
                return hit
        self.linked_now = False
        return ps

    # ------------------------------------------------------------------ learning

    def _sig(self):
        g = self.m.goal
        return None if g is None else (g.colour, len(g.cells), g.why)

    def observe(self, grid, action=None):
        cur = self.cursors_of(grid)
        # When the goal hypothesis changes, the cursor set changes with it, so the
        # previous frame's cursors are not comparable. One step is spent re-syncing
        # instead of learning a displacement between two different things.
        sig = self._sig()
        if sig != self._goal_sig:
            self._goal_sig = sig
            self._last = cur
            self.cursors = cur
            return cur
        if action and self._last:
            a = action.split("@")[0]
            before = {p.colour: p for p in self._last}
            after = {p.colour: p for p in cur}
            votes = collections.Counter()
            for col, p in before.items():
                q = after.get(col)
                if q is None:
                    continue
                if self.linked_colour is not None and col != self.linked_colour:
                    continue
                # tolerate a few pixels being absorbed: a size change of a cell or
                # two is the game eating the piece, not a different piece
                if abs(p.size - q.size) > max(3, p.size // 5):
                    continue
                votes[(q.at[0] - p.at[0], q.at[1] - p.at[1])] += 1
            if votes:
                (d, n), = votes.most_common(1)
                self.votes.setdefault(a, collections.Counter())[d] += n
                if d == (0, 0):
                    # the piece did not move: a wall, a blocked key, or a non-moving
                    # key. Never store this as "the key translates by zero".
                    self.blocked.add(a)
                else:
                    self.blocked.discard(a)
                    self.disp[a] = d
                self.last_delta = (a, d)
        self._last = cur
        self.cursors = cur
        return cur

    # ------------------------------------------------------------------ ranking

    def dist(self, pieces, targets):
        if not pieces or not targets:
            return None
        best = None
        ty = [t[0] for t in targets]
        tx = [t[1] for t in targets]
        for p in pieces:
            for y, x in p.cells:
                d = max(min(abs(y - a) for a in ty), min(abs(x - b) for b in tx))
                if best is None or d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    def rank(self, grid, actions, bounds=None):
        """{action: (overlap_after, distance_after, known)}.

        `overlap` is the number of target cells the piece would sit on if the key
        were pressed -- the only thing that actually wins. `distance` is the
        tie-breaker and the approach signal: on ar25 overlap stays 0 for eleven moves
        while distance falls from 28 to 1, so ranking on overlap alone would never
        start, and ranking on distance alone stops at "touching" instead of
        "aligned" (it reached d=0 at 18% coverage and then had nothing to prefer).
        """
        targets = self.m.uncovered(grid)
        if not targets or not self.cursors:
            return {}
        goal_cells = self.m.goal_mask or frozenset()
        h, w = (len(grid), len(grid[0])) if grid else (0, 0)
        if bounds:
            h, w = bounds
        out = {}
        for a in actions:
            a = a.split("@")[0]
            d = self.disp.get(a)
            if d is None:
                out[a] = (None, None, False)
                continue
            moved = []
            off = False
            for p in self.cursors:
                s = p.shifted(d)
                for y, x in s:
                    if y < 0 or x < 0 or y >= h or x >= w:
                        off = True
                        break
                if off:
                    break
                moved.append(Piece(p.colour, s))
            if off:
                out[a] = (0, float("inf"), True)     # walks off the board: blocked
                continue
            cov = max(len(p.cells & goal_cells) for p in moved)
            out[a] = (cov, self.dist(moved, targets), True)
        return out

    def best(self, grid, actions, bounds=None):
        """The key that covers most of the target; ties go to the shorter distance."""
        r = self.rank(grid, actions, bounds)
        known = [(-cov, d, a) for a, (cov, d, ok) in r.items()
                 if ok and cov is not None]
        if not known:
            return None, r
        known.sort()
        return known[0][2], r

    def text(self):
        parts = [f"{a.split('ACTION')[-1]}{d}" for a, d in sorted(self.disp.items())]
        tag = "linked" if self.linked_now else "generic"
        return (f"policy[{tag}]: {len(self.cursors)} cursors, "
                + (" ".join(parts) if parts else "no key semantics yet"))


def decompose(vec, disp, limit: int = 64):
    """Keys whose translations sum to `vec`.

    `disp` maps action -> (dy, dx). Greedy is enough and is also the honest choice:
    the key set is a handful of axis moves, so any order that cancels the vector
    works, and greedy re-derives the order from the *current* frame every step, which
    is what makes a blocked key harmless.
    """
    dy, dx = int(vec[0]), int(vec[1])
    out = []
    keys = list(disp.items())
    for _ in range(limit):
        if dy == 0 and dx == 0:
            break
        best = None
        for a, (ay, ax) in keys:
            if not ay and not ax:
                continue
            gy = abs(ay) if ay and (ay > 0) == (dy > 0) and dy else 0
            gx = abs(ax) if ax and (ax > 0) == (dx > 0) and dx else 0
            gain = gy + gx
            if gain and (best is None or gain > best[0]):
                best = (gain, a, ay, ax)
        if best is None:
            break
        _g, a, ay, ax = best
        if ay and (ay > 0) == (dy > 0):
            n = max(1, min(abs(dy) // abs(ay), 32))
            out.extend([a] * n)
            dy -= ay * n
        elif ax and (ax > 0) == (dx > 0):
            n = max(1, min(abs(dx) // abs(ax), 32))
            out.extend([a] * n)
            dx -= ax * n
        else:
            break
    return out


# ---------------------------------------------------------------------- notes
# 把一层内部状态翻译成一句人话。这不是日志美化：下一关开局要读它，所以
# 它必须是**可执行的事实**（键语义）和**可诊断的结论**（卡在哪）两部分。


def dir_name(d):
    """像素位移 -> 人话。不假设一格几像素，所以只说方向 + 距离。"""
    dy, dx = int(d[0]), int(d[1])
    if not dy and not dx:
        return "不动"
    parts = []
    if dy:
        parts.append("下" if dy > 0 else "上")
    if dx:
        parts.append("右" if dx > 0 else "左")
    return "".join(parts)


def goal_text(goal):
    if goal is None or not goal.cells:
        return "没认出目标"
    y0, x0, y1, x1 = goal.bbox
    return (f"颜色 {goal.colour}，{len(goal.cells)} 格，"
            f"占 {y1 - y0 + 1}×{x1 - x0 + 1} 框，左上角 ({y0},{x0})")


def keys_text(disp, blocked):
    if not disp and not blocked:
        return "还没学到"
    parts = []
    for a, d in sorted(disp.items()):
        parts.append(f"{a.replace('ACTION', '')}号={dir_name(d)}{max(abs(d[0]), abs(d[1]))}")
    for a in sorted(blocked):
        if a not in disp:
            parts.append(f"{a.replace('ACTION', '')}号=被挡")
    return "｜".join(parts)


def keys_kind(disp):
    """给程序读的键语义。json 友好，跨进程复用。"""
    return {str(a): [int(d[0]), int(d[1])] for a, d in disp.items()}


def diagnose(goal, cursors, blocked, cov, linked):
    """一句话说清卡在哪。顺序即优先级：越前面的问题越根本。"""
    if goal is None or not goal.cells:
        return "没认出目标，后面全部无从谈起"
    if not cursors:
        return "认出了目标，但找不到可动的方块"
    if cov >= 0.99:
        return "目标已被完全覆盖，但引擎没判过关"
    if len(blocked) >= 2:
        return f"{len(blocked)} 个方向被挡住，可能是墙或边界"
    if cov > 0:
        return f"只覆盖 {cov:.0%}，形状可能对不上"
    if not linked:
        return "没有方块的形状跟目标一致，这可能不是'推过去'型的关"
    return "一直没靠近目标，路线可能算错"
