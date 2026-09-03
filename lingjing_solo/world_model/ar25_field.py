"""AR25 Layer-1 primitives: reflection closure, cover check, config search.

Search the configuration space (piece / axis poses), not the action tree.
Bounce depth matches the official engine (`ythhvclqmk = 12`).
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Optional

from ..core.types import Ar25Axis, Ar25Config, Ar25CoverReport, Ar25Obs, Ar25Piece

BOUNCE_LIMIT = 12
TRANSPARENT = -1


@dataclass
class Ar25Field:
    """World-model field helpers for the kaleidoscope mirror puzzle."""

    bounce_limit: int = BOUNCE_LIMIT
    _cache: dict = field(default_factory=dict, repr=False)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ---- reflection / cover -------------------------------------------------

    def reflect_closure(
        self,
        pieces: Iterable[Ar25Piece],
        axes: Iterable[Ar25Axis],
        *,
        grid_w: int,
        grid_h: int,
        bounce_limit: Optional[int] = None,
    ) -> set[tuple[int, int]]:
        """All cells covered by piece pixels and their axis reflections."""
        limit = self.bounce_limit if bounce_limit is None else bounce_limit
        axes_list = list(axes)
        pieces_list = list(pieces)
        cache_key = (
            tuple((p.id, p.x, p.y, p.pixels_hash, tuple(sorted(p.tags))) for p in pieces_list),
            tuple((a.id, a.kind, a.x, a.y) for a in axes_list),
            grid_w,
            grid_h,
            limit,
        )
        hit = self._cache.get(cache_key)
        if hit is not None:
            return set(hit)

        covered: set[tuple[int, int]] = set()
        for piece in pieces_list:
            seeds = _piece_pixels(piece)
            for x, y in seeds:
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    covered.add((x, y))

            queue: deque[tuple[tuple[int, int], int]] = deque()
            visited: set[tuple[int, int]] = set()
            for pos in seeds:
                if pos not in visited:
                    visited.add(pos)
                    queue.append((pos, 0))

            while queue:
                (px, py), depth = queue.popleft()
                if depth > limit:
                    continue
                for axis in axes_list:
                    bounced = _bounce(px, py, axis, piece)
                    if bounced is None:
                        continue
                    rx, ry = bounced
                    if (rx, ry) in visited:
                        continue
                    visited.add((rx, ry))
                    if 0 <= rx < grid_w and 0 <= ry < grid_h:
                        covered.add((rx, ry))
                    queue.append(((rx, ry), depth + 1))

        self._cache[cache_key] = frozenset(covered)
        return covered

    def cover_report(self, obs: Ar25Obs, *, bounce_limit: Optional[int] = None) -> Ar25CoverReport:
        covered = self.reflect_closure(
            obs.pieces,
            obs.axes,
            grid_w=obs.grid_w,
            grid_h=obs.grid_h,
            bounce_limit=bounce_limit,
        )
        uncovered = [t for t in obs.targets if t not in covered]
        return Ar25CoverReport(
            covered=covered,
            uncovered=uncovered,
            ok=len(uncovered) == 0,
        )

    def check_cover(self, obs: Ar25Obs) -> bool:
        return self.cover_report(obs).ok

    # ---- config apply / cost / prune ----------------------------------------

    def apply_config(self, obs: Ar25Obs, config: Ar25Config) -> Ar25Obs:
        """Return a copy of obs with piece/axis poses from config."""
        piece_map = {p.id: p for p in obs.pieces}
        axis_map = {a.id: a for a in obs.axes}
        new_pieces = []
        for p in obs.pieces:
            if p.id in config.piece_xy:
                x, y = config.piece_xy[p.id]
                new_pieces.append(replace(p, x=x, y=y))
            else:
                new_pieces.append(deepcopy(p))
        new_axes = []
        for a in obs.axes:
            if a.id in config.axis_coord:
                coord = config.axis_coord[a.id]
                if a.kind == "V":
                    new_axes.append(replace(a, x=coord))
                else:
                    new_axes.append(replace(a, y=coord))
            else:
                new_axes.append(deepcopy(a))
        # keep maps referenced so unused locals don't confuse linters
        _ = (piece_map, axis_map)
        return replace(obs, pieces=new_pieces, axes=new_axes)

    def estimate_path_cost(self, obs: Ar25Obs, config: Ar25Config) -> int:
        """Lower bound: Manhattan moves + one SELECT between each moved object."""
        moves = 0
        objects_moved = 0
        for p in obs.pieces:
            if p.fixed or p.id not in config.piece_xy:
                continue
            tx, ty = config.piece_xy[p.id]
            d = abs(tx - p.x) + abs(ty - p.y)
            if d:
                moves += d
                objects_moved += 1
        for a in obs.axes:
            if a.fixed or a.id not in config.axis_coord:
                continue
            target = config.axis_coord[a.id]
            cur = a.x if a.kind == "V" else a.y
            d = abs(target - cur)
            if d:
                moves += d
                objects_moved += 1
        selects = max(0, objects_moved - 1)
        return moves + selects

    def is_futile(self, obs: Ar25Obs, config: Ar25Config) -> bool:
        return self.estimate_path_cost(obs, config) > obs.steps_left

    # ---- configuration search -----------------------------------------------

    def enumerate_covering_configs(
        self,
        obs: Ar25Obs,
        *,
        max_solutions: int = 32,
        joint_pieces: bool = True,
    ) -> list[tuple[Ar25Config, int]]:
        """Search axis then piece poses; return (config, cost_lb) sorted by cost.

        Does not simulate collisions / auto-rotate — callers must engine-replay.
        """
        if self.check_cover(obs):
            empty = Ar25Config()
            return [(empty, 0)]

        solutions: list[tuple[Ar25Config, int]] = []

        movable_axes = [a for a in obs.axes if not a.fixed]
        movable_pieces = [p for p in obs.pieces if not p.fixed]

        def consider(cfg: Ar25Config) -> None:
            if self.is_futile(obs, cfg):
                return
            cand = self.apply_config(obs, cfg)
            if self.check_cover(cand):
                solutions.append((cfg, self.estimate_path_cost(obs, cfg)))

        # Phase A: axes only
        for axis_cfg in _axis_pose_product(movable_axes, obs.grid_w, obs.grid_h):
            consider(Ar25Config(axis_coord=axis_cfg))
            if len(solutions) >= max_solutions:
                return _sorted_unique(solutions)

        # Phase B: single-piece moves (with current axes)
        if not solutions and movable_pieces:
            for piece in movable_pieces:
                for x, y in _piece_pose_candidates(piece, obs):
                    consider(Ar25Config(piece_xy={piece.id: (x, y)}))
                    if len(solutions) >= max_solutions:
                        return _sorted_unique(solutions)

        # Phase C: joint axes × pieces (L7+)
        if not solutions and joint_pieces and movable_axes and movable_pieces:
            for axis_cfg in _axis_pose_product(movable_axes, obs.grid_w, obs.grid_h):
                base = Ar25Config(axis_coord=axis_cfg)
                # try moving pieces one-by-one under this axis layout first
                for piece in movable_pieces:
                    for x, y in _piece_pose_candidates(piece, obs):
                        cfg = Ar25Config(
                            axis_coord=dict(axis_cfg),
                            piece_xy={piece.id: (x, y)},
                        )
                        consider(cfg)
                        if len(solutions) >= max_solutions:
                            return _sorted_unique(solutions)
                if len(movable_pieces) == 2:
                    p1, p2 = movable_pieces
                    for x1, y1 in _piece_pose_candidates(p1, obs):
                        for x2, y2 in _piece_pose_candidates(p2, obs):
                            cfg = Ar25Config(
                                axis_coord=dict(axis_cfg),
                                piece_xy={p1.id: (x1, y1), p2.id: (x2, y2)},
                            )
                            # cheap prune before cover
                            if self.is_futile(obs, cfg):
                                continue
                            consider(cfg)
                            if len(solutions) >= max_solutions:
                                return _sorted_unique(solutions)
                # keep base referenced
                _ = base

        return _sorted_unique(solutions)


def _piece_pixels(piece: Ar25Piece) -> list[tuple[int, int]]:
    out = []
    for r, row in enumerate(piece.pixels):
        for c, val in enumerate(row):
            if int(val) != TRANSPARENT:
                out.append((piece.x + c, piece.y + r))
    return out


def _bounce(
    px: int,
    py: int,
    axis: Ar25Axis,
    piece: Ar25Piece,
) -> Optional[tuple[int, int]]:
    """Mirror one bounce. Tag filters match engine `skqtojxvbv` / `nloqvbouxu`."""
    tags = piece.tags
    if axis.kind == "V":
        # reflect_horizontal_only → only bounce on H axis
        if "reflect_horizontal_only" in tags:
            return None
        return (2 * axis.x - px, py)
    if axis.kind == "H":
        # 0038pnuzypawco → only bounce on V axis
        if "0038pnuzypawco" in tags:
            return None
        return (px, 2 * axis.y - py)
    return None


def _axis_pose_product(
    axes: list[Ar25Axis],
    grid_w: int,
    grid_h: int,
) -> Iterator[dict[str, int]]:
    if not axes:
        yield {}
        return
    if len(axes) == 1:
        a = axes[0]
        rng = range(grid_w) if a.kind == "V" else range(grid_h)
        for pos in rng:
            yield {a.id: pos}
        return
    if len(axes) == 2:
        a1, a2 = axes[0], axes[1]
        r1 = range(grid_w) if a1.kind == "V" else range(grid_h)
        r2 = range(grid_w) if a2.kind == "V" else range(grid_h)
        for p1 in r1:
            for p2 in r2:
                yield {a1.id: p1, a2.id: p2}
        return
    # >2 axes: sequential product would explode; yield identity only
    yield {a.id: (a.x if a.kind == "V" else a.y) for a in axes}


def _piece_pose_candidates(piece: Ar25Piece, obs: Ar25Obs) -> Iterator[tuple[int, int]]:
    """Candidate top-left positions; full board for small grids."""
    pw = len(piece.pixels[0]) if piece.pixels else 1
    ph = len(piece.pixels) if piece.pixels else 1
    max_x = max(0, obs.grid_w - pw)
    max_y = max(0, obs.grid_h - ph)
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            yield (x, y)


def _sorted_unique(solutions: list[tuple[Ar25Config, int]]) -> list[tuple[Ar25Config, int]]:
    seen: set[str] = set()
    out: list[tuple[Ar25Config, int]] = []
    for cfg, cost in sorted(solutions, key=lambda t: t[1]):
        key = cfg.state_key()
        if key in seen:
            continue
        seen.add(key)
        out.append((cfg, cost))
    return out
