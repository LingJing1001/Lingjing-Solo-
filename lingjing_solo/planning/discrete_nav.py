"""Discrete 4-way navigation for ls20-like games (no ACTION6).

Learns action→(dx,dy) from frame deltas, marks walls on noop moves,
seeks small goal blobs, avoids blind clicking.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import numpy as np

from ..core import SoloConfig, Logger, canonicalize


def _grid(frame) -> np.ndarray | None:
    raw = getattr(frame, "frame", None) or getattr(frame, "grid", None)
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw
    layer = raw[-1] if raw and isinstance(raw[0][0], list) else raw
    return np.array(layer, dtype=np.int8)


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(grid.flatten()).most_common(1)[0][0])


def _centroid_delta(prev: np.ndarray, curr: np.ndarray) -> Optional[tuple[float, float, int]]:
    if prev is None or curr is None or prev.shape != curr.shape:
        return None
    ys, xs = np.where(prev != curr)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), int(len(xs))


def _connected_goals(grid: np.ndarray, bg: int, max_size: int = 80) -> list[tuple[float, float]]:
    H, W = grid.shape
    seen = set()
    hist = Counter(grid.flatten())
    goals: list[tuple[float, float]] = []
    for y in range(H):
        for x in range(W):
            if (x, y) in seen:
                continue
            c = int(grid[y, x])
            if c == bg or hist[c] >= hist[bg]:
                continue
            stack = [(x, y)]
            pts = []
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in seen:
                    continue
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                if int(grid[cy, cx]) != c:
                    continue
                seen.add((cx, cy))
                pts.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    stack.append((cx + dx, cy + dy))
            if 2 <= len(pts) <= max_size:
                goals.append((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    cleaned: list[tuple[float, float]] = []
    for g in goals:
        if all((g[0] - c[0]) ** 2 + (g[1] - c[1]) ** 2 > 16 for c in cleaned):
            cleaned.append(g)
    return cleaned[:24]


class DiscreteNavPlanner:
    """Online 4-way nav: learn vectors, walls, seek goals."""

    AXIS_NAMES = {
        "up": "ACTION1",
        "down": "ACTION2",
        "left": "ACTION3",
        "right": "ACTION4",
    }

    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self.player_xy: Optional[tuple[float, float]] = None
        self.goals: list[tuple[float, float]] = []
        self.seek_target: Optional[tuple[float, float]] = None
        self.goal_queue: deque = deque()
        self.visited_goals: set = set()
        self.seek_ttl = 0
        self.move_samples: dict[str, list[tuple[float, float]]] = {}
        self.walls: set[tuple[str, str]] = set()  # (state_hash, action)
        self.death_actions: Counter = Counter()
        self.steps_on_level = 0
        self.best_levels = 0
        self.stall = 0
        self.last_action: Optional[str] = None
        self.streak = 0
        self._prev_grid: np.ndarray | None = None
        self._state_hash_fn = None

    def reset_level(self):
        self.steps_on_level = 0
        self.stall = 0
        self.seek_target = None
        self.seek_ttl = 0

    def bind_hasher(self, fn):
        self._state_hash_fn = fn

    def observe(
        self,
        prev_frame,
        curr_frame,
        action: str | None,
        state_hash: str,
        levels: int,
    ):
        action = canonicalize(action) if action else None
        prev = _grid(prev_frame)
        curr = _grid(curr_frame)
        if prev is not None:
            self._prev_grid = prev.copy()
        cen = _centroid_delta(prev, curr) if prev is not None and curr is not None else None
        if cen and action:
            cx, cy, n = cen
            if self.player_xy is not None:
                dx, dy = cx - self.player_xy[0], cy - self.player_xy[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if 0.5 < dist < 20:
                    self.move_samples.setdefault(action, []).append((dx, dy))
                    self.move_samples[action] = self.move_samples[action][-24:]
                elif dist < 0.6:
                    self.walls.add((state_hash, action))
                    self.death_actions[action] += 1
            self.player_xy = (cx, cy)
        elif action and state_hash:
            self.walls.add((state_hash, action))

        if levels > self.best_levels:
            self.best_levels = levels
            self.stall = 0
            self.reset_level()
        else:
            self.steps_on_level += 1
            if cen is None or (cen and cen[2] < 3):
                self.stall += 1
            else:
                self.stall = max(0, self.stall - 1)

        if action == self.last_action:
            self.streak += 1
        else:
            self.streak = 1
            self.last_action = action

        if curr is not None and self.steps_on_level % 3 == 0:
            self.refresh_goals(curr)

    def refresh_goals(self, grid: np.ndarray):
        bg = _bg_color(grid)
        self.goals = _connected_goals(grid, bg)
        if self.player_xy and self.goals:
            if self.seek_target is None or self.seek_ttl <= 0 or self.stall >= 12:
                # 优先最远 blob（ls20 目标通常在上方）
                self.seek_target = max(
                    self.goals,
                    key=lambda g: (g[0] - self.player_xy[0]) ** 2
                    + (g[1] - self.player_xy[1]) ** 2,
                )
                self.seek_ttl = 48
                self.stall = max(0, self.stall - 4)

    def axis_actions(self) -> dict[str, Optional[str]]:
        best = {k: (None, 0.0) for k in self.AXIS_NAMES}
        for name, samples in self.move_samples.items():
            if len(samples) < 2:
                continue
            dx = sum(s[0] for s in samples) / len(samples)
            dy = sum(s[1] for s in samples) / len(samples)
            n = float(len(samples))
            if abs(dy) >= abs(dx) and abs(dy) > 0.3:
                key = "up" if dy < 0 else "down"
                if abs(dy) > best[key][1]:
                    best[key] = (name, abs(dy))
            elif abs(dx) > 0.3:
                key = "left" if dx < 0 else "right"
                if abs(dx) > best[key][1]:
                    best[key] = (name, abs(dx))
        return {k: v[0] for k, v in best.items()}

    def is_wall(self, state_hash: str, action: str) -> bool:
        return (state_hash, canonicalize(action)) in self.walls

    def plan(
        self,
        valid_actions: list[str],
        state_hash: str,
        levels: int,
    ) -> Optional[str]:
        valid = [canonicalize(a) for a in valid_actions if canonicalize(a) != "ACTION6"]
        if not valid or not self.player_xy:
            return None
        if self.seek_ttl > 0:
            self.seek_ttl -= 1
        axis = self.axis_actions()
        if not any(axis.values()):
            # bootstrap: try untried directions
            for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                if a in valid and not self.is_wall(state_hash, a):
                    return a
            return valid[0]

        if self.seek_target:
            px, py = self.player_xy
            tx, ty = self.seek_target
            dx, dy = tx - px, ty - py
            dist2 = dx * dx + dy * dy
            if dist2 < 12:
                # near target — sweep adjacent to hit modifier / goal tile
                for a in valid:
                    if not self.is_wall(state_hash, a) and self.death_actions[a] < 3:
                        return a
                self.seek_target = None
            else:
                order = (
                    (["right", "left"] if dx > 0 else ["left", "right"])
                    + (["down", "up"] if dy > 0 else ["up", "down"])
                    if abs(dx) >= abs(dy)
                    else (["down", "up"] if dy > 0 else ["up", "down"])
                    + (["right", "left"] if dx > 0 else ["left", "right"])
                )
                for card in order:
                    aname = axis.get(card)
                    if aname and aname in valid and not self.is_wall(state_hash, aname):
                        if aname == self.last_action and self.streak >= 6:
                            continue
                        return aname

        # fallback: least death-count open move
        open_ = [a for a in valid if not self.is_wall(state_hash, a)]
        if open_:
            open_.sort(key=lambda a: (self.death_actions[a], self.streak if a == self.last_action else 0))
            return open_[0]
        return None

    @staticmethod
    def is_discrete_game(valid_actions: list[str]) -> bool:
        v = {canonicalize(a) for a in valid_actions}
        simple = v & {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}
        return bool(simple) and "ACTION6" not in v and len(simple) <= 5
