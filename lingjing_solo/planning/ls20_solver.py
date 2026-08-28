"""Conservative LS20 route execution primitives.

The ARC frame does not expose semantic labels for every LS20 object.  This module
therefore owns the safe execution contract first: a route is invalidated whenever
the observed scene changes, so a future semantic planner cannot execute a stale
route through a moving platform.
"""
from dataclasses import dataclass, field
from typing import Any

from ..core import hash_grid
from .ls20_perception import observe_motion


@dataclass
class LS20State:
    """Structured per-level state reserved for semantic LS20 perception."""

    player: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    shape: Any = None
    color: int | None = None
    rotation: int | None = None
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    dynamic_obstacles: list[tuple[int, int, int, int]] = field(default_factory=list)


class LS20Solver:
    """Execute only a legal, still-valid one-step route.

    ``set_plan`` is intentionally separate from perception.  A semantic LS20
    planner may provide a route, while this class guarantees that a route is
    discarded when the next frame differs from the frame on which it was based.
    """

    def __init__(self) -> None:
        self._plan: list[str] = []
        self._last_hash: str | None = None
        self.replan_required = False
        self.state = LS20State()

    def reset(self) -> None:
        self._plan.clear()
        self._last_hash = None
        self.replan_required = False
        self.state = LS20State()

    def set_plan(self, actions: list[str]) -> None:
        self._plan = list(actions)
        self.replan_required = False

    def observe_transition(
        self, previous, current, *, player: tuple[int, int] | None = None
    ) -> list[tuple[float, float]]:
        """Record motion and invalidate only when it blocks the next step."""
        motions = observe_motion(previous, current)
        self.state.dynamic_obstacles = [motion.object.bbox for motion in motions]
        self.observe(previous)
        self.observe(current, invalidate=False)
        if player is not None:
            self.state.player = player
        self.replan_required = self._next_step_blocked()
        return [motion.displacement for motion in motions]

    def _next_step_blocked(self) -> bool:
        if not self._plan or self.state.player is None:
            return False
        row, col = self.state.player
        offsets = {
            "ACTION1": (5, 0), "ACTION2": (0, -5),
            "ACTION3": (0, 5), "ACTION4": (-5, 0),
        }
        dr, dc = offsets.get(self._plan[0], (0, 0))
        next_row, next_col = row + dr, col + dc
        return any(
            top <= next_row <= bottom and left <= next_col <= right
            for top, left, bottom, right in self.state.dynamic_obstacles
        )

    def observe(self, grid, *, invalidate: bool = True) -> bool:
        """Record a frame and optionally invalidate on scene changes."""
        current_hash = hash_grid(grid)
        if invalidate and self._last_hash is not None and current_hash != self._last_hash:
            self._plan.clear()
            self.replan_required = True
        self._last_hash = current_hash
        return not self.replan_required

    def plan_waypoints(
        self,
        start: tuple[int, int],
        waypoints: list[tuple[int, int]],
        *,
        action_map: dict[str, str] | None = None,
        step_size: int = 5,
    ) -> list[str]:
        """Build a Manhattan route through semantic LS20 waypoints.

        Coordinates and action names are supplied by a game-specific perception
        layer; no ARC color is interpreted here.
        """
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        actions = action_map or {
            "up": "ACTION4", "down": "ACTION1",
            "left": "ACTION2", "right": "ACTION3",
        }
        route: list[str] = []
        current = start
        for target in waypoints:
            dy, dx = target[0] - current[0], target[1] - current[1]
            if dy % step_size or dx % step_size:
                raise ValueError("waypoints must align to step_size")
            vertical = actions["down"] if dy > 0 else actions["up"]
            horizontal = actions["right"] if dx > 0 else actions["left"]
            route.extend([vertical] * (abs(dy) // step_size))
            route.extend([horizontal] * (abs(dx) // step_size))
            current = target
        self.set_plan(route)
        return list(route)

    def next_action(self, grid, valid_actions: list[str]) -> str | None:
        """Return one legal action, or ``None`` when replanning is required."""
        self.observe(grid, invalidate=False)
        if self.replan_required:
            return None
        allowed = set(valid_actions)
        while self._plan:
            action = self._plan.pop(0)
            if action in allowed:
                return action
        return None
