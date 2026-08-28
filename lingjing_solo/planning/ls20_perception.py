"""Geometry-only LS20 perception primitives.

ARC colors are intentionally kept as data.  Semantic labels must be learned from
fixtures or supplied by a game-specific policy, not guessed from one recording.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridObject:
    color: int
    area: int
    bbox: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[float, float]:
        y0, x0, y1, x1 = self.bbox
        return ((y0 + y1) / 2, (x0 + x1) / 2)


def extract_objects(grid, *, min_area: int = 1) -> list[GridObject]:
    """Return four-connected non-background objects from a 2-D grid."""
    array = np.asarray(grid)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("grid must be a 2-D array or a single-frame 3-D array")
    values, counts = np.unique(array, return_counts=True)
    background = values[int(np.argmax(counts))]
    seen: set[tuple[int, int]] = set()
    objects: list[GridObject] = []
    height, width = array.shape
    for y, x in zip(*np.where(array != background)):
        start = (int(y), int(x))
        if start in seen:
            continue
        color = int(array[start])
        queue = deque([start])
        seen.add(start)
        points = []
        while queue:
            cy, cx = queue.popleft()
            points.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cy + dy, cx + dx)
                if (
                    0 <= neighbor[0] < height
                    and 0 <= neighbor[1] < width
                    and neighbor not in seen
                    and array[neighbor] == color
                ):
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(points) >= min_area:
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            objects.append(GridObject(color, len(points), (min(ys), min(xs), max(ys), max(xs))))
    return objects


@dataclass(frozen=True)
class MotionObject:
    """A current object paired with its measured frame-to-frame displacement."""

    object: GridObject
    displacement: tuple[float, float]


def observe_motion(previous, current, *, max_distance: float = 2.0) -> list[MotionObject]:
    """Return moved objects with a deterministic nearest same-color displacement."""
    before = extract_objects(previous)
    motions = []
    for candidate in extract_objects(current):
        matches = [obj for obj in before if obj.color == candidate.color]
        if not matches:
            continue
        predecessor = min(matches, key=lambda obj: (
            (candidate.centroid[0] - obj.centroid[0]) ** 2
            + (candidate.centroid[1] - obj.centroid[1]) ** 2
        ))
        displacement = (
            candidate.centroid[0] - predecessor.centroid[0],
            candidate.centroid[1] - predecessor.centroid[1],
        )
        if (displacement[0] ** 2 + displacement[1] ** 2) ** 0.5 > max_distance:
            motions.append(MotionObject(candidate, displacement))
    return motions
def moved_objects(previous, current, *, max_distance: float = 8.0) -> list[GridObject]:
    """Find current objects whose nearest same-color predecessor moved."""
    return [
        motion.object
        for motion in observe_motion(previous, current, max_distance=max_distance)
    ]
