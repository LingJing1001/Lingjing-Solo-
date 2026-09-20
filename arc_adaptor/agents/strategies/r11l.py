"""Observation-driven R11L click strategy.

The route is intentionally not hard-coded. R11L exposes clickable sprites in
rendered frame colour 3; colour 6 is UI state, not a stable target coordinate.
The strategy therefore extracts connected clickable components from the
normalised 2-D frame and returns their display-space centre.
"""
from __future__ import annotations

from typing import Any


class R11LStrategy:
    def __init__(self) -> None:
        self.last_level: int | None = None
        self._animation_tick_pending = False

    def reset(self, frame: Any) -> None:
        self.last_level = int(getattr(frame, "levels_completed", 0) or 0)
        self._animation_tick_pending = False

    def choose_action(
        self,
        frames: list[Any],
        frame: Any,
        grid: Any,
        legal_names: list[str],
        levels_completed: int,
    ) -> dict[str, int | str] | str | None:
        del frames, frame
        if "ACTION6" not in legal_names:
            return legal_names[0] if legal_names else None
        self.last_level = levels_completed
        # The game consumes one engine tick to finish a click-driven move.  A
        # new ACTION6 during that tick is ignored by R11L, but its coordinate
        # can still be interpreted as a second selection by a replay driver.
        # Emit one safe out-of-target tick before selecting another marker.
        if self._animation_tick_pending:
            self._animation_tick_pending = False
            return {"name": "ACTION6", "x": 0, "y": 0}
        # FrameData is normalised by the adapter to a 2-D colour grid. The
        # game source remaps unselected sys_click sprites to colour 3; colour
        # 6 belongs to UI state and was the source of the old (0,0) route.
        try:
            import numpy as np

            array = np.asarray(grid)
            if array.ndim == 3:
                array = array[0]
            mask = array == 3
            height, width = mask.shape
            seen = np.zeros_like(mask, dtype=bool)
            components: list[tuple[int, int, int]] = []
            for row in range(height):
                for col in range(width):
                    if not mask[row, col] or seen[row, col]:
                        continue
                    stack = [(row, col)]
                    seen[row, col] = True
                    cells = []
                    while stack:
                        r, c = stack.pop()
                        cells.append((r, c))
                        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    if len(cells) >= 4:
                        components.append((len(cells), sum(r for r, _ in cells), sum(c for _, c in cells)))
            if components:
                area, row_sum, col_sum = max(components)
                self._animation_tick_pending = True
                return {"name": "ACTION6", "x": col_sum // area, "y": row_sum // area}
        except (TypeError, ValueError):
            pass
        return {"name": "ACTION6", "x": 32, "y": 32}
