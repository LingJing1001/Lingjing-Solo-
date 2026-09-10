"""Coverage bootstrap — first-level rush for games without a plugin.

Goal (P1): maximize number of games that reach levels_completed >= 1
by aggressively probing legal actions + sparse clicks before deep search.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional


class CoverageBootstrap:
    """Cheap universal prior: try unused legal actions, then click grid."""

    def __init__(self, max_probe_steps: int = 48, click_grid: int = 8) -> None:
        self.max_probe_steps = max_probe_steps
        self.click_grid = click_grid
        self._tried: Counter = Counter()
        self._step = 0
        self._got_level = False
        self._click_i = 0
        self._clicks = self._build_clicks()

    def _build_clicks(self) -> list[tuple[int, int]]:
        step = max(4, int(self.click_grid))
        pts: list[tuple[int, int]] = []
        for y in range(8, 56, step):
            for x in range(8, 56, step):
                pts.append((x, y))
        # center-first
        pts.sort(key=lambda p: (p[0] - 32) ** 2 + (p[1] - 32) ** 2)
        return pts

    def reset(self) -> None:
        self._tried.clear()
        self._step = 0
        self._got_level = False
        self._click_i = 0

    def note_progress(self, levels: int) -> None:
        if levels > 0:
            self._got_level = True

    def suggest(
        self,
        valid_actions: list[str],
        levels_completed: int,
    ) -> Optional[tuple[str, Optional[tuple[int, int]], str]]:
        """Return (action, click_xy|None, rationale) or None to defer to Solo."""
        self.note_progress(int(levels_completed))
        if self._got_level or self._step >= self.max_probe_steps:
            return None
        self._step += 1
        valid = [a.upper() for a in valid_actions if a and a.upper() != "RESET"]
        if not valid:
            return None

        # 1) prefer least-tried simple directional / select actions
        simple = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5") if a in valid]
        if simple:
            simple.sort(key=lambda a: (self._tried[a], a))
            pick = simple[0]
            self._tried[pick] += 1
            return pick, None, f"coverage_probe:{pick}"

        # 2) click sweep if ACTION6 available
        if "ACTION6" in valid and self._click_i < len(self._clicks):
            xy = self._clicks[self._click_i]
            self._click_i += 1
            self._tried["ACTION6"] += 1
            return "ACTION6", xy, f"coverage_click:{xy[0]},{xy[1]}"

        # 3) any remaining least-tried
        rest = sorted(valid, key=lambda a: (self._tried[a], a))
        pick = rest[0]
        self._tried[pick] += 1
        return pick, None, f"coverage_fallback:{pick}"


__all__ = ["CoverageBootstrap"]
