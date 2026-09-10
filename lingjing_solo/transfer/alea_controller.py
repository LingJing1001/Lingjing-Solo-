"""ALEA controller — synced from AGI进化研究_ALEA_v1/bridge (import path adapted)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..alea import AleaMind
from .ceax_controller import CeaxController, ClickTarget


class AleaCeaxController(CeaxController):
    def __init__(self, cfg=None, logger=None):
        super().__init__(cfg=cfg, logger=logger)
        self.alea = AleaMind(dim=64)
        self._alea_prior: Dict[Tuple, float] = {}
        self._last_shape: Optional[Tuple[int, int]] = None

    def reset_game(self, *, import_skills: Optional[dict] = None):
        super().reset_game(import_skills=import_skills)
        try:
            self.alea.hard_reset()
        except Exception:
            self.alea = AleaMind(dim=64)
        self._alea_prior.clear()
        self._last_shape = None

    def on_level_up(self):
        super().on_level_up()
        try:
            self.alea.reset_episode()
        except Exception:
            pass
        self._alea_prior.clear()

    def observe_outcome(self, *, delta_pixels: int, progressed: bool, grid_hash: str = "", levels: int = 0, prev_grid=None, grid=None):
        super().observe_outcome(delta_pixels=delta_pixels, progressed=progressed, grid_hash=grid_hash, levels=levels, prev_grid=prev_grid, grid=grid)
        try:
            self.alea.observe(delta_pixels=delta_pixels, progressed=progressed, grid=grid)
        except Exception:
            pass

    def choose(self, *, valid_actions: Sequence[str], objects: Optional[Sequence[Any]] = None, grid=None, grid_hash: str = ""):
        try:
            if grid is not None:
                g = np.asarray(grid)
                self._last_shape = (int(g.shape[0]), int(g.shape[1]))
                self.alea.encode(g)
        except Exception:
            pass
        act, xy, why = super().choose(valid_actions=valid_actions, objects=objects, grid=grid, grid_hash=grid_hash)
        try:
            self.alea.note_choice(act, xy, self._last_shape)
            why = f"alea+{why}"
        except Exception:
            pass
        return act, xy, why

    def _ingest_objects(self, objects: Sequence[Any]) -> None:
        super()._ingest_objects(objects)
        try:
            cands = [(key, t.xy) for key, t in self.targets.items() if t.trials == 0 and t.xy is not None]
            scores = self.alea.score_clicks(cands, self._last_shape)
            self._alea_prior = {k: float(scores.get(k, 0.0)) for k, _ in cands}
        except Exception:
            self._alea_prior = {}

    def _next_experiment_click(self):
        try:
            untried = [t for t in self.targets.values() if t.trials == 0]
            if len(untried) >= 2 and self._alea_prior:
                def sk(t):
                    return (-float(self._alea_prior.get(t.key, 0.0)), 0 if t.color in self._preferred_colors else 1, t.area, t.color)
                untried.sort(key=sk)
                t = untried[0]
                return t.xy, t.key
        except Exception:
            pass
        return super()._next_experiment_click()

    def _rank_simple(self, actions: Sequence[str], grid_hash: str):
        base = super()._rank_simple(actions, grid_hash)
        try:
            scored = self.alea.score_actions(base, self._last_shape)
            if not scored:
                return base
            vals = sorted(scored.values())
            lo, hi = vals[0], vals[-1]
            span = (hi - lo) or 1.0
            def blend(i, name):
                sp = (float(scored.get(name, 0.0)) - lo) / span
                return (i + 0.4 * sp, name)
            return [n for _, n in sorted((blend(i, n) for i, n in enumerate(base)), reverse=True)]
        except Exception:
            return base

    def snapshot(self) -> dict:
        snap = super().snapshot()
        try:
            snap["alea"] = self.alea.snapshot()
        except Exception:
            snap["alea"] = {"error": True}
        return snap
