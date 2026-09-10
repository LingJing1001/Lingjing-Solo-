"""Neural × CEAX graft: CNN/DNN reason loop injected into hypothesis-experiment cycle.

NeuralCeaxController(CeaxController) is additive only:
  1) encode grid with NumpyCNN (torch if available)
  2) observe_outcome → NeuralReasonLoop.observe (learn + failure summary)
  3) untried click targets ranked by neural hypothesis score
  4) simple actions blended with neural action scores
  5) any neural fault → fall back to base CeaxController

Loop realized:
  Observation → CNN embed → Hypothesize(effect/progress)
  → Experiment(choose) → Outcome → DNN update + FailureMemory summarize
  → Retry with epistemic / success-bonus / fail-penalty
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..neural import NeuralReasonLoop
from .ceax_controller import CeaxController, ClickTarget


class NeuralCeaxController(CeaxController):
    """CEAX + online CNN/DNN reasoner (no game-id hardcoding)."""

    def __init__(self, cfg=None, logger=None):
        super().__init__(cfg=cfg, logger=logger)
        num_colors = int(getattr(cfg, "num_colors", 16) or 16) if cfg else 16
        embed = 64
        self.reasoner = NeuralReasonLoop(num_colors=num_colors, embed_dim=embed)
        self._neural_prior: Dict[Tuple, float] = {}
        self._last_shape: Optional[Tuple[int, int]] = None
        self._neural_step = 0

    # ---------- lifecycle ----------

    def reset_game(self, *, import_skills: Optional[dict] = None):
        super().reset_game(import_skills=import_skills)
        try:
            self.reasoner.hard_reset()
        except Exception:
            self.reasoner = NeuralReasonLoop()
        self._neural_prior.clear()
        self._last_shape = None
        self._neural_step = 0

    def on_level_up(self):
        super().on_level_up()
        # Keep predictor weights across levels (transfer); clear episodic failures
        try:
            self.reasoner.reset()
        except Exception:
            pass
        self._neural_prior.clear()

    # ---------- observe ----------

    def observe_outcome(
        self,
        *,
        delta_pixels: int,
        progressed: bool,
        grid_hash: str = "",
        levels: int = 0,
        prev_grid=None,
        grid=None,
    ):
        super().observe_outcome(
            delta_pixels=delta_pixels,
            progressed=progressed,
            grid_hash=grid_hash,
            levels=levels,
            prev_grid=prev_grid,
            grid=grid,
        )
        try:
            self.reasoner.observe(
                delta_pixels=delta_pixels,
                progressed=progressed,
                grid=grid,
            )
        except Exception:
            pass
        self._neural_step += 1

    # ---------- decide ----------

    def choose(
        self,
        *,
        valid_actions: Sequence[str],
        objects: Optional[Sequence[Any]] = None,
        grid=None,
        grid_hash: str = "",
    ) -> Tuple[str, Optional[Tuple[int, int]], str]:
        try:
            if grid is not None:
                g = np.asarray(grid)
                self._last_shape = (int(g.shape[0]), int(g.shape[1]))
                self.reasoner.encode(g)
        except Exception:
            pass

        act, xy, why = super().choose(
            valid_actions=valid_actions,
            objects=objects,
            grid=grid,
            grid_hash=grid_hash,
        )
        try:
            self.reasoner.note_choice(act, xy, self._last_shape)
            if not str(why).startswith("neural_"):
                why = f"neural+{why}"
        except Exception:
            pass
        return act, xy, why

    # ---------- injection points ----------

    def _ingest_objects(self, objects: Sequence[Any]) -> None:
        super()._ingest_objects(objects)
        try:
            cands = [
                (key, t.xy)
                for key, t in self.targets.items()
                if t.trials == 0 and t.xy is not None
            ]
            scores = self.reasoner.score_click_candidates(cands, self._last_shape)
            self._neural_prior = {
                key: float(scores.get(key, 0.0)) for key, _xy in cands
            }
        except Exception:
            self._neural_prior = {}

    def _next_experiment_click(self) -> Optional[Tuple[Tuple[int, int], Tuple]]:
        try:
            untried = [t for t in self.targets.values() if t.trials == 0]
            if len(untried) >= 2 and self._neural_prior:
                def sort_key(t: ClickTarget):
                    sp = float(self._neural_prior.get(t.key, 0.0))
                    pref = 0 if t.color in self._preferred_colors else 1
                    return (-sp, pref, t.area, t.color)

                untried.sort(key=sort_key)
                t = untried[0]
                return t.xy, t.key
        except Exception:
            pass
        return super()._next_experiment_click()

    def _rank_simple(self, actions: Sequence[str], grid_hash: str) -> List[str]:
        base = super()._rank_simple(actions, grid_hash)
        try:
            neural = self.reasoner.score_actions(base, self._last_shape)
            if not neural:
                return base
            vals = sorted(neural.values())
            lo, hi = vals[0], vals[-1]
            span = (hi - lo) or 1.0

            def blend(i: int, name: str) -> tuple:
                sp = (float(neural.get(name, 0.0)) - lo) / span
                return (i + 0.35 * sp, name)

            ranked = sorted((blend(i, n) for i, n in enumerate(base)), reverse=True)
            return [n for _s, n in ranked]
        except Exception:
            return base

    def snapshot(self) -> dict:
        snap = super().snapshot()
        try:
            snap["neural"] = self.reasoner.snapshot()
        except Exception:
            snap["neural"] = {"error": True}
        return snap
