"""Unified CEAX × SSA × ALEA — 未知环境总成控制器（只加不改基类）。

合成策略（保 CEAX 底盘，叠加谱 + 验证式代数）：
  1) observe：CEAX + SpectralMind + AleaMind 并行学习
  2) 点击实验排序：先 CEAX 偏好色，再 0.5·谱先验 + 0.5·ALEA 验证分
  3) 简单动作排序：基类次序上叠加谱/ALEA 轻权重（不推翻 CEAX）
  4) 任一层异常 → 自动退化到下一层 / 纯 CEAX

目标：focus 易局不低于 CEAX；难局用验证规则与谱杠杆试破零。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..alea import AleaMind
from ..lincore import MatrixMemory, SpectralMind
from ..lincore.ttt import SharedBasis
from .ceax_controller import CeaxController, ClickTarget


def _norm_scores(d: Dict[Any, float]) -> Dict[Any, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (float(v) - lo) / span for k, v in d.items()}


class UnifiedCeaxController(CeaxController):
    """CEAX 执行环 × SSA 谱内核 × ALEA 验证进化。"""

    def __init__(self, cfg=None, logger=None, game_sig: str = "unknown"):
        super().__init__(cfg=cfg, logger=logger)
        self.game_sig = str(game_sig or "unknown")
        self.mind = SpectralMind(cfg=cfg, game_sig=self.game_sig, memory=MatrixMemory.shared())
        try:
            sb = os.environ.get("LINCORE_SHARED_BASIS", "")
            if sb:
                self.mind.warm_start(SharedBasis.load(sb))
        except Exception:
            pass
        self.alea = AleaMind(dim=64)
        self._spectral_prior: Dict[Tuple, float] = {}
        self._alea_prior: Dict[Tuple, float] = {}
        self._fusion_prior: Dict[Tuple, float] = {}
        self._last_shape: Optional[Tuple[int, int]] = None
        self._step_u = 0

    # ---------- lifecycle ----------

    def reset_game(self, *, import_skills: Optional[dict] = None):
        super().reset_game(import_skills=import_skills)
        try:
            self.mind = SpectralMind(
                cfg=self.cfg, game_sig=self.game_sig, memory=MatrixMemory.shared()
            )
            sb = os.environ.get("LINCORE_SHARED_BASIS", "")
            if sb:
                self.mind.warm_start(SharedBasis.load(sb))
        except Exception:
            pass
        try:
            self.alea.hard_reset()
        except Exception:
            self.alea = AleaMind(dim=64)
        self._spectral_prior.clear()
        self._alea_prior.clear()
        self._fusion_prior.clear()
        self._last_shape = None
        self._step_u = 0

    def on_level_up(self):
        super().on_level_up()
        try:
            self.mind = SpectralMind(
                cfg=self.cfg, game_sig=self.game_sig, memory=MatrixMemory.shared()
            )
        except Exception:
            pass
        try:
            self.alea.reset_episode()
        except Exception:
            pass
        self._spectral_prior.clear()
        self._alea_prior.clear()
        self._fusion_prior.clear()

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
            self.mind.observe(
                prev_grid, grid, self._last_action, levels=levels, progressed=progressed
            )
        except Exception:
            pass
        try:
            self.alea.observe(
                delta_pixels=delta_pixels, progressed=progressed, grid=grid
            )
        except Exception:
            pass
        self._step_u += 1
        if self._step_u % 16 == 0:
            try:
                self.mind.maintenance()
            except Exception:
                pass

    # ---------- choose ----------

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
                self.alea.encode(g)
                self.mind.set_state_context(g)
        except Exception:
            pass

        act, xy, why = super().choose(
            valid_actions=valid_actions,
            objects=objects,
            grid=grid,
            grid_hash=grid_hash,
        )
        try:
            self.alea.note_choice(act, xy, self._last_shape)
            shape = np.asarray(grid).shape if grid is not None else None
            self.mind.note_choice(act, xy, shape)
            self.mind.note_action_taken(act)
            why = f"unified+{why}"
        except Exception:
            pass
        return act, xy, why

    # ---------- fusion inject ----------

    def _ingest_objects(self, objects: Sequence[Any]) -> None:
        super()._ingest_objects(objects)
        cands = [
            (key, t.xy)
            for key, t in self.targets.items()
            if t.trials == 0 and t.xy is not None
        ]
        try:
            sp = self.mind.score_click_candidates(cands)
            self._spectral_prior = {k: float(sp.get(k, 0.0)) for k, _ in cands}
        except Exception:
            self._spectral_prior = {}
        try:
            ap = self.alea.score_clicks(cands, self._last_shape)
            self._alea_prior = {k: float(ap.get(k, 0.0)) for k, _ in cands}
        except Exception:
            self._alea_prior = {}

        sn = _norm_scores(self._spectral_prior)
        an = _norm_scores(self._alea_prior)
        # verified ALEA gets a mild boost when pass_rate high
        alea_w = 0.45
        try:
            pr = float(self.alea.verify.stats.get("pass_rate", 0.0))
            if pr >= 0.4:
                alea_w = 0.55
        except Exception:
            pass
        spec_w = 1.0 - alea_w
        keys = {k for k, _ in cands}
        self._fusion_prior = {
            k: spec_w * sn.get(k, 0.0) + alea_w * an.get(k, 0.0) for k in keys
        }

    def _next_experiment_click(self) -> Optional[Tuple[Tuple[int, int], Tuple]]:
        # CEAX-first: do not reorder until ALEA has verified structure
        # (early fusion previously cost r11l L1).
        base = super()._next_experiment_click()
        try:
            accepted = int(self.alea.verify.stats.get("accepted", 0) or 0)
            if accepted < 3 or not self._fusion_prior:
                return base
            untried = [t for t in self.targets.values() if t.trials == 0]
            if len(untried) < 2:
                return base

            def sort_key(t: ClickTarget):
                pref = 0 if t.color in self._preferred_colors else 1
                fus = float(self._fusion_prior.get(t.key, 0.0))
                return (pref, -fus, t.area, t.color)

            untried.sort(key=sort_key)
            t = untried[0]
            return t.xy, t.key
        except Exception:
            return base

    def _rank_simple(self, actions: Sequence[str], grid_hash: str) -> List[str]:
        base = super()._rank_simple(actions, grid_hash)
        try:
            accepted = int(self.alea.verify.stats.get("accepted", 0) or 0)
            if accepted < 3:
                return base  # pure CEAX until verification warms up
            spectral = self.mind.score_actions(base) or {}
            alea = self.alea.score_actions(base, self._last_shape) or {}
            sn = _norm_scores({a: float(spectral.get(a, 0.0)) for a in base})
            an = _norm_scores({a: float(alea.get(a, 0.0)) for a in base})

            def blend(i: int, name: str) -> tuple:
                sp = 0.5 * sn.get(name, 0.0) + 0.5 * an.get(name, 0.0)
                return (i + 0.15 * sp, name)

            ranked = sorted((blend(i, n) for i, n in enumerate(base)), reverse=True)
            return [n for _s, n in ranked]
        except Exception:
            return base

    def snapshot(self) -> dict:
        snap = super().snapshot()
        try:
            snap["spectral"] = self.mind.snapshot()
        except Exception:
            snap["spectral"] = {"error": True}
        try:
            snap["alea"] = self.alea.snapshot()
        except Exception:
            snap["alea"] = {"error": True}
        snap["unified"] = {
            "fusion_targets": len(self._fusion_prior),
            "step": self._step_u,
        }
        return snap
