"""ALEA Mind · 编排：感知摘要 → PCA新奇 → DNN假设 → VerifyWM → Gram进化。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cov_pca import CovPCA
from .gram_evolve import GramVolume
from .lu_identify import HypothesisEliminator
from .piecewise_dnn import ActionPiecewiseHead, DEFAULT_ACTIONS
from .verify_wm import VerifyWM


def grid_embed(grid, dim: int = 64) -> np.ndarray:
    """Fast classical+spatial embed (no torch). Compatible with ARC 64x64."""
    g = np.asarray(grid)
    if g.ndim != 2:
        return np.zeros(dim, dtype=np.float32)
    play = g[:58] if g.shape[0] >= 58 else g
    # downsample 8x8 blocks
    bh, bw = 8, 8
    H, W = play.shape
    feat = []
    for i in range(bh):
        for j in range(bw):
            y0, y1 = int(H * i / bh), int(H * (i + 1) / bh)
            x0, x1 = int(W * j / bw), int(W * (j + 1) / bw)
            block = play[y0:y1, x0:x1]
            feat.append(float(block.mean()) / 15.0)
            feat.append(float(len(np.unique(block))) / 16.0)
    z = np.zeros(dim, dtype=np.float32)
    n = min(dim, len(feat))
    z[:n] = np.asarray(feat[:n], dtype=np.float32)
    # L2 normalize
    z /= float(np.linalg.norm(z) + 1e-8)
    return z


class AleaMind:
    """Algebraic Learning–Evolution mind (CPU / numpy)."""

    def __init__(self, dim: int = 64, actions: Sequence[str] = DEFAULT_ACTIONS):
        self.dim = int(dim)
        self.pca = CovPCA(dim=dim, k=8)
        self.dnn = ActionPiecewiseHead(embed_dim=dim, actions=actions)
        self.verify = VerifyWM(dim=dim, tol=0.50)
        self.gram = GramVolume(dim=dim)
        self.elim = HypothesisEliminator(dim=dim)
        self._embed: Optional[np.ndarray] = None
        self._prev_embed: Optional[np.ndarray] = None
        self._pending: Optional[Dict[str, Any]] = None
        self._step = 0
        self.metrics = {
            "learns": 0,
            "verified_rules": 0,
            "volume": 0.0,
            "novelty": 0.0,
        }

    def encode(self, grid) -> np.ndarray:
        z = grid_embed(grid, self.dim)
        self._prev_embed = self._embed
        self._embed = z
        self.pca.observe(z)
        if self._step % 8 == 0:
            self.pca.refit()
        return z

    def hypothesize(
        self,
        action: str,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        z = self._embed if self._embed is not None else np.zeros(self.dim, np.float32)
        eff, prog = self.dnn.predict(z, action, xy, shape)
        nov = self.pca.novelty(z)
        verified = self.verify.score_action_context(z, action)
        # prefer verified structure; explore when novel
        score = 0.45 * eff + 0.85 * prog + 0.35 * verified + 0.20 * nov
        # if elimination says need rank experiment, boost novelty actions
        if self.elim.need_rank_experiment():
            score += 0.15 * nov
        return {
            "effect": eff,
            "progress": prog,
            "novelty": nov,
            "verified": verified,
            "score": float(score),
        }

    def score_actions(self, actions: Sequence[str], shape=None) -> Dict[str, float]:
        return {a: self.hypothesize(a, None, shape)["score"] for a in actions}

    def score_clicks(
        self,
        cands: Sequence[Tuple[Any, Tuple[int, int]]],
        shape=None,
    ) -> Dict[Any, float]:
        out = {}
        for key, xy in cands:
            out[key] = self.hypothesize("ACTION6", xy, shape)["score"]
        return out

    def note_choice(self, action: str, xy=None, shape=None) -> None:
        self._pending = {
            "embed": None if self._embed is None else self._embed.copy(),
            "action": str(action).upper(),
            "xy": xy,
            "shape": shape,
        }
        self._step += 1

    def observe(
        self,
        *,
        delta_pixels: int = 0,
        progressed: bool = False,
        grid=None,
    ) -> Dict[str, Any]:
        if grid is not None:
            self.encode(grid)
        pending = self._pending
        self._pending = None
        if not pending or pending.get("embed") is None:
            return {"learned": False}

        prev = pending["embed"]
        cur = self._embed if self._embed is not None else prev
        delta = (cur - prev).astype(np.float32)
        effect = min(1.0, float(delta_pixels) / 64.0 + (0.8 if progressed else 0.0))

        # DNN learn
        loss = self.dnn.learn(
            prev,
            pending["action"],
            effect,
            progressed,
            pending.get("xy"),
            pending.get("shape"),
        )
        self.metrics["learns"] += 1

        # elimination
        self.elim.observe(delta, progressed, pending["action"])

        # verify / evolve
        info: Dict[str, Any] = {"learned": True, "loss": loss, "effect": effect}
        if effect >= 0.08 or progressed:
            rule = self.verify.propose_and_gate(prev, pending["action"], delta)
            if rule is not None:
                self.metrics["verified_rules"] = self.verify.bank.snapshot()["n_rules"]
                gained = self.gram.try_add(rule.c)
                info["rule"] = rule.name
                info["volume_gain"] = gained
            else:
                self.verify.reinforce(pending["action"], prev, delta)
        else:
            self.verify.reinforce(pending["action"], prev, delta)

        self.metrics["volume"] = self.gram.volume()
        self.metrics["novelty"] = self.pca.novelty(cur)
        info["verify"] = self.verify.snapshot()
        info["elim"] = self.elim.snapshot()
        info["gram"] = self.gram.snapshot()
        return info

    def reset_episode(self) -> None:
        # keep DNN/PCA soft knowledge; clear episodic verify buffer lightly
        self.verify.history.clear()
        self._pending = None
        self.elim = HypothesisEliminator(dim=self.dim)

    def hard_reset(self) -> None:
        actions = list(self.dnn.actions)
        self.__init__(dim=self.dim, actions=actions)

    def snapshot(self) -> dict:
        return {
            "step": self._step,
            "metrics": dict(self.metrics),
            "verify": self.verify.snapshot(),
            "pca": self.pca.snapshot(),
            "gram": self.gram.snapshot(),
            "elim": self.elim.snapshot(),
            "dnn_updates": self.dnn.net.updates,
            "dnn_loss_ema": self.dnn.net.loss_ema,
        }
