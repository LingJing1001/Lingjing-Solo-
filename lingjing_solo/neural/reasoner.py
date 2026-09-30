"""Neural reason loop: observe → hypothesize → experiment → summarize → retry.

Closed loop (plugs into CEAX without game-id hardcoding):

  encode(grid) ──► embedding z
       │
       ├─ score experiments (effect_hat, progress_hat, epistemic)
       ├─ choose / rank clicks & actions
       └─ on outcome: learn DNN + write FailureMemory summary
            ├─ success → boost similar future exploits
            └─ fail/noop → penalize similar retries, raise exploration

This is the "推理、实验、得出结果、总结、失败再总结直至成功" layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .conv import NumpyCNN, try_torch_encode
from .mlp import ActionEffectNet, FailureMemory, DEFAULT_ACTIONS


class NeuralReasonLoop:
    """Online CNN+DNN reasoner for unknown ARC environments."""

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 64,
        actions: Sequence[str] = DEFAULT_ACTIONS,
        prefer_torch: bool = True,
    ):
        self.embed_dim = int(embed_dim)
        self.cnn = NumpyCNN(num_colors=num_colors, embed_dim=self.embed_dim)
        self.net = ActionEffectNet(embed_dim=self.embed_dim, actions=actions)
        self.memory = FailureMemory(capacity=160, dim=self.embed_dim)
        self.prefer_torch = bool(prefer_torch)
        self._embed: Optional[np.ndarray] = None
        self._prev_embed: Optional[np.ndarray] = None
        self._pending: Optional[Dict[str, Any]] = None
        self._step = 0
        self._backend = "numpy"
        self.metrics = {
            "encodes": 0,
            "learns": 0,
            "hypotheses": 0,
            "failures_logged": 0,
            "successes_logged": 0,
            "loss_ema": 0.0,
        }

    # ---------- perception ----------

    def encode(self, grid) -> np.ndarray:
        if grid is None:
            z = np.zeros(self.embed_dim, dtype=np.float32)
            self._embed = z
            return z
        z = None
        if self.prefer_torch:
            z = try_torch_encode(grid, self.embed_dim)
            if z is not None:
                self._backend = "torch"
        if z is None:
            z = self.cnn.encode(grid)
            self._backend = "numpy"
        self._prev_embed = self._embed
        self._embed = z
        self.metrics["encodes"] += 1
        return z

    # ---------- hypothesize / score ----------

    def hypothesize_action(
        self,
        action: str,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        """Predict effect & progress; fold failure summaries into epistemic score."""
        z = self._embed if self._embed is not None else np.zeros(self.embed_dim, dtype=np.float32)
        eff, prog = self.net.predict(z, action, xy, shape)
        fail_pen = self.memory.failure_penalty(z, action)
        suc_bon = self.memory.success_bonus(z, action)
        # epistemic: prefer uncertain / under-tried regions (high when mid probs)
        uncertainty = float(4.0 * eff * (1.0 - eff) + 4.0 * prog * (1.0 - prog))
        score = (
            0.55 * eff
            + 0.90 * prog
            + 0.25 * uncertainty
            + 0.35 * suc_bon
            - 0.55 * fail_pen
        )
        self.metrics["hypotheses"] += 1
        return {
            "effect": eff,
            "progress": prog,
            "uncertainty": uncertainty,
            "fail_penalty": fail_pen,
            "success_bonus": suc_bon,
            "score": float(score),
        }

    def score_actions(
        self,
        actions: Sequence[str],
        shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        return {
            a: self.hypothesize_action(a, None, shape)["score"] for a in actions
        }

    def score_click_candidates(
        self,
        candidates: Sequence[Tuple[Any, Tuple[int, int]]],
        shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[Any, float]:
        """candidates: iterable of (key, xy)."""
        out: Dict[Any, float] = {}
        for key, xy in candidates:
            h = self.hypothesize_action("ACTION6", xy, shape)
            out[key] = h["score"]
        return out

    # ---------- experiment bookkeeping ----------

    def note_choice(
        self,
        action: str,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._pending = {
            "embed": None if self._embed is None else self._embed.copy(),
            "action": str(action).upper(),
            "xy": xy,
            "shape": shape,
            "hyp": self.hypothesize_action(action, xy, shape),
        }
        self._step += 1

    def observe(
        self,
        *,
        delta_pixels: int = 0,
        progressed: bool = False,
        grid=None,
    ) -> Dict[str, Any]:
        """Learn from last experiment; summarize success/failure."""
        if grid is not None:
            self.encode(grid)
        pending = self._pending
        self._pending = None
        if not pending or pending.get("embed") is None:
            return {"learned": False}

        effect = min(1.0, float(delta_pixels) / 64.0) + (0.8 if progressed else 0.0)
        effect = float(min(1.0, effect))
        loss = self.net.learn(
            pending["embed"],
            pending["action"],
            effect,
            progressed,
            pending.get("xy"),
            pending.get("shape"),
        )
        self.metrics["learns"] += 1
        self.metrics["loss_ema"] = self.net.loss_ema

        hyp = pending.get("hyp") or {}
        predicted = float(hyp.get("progress", 0.0))
        # Summarize outcome relative to hypothesis
        if progressed or effect >= 0.25:
            outcome = "success"
            note = f"Δ={delta_pixels} prog={int(progressed)} pred_p={predicted:.2f}"
            self.metrics["successes_logged"] += 1
        elif effect < 0.05:
            outcome = "noop"
            note = f"noop Δ={delta_pixels} pred_e={float(hyp.get('effect', 0)):.2f}"
            self.metrics["failures_logged"] += 1
        else:
            outcome = "fail"
            note = (
                f"weak Δ={delta_pixels} pred_e={float(hyp.get('effect', 0)):.2f} "
                f"pred_p={predicted:.2f}"
            )
            self.metrics["failures_logged"] += 1

        self.memory.add(
            pending["embed"],
            pending["action"],
            outcome=outcome,
            note=note,
        )
        return {
            "learned": True,
            "loss": loss,
            "effect": effect,
            "outcome": outcome,
            "note": note,
            "lessons": self.memory.summarize(3),
        }

    def reset(self) -> None:
        # Keep CNN/DNN weights (cross-level transfer of perception+predictor);
        # clear episodic failure buffer that is level-dynamics specific.
        self.memory = FailureMemory(capacity=160, dim=self.embed_dim)
        self._embed = None
        self._prev_embed = None
        self._pending = None
        self._step = 0

    def hard_reset(self) -> None:
        """Full wipe including network (new game dynamics)."""
        actions = list(self.net.actions)
        self.__init__(
            num_colors=self.cnn.num_colors,
            embed_dim=self.embed_dim,
            actions=actions,
            prefer_torch=self.prefer_torch,
        )

    def snapshot(self) -> dict:
        return {
            "backend": self._backend,
            "step": self._step,
            "metrics": dict(self.metrics),
            "memory": self.memory.snapshot(),
            "net_updates": self.net.mlp.updates,
        }
