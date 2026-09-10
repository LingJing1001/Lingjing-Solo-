"""Online DNN heads (pure numpy): effect / progress / epistemic scoring.

Two-layer MLP with SGD + momentum. Used as:
  z_state ⊕ action_onehot ⊕ click_xy_norm  →  [effect_hat, progress_hat]
and a sibling head for action preference logits.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ACTIONS = (
    "ACTION1", "ACTION2", "ACTION3", "ACTION4",
    "ACTION5", "ACTION6", "ACTION7",
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


class OnlineMLP:
    """Two-layer MLP trained online with SGD+momentum; L2 regularization."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 2,
        lr: float = 0.05,
        momentum: float = 0.85,
        l2: float = 1e-4,
        seed: int = 11,
    ):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / max(1, self.in_dim))
        s2 = np.sqrt(2.0 / max(1, self.hidden))
        self.w1 = (rng.standard_normal((self.in_dim, self.hidden)) * s1).astype(np.float32)
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((self.hidden, self.out_dim)) * s2).astype(np.float32)
        self.b2 = np.zeros(self.out_dim, dtype=np.float32)
        self.vw1 = np.zeros_like(self.w1)
        self.vb1 = np.zeros_like(self.b1)
        self.vw2 = np.zeros_like(self.w2)
        self.vb2 = np.zeros_like(self.b2)
        self.updates = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.in_dim:
            xx = np.zeros(self.in_dim, dtype=np.float32)
            n = min(self.in_dim, x.shape[0])
            xx[:n] = x[:n]
            x = xx
        h_pre = x @ self.w1 + self.b1
        h = _relu(h_pre)
        y = h @ self.w2 + self.b2
        return x, h, y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[2]

    def update(self, x: np.ndarray, target: np.ndarray) -> float:
        """MSE on outputs; returns loss."""
        x, h, y = self.forward(x)
        t = np.asarray(target, dtype=np.float32).reshape(-1)
        if t.shape[0] != self.out_dim:
            tt = np.zeros(self.out_dim, dtype=np.float32)
            tt[: min(self.out_dim, t.shape[0])] = t[: self.out_dim]
            t = tt
        err = y - t
        loss = float(0.5 * np.dot(err, err))
        # backprop
        dy = err  # dL/dy
        dw2 = np.outer(h, dy) + self.l2 * self.w2
        db2 = dy
        dh = self.w2 @ dy
        dh_pre = dh * (h > 0)
        dw1 = np.outer(x, dh_pre) + self.l2 * self.w1
        db1 = dh_pre
        # momentum SGD
        self.vw2 = self.momentum * self.vw2 - self.lr * dw2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.vw1 = self.momentum * self.vw1 - self.lr * dw1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.w2 = (self.w2 + self.vw2).astype(np.float32)
        self.b2 = (self.b2 + self.vb2).astype(np.float32)
        self.w1 = (self.w1 + self.vw1).astype(np.float32)
        self.b1 = (self.b1 + self.vb1).astype(np.float32)
        self.updates += 1
        return loss


class ActionEffectNet:
    """State embedding + action (+ optional click) → predicted effect & progress."""

    def __init__(
        self,
        embed_dim: int = 64,
        actions: Sequence[str] = DEFAULT_ACTIONS,
        hidden: int = 64,
    ):
        self.actions = [str(a).upper() for a in actions]
        self.action_index = {a: i for i, a in enumerate(self.actions)}
        self.embed_dim = int(embed_dim)
        self.in_dim = self.embed_dim + len(self.actions) + 3  # + xy_norm + has_xy
        self.mlp = OnlineMLP(self.in_dim, hidden=hidden, out_dim=2, lr=0.06)
        self.loss_ema = 0.0

    def _pack(
        self,
        embed: np.ndarray,
        action: str,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        z = np.asarray(embed, dtype=np.float32).reshape(-1)
        vec = np.zeros(self.in_dim, dtype=np.float32)
        n = min(self.embed_dim, z.shape[0])
        vec[:n] = z[:n]
        a = str(action or "").upper()
        if a in self.action_index:
            vec[self.embed_dim + self.action_index[a]] = 1.0
        base = self.embed_dim + len(self.actions)
        if xy is not None and shape is not None and shape[0] > 0 and shape[1] > 0:
            vec[base] = float(xy[0]) / float(shape[1])
            vec[base + 1] = float(xy[1]) / float(shape[0])
            vec[base + 2] = 1.0
        return vec

    def predict(
        self,
        embed: np.ndarray,
        action: str,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        y = self.mlp.predict(self._pack(embed, action, xy, shape))
        return float(_sigmoid(y[0:1])[0]), float(_sigmoid(y[1:2])[0])

    def learn(
        self,
        embed: np.ndarray,
        action: str,
        effect: float,
        progressed: bool,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> float:
        # train in logit space via targets in [0,1] mapped through soft labels
        # use raw MSE against [effect, progress] then sigmoid at predict time:
        # easier: store targets as logit-friendly soft values in [0,1] and
        # train output as pre-sigmoid via BCE-ish approx: target in [0,1],
        # use y as logits → convert target to soft logit via inverse-sigmoid clamp.
        eff = float(np.clip(effect, 0.0, 1.0))
        prog = 1.0 if progressed else 0.0
        # train outputs as probabilities directly (linear head + clip in loss targets)
        target = np.array([eff, prog], dtype=np.float32)
        # map network raw output through sigmoid in loss by transforming:
        x = self._pack(embed, action, xy, shape)
        _, _, y = self.mlp.forward(x)
        # soft BCE gradient: (sigmoid(y) - t)
        p = _sigmoid(y)
        # fake a linear target so OnlineMLP MSE ≈ BCE: set t_lin = y - (p-t)
        # then err = y - t_lin = p - t  ✓
        t_lin = y - (p - target)
        loss = self.mlp.update(x, t_lin)
        self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        return loss


class FailureMemory:
    """Summarize failed experiments; boost epistemic drive near similar states."""

    def __init__(self, capacity: int = 128, dim: int = 64):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.embeds: List[np.ndarray] = []
        self.actions: List[str] = []
        self.outcomes: List[str] = []  # "fail" | "noop" | "success"
        self.notes: List[str] = []

    def add(
        self,
        embed: np.ndarray,
        action: str,
        *,
        outcome: str,
        note: str = "",
    ) -> None:
        z = np.asarray(embed, dtype=np.float32).reshape(-1)
        zz = np.zeros(self.dim, dtype=np.float32)
        zz[: min(self.dim, z.shape[0])] = z[: self.dim]
        self.embeds.append(zz)
        self.actions.append(str(action).upper())
        self.outcomes.append(outcome)
        self.notes.append(note[:120])
        if len(self.embeds) > self.capacity:
            self.embeds = self.embeds[-self.capacity :]
            self.actions = self.actions[-self.capacity :]
            self.outcomes = self.outcomes[-self.capacity :]
            self.notes = self.notes[-self.capacity :]

    def failure_penalty(self, embed: np.ndarray, action: str) -> float:
        """Higher = more similar to past failures for this action → avoid / explore alt."""
        if not self.embeds:
            return 0.0
        z = np.asarray(embed, dtype=np.float32).reshape(-1)
        zz = np.zeros(self.dim, dtype=np.float32)
        zz[: min(self.dim, z.shape[0])] = z[: self.dim]
        a = str(action).upper()
        pen = 0.0
        for e, act, out in zip(self.embeds, self.actions, self.outcomes):
            if act != a or out == "success":
                continue
            sim = float(np.dot(zz, e))  # both ~unit
            if sim > 0.55:
                pen += (sim - 0.55) * (1.5 if out == "fail" else 0.8)
        return float(min(2.0, pen))

    def success_bonus(self, embed: np.ndarray, action: str) -> float:
        if not self.embeds:
            return 0.0
        z = np.asarray(embed, dtype=np.float32).reshape(-1)
        zz = np.zeros(self.dim, dtype=np.float32)
        zz[: min(self.dim, z.shape[0])] = z[: self.dim]
        a = str(action).upper()
        bonus = 0.0
        for e, act, out in zip(self.embeds, self.actions, self.outcomes):
            if act != a or out != "success":
                continue
            sim = float(np.dot(zz, e))
            if sim > 0.55:
                bonus += (sim - 0.55) * 1.2
        return float(min(2.0, bonus))

    def summarize(self, top: int = 5) -> List[Dict[str, str]]:
        """Human-readable recent failure lessons (for logs / skills)."""
        out = []
        for e, a, o, n in zip(
            self.embeds[::-1], self.actions[::-1], self.outcomes[::-1], self.notes[::-1]
        ):
            if o == "success":
                continue
            out.append({"action": a, "outcome": o, "note": n or "no_effect"})
            if len(out) >= top:
                break
        return out

    def snapshot(self) -> dict:
        return {
            "n": len(self.embeds),
            "fails": sum(1 for o in self.outcomes if o != "success"),
            "successes": sum(1 for o in self.outcomes if o == "success"),
            "lessons": self.summarize(3),
        }
