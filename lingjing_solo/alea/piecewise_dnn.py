"""Ch9–Ch10 · 分段线性 DNN（ReLU）+ 动量 SGD 反传。

Strang §10.1：深层网 = 分段线性函数 F(v)。
§9.1–9.2：动量 + 链式法则反传。
输出：[effect_logit, progress_logit]
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

DEFAULT_ACTIONS = (
    "ACTION1", "ACTION2", "ACTION3", "ACTION4",
    "ACTION5", "ACTION6", "ACTION7",
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


class PiecewiseDNN:
    """2-hidden-layer ReLU net; online backprop with momentum (heavy ball)."""

    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, int] = (64, 32),
        out_dim: int = 2,
        lr: float = 0.04,
        momentum: float = 0.9,
        l2: float = 1e-4,
        seed: int = 3,
    ):
        self.in_dim = int(in_dim)
        self.h1, self.h2 = int(hidden[0]), int(hidden[1])
        self.out_dim = int(out_dim)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)

        def init(a, b):
            return (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)

        self.W1, self.b1 = init(self.in_dim, self.h1), np.zeros(self.h1, np.float32)
        self.W2, self.b2 = init(self.h1, self.h2), np.zeros(self.h2, np.float32)
        self.W3, self.b3 = init(self.h2, self.out_dim), np.zeros(self.out_dim, np.float32)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)
        self.updates = 0
        self.loss_ema = 0.0

    def _pack_in(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        out = np.zeros(self.in_dim, dtype=np.float32)
        out[: min(self.in_dim, v.shape[0])] = v[: self.in_dim]
        return out

    def forward(self, x: np.ndarray):
        x0 = self._pack_in(x)
        z1 = x0 @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = _relu(z2)
        y = h2 @ self.W3 + self.b3
        return x0, h1, h2, y

    def predict_probs(self, x: np.ndarray) -> Tuple[float, float]:
        y = self.forward(x)[-1]
        p = _sigmoid(y)
        return float(p[0]), float(p[1])

    def update(self, x: np.ndarray, effect: float, progress: float) -> float:
        x0, h1, h2, y = self.forward(x)
        target = np.array(
            [float(np.clip(effect, 0, 1)), float(np.clip(progress, 0, 1))],
            dtype=np.float32,
        )
        p = _sigmoid(y)
        # BCE grad wrt logits: p - t
        dy = p - target
        loss = float(-np.mean(target * np.log(p + 1e-8) + (1 - target) * np.log(1 - p + 1e-8)))

        dW3 = np.outer(h2, dy) + self.l2 * self.W3
        db3 = dy
        dh2 = self.W3 @ dy
        dz2 = dh2 * (h2 > 0)
        dW2 = np.outer(h1, dz2) + self.l2 * self.W2
        db2 = dz2
        dh1 = self.W2 @ dz2
        dz1 = dh1 * (h1 > 0)
        dW1 = np.outer(x0, dz1) + self.l2 * self.W1
        db1 = dz1

        self.vW3 = self.momentum * self.vW3 - self.lr * dW3
        self.vb3 = self.momentum * self.vb3 - self.lr * db3
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1

        self.W3 += self.vW3
        self.b3 += self.vb3
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.updates += 1
        self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        return loss


class ActionPiecewiseHead:
    """embed ⊕ action_onehot ⊕ xy → PiecewiseDNN."""

    def __init__(self, embed_dim: int = 64, actions: Sequence[str] = DEFAULT_ACTIONS):
        self.actions = [a.upper() for a in actions]
        self.embed_dim = int(embed_dim)
        self.in_dim = self.embed_dim + len(self.actions) + 3
        self.net = PiecewiseDNN(self.in_dim, hidden=(64, 32), out_dim=2)

    def _x(
        self,
        embed: np.ndarray,
        action: str,
        xy: Optional[Tuple[int, int]] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        v = np.zeros(self.in_dim, dtype=np.float32)
        e = np.asarray(embed, dtype=np.float32).reshape(-1)
        v[: min(self.embed_dim, e.shape[0])] = e[: self.embed_dim]
        a = str(action).upper()
        if a in self.actions:
            v[self.embed_dim + self.actions.index(a)] = 1.0
        base = self.embed_dim + len(self.actions)
        if xy is not None and shape and shape[0] > 0 and shape[1] > 0:
            v[base] = xy[0] / shape[1]
            v[base + 1] = xy[1] / shape[0]
            v[base + 2] = 1.0
        return v

    def predict(self, embed, action, xy=None, shape=None):
        return self.net.predict_probs(self._x(embed, action, xy, shape))

    def learn(self, embed, action, effect, progressed, xy=None, shape=None):
        return self.net.update(
            self._x(embed, action, xy, shape),
            effect,
            1.0 if progressed else 0.0,
        )
