"""SSA-L5 · 梯度在线精调与对偶探索预算（[S6] Ch9 最优化中的线性代数）。

- LogisticSGD：流式逻辑回归（SGD + 动量 + L2，[S6] §9.1.1 最速下降 /
  §9.1.5 动量与重球 / §9.2 随机梯度下降）—— 进度概率预测器。
- DualBudgetScheduler：探索预算的拉格朗日对偶上升（[S6] §9.3–9.4）——
  约束 Σ探索 ≤ B·份额，对偶变量 λ 随预算消耗率上升 → 探索概率自动衰减。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(60.0, z)))
    ez = math.exp(max(-60.0, z))
    return ez / (1.0 + ez)


class LogisticSGD:
    """w ← w − η[(σ(wᵀx) − y)x + λw]，动量 v = μv − ηg（[S6] §9.1.5/§9.2）。"""

    def __init__(self, dim: int, lr: float = 0.08, momentum: float = 0.9, l2: float = 1e-4) -> None:
        self.dim = int(dim)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.l2 = float(l2)
        self.w = np.zeros(self.dim, dtype=np.float64)
        self.v = np.zeros(self.dim, dtype=np.float64)
        self.n = 0

    def predict_proba(self, x) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.dim:
            return 0.5
        return _sigmoid(float(x @ self.w))

    def partial_fit(self, x, y: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.dim or not (0.0 <= float(y) <= 1.0):
            return
        p = self.predict_proba(x)
        grad = (p - float(y)) * x + self.l2 * self.w
        self.v = self.momentum * self.v - self.lr * grad
        self.w = self.w + self.v
        self.n += 1

    @property
    def ready(self) -> bool:
        return self.n >= 5


class DualBudgetScheduler:
    """探索/利用预算的对偶调度器。

    原问题：max 进度 s.t. 探索步数 ≤ B·share。
    对偶上升：λ ← max(0, λ + η·(已用探索比例 − 已过时间比例))；
    探索门 p = σ(base − λ)：预算超支越快，探索越快收敛到利用。
    """

    def __init__(self, horizon: int = 800, explore_share: float = 0.25, eta: float = 0.03) -> None:
        self.horizon = max(1, int(horizon))
        self.budget = max(1.0, self.horizon * float(explore_share))
        self.used = 0
        self.t = 0
        self.lam = 0.0
        self.eta = float(eta)

    def explore_gate(self, base: float = 0.6) -> float:
        """当前探索概率 p = σ(base − λ)（λ=0 时 ≈ 0.65，超支后指数衰减）。"""
        return _sigmoid(base - self.lam)

    def step(self, explored: bool) -> None:
        self.t += 1
        if explored:
            self.used += 1
        used_ratio = self.used / self.budget
        time_ratio = self.t / self.horizon
        self.lam = max(0.0, self.lam + self.eta * (used_ratio - time_ratio))
        self.lam = float(min(self.lam, 8.0))  # 有界：绝不把探索压到 0（保持最小好奇）

    @property
    def pressure(self) -> float:
        """对偶压力（诊断用）。"""
        return float(self.lam)
