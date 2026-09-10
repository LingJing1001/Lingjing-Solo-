"""Ch5 · Gram 行列式体积 = 进化多样性度量。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class GramVolume:
    """Track basis vectors; evolve_score = log det(G + εI)."""

    def __init__(self, dim: int = 64, eps: float = 1e-3, max_basis: int = 24):
        self.dim = int(dim)
        self.eps = float(eps)
        self.max_basis = int(max_basis)
        self.basis: List[np.ndarray] = []
        self.history: List[float] = []

    def _pad(self, v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=np.float64).reshape(-1)
        out = np.zeros(self.dim, dtype=np.float64)
        out[: min(self.dim, x.shape[0])] = x[: self.dim]
        n = float(np.linalg.norm(out) + 1e-12)
        return out / n

    def volume(self) -> float:
        if not self.basis:
            return 0.0
        B = np.stack(self.basis, axis=1)  # dim × m
        G = B.T @ B
        G = G + self.eps * np.eye(G.shape[0])
        sign, logdet = np.linalg.slogdet(G)
        if sign <= 0:
            return -1e9
        return float(logdet)

    def try_add(self, v: np.ndarray, min_gain: float = 1e-4) -> bool:
        """Add vector if it increases log-det volume (orthogonal diversity)."""
        u = self._pad(v)
        before = self.volume()
        # residual against current span
        if self.basis:
            B = np.stack(self.basis, axis=1)
            proj = B @ np.linalg.lstsq(B, u, rcond=None)[0]
            resid = u - proj
            if float(np.linalg.norm(resid)) < 0.08:
                return False
            u = resid / (np.linalg.norm(resid) + 1e-12)
        self.basis.append(u.astype(np.float64))
        after = self.volume()
        if after < before + min_gain and len(self.basis) > 1:
            self.basis.pop()
            return False
        if len(self.basis) > self.max_basis:
            # drop lowest leverage: rebuild by greedy volume
            self._prune()
        self.history.append(after)
        return True

    def _prune(self) -> None:
        vecs = list(self.basis)
        self.basis = []
        # greedy: add in order of singular contribution
        for v in vecs:
            self.try_add(v, min_gain=0.0)
            if len(self.basis) >= self.max_basis:
                break

    def gain_if_add(self, v: np.ndarray) -> float:
        before = self.volume()
        ok = self.try_add(v, min_gain=-1e9)
        after = self.volume()
        if ok and self.basis:
            # revert trial
            self.basis.pop()
            if self.history:
                self.history.pop()
        return float(after - before)

    def snapshot(self) -> dict:
        return {
            "basis": len(self.basis),
            "volume": self.volume(),
            "history_tail": self.history[-5:],
        }
