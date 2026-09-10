"""Ch7 / Ch10.3 · 协方差、PCA、新奇度（Eckart–Young 残差）。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class CovPCA:
    """Online covariance + top-k PCA basis; novelty = residual energy."""

    def __init__(self, dim: int = 64, k: int = 8, eps: float = 1e-4):
        self.dim = int(dim)
        self.k = int(k)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.C = np.eye(self.dim, dtype=np.float64) * eps  # unnormalized scatter
        self.basis: Optional[np.ndarray] = None  # (dim, k)
        self.evals: Optional[np.ndarray] = None

    def _pad(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.zeros(self.dim, dtype=np.float64)
        out[: min(self.dim, v.shape[0])] = v[: self.dim]
        return out

    def observe(self, x: np.ndarray) -> None:
        v = self._pad(x)
        self.n += 1
        # Welford-ish mean + scatter
        delta = v - self.mean
        self.mean += delta / self.n
        delta2 = v - self.mean
        self.C += np.outer(delta, delta2)

    def refit(self) -> None:
        if self.n < 2:
            return
        cov = self.C / max(1, self.n - 1)
        # symmetric eigendecomposition (PSD)
        cov = 0.5 * (cov + cov.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        kk = min(self.k, self.dim)
        self.evals = evals[:kk].astype(np.float32)
        self.basis = evecs[:, :kk].astype(np.float32)

    def project(self, x: np.ndarray) -> np.ndarray:
        v = self._pad(x).astype(np.float32)
        if self.basis is None:
            self.refit()
        if self.basis is None:
            return np.zeros(self.k, dtype=np.float32)
        return (self.basis.T @ (v - self.mean.astype(np.float32))).astype(np.float32)

    def novelty(self, x: np.ndarray) -> float:
        """‖(I − V Vᵀ)(x−μ)‖ / ‖x−μ‖  — residual subspace distance."""
        v = self._pad(x).astype(np.float32)
        d = v - self.mean.astype(np.float32)
        nrm = float(np.linalg.norm(d) + 1e-8)
        if self.basis is None:
            self.refit()
        if self.basis is None:
            return 1.0
        recon = self.basis @ (self.basis.T @ d)
        resid = float(np.linalg.norm(d - recon))
        return float(np.clip(resid / nrm, 0.0, 1.0))

    def snapshot(self) -> dict:
        return {
            "n": self.n,
            "k": self.k,
            "top_evals": None if self.evals is None else [float(x) for x in self.evals[:5]],
        }
