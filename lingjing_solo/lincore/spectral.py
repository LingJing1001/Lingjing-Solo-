"""SSA-L1/L2 · 稳态谱工具：SVD 岭回归、增量正规方程、周期检测、Krylov 前瞻。

- ridge_solve：SVD 截断解最小二乘（rank-deficient 安全，[S6] §4.4/§4.5）
- RidgeLinear：正规方程 G = XᵀX + λI 流式累加 + 惰性求逆（[S6] §3.4/§4.3）
- detect_period：自相关周期检测 —— 循环矩阵共享傅里叶特征基的离散化体现（[S6] §6.4.6–6.4.9）
- krylov_horizon：线性前瞻算子的 Krylov 序列迭代（[S6] Ch6 幂迭代思想）
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as np


def ridge_solve(A, y, lam: float = 1e-2) -> Optional[np.ndarray]:
    """稳定岭回归：w = V Σ/(Σ²+λ) Uᵀ y（处理秩亏，最小范数解）。"""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if A.ndim != 2 or A.shape[0] == 0:
        return None
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    filt = s / (s * s + max(lam, 1e-12))
    return Vt.T @ (filt * (U.T @ y))


class RidgeLinear:
    """增量最小二乘回归器（小维度、闭式解、无迭代训练）。"""

    def __init__(self, dim: int, lam: float = 1e-1) -> None:
        self.dim = int(dim)
        self.lam = float(lam)
        self.G = np.eye(self.dim) * self.lam
        self.b = np.zeros(self.dim)
        self.n = 0
        self._Ginv: Optional[np.ndarray] = None

    def update(self, x, y: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.dim or not np.isfinite(float(y)):
            return
        self.G += np.outer(x, x)
        self.b += x * float(y)
        self.n += 1
        self._Ginv = None

    def _inverse(self) -> np.ndarray:
        if self._Ginv is None:
            try:
                self._Ginv = np.linalg.inv(self.G)
            except np.linalg.LinAlgError:
                self._Ginv = np.linalg.pinv(self.G)
        return self._Ginv

    def weights(self) -> Optional[np.ndarray]:
        if self.n == 0:
            return None
        return self._inverse() @ self.b

    def predict(self, x) -> tuple:
        """返回 (预测值, 不确定度)；不确定度 = 归一化杠杆 h = xᵀG⁻¹x（远离已见 x → 高）。"""
        w = self.weights()
        x = np.asarray(x, dtype=np.float64).ravel()
        if w is None or x.shape[0] != self.dim:
            return 0.0, 1.0
        val = float(x @ w)
        h = float(x @ (self._inverse() @ x))
        return val, float(min(1.0, max(0.05, h)))


def detect_period(
    series: Sequence[float],
    min_p: int = 2,
    max_p: int = 64,
    corr_thresh: float = 0.45,
    match_thresh: float = 0.7,
) -> Optional[int]:
    """变化序列的周期检测（自相关法，循环矩阵/FFT 视角的离散实现）。

    从短到长扫描滞后，取第一个通过双重校验（归一化自相关 + 逐位匹配率）
    的滞后为基周期 —— 排除倍周期假峰。
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < max(6, 3 * min_p):
        return None
    x = x - x.mean()
    scale = float(np.sqrt(np.mean(x * x)))
    if scale < 1e-9:
        return None
    upper = int(min(max_p, n // 2))
    for p in range(min_p, upper + 1):
        corr = float(np.mean(x[:-p] * x[p:])) / scale
        if corr <= corr_thresh:
            continue
        match = float(
            np.mean(np.abs(x[:-p] - x[p:]) <= 0.5 * scale + 1e-9)
        )
        if match >= match_thresh:
            return p
    return None


def krylov_horizon(
    step_fn: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    goal_axis: np.ndarray,
    max_k: int = 16,
    thresh: float = 0.5,
) -> Optional[int]:
    """Krylov 前瞻：v ← T̂v 逐次迭代，返回目标轴对齐首次超阈值的步数。

    规划深度的线性代数代理：不展开整棵动作树，只估计
    "沿当前主算子走多少步能进入目标方向"（[S6] Krylov 子空间 K_k(T,b) 思想）。
    """
    v = np.asarray(start, dtype=np.float64).ravel()
    g = np.asarray(goal_axis, dtype=np.float64).ravel()
    gn = float(np.linalg.norm(g))
    if v.size == 0 or gn < 1e-12:
        return None
    g = g / gn
    for k in range(1, max_k + 1):
        try:
            v = np.asarray(step_fn(v), dtype=np.float64).ravel()
        except Exception:
            return None
        if v.size != g.size or not np.all(np.isfinite(v)):
            return None
        vn = float(np.linalg.norm(v))
        if vn < 1e-12:
            return None
        if float(v @ g) / vn >= thresh:
            return k
    return None


def robust_rank(
    names: Sequence[str],
    means: Sequence[float],
    uncertainties: Sequence[float],
    z: float = 1.0,
) -> List[str]:
    """minimax 稳健排序：score = mean − z·uncertainty（[S6] §9.4.4 博弈论稳健化）。"""
    rows = sorted(
        zip(names, means, uncertainties),
        key=lambda t: float(t[1]) - z * float(t[2]),
        reverse=True,
    )
    return [n for n, _m, _u in rows]
