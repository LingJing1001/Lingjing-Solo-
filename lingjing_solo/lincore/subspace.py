"""SSA-L2 · 四大基本子空间世界模型（[S6] Ch3/4/7 的认知化）。

效应矩阵 E ∈ R^{m×d}：第 i 行 = 第 i 次「动作→效应」Δ_i = φ(x_{t+1}) − φ(x_t)。
SVD: E = U Σ Vᵀ 后：
  C(E)  列空间   = 已证实的效应能力边界
  N(E)  零空间   = 不变量 / 背景法则（从未变化的坐标）
  C(Eᵀ) 行空间   = 效应主轴（可预测方向）
  N(Eᵀ) 左零空间 = 冗余实验（重复信息，不奖励重复）

三信号（SSA 白皮书 §0）：
  r(Δ) 惊讶   = ‖(I − V_k V_kᵀ)Δ‖ / ‖Δ‖       → 探索采样
  g(a) 可控   = EMA‖Δ_a‖                        → 因果杠杆
  s(Δ) 进度   = ⟨Δ, p*⟩/(‖Δ‖‖p*‖+ε)            → 目标对齐
其中 p* 由岭回归 w = argmin‖Ew − y‖² + λ‖w‖² 给出，y = 关卡进度指示
（[S6] §4.3 最小二乘 + §6.3.7 最优化与机器学习；SVD 稳定解见 spectral.ridge_solve）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class SubspaceModel:
    """在线四子空间世界模型：流式更新、按需 SVD、进度轴、三信号。"""

    def __init__(
        self,
        dim: int,
        k: int = 8,
        max_rows: int = 512,
        ridge: float = 1e-2,
        invariant_tol: float = 1e-9,
    ) -> None:
        self.dim = int(dim)
        self.k = max(1, min(int(k), self.dim))
        self.max_rows = int(max_rows)
        self.ridge = float(ridge)
        self.invariant_tol = float(invariant_tol)

        self.buffer = np.zeros((0, self.dim), dtype=np.float64)
        self.progress_flags = np.zeros(0, dtype=np.float64)
        self._coord_energy = np.zeros(self.dim, dtype=np.float64)
        self._n_total = 0
        self._svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._dirty = True

        # 每动作效应强度 EMA（可控增益 g(a) 的流式估计）
        self.action_gain: Dict[str, float] = {}
        self.action_count: Dict[str, int] = {}

    # ---------- 观测 ----------

    def update(self, delta, progressed: bool = False) -> bool:
        """追加一条效应样本；超过容量时保留较新一半（侧重近期动力学 + 有界算力）。"""
        if delta is None:
            return False
        d = np.asarray(delta, dtype=np.float64).ravel()
        if d.shape[0] != self.dim or not np.all(np.isfinite(d)):
            return False
        if self.buffer.shape[0] >= self.max_rows:
            keep = max(1, self.max_rows // 2)
            self.buffer = self.buffer[-keep:]
            self.progress_flags = self.progress_flags[-keep:]
        self.buffer = np.vstack([self.buffer, d[None, :]])
        self.progress_flags = np.append(self.progress_flags, 1.0 if progressed else 0.0)
        self._coord_energy += d * d
        self._n_total += 1
        self._dirty = True
        return True

    def note_action(self, action: str, delta_norm: float) -> None:
        a = str(action or "").upper()
        ema = self.action_gain.get(a, 0.0)
        self.action_gain[a] = 0.8 * ema + 0.2 * max(0.0, float(delta_norm))
        self.action_count[a] = self.action_count.get(a, 0) + 1

    # ---------- 谱分解 ----------

    def fit(self, force: bool = False) -> bool:
        """计算/刷新 top-k SVD（Eckart–Young 最优截断，[S6] §7.1.2）。

        关键：只保留非零奇异值的右奇异向量 —— C(Eᵀ) 由它们张成；
        数值秩亏补出的正交方向属于 N(E) 一侧，若计入会把"从未见过"
        误判为"已辨识"（novelty 失真）。
        """
        if not self._dirty and not force:
            return False
        m = self.buffer
        if m.shape[0] == 0:
            self._svd = None
            self._dirty = False
            return True
        U, s, Vt = np.linalg.svd(m, full_matrices=False)
        smax = float(s[0]) if s.size else 0.0
        eps = 1e-8 * max(smax, 1.0)
        kk = int(np.sum(s > eps))
        kk = max(1, min(self.k, kk))
        self._svd = (U[:, :kk], s[:kk], Vt[:kk])
        self._dirty = False
        return True

    @property
    def n_samples(self) -> int:
        return self._n_total

    def row_basis(self) -> Optional[np.ndarray]:
        """当前已辨识行空间基 Vk（k×d，仅含非零奇异方向）；无数据返回 None。"""
        self.fit()
        if not self._svd:
            return None
        return self._svd[2]

    @property
    def effective_rank(self) -> int:
        """稳定秩：奇异能量占比 99% 所需维数（能力边界的维数）。"""
        self.fit()
        if not self._svd:
            return 0
        s = self._svd[1]
        if s.size == 0 or s[0] <= 0:
            return 0
        energy = np.cumsum(s * s)
        total = energy[-1]
        if total <= 0:
            return 0
        return int(np.searchsorted(energy, 0.99 * total) + 1)

    # ---------- 三信号 ----------

    def novelty(self, delta) -> float:
        """r(Δ)：Δ 落在已辨识行空间 C(Eᵀ) 之外的程度（残差子空间距离）。"""
        if delta is None:
            return 0.0
        d = np.asarray(delta, dtype=np.float64).ravel()
        if d.shape[0] != self.dim or not np.all(np.isfinite(d)):
            return 0.0
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            return 0.0
        if not self._svd:
            return 1.0  # 无任何已辨识方向 → 完全新奇
        _U, _s, Vk = self._svd  # Vk: (k, d)
        proj = Vk @ d           # (k,) 行空间坐标
        residual = d - (Vk.T @ proj)  # 行空间重构后的正交补残差
        r = float(np.linalg.norm(residual)) / norm
        return float(min(1.0, max(0.0, r)))

    def progress_axis(self) -> Optional[np.ndarray]:
        """进度方向 p*：岭回归 SVD 稳定解 w = V Σ/(Σ²+λ) Uᵀ y（[S6] §4.3/§4.5）。"""
        self.fit()
        m = self.buffer
        if m.shape[0] < 3:
            return None
        y = self.progress_flags
        if float(y.max()) <= 0.0:
            return None
        U, s, Vt = np.linalg.svd(m, full_matrices=False)
        filt = s / (s * s + self.ridge)
        w = Vt.T @ (filt * (U.T @ y))
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-12:
            return None
        return w / nrm

    def progress_alignment(self, delta, axis: Optional[np.ndarray] = None) -> float:
        """s(Δ) = ⟨Δ, p*⟩/(‖Δ‖‖p*‖+ε)，无进度轴时返回 0（诚实退化）。"""
        if delta is None:
            return 0.0
        if axis is None:
            axis = self.progress_axis()
        if axis is None:
            return 0.0
        d = np.asarray(delta, dtype=np.float64).ravel()
        if d.shape[0] != self.dim:
            return 0.0
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            return 0.0
        return float(np.clip(d @ axis / norm, -1.0, 1.0))

    def action_gain_score(self, action: str) -> float:
        """g(a)：每动作平均效应强度的 EMA（empowerment 的确定性代理）。"""
        return float(self.action_gain.get(str(action or "").upper(), 0.0))

    # ---------- 不变量 ----------

    def coord_energy(self) -> np.ndarray:
        return self._coord_energy.copy()

    def invariant_dims(self) -> np.ndarray:
        """零空间支撑集：能量相对峰值近零的坐标（保守：只排除确证不变者）。"""
        e = self._coord_energy
        if e.size == 0 or self._n_total < 3:
            return np.zeros(e.size, dtype=bool)
        peak = float(e.max())
        if peak <= 0:
            return np.ones(e.size, dtype=bool)
        return e <= self.invariant_tol * peak

    def invariant_block_mask(self, blocks: int = 8) -> np.ndarray:
        """块级不变掩码（False = 已证不变/死区）：占用与多样性维联合判定。"""
        dead = self.invariant_dims()
        b2 = blocks * blocks
        if dead.size < 2 * b2:
            return np.ones((blocks, blocks), dtype=bool)
        occ_live = dead[:b2]
        div_live = dead[b2:2 * b2]
        live = ~(occ_live & div_live)  # 占用与多样性都近零才算死块
        return live.reshape(blocks, blocks)
