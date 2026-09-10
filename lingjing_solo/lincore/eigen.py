"""SSA-L4 · 特征技能图（Eigenoptions / empowerment 的闭式线性实现）。

区域×动作效应矩阵 S ∈ R^{B²×K}（[row, col] = 该区域对该动作的效应强度 EMA）。
SVD: S = U Σ Vᵀ（[S6] Ch7）：
  - U 的列 = "特征区域轴"（eigenregions）：屏幕上可控性的主方向
  - V 的列 = "特征动作轴"：哪些动作组合承载同一效应轴
  - σ_j = 第 j 条可控性轴的强度（empowerment 代理）

与 Gregor et al. (eigenoption discovery) 的差异：无策略网络，
特征选项 = 贪心选取 U 列投影最大的未尝试区域 —— 毫秒级、可解释、可证伪。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class EigenSkillMap:
    """流式区域-动作杠杆图 + 特征选项推荐。"""

    def __init__(self, n_regions: int, actions: Sequence[str], decay: float = 0.985) -> None:
        self.actions: List[str] = [str(a).upper() for a in actions]
        self.action_index: Dict[str, int] = {a: i for i, a in enumerate(self.actions)}
        self.S = np.zeros((int(n_regions), max(1, len(self.actions))), dtype=np.float64)
        self.trials = np.zeros((int(n_regions), max(1, len(self.actions))), dtype=np.float64)
        self.decay = float(decay)
        self._u: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def _idx(self, action: str) -> Optional[int]:
        return self.action_index.get(str(action or "").upper())

    def note(self, action: str, region: int, magnitude: float) -> None:
        """记录一次「动作作用于区域」的效应强度（含时间衰减以适应非平稳关卡）。"""
        j = self._idx(action)
        if j is None or not (0 <= int(region) < self.S.shape[0]):
            return
        self.S[:, j] *= self.decay
        self.trials[:, j] *= self.decay
        self.S[int(region), j] += max(0.0, float(magnitude))
        self.trials[int(region), j] += 1.0
        self._u = None

    def eigen_decompose(self, k: int = 4) -> bool:
        if float(self.S.max()) <= 0.0:
            return False
        U, s, _Vt = np.linalg.svd(self.S, full_matrices=False)
        kk = min(k, s.size)
        self._u = U[:, :kk]
        self._sigma = s[:kk]
        return True

    def leverage(self, region: int) -> float:
        """区域的总可控杠杆 ‖S[row]‖（empowerment 代理）。"""
        if not (0 <= int(region) < self.S.shape[0]):
            return 0.0
        return float(np.linalg.norm(self.S[int(region)]))

    def action_leverage(self, action: str) -> float:
        """动作列的最大区域杠杆（该动作在全屏最强效应）。"""
        j = self._idx(action)
        if j is None:
            return 0.0
        return float(np.linalg.norm(self.S[:, j]))

    def eigen_projection(self, region: int, axis: int = 0) -> float:
        """区域在第 axis 条特征区域轴上的投影（特征选项价值）。"""
        if self._u is None and not self.eigen_decompose():
            return 0.0
        if not (0 <= int(region) < self._u.shape[0]):
            return 0.0
        axis = min(axis, self._u.shape[1] - 1)
        return float(self._u[int(region), axis])

    def trial_count(self, region: int, action: Optional[str] = None) -> float:
        if not (0 <= int(region) < self.trials.shape[0]):
            return 0.0
        if action is None:
            return float(self.trials[int(region)].sum())
        j = self._idx(action)
        return float(self.trials[int(region), j]) if j is not None else 0.0

    def rank_candidates(
        self,
        candidates: Sequence[int],
        progress_block: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[int, float]]:
        """对候选区域综合评分并降序返回 [(region, score)]。

        score = 0.40·杠杆 + 0.30·未尝试 + 0.30·特征轴投影
        （progress_block 可选：块级进度对齐 {region: s}，并入 0.25 权重）
        """
        progress_block = progress_block or {}
        cand = [int(c) for c in candidates if 0 <= int(c) < self.S.shape[0]]
        if not cand:
            return []
        lev = np.array([self.leverage(c) for c in cand])
        lmax = float(lev.max()) if lev.size and lev.max() > 0 else 1.0
        eig = np.array([self.eigen_projection(c) for c in cand])
        emax = float(eig.max()) if eig.size and eig.max() > 0 else 1.0
        out: List[Tuple[int, float]] = []
        for i, c in enumerate(cand):
            untried = 1.0 / (1.0 + self.trial_count(c))
            score = 0.40 * lev[i] / lmax + 0.30 * untried + 0.30 * max(0.0, eig[i]) / emax
            if c in progress_block:
                score += 0.25 * float(progress_block[c])
            out.append((c, float(score)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out
