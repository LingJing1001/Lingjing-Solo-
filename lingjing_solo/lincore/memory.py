"""SSA-L5 · 矩阵记忆：跨游戏低秩补全迁移（[S6] §4.5 广义逆 + §7 SVD）。

游戏×统计矩阵 R（行 = 游戏签名，列 = 动作族统计量）。
新游戏先验 = 行空间内的最小范数插值：
  1) 签名同族 → 直接取该行（最优先）
  2) 否则迭代 SVD 硬阈值补全缺失项（收敛于观测约束下秩-k 最小 Frobenius 解）
最小范数保证：迁移绝不臆造旧数据张成子空间之外的新方向（防迁移幻觉，P3 命题）。

进程级单例：同一 Kaggle 会话内多局共享；可选 JSON 持久化（环境变量 LINCORE_MEMORY_PATH）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

COLUMNS: tuple = (
    "a1_gain", "a2_gain", "a3_gain", "a4_gain",
    "a5_gain", "a6_gain", "progress_corridor", "progress_click",
)
_COL_I = {c: i for i, c in enumerate(COLUMNS)}


class MatrixMemory:
    """跨游戏统计记忆 + 低秩补全先验。"""

    def __init__(self, rank: int = 2) -> None:
        self.rows: Dict[str, List[float]] = {}
        self.rank = max(1, int(rank))

    # ---------- 记录 ----------

    def record(self, game_sig: str, updates: Dict[str, float]) -> None:
        key = str(game_sig or "unknown")
        row = self.rows.setdefault(key, [float("nan")] * len(COLUMNS))
        for col, val in (updates or {}).items():
            i = _COL_I.get(col)
            if i is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if np.isnan(v):
                continue
            row[i] = v if np.isnan(row[i]) else 0.7 * row[i] + 0.3 * v

    # ---------- 检索 ----------

    def _matrix(self) -> tuple:
        sigs = sorted(self.rows.keys())
        if not sigs:
            return sigs, None, None
        M = np.vstack([np.asarray(self.rows[s], dtype=np.float64) for s in sigs])
        return sigs, M, ~np.isnan(M)

    def prior_for(self, game_sig: str) -> Dict[str, float]:
        """新游戏先验：本族已观测值 > 低秩补全 > 极相似邻居行 > 全局均值。

        部分观测行必须走补全路径（其自身观测项即约束），余弦捷径只用于
        完全未知且与某行几乎同向的情形。
        """
        sigs, M, W = self._matrix()
        if M is None or M.shape[0] == 0:
            return {}
        key = str(game_sig or "unknown")
        if key in sigs:
            i = sigs.index(key)
            vec = M[i]
            if not np.isnan(vec).any():
                return self._to_dict(vec)
            completed = self._complete(M, W)
            if completed is not None:
                return self._to_dict(completed[i])
            return self._to_dict(vec)
        # 完全未知的游戏：几乎同向 → 取邻居行；否则全局补全均值行
        best_row, best_sim = None, 0.0
        for i, s in enumerate(sigs):
            r = M[i]
            if np.isnan(r).any():
                continue
            na = float(np.linalg.norm(r))
            if na < 1e-9:
                continue
            sim = 1.0  # 与"零观测新游戏"同向性以行自身置信度衡量
            conf = sim * min(1.0, na / 4.0)
            if conf > best_sim:
                best_sim, best_row = conf, r
        if best_row is not None and best_sim >= 0.9:
            return self._to_dict(best_row)
        completed = self._complete(M, W)
        if completed is None:
            return {}
        return self._to_dict(np.nanmean(completed, axis=0))

    def _complete(self, M: np.ndarray, W: np.ndarray, iters: int = 200) -> Optional[np.ndarray]:
        """迭代 SVD 硬阈值补全：X ← 观测约束投影 ∘ 秩-k 截断。

        k 必须严格小于完全观测锚定行数，补全才有纠错能力（k=锚定行数时
        可精确表示任意缺失初始化，迭代不收敛到结构解）。
        硬阈值迭代线性收敛：给足迭代数与小容差直到不动点。
        """
        if W is None or not W.any():
            return None
        if W.all():
            return M
        anchor = int(np.sum(~np.isnan(M).any(axis=1)))
        k = max(1, min(self.rank, anchor - 1)) if anchor >= 2 else 1
        k = max(1, min(k, min(M.shape) - 1))
        col_mean = np.nanmean(np.where(W, M, np.nan), axis=0)
        col_mean = np.nan_to_num(col_mean)
        X = np.where(W, M, col_mean[None, :].repeat(M.shape[0], axis=0))
        for _ in range(iters):
            try:
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
            except np.linalg.LinAlgError:
                return None
            Xk = (U[:, :k] * s[:k]) @ Vt[:k]
            Xn = np.where(W, M, Xk)
            if np.max(np.abs(Xn - X)) < 1e-10:
                X = Xn
                break
            X = Xn
        return X

    @staticmethod
    def _to_dict(vec: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for col, i in _COL_I.items():
            v = float(vec[i]) if i < vec.size else float("nan")
            if not np.isnan(v):
                out[col] = round(v, 6)
        return out

    # ---------- 持久化 / 单例 ----------

    def to_dict(self) -> dict:
        return {"rank": self.rank, "rows": {k: list(v) for k, v in self.rows.items()}}

    def from_dict(self, data: dict) -> None:
        try:
            self.rank = max(1, int(data.get("rank", self.rank)))
            for k, row in (data.get("rows") or {}).items():
                vec = [float("nan")] * len(COLUMNS)
                for i, v in enumerate(list(row)[: len(COLUMNS)]):
                    try:
                        vec[i] = float(v)
                    except (TypeError, ValueError):
                        vec[i] = float("nan")
                self.rows[str(k)] = vec
        except Exception:
            pass

    _shared: Optional["MatrixMemory"] = None

    @classmethod
    def shared(cls) -> "MatrixMemory":
        """进程级共享记忆（Kaggle 一次 notebook 跑多局时跨局积累）。"""
        if cls._shared is not None:
            return cls._shared
        mem = cls()
        path = os.environ.get("LINCORE_MEMORY_PATH", "")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    mem.from_dict(json.load(f))
            except Exception:
                pass
        cls._shared = mem
        return mem

    def flush(self) -> None:
        path = os.environ.get("LINCORE_MEMORY_PATH", "")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
        except Exception:
            pass
