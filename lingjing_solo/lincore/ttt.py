"""SSA-E3 · 谱-TTT：跨局共享基（无 GPU 版测试时训练）。

ARC 冠军路线（MindsAI / the ARChitects）的"测试时训练"本质是：
    预训练先验 → 每任务少量梯度步适配。
SSA 的线性代数版本（凸代理，闭式解）：
    预训练 = 跨游戏全部转移的 PCA（全局 SVD → 世界先验基 V_world）
    测试时适配 = 每局在线岭回归（已有 SubspaceModel）
    每晚离线重训 = 重放 JSONL 里的全部 Δ → 刷新 V_world

数学依据：[S6] §7.3 PCA / Eckart–Young；白皮书 §4 TTT 行、§7 E3。
新颖度对"已见"的判定随之升级：已辨识 = 本局行空间 ∪ 世界先验行空间
（QR 正交化后投影，[S6] §4.4 Gram–Schmidt/QR）。
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


class TransitionReplay:
    """按 JSONL 追加的效应转移重放库（行 = Δ 向量，附游戏签名与时间戳）。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = str(path or os.environ.get("LINCORE_REPLAY_PATH", "") or "")
        self._lock = threading.Lock()

    def append(self, game_sig: str, delta: np.ndarray) -> bool:
        if not self.path or delta is None:
            return False
        d = np.asarray(delta, dtype=np.float64).ravel()
        if not np.all(np.isfinite(d)) or not np.any(d):
            return False
        rec = {"g": str(game_sig), "d": [round(float(x), 6) for x in d.tolist()]}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            return True
        except Exception:
            return False

    def load(self) -> Tuple[np.ndarray, List[str]]:
        """返回 (Δ 矩阵 [m×d], 游戏签名列表)；文件缺失/损坏返回空。"""
        rows: List[np.ndarray] = []
        sigs: List[str] = []
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            rows.append(np.asarray(rec["d"], dtype=np.float64))
                            sigs.append(str(rec.get("g", "unknown")))
                        except Exception:
                            continue
            except Exception:
                pass
        if not rows:
            return np.zeros((0, 0)), []
        d = rows[0].shape[0]
        M = np.vstack([r for r in rows if r.shape[0] == d])
        return M, sigs


@dataclass
class SharedBasis:
    """世界先验基 V_world ∈ R^{k×d}（行 = 主奇异方向，跨游戏共享）。"""

    V: Optional[np.ndarray] = None  # (k, d)
    n_transitions: int = 0
    n_games: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.V is not None and self.V.size > 0

    @classmethod
    def fit(cls, deltas: np.ndarray, k: int = 12, sigs: Optional[List[str]] = None) -> "SharedBasis":
        """全局 PCA：Δ 矩阵的 top-k 右奇异向量（Eckart–Young 最优）。"""
        M = np.asarray(deltas, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] < 4 or M.shape[1] == 0:
            return cls()
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        smax = float(s[0]) if s.size else 0.0
        eps = 1e-8 * max(smax, 1.0)
        kk = max(1, min(int(k), int(np.sum(s > eps))))
        games = len(set(sigs)) if sigs else 0
        return cls(V=Vt[:kk], n_transitions=int(M.shape[0]), n_games=int(games),
                   meta={"sigma_top": round(float(s[0]), 6), "k": kk})

    def save(self, path: str) -> bool:
        if not self.ready:
            return False
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "V": self.V.tolist(),
                "n_transitions": self.n_transitions,
                "n_games": self.n_games,
                "meta": self.meta,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path: str) -> "SharedBasis":
        try:
            if not os.path.exists(path):
                return cls()
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            V = np.asarray(data.get("V"), dtype=np.float64)
            if V.ndim != 2 or V.size == 0 or not np.all(np.isfinite(V)):
                return cls()
            return cls(
                V=V,
                n_transitions=int(data.get("n_transitions", 0)),
                n_games=int(data.get("n_games", 0)),
                meta=dict(data.get("meta") or {}),
            )
        except Exception:
            return cls()


def combined_row_space(bases: Iterable[np.ndarray], max_dim: int = 32) -> Optional[np.ndarray]:
    """合并多个行空间基并 QR 正交化（[S6] §4.4.2 格拉姆-施密特的数值稳定版）。"""
    rows: List[np.ndarray] = []
    for B in bases:
        if B is None:
            continue
        arr = np.asarray(B, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] > 0 and np.all(np.isfinite(arr)):
            rows.append(arr)
    if not rows:
        return None
    stacked = np.vstack(rows)
    dim = stacked.shape[1]
    max_dim = max(1, min(int(max_dim), dim))
    try:
        Q, R = np.linalg.qr(stacked.T)  # 列正交基张成 span(stacked rows)
    except np.linalg.LinAlgError:
        return None
    diag = np.abs(np.diag(R))
    if diag.size and diag.max() > 0:
        keep = diag > 1e-10 * diag.max()
    else:
        keep = np.ones(Q.shape[1], dtype=bool)
    Q = Q[:, keep]
    return Q[:, :max_dim].T  # (≤max_dim, d)


def residual_novelty(vec: np.ndarray, orthonormal_basis: Optional[np.ndarray]) -> float:
    """r(v) = ‖(I − QQᵀ)v‖ / ‖v‖（对正交列基的残差新奇度）。"""
    v = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return 0.0
    if orthonormal_basis is None or orthonormal_basis.size == 0:
        return 1.0
    Q = np.asarray(orthonormal_basis, dtype=np.float64)
    if Q.shape[0] != v.shape[0]:  # 期望 (d, k)
        Q = Q.T if Q.shape[1] == v.shape[0] else None
        if Q is None:
            return 1.0
    proj = Q @ (Q.T @ v)
    residual = v - proj
    r = float(np.linalg.norm(residual)) / norm
    return float(min(1.0, max(0.0, r)))
