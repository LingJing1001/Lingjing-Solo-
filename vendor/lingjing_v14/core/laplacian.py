"""
灵境引擎 V14.0 — core.laplacian
================================
Laplacian 算子的两种实现（面积律核心）：

1. 体积 7 点离散：O(N³) 全网格，含 1/h²（V12 量纲修正）
2. 泡壁 CSR 稀疏：O(R²) 活动节点，与 N 脱钩

两者在泡壁**内部点**（子图度数 = 6）上逐点一致（max_err ~ 1e-13），
边界层刻意不一致（零通量 ≠ 周期），这是契约的一部分。
"""
from __future__ import annotations
from typing import Optional
import numpy as np
try:
    from scipy.sparse import csr_matrix
except ImportError:  # optional: volume evolution does not require SciPy
    csr_matrix = None  # type: ignore[assignment]


def _require_scipy():
    if csr_matrix is None:
        raise ImportError(
            "SciPy is required for CSR Laplacian/bubble operations; "
            "volume Field evolution only needs NumPy"
        )
    return csr_matrix


def build_volume_laplacian_csr(
    shape: tuple[int, int, int], h: float = 1.0
) -> csr_matrix:
    """
    体积 7 点 Laplacian 的 CSR 矩阵（供泡壁 CSR 对照）。
    仅用于测试/对照，主演化用 field._laplacian_volume（向量化更快）。
    """
    matrix_type = _require_scipy()
    nx, ny, nz = shape
    n = nx * ny * nz
    h2 = h * h
    # 每个内点 7 个非零元；边界用周期
    row, col, data = [], [], []
    for idx in range(n):
        i, j, k = idx // (ny * nz), (idx % (ny * nz)) // nz, idx % nz
        row.append(idx); col.append(idx); data.append(-6.0 / h2)
        for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            ni = (i + di) % nx; nj = (j + dj) % ny; nk = (k + dk) % nz
            nidx = ni * ny * nz + nj * nz + nk
            row.append(idx); col.append(nidx); data.append(1.0 / h2)
    return matrix_type((data, (row, col)), shape=(n, n))


def build_bubble_laplacian_csr(
    mask: np.ndarray, h: float = 1.0
) -> csr_matrix:
    """
    泡壁子图 Laplacian（CSR 稀疏，O(R²) 活动节点）。

    构造：对活动节点（mask=True）的诱导子图，度数 d_v ≤ 6，
    L = D - A，再乘 1/h²（V12 修正：CSR 此前缺 1/h²，已修复）。

    活动节点索引：按 C-order 编号 0..M-1。
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    matrix_type = _require_scipy()
    idxs = np.flatnonzero(mask)  # 活动节点全局索引
    M = len(idxs)
    if M == 0:
        return matrix_type((M, M))
    # 全局 → 局部
    local = {int(g): l for l, g in enumerate(idxs.tolist())}
    nx, ny, nz = mask.shape
    h2 = h * h
    row, col, data = [], [], []
    for g in idxs.tolist():
        i, j, k = g // (ny * nz), (g % (ny * nz)) // nz, g % nz
        l = local[g]
        deg = 0
        for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            # 周期边界
            ni %= nx; nj %= ny; nk %= nz
            ng = ni * ny * nz + nj * nz + nk
            if ng in local:
                deg += 1
                row.append(l); col.append(local[ng]); data.append(1.0 / h2)
        row.append(l); col.append(l); data.append(-float(deg) / h2)
    return matrix_type((data, (row, col)), shape=(M, M))


def laplacian_7point(phi: np.ndarray, h: float = 1.0) -> np.ndarray:
    """向量化体积 7 点 Laplacian（含 1/h²）。主演化使用。"""
    h2 = h * h
    xp, xm = np.roll(phi, -1, 0), np.roll(phi, 1, 0)
    yp, ym = np.roll(phi, -1, 1), np.roll(phi, 1, 1)
    zp, zm = np.roll(phi, -1, 2), np.roll(phi, 1, 2)
    return (xm + xp + ym + yp + zm + zp - 6.0 * phi) / h2


def bubble_mask_spherical(
    shape: tuple[int, int, int], center: tuple[int, int, int], radius: int
) -> np.ndarray:
    """构造球形泡壁掩码（用于面积律测试 / 场景初始化）。"""
    nx, ny, nz = shape
    cx, cy, cz = center
    i = np.arange(nx)[:, None, None]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[None, None, :]
    dist2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
    mask = (dist2 <= radius ** 2)
    return mask
