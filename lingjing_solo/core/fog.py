"""
灵境引擎 V14.4 — core.fog
============================
迷雾区展开（v13.2 协议 9）。

当新区域从迷雾进入清晰区时，内部值未知，只有边界已知。
用拉普拉斯方程 ∇²φ = 0（无内源）在 Dirichlet 边界条件下求解，
得到内部平滑插值（最大熵 / 最小曲率重构）。

面积律保证：展开只发生在新进入清晰区的薄壳层，O(R²) 而非 O(N³)。

守恒：展开前后 Σφ 差值应 < ε（拉普拉斯方程是调和函数，
边界值固定时内部插值是边界值的加权平均，不创造新信息量）。
"""
from __future__ import annotations
import numpy as np


def expand_fog_laplacian(
    phi: np.ndarray,
    unknown_mask: np.ndarray,
    h: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    迷雾区拉普拉斯展开。

    参数：
        phi: 当前场。boundary 节点已有值；unknown_mask 内节点值待求解。
        unknown_mask: bool ndarray，True = 待展开节点（值未知）。
                      边界节点 = ~unknown_mask 中与 unknown_mask 相邻的节点。
        h: 网格间距。

    返回：
        (phi_new, conservation_error)
        phi_new: 展开后的场（unknown 节点被拉普拉斯插值填充）。
        conservation_error: |Σφ_new − Σφ_old|（应 < ε）。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    nx, ny, nz = phi.shape
    n_total = nx * ny * nz

    # 索引映射
    unknown_flat = np.flatnonzero(unknown_mask.ravel())
    n_unknown = len(unknown_flat)

    if n_unknown == 0:
        return phi.copy(), 0.0

    # 全局索引 → 局部索引（仅未知节点）
    local_idx = {int(g): i for i, g in enumerate(unknown_flat.tolist())}

    rows, cols, vals = [], [], []
    b = np.zeros(n_unknown, dtype=np.float64)

    for g in unknown_flat.tolist():
        i = g // (ny * nz)
        j = (g % (ny * nz)) // nz
        k = g % nz
        li = local_idx[g]

        deg = 0  # 总度数（所有域内邻居，含 boundary）
        for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            ng = ni * ny * nz + nj * nz + nk
            deg += 1  # 所有域内邻居都计入总度数
            if ng in local_idx:
                # 未知邻居：进入矩阵
                rows.append(li); cols.append(local_idx[ng]); vals.append(1.0 / h ** 2)
            else:
                # 已知边界节点：值移到右端
                b[li] -= phi.ravel()[ng] / h ** 2

        # 对角元：-总度数/h²（拉普拉斯方程 Σ_neighbors(φ_neighbor - φ_i) = 0）
        rows.append(li); cols.append(li); vals.append(-float(deg) / h ** 2)

    A = csr_matrix((vals, (rows, cols)), shape=(n_unknown, n_unknown))

    # 求解 A·x = b（拉普拉斯方程 Dirichlet）
    x = spsolve(A.tocsr(), b)

    # 填回
    phi_new = phi.copy()
    phi_flat = phi_new.ravel()
    for li, g in enumerate(unknown_flat.tolist()):
        phi_flat[g] = x[li]

    phi_new = phi_flat.reshape(phi.shape)

    # 守恒误差
    conservation_error = float(abs(np.sum(phi_new) - np.sum(phi)))

    return phi_new, conservation_error


def identify_fog_shell(
    clear_mask: np.ndarray,
    new_clear_mask: np.ndarray,
) -> np.ndarray:
    """
    识别新进入清晰区的薄壳层（面积律 O(R²)）。

    参数：
        clear_mask: 之前的清晰区。
        new_clear_mask: 扩展后的清晰区。

    返回：
        shell_mask: True = 新进入的节点（待展开）。
    """
    shell = new_clear_mask & ~clear_mask
    return shell
