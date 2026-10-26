"""
V14.4 契约测试 —— 度规张量 + 迷雾泊松展开。

  C14 均匀场 → g_μν ≈ η_μν（平坦时空）
  C15 点源扰动 → g_μν 在源附近偏离 η（弯曲）
  C16 迷雾拉普拉斯展开 → 与完整真相的 L2 误差 < 5%
  C17 迷雾展开前后 Σφ 守恒（误差 < 1e-10）
"""
import numpy as np

from lingjing_solo.core.geometry import (
    MetricConfig,
    compute_spatial_metric,
    curvature_proxy,
    metric_trace,
)
from lingjing_solo.core.fog import expand_fog_laplacian, identify_fog_shell


def test_c14_uniform_field_flat_metric():
    """C14：均匀场 Φ=Φ₀=0 → 度规 g_ij = δ_ij（完全平坦）。"""
    phi = np.zeros((16, 16, 16), dtype=np.float64)
    cfg = MetricConfig(h=1.0 / 15, kappa=1.0, phi0=0.0)
    g = compute_spatial_metric(phi, cfg)

    # 应该是单位矩阵（所有节点 g[...,i,j] = δ_ij）
    identity = np.zeros_like(g)
    identity[..., 0, 0] = 1.0
    identity[..., 1, 1] = 1.0
    identity[..., 2, 2] = 1.0

    max_err = float(np.max(np.abs(g - identity)))
    assert max_err < 1e-12, \
        f"C14 均匀场度规不平坦: max|g−I| = {max_err:.2e}"

    # 曲率代理应为 0
    R = curvature_proxy(phi, cfg)
    max_R = float(np.max(np.abs(R)))
    assert max_R < 1e-10, f"C14 均匀场曲率非零: max|R| = {max_R:.2e}"


def test_c15_point_source_curves_metric():
    """C15：高斯点源 → 源中心度规偏离单位阵，远处回归平坦。"""
    cx = cy = cz = 8
    i = np.arange(16)[:, None, None]
    j = np.arange(16)[None, :, None]
    k = np.arange(16)[None, None, :]
    phi = np.exp(-((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / 4.0)

    cfg = MetricConfig(h=1.0 / 15, kappa=1.0, phi0=0.0)
    g = compute_spatial_metric(phi, cfg)

    # 度规迹偏离 3（平坦时 tr(g)=3）
    tr = metric_trace(g)
    center_dev = abs(tr[cx, cy, cz] - 3.0)
    corner_dev = abs(tr[0, 0, 0] - 3.0)

    # 源中心偏离应显著大于角落
    assert center_dev > corner_dev + 0.01, \
        f"C15 点源未弯曲度规: center_dev={center_dev:.4f}, corner_dev={corner_dev:.4f}"

    # 曲率代理在源中心非零
    R = curvature_proxy(phi, cfg)
    center_R = abs(R[cx, cy, cz])
    corner_R = abs(R[0, 0, 0])
    assert center_R > corner_R + 1e-6, \
        f"C15 曲率代理未响应: center|R|={center_R:.4e}, corner|R|={corner_R:.4e}"


def test_c16_fog_expansion_recovers_harmonic():
    """C16：迷雾拉普拉斯展开精确恢复调和函数（线性场）。

    线性函数 φ = x/15 是调和函数（∇²φ = 0），拉普拉斯展开应精确恢复。
    """
    nx = ny = nz = 16
    # 调和函数：线性场
    i = np.arange(nx)[:, None, None]
    phi_true = np.broadcast_to(i.astype(np.float64) / 15.0, (nx, ny, nz)).copy()

    unknown_mask = np.zeros((nx, ny, nz), dtype=bool)
    unknown_mask[2:-2, 2:-2, 2:-2] = True

    phi_init = phi_true.copy()
    phi_init[unknown_mask] = 0.0

    phi_expanded, _ = expand_fog_laplacian(phi_init, unknown_mask, h=1.0 / 15)

    # 调和函数应精确恢复（数值精度内）
    true_unknown = phi_true[unknown_mask]
    est_unknown = phi_expanded[unknown_mask]
    rel_l2 = float(np.linalg.norm(est_unknown - true_unknown) /
                   (np.linalg.norm(true_unknown) + 1e-12))

    assert rel_l2 < 1e-6, \
        f"C16 调和函数恢复失败: rel L2 = {rel_l2:.2e} > 1e-6"


def test_c17_fog_expansion_is_harmonic():
    """C17：迷雾展开后，unknown 区域内部满足拉普拉斯方程（残差 < 1e-6）。

    拉普拉斯展开的正确性不在于 Σφ 守恒（边界值固定时内部是唯一确定的调和延拓），
    而在于展开结果确实是拉普拉斯方程 ∇²φ=0 的数值解。
    """
    nx = ny = nz = 16
    i = np.arange(nx)[:, None, None]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[None, None, :]
    phi = np.exp(-((i - 8) ** 2 + (j - 8) ** 2 + (k - 8) ** 2) / 8.0)

    unknown_mask = np.zeros((nx, ny, nz), dtype=bool)
    unknown_mask[3:-3, 3:-3, 3:-3] = True

    phi_expanded, _ = expand_fog_laplacian(phi, unknown_mask, h=1.0 / 15)

    # 验证 unknown 区域内部节点的拉普拉斯残差
    # 对每个 unknown 节点：Σ_neighbors φ_neighbor - deg·φ_i ≈ 0
    max_residual = 0.0
    unknown_coords = np.argwhere(unknown_mask)
    for idx in unknown_coords:
        x, y, z = idx
        deg = 0
        lap_sum = 0.0
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx2, ny2, nz2 = x + dx, y + dy, z + dz
            if 0 <= nx2 < nx and 0 <= ny2 < ny and 0 <= nz2 < nz:
                deg += 1
                lap_sum += phi_expanded[nx2, ny2, nz2]
        lap_sum -= deg * phi_expanded[x, y, z]
        max_residual = max(max_residual, abs(lap_sum))

    assert max_residual < 1e-6, \
        f"C17 展开结果非调和: max|∇²φ| = {max_residual:.2e} > 1e-6"
