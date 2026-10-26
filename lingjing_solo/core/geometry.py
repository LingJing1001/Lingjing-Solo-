"""
灵境引擎 V14.4 — core.geometry
================================
度规张量生成（v13.2 §3）。

公式（v13.2）：
    g_μν = η_μν + (1/I_Pl)·∂_μΦ·∂_νΦᵀ + κ·(Φ²−Φ₀²)·η_μν

其中：
    η_μν    = 欧几里得度规 diag(1,1,1,1)（信息场不做相对论洛伦兹号差）
    I_Pl    = Planck 尺度（自然单位下取 1.0）
    κ       = 曲率耦合常数
    Φ₀      = 真空基线

空间度规 g_ij (i,j ∈ {1,2,3})：
    g_ij = (1 + κ·(Φ²−Φ₀²))·δ_ij + (1/I_Pl)·∂_iΦ·∂_jΦ

曲率代理（弱场 Ricci scalar 近似）：
    R ≈ −∇²[κ·(Φ²−Φ₀²) + (1/(2·I_Pl))·‖∇Φ‖²]
    均匀场 R=0；点源附近 R 偏离 0。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricConfig:
    """度规计算配置。"""
    h: float = 1.0           # 网格间距（与 FieldConfig.h 一致）
    I_Pl: float = 1.0        # Planck 尺度（自然单位）
    kappa: float = 1.0       # 曲率耦合 κ
    phi0: float = 0.0        # 真空基线 Φ₀


def spatial_gradient(phi: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 Φ 的空间梯度 (∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z)。
    内部中心差分，边界单侧差分（与 Neumann 一致）。
    薄切片（某方向 N=1）时该方向梯度为 0（无信息）。

    返回三个 (Nx,Ny,Nz) 数组。
    """
    Nx, Ny, Nz = phi.shape

    # x 方向
    dphi_dx = np.zeros_like(phi)
    if Nx >= 2:
        dphi_dx[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / h
        dphi_dx[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / h
    if Nx >= 3:
        dphi_dx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h)

    # y 方向
    dphi_dy = np.zeros_like(phi)
    if Ny >= 2:
        dphi_dy[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / h
        dphi_dy[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / h
    if Ny >= 3:
        dphi_dy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * h)

    # z 方向
    dphi_dz = np.zeros_like(phi)
    if Nz >= 2:
        dphi_dz[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / h
        dphi_dz[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / h
    if Nz >= 3:
        dphi_dz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * h)

    return dphi_dx, dphi_dy, dphi_dz


def compute_spatial_metric(phi: np.ndarray, cfg: MetricConfig) -> np.ndarray:
    """
    计算空间度规 g_ij (i,j=1,2,3)。

    g_ij = (1 + κ·(Φ²−Φ₀²))·δ_ij + (1/I_Pl)·∂_iΦ·∂_jΦ

    返回 shape (Nx,Ny,Nz,3,3) 的 ndarray。
    g[...,i,j] = g_ij 在该节点的值。
    """
    dphi_dx, dphi_dy, dphi_dz = spatial_gradient(phi, cfg.h)

    # 标量标量项：(1 + κ·(Φ²−Φ₀²))·δ_ij
    scalar = 1.0 + cfg.kappa * (phi ** 2 - cfg.phi0 ** 2)

    # 梯度外积项：(1/I_Pl)·∂_iΦ·∂_jΦ
    g = np.zeros(phi.shape + (3, 3), dtype=np.float64)
    # 对角元
    g[..., 0, 0] = scalar + (1.0 / cfg.I_Pl) * dphi_dx ** 2
    g[..., 1, 1] = scalar + (1.0 / cfg.I_Pl) * dphi_dy ** 2
    g[..., 2, 2] = scalar + (1.0 / cfg.I_Pl) * dphi_dz ** 2
    # 非对角元（对称）
    g[..., 0, 1] = g[..., 1, 0] = (1.0 / cfg.I_Pl) * dphi_dx * dphi_dy
    g[..., 0, 2] = g[..., 2, 0] = (1.0 / cfg.I_Pl) * dphi_dx * dphi_dz
    g[..., 1, 2] = g[..., 2, 1] = (1.0 / cfg.I_Pl) * dphi_dy * dphi_dz

    return g


def curvature_proxy(phi: np.ndarray, cfg: MetricConfig) -> np.ndarray:
    """
    弱场 Ricci scalar 近似（曲率代理场）。

    R ≈ −∇²[κ·(Φ²−Φ₀²) + (1/(2·I_Pl))·‖∇Φ‖²]

    均匀场 → R=0；点源附近 → R 偏离 0。
    返回 shape (Nx,Ny,Nz) 的 ndarray。
    """
    dphi_dx, dphi_dy, dphi_dz = spatial_gradient(phi, cfg.h)

    # 标量曲率势：V = κ·(Φ²−Φ₀²) + (1/(2·I_Pl))·‖∇Φ‖²
    grad_sq = dphi_dx ** 2 + dphi_dy ** 2 + dphi_dz ** 2
    V = cfg.kappa * (phi ** 2 - cfg.phi0 ** 2) + (1.0 / (2.0 * cfg.I_Pl)) * grad_sq

    # ∇²V（7 点 Laplacian，Neumann 边界）
    h2 = cfg.h ** 2
    nx, ny, nz = phi.shape
    lap_V = -6.0 * V
    lap_V += np.concatenate([V[1:, :, :], V[nx-1:nx, :, :]], axis=0)
    lap_V += np.concatenate([V[0:1, :, :], V[:-1, :, :]], axis=0)
    lap_V += np.concatenate([V[:, 1:, :], V[:, ny-1:ny, :]], axis=1)
    lap_V += np.concatenate([V[:, 0:1, :], V[:, :-1, :]], axis=1)
    lap_V += np.concatenate([V[:, :, 1:], V[:, :, nz-1:nz]], axis=2)
    lap_V += np.concatenate([V[:, :, 0:1], V[:, :, :-1]], axis=2)
    lap_V /= h2

    return -lap_V


def metric_trace(g: np.ndarray) -> np.ndarray:
    """度规迹 tr(g) = g_ii（空间度规的标量迹）。"""
    return g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
