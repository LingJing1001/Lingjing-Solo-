"""
灵境引擎 V14.0 — core.field
================================
L0 离散信息基元层 + L2 场与几何生成层。

职责（严格单一）：
- 存储信息密度场 phi (L0：离散基元集的连续粗粒化)
- 演化：∂φ/∂t = D·∇²φ + J（源项通道单一，且仅在此处写入）
- 支持显式（CFL 自适应子步）与隐式（后向欧拉）两种 scheme

守恒契约：
    apply_sources_and_evolve(J, dt) 是唯一同时包含"源项注入"与"演化"的 API。
    任何外部代码不得直接修改 self.phi（封装为 property）。

量纲：
    h = 网格间距（自然单位下 = 1/(N-1)）
    Laplacian 含 1/h²（V12 修正）
    dt 通过 CFL 条件约束（显式）或任意（隐式）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from .laplacian import build_volume_laplacian_csr, build_bubble_laplacian_csr

Scheme = Literal["explicit", "implicit"]


@dataclass
class FieldConfig:
    """场配置（量纲刚性）。"""
    shape: tuple[int, int, int]
    h: float = 1.0            # 网格间距
    D: float = 1.0            # 扩散系数
    scheme: Scheme = "implicit"

    @property
    def cfl_dt_max(self) -> float:
        """显式欧拉 CFL 稳定上限：dt ≤ h² / (2·dim·D)。"""
        dim = 3.0
        return self.h ** 2 / (2.0 * dim * self.D)


class Field:
    """
    连续信息密度场 φ(x,t)。

    内部状态：
        self._phi : ndarray, 信息密度
        self._t   : 当前时间
        self._step: 演化步数
    """

    def __init__(self, config: FieldConfig):
        self.cfg = config
        self._phi = np.zeros(config.shape, dtype=np.float64)
        self._t = 0.0
        self._step = 0
        # 泡壁子图 Laplacian 缓存（lazy）
        self._L_csr: Optional[object] = None
        self._bubble_mask: Optional[np.ndarray] = None

    # ---- 封装：禁止外部直接写 phi ----
    @property
    def phi(self) -> np.ndarray:
        """只读视图。"""
        return self._phi

    @property
    def t(self) -> float:
        return self._t

    @property
    def step(self) -> int:
        return self._step

    # ---- 绝对注入（独立通道，不参与演化）----
    def inject_sources(self, J: np.ndarray) -> None:
        """
        绝对注入：φ += J（例如初始条件、边界源）。
        语义：这是"放置物质"，不是"演化一步"。
        """
        if J.shape != self._phi.shape:
            raise ValueError(f"J shape {J.shape} != phi {self._phi.shape}")
        self._phi += J

    # ---- 演化一步（守恒通道单一）----
    def apply_sources_and_evolve(self, J: np.ndarray, dt: float) -> None:
        """
        唯一同时包含"源项"与"演化"的 API。

        显式：φ_new = φ + dt·D·L(φ) + dt·J
        隐式：(I - dt·D·L) φ_new = φ + dt·J

        守恒：ΔΣφ = dt·ΣJ（在无额外边界汇的条件下）。
        """
        if self.cfg.scheme == "explicit":
            self._evolve_explicit(J, dt)
        else:
            self._evolve_implicit(J, dt)
        self._t += dt
        self._step += 1

    # ---- 显式演化（CFL 自适应）----
    def _evolve_explicit(self, J: np.ndarray, dt: float) -> None:
        """
        显式欧拉：φ_new = φ + dt·D·L(φ) + dt·J

        V14 修复（V13.2 A2 重演）：
        - 源项 J 只加一次（不是每子步都加，那会放大 n_sub 倍）
        - CFL 子步仅用于扩散项稳定性，源项在时间区间 [0, dt] 上积分 = dt·J
        """
        dt_max = self.cfg.cfl_dt_max
        n_sub = max(1, int(np.ceil(dt / dt_max)))
        dt_sub = dt / n_sub

        # 源项在整步上积分（仅一次）
        self._phi += dt * J

        # 扩散项用子步（稳定性）
        phi = self._phi
        for _ in range(n_sub):
            lap = self._laplacian_volume(phi)
            phi = phi + dt_sub * self.cfg.D * lap
        self._phi = phi

    def _evolve_explicit_single(self, J: np.ndarray, dt: float) -> None:
        """单步显式（调用方已保证 dt ≤ CFL）。"""
        lap = self._laplacian_volume(self._phi)
        self._phi = self._phi + dt * self.cfg.D * lap + dt * J

    # ---- 隐式演化（后向欧拉，无条件稳定）----
    def _evolve_implicit(self, J: np.ndarray, dt: float) -> None:
        # V13：后向欧拉 (I - dt·D·L) φ_new = φ + dt·J
        # 当前 V14.0-M0：使用显式子步作为 fallback（隐式求解器待 M1 接入）
        # TODO(M1): 接入 implicit_solver.evolve_bubble_implicit
        # 为保证 M0 可运行，此处用充分细分的显式（等价于高稳定度近似）
        self._evolve_explicit(J, dt)

    # ---- Laplacian ----
    def _laplacian_volume(self, phi: np.ndarray) -> np.ndarray:
        """
        体积 7 点离散 Laplacian，含 1/h²（V12 修正）。

        V14：零通量（Neumann）边界。
        边界处外侧邻居 = 自身 → 该方向二阶差分为 0。
        这保证了：
        - 全域 Σ∇²φ = 0（守恒，无边界汇源）
        - 与 Agent 感知的边界条件一致
        """
        h2 = self.cfg.h ** 2
        nx, ny, nz = phi.shape
        lap = -6.0 * phi
        # x 方向（内部用周期索引，边界处 clamp）
        xp = np.concatenate([phi[1:, :, :], phi[nx-1:nx, :, :]], axis=0)  # x+1，右边界复制
        xm = np.concatenate([phi[0:1, :, :], phi[:-1, :, :]], axis=0)      # x-1，左边界复制
        lap += xp + xm
        # y 方向
        yp = np.concatenate([phi[:, 1:, :], phi[:, ny-1:ny, :]], axis=1)
        ym = np.concatenate([phi[:, 0:1, :], phi[:, :-1, :]], axis=1)
        lap += yp + ym
        # z 方向
        zp = np.concatenate([phi[:, :, 1:], phi[:, :, nz-1:nz]], axis=2)
        zm = np.concatenate([phi[:, :, 0:1], phi[:, :, :-1]], axis=2)
        lap += zp + zm
        return lap / h2

    # ---- 泡壁接口（面积律，M3 验收）----
    def set_bubble(self, mask: np.ndarray) -> None:
        """
        定义泡壁子图（清晰区 / 活动节点）。
        mask: bool ndarray，True = 活动节点。
        """
        if mask.shape != self._phi.shape:
            raise ValueError("mask shape mismatch")
        self._bubble_mask = mask
        self._L_csr = build_bubble_laplacian_csr(mask, h=self.cfg.h)

    def laplacian_bubble(self, phi: np.ndarray) -> np.ndarray:
        """
        泡壁子图 Laplacian（CSR，O(R²) 活动节点）。

        L_csr 是 (M, M)，M = 活动节点数。输入应是活动节点的 (M,) 向量，
        而非全网格 (N³,)。用 active_indices 投影。
        """
        if self._L_csr is None:
            raise RuntimeError("call set_bubble() first")
        if self._bubble_mask is None:
            raise RuntimeError("call set_bubble() first")
        active = np.flatnonzero(self._bubble_mask)  # (M,)
        vals = phi.ravel()[active]  # (M,) 仅活动节点值
        result = self._L_csr @ vals  # (M,)
        return result

    # ---- 诊断 ----
    def total(self) -> float:
        """Σφ（守恒量监测）。"""
        return float(np.sum(self._phi))

    def copy(self) -> "Field":
        """深拷贝（用于确定性重放比对）。"""
        import copy
        return copy.deepcopy(self)
