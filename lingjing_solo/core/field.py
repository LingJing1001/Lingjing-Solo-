"""
灵境引擎 V14.3 — core.field
================================
L0 离散信息基元层 + L2 场与几何生成层。

职责（严格单一）：
- 存储信息密度场 phi (L0：离散基元集的连续粗粒化)
- 演化：dΦ/dt = D·∇²Φ − γΦ − λΦ(Φ²−Φ₀²) + J（v13.2 完整场方程）
- 支持显式（CFL 自适应子步）与隐式（半隐式后向欧拉）两种 scheme

V14.3 变更（场方程补全，对照 v13.2 §7.1）：
- FieldConfig 新增 gamma（耗散系数）、lam（自耦合）、phi0（真空基线）
- 默认 gamma=0, lam=0 → 退化为 V14.2 的纯扩散方程，向后兼容
- 隐式求解器从 TODO 落地为 scipy 半隐式求解（线性项隐式，非线性项显式）
- Neumann 零通量边界在稀疏矩阵层面复现（与显式路径逐点一致）

守恒契约（V14.3 修正）：
    apply_sources_and_evolve(J, dt) 是唯一同时包含"源项注入"与"演化"的 API。
    无耗散时（γ=0）：ΔΣφ = dt·ΣJ
    有耗散时（γ>0）：ΔΣφ = dt·(ΣJ − γ·Σφ)，差额计入累计耗散。

量纲：
    h = 网格间距（自然单位下 = 1/(N-1)）
    Laplacian 含 1/h²（V12 修正）
    dt 通过 CFL 条件约束（显式）或任意（隐式）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from .laplacian import build_bubble_laplacian_csr

Scheme = Literal["explicit", "implicit"]


@dataclass
class FieldConfig:
    """场配置（量纲刚性）。V14.3 补全 v13.2 场方程参数。"""
    shape: tuple[int, int, int]
    h: float = 1.0            # 网格间距
    D: float = 1.0            # 扩散系数（波动项 c² 在自然单位下记为 D）
    gamma: float = 0.0        # 耗散系数 γ（v13.2 新增；0 = 关闭，向后兼容）
    lam: float = 0.0          # 自耦合 λ（v13.2 新增；0 = 关闭，向后兼容）
    phi0: float = 0.0         # 真空基线 Φ₀（v13.2 新增；0 = 无基线）
    scheme: Scheme = "implicit"

    @property
    def cfl_dt_max(self) -> float:
        """显式欧拉 CFL 稳定上限：dt ≤ h² / (2·dim·D)。
        耗散项（一阶）不改变扩散 CFL；非线性自作用项由半隐式吸收。"""
        dim = 3.0
        return self.h ** 2 / (2.0 * dim * self.D)


def _build_neumann_laplacian_csr(shape: tuple[int, int, int], h: float):
    """
    构造 Neumann 零通量边界的体积 7 点 Laplacian CSR 矩阵。

    与 field._laplacian_volume（向量化）边界处理完全一致：
    - 内部节点：对角 -6/h²，6 邻居各 +1/h²
    - 边界面节点：有效度数 5，对角 -5/h²
    - 边棱节点：有效度数 4，对角 -4/h²
    - 角节点：有效度数 3，对角 -3/h²

    返回 scipy.sparse.csr_matrix，shape (N³, N³)。
    """
    from scipy.sparse import csr_matrix
    nx, ny, nz = shape
    n = nx * ny * nz
    h2 = h * h
    rows, cols, vals = [], [], []
    for idx in range(n):
        i = idx // (ny * nz)
        j = (idx % (ny * nz)) // nz
        k = idx % nz
        deg = 0
        for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                deg += 1
                nidx = ni * ny * nz + nj * nz + nk
                rows.append(idx); cols.append(nidx); vals.append(1.0 / h2)
        rows.append(idx); cols.append(idx); vals.append(-float(deg) / h2)
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


class Field:
    """
    连续信息密度场 φ(x,t)。

    内部状态：
        self._phi : ndarray, 信息密度
        self._t   : 当前时间
        self._step: 演化步数
        self._cum_dissipated: V14.3 累计耗散掉的 Σφ（γ>0 时非零）
    """

    def __init__(self, config: FieldConfig):
        self.cfg = config
        self._phi = np.zeros(config.shape, dtype=np.float64)
        self._t = 0.0
        self._step = 0
        self._cum_dissipated = 0.0
        # 泡壁子图 Laplacian 缓存（lazy）
        self._L_csr: Optional[object] = None
        self._bubble_mask: Optional[np.ndarray] = None
        # V14.3：隐式求解用的 Neumann Laplacian 缓存（lazy）
        self._L_neumann_csr: Optional[object] = None

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

    @property
    def cumulative_dissipation(self) -> float:
        """V14.3：累计被耗散项抽走的信息量 Σφ（γ=0 时恒为 0）。"""
        return self._cum_dissipated

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

        显式：φ_new = φ + dt·D·L(φ) − dt·γφ − dt·λφ(φ²−Φ₀²) + dt·J
        隐式（半隐式）：
            (I − dt·D·L + dt·γ·I) φ_new = φ + dt·J − dt·λφ(φ²−Φ₀²)
            （线性项隐式，非线性自作用项显式评估在右端）

        守恒（V14.3）：
            γ=0 时：ΔΣφ = dt·ΣJ
            γ>0 时：ΔΣφ = dt·(ΣJ − γ·Σφ)，差额计入 _cum_dissipated。
        """
        if self.cfg.scheme == "explicit":
            self._evolve_explicit(J, dt)
        else:
            self._evolve_implicit(J, dt)
        self._t += dt
        self._step += 1

    # ---- V14.3：v13.2 场方程的非线性项 ----
    def _self_interaction(self, phi: np.ndarray) -> np.ndarray:
        """−λ·Φ·(Φ² − Φ₀²)。lam=0 时返回 0。"""
        if self.cfg.lam == 0.0:
            return np.zeros_like(phi)
        return -self.cfg.lam * phi * (phi * phi - self.cfg.phi0 * self.cfg.phi0)

    def _dissipation(self, phi: np.ndarray) -> np.ndarray:
        """−γ·Φ。gamma=0 时返回 0。"""
        if self.cfg.gamma == 0.0:
            return np.zeros_like(phi)
        return -self.cfg.gamma * phi

    # ---- 显式演化（CFL 自适应）----
    def _evolve_explicit(self, J: np.ndarray, dt: float) -> None:
        """
        显式欧拉：φ_new = φ + dt·[D·L(φ) − γφ − λφ(φ²−Φ₀²) + J]

        V14 修复（V13.2 A2 重演）：
        - 源项 J 只加一次（不是每子步都加，那会放大 n_sub 倍）
        - CFL 子步仅用于扩散项稳定性，源项在时间区间 [0, dt] 上积分 = dt·J
        V14.3：耗散与自作用项也随扩散子步一起积分。
        """
        dt_max = self.cfg.cfl_dt_max
        n_sub = max(1, int(np.ceil(dt / dt_max)))
        dt_sub = dt / n_sub

        # 源项在整步上积分（仅一次）
        self._phi += dt * J

        # 扩散 + 耗散 + 自作用用子步（稳定性）
        phi = self._phi
        for _ in range(n_sub):
            lap = self._laplacian_volume(phi)
            rhs = self.cfg.D * lap + self._dissipation(phi) + self._self_interaction(phi)
            phi = phi + dt_sub * rhs
        self._phi = phi

        # 累计耗散追踪（V14.3）
        if self.cfg.gamma != 0.0:
            self._cum_dissipated += dt * self.cfg.gamma * float(np.sum(phi))

    def _evolve_explicit_single(self, J: np.ndarray, dt: float) -> None:
        """单步显式（调用方已保证 dt ≤ CFL）。"""
        lap = self._laplacian_volume(self._phi)
        rhs = self.cfg.D * lap + self._dissipation(self._phi) + self._self_interaction(self._phi)
        self._phi = self._phi + dt * rhs + dt * J

    # ---- 隐式演化（半隐式后向欧拉，无条件稳定）----
    def _evolve_implicit(self, J: np.ndarray, dt: float) -> None:
        """
        V14.3 落地：半隐式后向欧拉。

        线性项（扩散 D·L + 耗散 γ·I）隐式求解：
            (I − dt·D·L + dt·γ·I) φ_new = φ + dt·J − dt·λ·φ·(φ²−Φ₀²)

        非线性自作用项 λΦ(Φ²−Φ₀²) 显式评估在右端（Picard 显式步）。
        Neumann 零通量边界与显式路径逐点一致。
        gamma=0, lam=0 时退化为标准后向欧拉 (I − dt·D·L) φ_new = φ + dt·J。
        """
        from scipy.sparse import eye as speye
        from scipy.sparse.linalg import spsolve

        # lazy 构造 Neumann Laplacian CSR
        if self._L_neumann_csr is None:
            self._L_neumann_csr = _build_neumann_laplacian_csr(
                self.cfg.shape, self.cfg.h
            )
        L = self._L_neumann_csr
        n = L.shape[0]

        phi_flat = self._phi.ravel()
        J_flat = J.ravel()

        # 左端矩阵 A = (1 + dt·γ)·I − dt·D·L
        # L 的对角元为负（-deg/h²），-dt·D·L 贡献 +dt·D·deg/h² 到对角
        A = (1.0 + dt * self.cfg.gamma) * speye(n, format='csr') - dt * self.cfg.D * L

        # 右端 b = φ + dt·J − dt·λ·φ·(φ²−Φ₀²)
        b = phi_flat + dt * J_flat
        if self.cfg.lam != 0.0:
            b = b - dt * self.cfg.lam * phi_flat * (phi_flat ** 2 - self.cfg.phi0 ** 2)

        # 求解
        phi_new_flat = spsolve(A.tocsr(), b)
        self._phi = phi_new_flat.reshape(self.cfg.shape).astype(np.float64)

        # 累计耗散追踪（V14.3）
        if self.cfg.gamma != 0.0:
            self._cum_dissipated += dt * self.cfg.gamma * float(np.sum(self._phi))

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
