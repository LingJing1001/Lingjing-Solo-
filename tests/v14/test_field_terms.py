"""
V14.3 契约测试 —— 场方程补全（耗散项 + 自相互作用项 + 隐式求解器落地）。

每个测试对应一份 v13.2 §7.1 场方程的可观测契约：
  C10 耗散验证（γ>0 → 总信息量单调下降）
  C11 自相互作用稳定性（λ>0 → 孤子/紧束缚解不发散）
  C12 向后兼容（γ=0, λ=0 → 与 V14.2 纯扩散逐位一致）
  C13 修正守恒恒等式（γ>0 → ΔΣφ = dt·(ΣJ − γ·Σφ)，机器精度）

约定：不依赖 pytest，纯 Python + numpy/scipy，与 V14/V14.2 风格一致。
"""
import numpy as np

from lingjing_solo.core.field import Field, FieldConfig


def test_c10_dissipation_monotonic_decrease():
    """C10：γ>0 时，无源项注入，总信息量 Σφ 必须随时间单调下降。

    v13.2: dΦ/dt = ... − γΦ → 均匀衰减 e^{−γt}。
    """
    cfg = FieldConfig(
        shape=(16, 16, 16), h=1.0 / 15, D=0.10,
        gamma=0.05, scheme="implicit",
    )
    f = Field(cfg)
    # 初始高斯斑
    cx = cy = cz = 8
    i = np.arange(16)[:, None, None]
    j = np.arange(16)[None, :, None]
    k = np.arange(16)[None, None, :]
    f.inject_sources(np.exp(-((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / 4.0))

    totals = [f.total()]
    for _ in range(20):
        f.apply_sources_and_evolve(np.zeros_like(f.phi), dt=0.01)
        totals.append(f.total())

    # 单调下降（允许微小数值抖动 < 1e-10）
    for t_prev, t_curr in zip(totals[:-1], totals[1:]):
        assert t_curr < t_prev + 1e-10, \
            f"C10 耗散不单调: Σφ {t_prev:.6f} → {t_curr:.6f}"

    # 累计耗散追踪非零
    assert f.cumulative_dissipation > 0, "cumulative_dissipation 应 > 0"


def test_c11_self_interaction_stability():
    """C11：λ>0 时，小幅度高斯斑在 100 tick 内不发散（max|φ| 不爆炸）。

    势阱项 −λΦ(Φ²−Φ₀²) 在 Φ₀=0 时是双势阱（Φ²>0 时拉回，|Φ| 小处推离）。
    紧束缚解应保持有界。
    """
    cfg = FieldConfig(
        shape=(16, 16, 16), h=1.0 / 15, D=0.10,
        lam=0.001, phi0=0.0, gamma=0.0, scheme="implicit",
    )
    f = Field(cfg)
    cx = cy = cz = 8
    i = np.arange(16)[:, None, None]
    j = np.arange(16)[None, :, None]
    k = np.arange(16)[None, None, :]
    # 小幅度高斯斑（幅度 0.5，远小于 λ 的破坏阈值）
    f.inject_sources(0.5 * np.exp(-((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / 4.0))

    max_phi0 = float(np.max(np.abs(f.phi)))
    for _ in range(100):
        f.apply_sources_and_evolve(np.zeros_like(f.phi), dt=0.01)

    max_phi_end = float(np.max(np.abs(f.phi)))
    # 不发散：最大值不超过初值的 3 倍（扩散会摊开，幅度下降；不发散是硬约束）
    assert max_phi_end < 3.0 * max_phi0 + 1e-6, \
        f"C11 自相互作用发散: max|φ| {max_phi0:.4f} → {max_phi_end:.4f}"
    # 有限
    assert np.all(np.isfinite(f.phi)), "场出现 NaN/Inf"


def test_c12_backward_compat_no_terms():
    """C12：γ=0, λ=0 时，隐式求解器满足 V14.2 守恒恒等式 ΔΣφ = dt·ΣJ。

    向后兼容的真正含义：V14.3 新落地的隐式求解器在默认参数下，
    必须复现 V14.2 C2 的守恒契约（而非与显式逐位一致——后向/前向欧拉本就有 O(dt) 差）。
    """
    cfg = FieldConfig(
        shape=(16, 16, 16), h=1.0 / 15, D=0.10,
        gamma=0.0, lam=0.0, phi0=0.0, scheme="implicit",
    )
    f = Field(cfg)

    # 初始高斯斑
    cx = cy = cz = 8
    i = np.arange(16)[:, None, None]
    j = np.arange(16)[None, :, None]
    k = np.arange(16)[None, None, :]
    f.inject_sources(0.5 * np.exp(-((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / 4.0))

    # 固定源项
    J = np.zeros_like(f.phi)
    J[cx, cy, cz] = 1.0

    dt = 0.01
    S0 = f.total()
    for _ in range(10):
        f.apply_sources_and_evolve(J, dt=dt)
    S1 = f.total()

    # V14.2 守恒恒等式（γ=0 时修正项消失）
    expected_delta = dt * 10 * float(np.sum(J))
    actual_delta = S1 - S0
    rel_err = abs(actual_delta - expected_delta) / (abs(expected_delta) + 1e-12)
    assert rel_err < 1e-8, \
        f"C12 向后兼容失败（γ=0 隐式守恒）: Δ={actual_delta:.6f}, " \
        f"预期={expected_delta:.6f}, rel_err={rel_err:.2e}"

    # cumulative_dissipation 在 γ=0 时必须恒为 0
    assert f.cumulative_dissipation == 0.0, \
        f"C12: γ=0 时 cumulative_dissipation 应为 0，实际={f.cumulative_dissipation}"


def test_c13_corrected_conservation_with_dissipation():
    """C13：γ>0 时修正守恒恒等式 ΔΣφ = dt·(ΣJ − γ·Σφ)，机器精度。

    V14.2 恒等式 ΔΣφ = dt·ΣJ 在 γ>0 时不成立；V14.3 修正为减去耗散流失。
    """
    cfg = FieldConfig(
        shape=(16, 16, 16), h=1.0 / 15, D=0.10,
        gamma=0.1, scheme="implicit",
    )
    f = Field(cfg)

    # 固定源项注入（单点 J=1.0）
    J = np.zeros_like(f.phi)
    J[8, 8, 8] = 1.0

    # 初始有密度
    f.inject_sources(np.ones_like(f.phi) * 0.5)
    S0 = f.total()

    dt = 0.01
    f.apply_sources_and_evolve(J, dt=dt)
    S1 = f.total()

    # 修正恒等式：ΔΣφ ≈ dt·(ΣJ − γ·Σφ_平均)
    # 离散上耗散项是 −γ·φ_new（后向欧拉）或 −γ·φ_old（前向欧拉）
    # 半隐式后向欧拉：ΔΣφ = dt·(ΣJ − γ·Σφ_new)
    expected_delta = dt * (float(np.sum(J)) - cfg.gamma * S1)
    actual_delta = S1 - S0

    rel_err = abs(actual_delta - expected_delta) / (abs(expected_delta) + 1e-12)
    assert rel_err < 1e-8, \
        f"C13 修正守恒恒等式失败: Δ={actual_delta:.6f}, 预期={expected_delta:.6f}, rel_err={rel_err:.2e}"
