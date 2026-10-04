"""宇宙常数校准接口：calibrate（附录 B B.3 + 附录 E E.2）。"""
from __future__ import annotations

from typing import List

import numpy as np

from .types import UniversalConstants


def calibrate(observations: List[np.ndarray]) -> UniversalConstants:
    """从观测序列自动校准宇宙常数与门禁阈值。

    Args:
        observations: 观测序列，长度 >= 1

    Returns:
        UniversalConstants: 17 个字段全部填充

    Raises:
        ValueError: 观测为空、含 NaN/Inf、形状不一致
    """
    # #0. 默认值（附录 E E.2.2）
    c = UniversalConstants()
    c.provenance = {k: "default" for k in c.__dataclass_fields__ if k != "provenance"}

    # #1. 输入合法性检查
    if len(observations) == 0:
        raise ValueError("observations 为空")
    for obs in observations:
        if not np.all(np.isfinite(obs)):
            raise ValueError("observations 含 NaN/Inf")

    if len(observations) < 2:
        # 样本不足，返回默认值并标记
        c.provenance["sample_size"] = "insufficient"
        return c

    # 形状一致性检查
    base_shape = observations[0].shape
    for obs in observations[1:]:
        if obs.shape != base_shape:
            raise ValueError("观测序列形状不一致")

    # #2. 逐帧差分
    delta_Is = [observations[t+1] - observations[t] for t in range(len(observations) - 1)]

    # #3. c_light：最大信息密度变化率
    max_dl = max(np.max(np.abs(dl)) for dl in delta_Is)
    c.c_light = float(max_dl * c.I_0 / c.l_0) if max_dl > 0 else 1.0
    c.provenance["c_light"] = "calibrated"

    # #4. h_planck：最小非零信息密度差
    nonzero_dls = []
    for dl in delta_Is:
        abs_dl = np.abs(dl)
        if np.any(abs_dl > 0):
            nonzero_dls.append(np.min(abs_dl[abs_dl > 0]))
    if nonzero_dls:
        c.h_planck = float(min(nonzero_dls))
        c.provenance["h_planck"] = "calibrated"

    # #5. delta_s_min：最小空间间隔
    c.delta_s_min = 1.0  # 默认网格间距 1
    c.provenance["delta_s_min"] = "grid_default"

    # #6. time_arrow：对称 KL 散度差（简化版）
    if len(observations) >= 2:
        c.time_arrow = _calibrate_time_arrow(observations)
        c.provenance["time_arrow"] = "calibrated"

    # #7. laws_set + conservation_stats：守恒律筛选（简化版）
    c.laws_set, c.conservation_stats = _calibrate_laws(observations)
    c.provenance["laws_set"] = "calibrated"
    c.provenance["conservation_stats"] = "calibrated"

    # #8. D：信息扩散系数拟合（简化版，默认 1.0）
    c.D = 1.0
    c.provenance["D"] = "default"

    # #9. K_max：曲率场 95% 分位数（2D 简化）
    c.K_max = _calibrate_K_max(observations, c.I_0, c.l_0)
    c.provenance["K_max"] = "calibrated"

    # #10. omega_max：涡旋场 95% 分位数（简化版）
    c.omega_max = _calibrate_omega_max(observations, c.D, c.I_0)
    c.provenance["omega_max"] = "calibrated"

    # #11. epsilon_gate：信息守恒阈值
    c.epsilon_gate = _calibrate_epsilon_gate(observations, c.D, c.I_0, c.l_0)
    c.provenance["epsilon_gate"] = "calibrated"

    # #12. delta_overlap：跨维隔离容许偏离量（简化默认）
    c.delta_overlap = 0.05
    c.provenance["delta_overlap"] = "default"

    return c


def _calibrate_time_arrow(observations: List[np.ndarray]) -> float:
    """时间箭头：正向与反向 KL 散度之差（简化版）。"""
    diff_values = []
    for t in range(len(observations) - 1):
        # 简化：用均值差代替 KL
        p_t = observations[t].flatten().astype(float)
        p_next = observations[t+1].flatten().astype(float)
        if np.std(p_t) < 1e-12 or np.std(p_next) < 1e-12:
            continue
        d_fwd = abs(np.mean(p_next) - np.mean(p_t))
        d_bwd = abs(np.mean(p_t) - np.mean(p_next))
        diff_values.append(d_fwd - d_bwd)
    return float(np.mean(diff_values)) if diff_values else 0.0


def _calibrate_laws(observations: List[np.ndarray]):
    """守恒律筛选 + 统计量存储。"""
    candidates = {
        "total_information": lambda o: float(np.sum(np.abs(o))),
        "total_gradient": lambda o: float(np.sum(np.abs(np.gradient(o)))) if o.ndim >= 2 else 0.0,
    }
    laws = []
    stats = {}
    for name, fn in candidates.items():
        values = [fn(o) for o in observations]
        mean = float(np.mean(values))
        std = float(np.std(values))
        stats[name] = {"mean": mean, "std": std, "count": len(values)}
        if abs(mean) > 1e-12 and std / abs(mean) < 0.01:
            laws.append(name)
    return laws, stats


def _calibrate_K_max(observations, I_0, l_0):
    """曲率场 95% 分位数（2D 简化版）。"""
    curvatures = []
    for obs in observations:
        I_norm = obs / I_0
        if I_norm.ndim != 2:
            continue
        Ix = np.gradient(I_norm, axis=0)
        Iy = np.gradient(I_norm, axis=1)
        Ixx = np.gradient(Ix, axis=0)
        Iyy = np.gradient(Iy, axis=1)
        Ixy = np.gradient(Ix, axis=1)
        det_H = Ixx * Iyy - Ixy**2
        denom = (1 + Ix**2 + Iy**2) ** 2
        K = det_H / (denom + 1e-12)
        curvatures.append(np.abs(K).flatten())
    if not curvatures:
        return 1.0
    all_K = np.concatenate(curvatures)
    return float(np.percentile(all_K, 95))


def _calibrate_omega_max(observations, D, I_0):
    """涡旋场 95% 分位数（2D 简化版）。"""
    vorticity_magnitudes = []
    for obs in observations:
        I_norm = obs / I_0
        if I_norm.ndim != 2:
            continue
        grads = np.gradient(I_norm)
        J_I = -D * np.stack(grads, axis=-1)
        v_I = J_I / (np.abs(I_norm)[..., None] + 1e-12)
        vx, vy = v_I[..., 0], v_I[..., 1]
        omega = np.gradient(vy, axis=0) - np.gradient(vx, axis=1)
        vorticity_magnitudes.append(np.abs(omega).flatten())
    if not vorticity_magnitudes:
        return 1.0
    all_omega = np.concatenate(vorticity_magnitudes)
    return float(np.percentile(all_omega, 95))


def _calibrate_epsilon_gate(observations, D, I_0, l_0):
    """信息守恒阈值（简化版）。"""
    flux_diffs = []
    for t in range(len(observations) - 1):
        I_norm = observations[t] / I_0
        grads = np.gradient(I_norm)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]
        J_I = -D * np.stack(grads, axis=-1)
        delta = float(np.max(np.abs(J_I)))
        flux_diffs.append(delta)
    if not flux_diffs:
        return 1e-3
    return float(np.percentile(flux_diffs, 95))
