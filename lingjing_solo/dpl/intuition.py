"""物理直觉推理引擎：infer 接口（附录 B B.6 + 附录 E E.3）。

四条推理链：
1. 引力直觉 gravitational_well：D3 密度升高 → D2 曲率增大
2. 周期直觉 periodic_dynamics：D4 FFT 谱峰占比
3. 守恒直觉 conservation：D1 全局散度为零
4. 拓扑直觉 topological：D3 欧拉示性数 χ≠2

主类型优先级：geometric_gravity → temporal_rhythm → topological_navigation → conservation_law → no_intuition
"""
from __future__ import annotations

from typing import List

import numpy as np

from .types import (
    ProjectedState, UniversalConstants,
    IntuitionRule, IntuitionResult,
)


def infer(projected: ProjectedState, constants: UniversalConstants) -> IntuitionResult:
    """物理直觉推理主接口。

    Args:
        projected: project 输出的投影状态
        constants: 已校准的宇宙常数

    Returns:
        IntuitionResult: rules / confidence / intuition_type
    """
    rules: List[IntuitionRule] = []

    # 1. 引力直觉
    g_rule = _infer_gravitational(projected, constants)
    if g_rule is not None:
        rules.append(g_rule)

    # 2. 周期直觉
    p_rule = _infer_periodic(projected, constants)
    if p_rule is not None:
        rules.append(p_rule)

    # 3. 守恒直觉
    c_rule = _infer_conservation(projected, constants)
    if c_rule is not None:
        rules.append(c_rule)

    # 4. 拓扑直觉
    t_rule = _infer_topological(projected, constants)
    if t_rule is not None:
        rules.append(t_rule)

    # 按置信度排序，选主类型
    if not rules:
        return IntuitionResult(
            rules=[],
            confidence=0.0,
            intuition_type="no_intuition",
        )

    # 主类型优先级：geometric_gravity → temporal_rhythm → topological_navigation → conservation_law
    priority_map = {
        "gravitational_well": "geometric_gravity",
        "periodic_dynamics": "temporal_rhythm",
        "topological": "topological_navigation",
        "conservation": "conservation_law",
    }

    # 按优先级选最高的非零置信规则
    priority_order = ["gravitational_well", "periodic_dynamics", "topological", "conservation"]
    main_rule = None
    for rule_type in priority_order:
        for r in rules:
            if r.rule_type == rule_type and r.confidence > 0.1:
                main_rule = r
                break
        if main_rule:
            break

    if main_rule is None:
        main_rule = max(rules, key=lambda r: r.confidence)

    main_type = priority_map.get(main_rule.rule_type, "no_intuition")

    return IntuitionResult(
        rules=rules,
        confidence=main_rule.confidence,
        intuition_type=main_type,
    )


# ============================================================
# 四条推理链
# ============================================================

def _infer_gravitational(projected: ProjectedState, c: UniversalConstants) -> IntuitionRule | None:
    """引力直觉：D3 密度升高 → D2 曲率增大。

    触发条件：D2 曲率显著正（类引力场），且 D3 密度存在。
    """
    if not projected.D2.available or projected.D2.curvature is None:
        return None

    curvature = projected.D2.curvature
    mean_K = float(np.mean(curvature))
    max_K = float(np.max(curvature))

    # 正曲率中心 → 引力阱
    # 置信度：曲率相对于 gravity_threshold_ratio 的比例
    threshold = c.gravity_threshold_ratio * c.K_max
    if max_K < threshold:
        return IntuitionRule(
            rule_type="gravitational_well",
            confidence=0.0,
            triggered_features=[],
        )

    confidence = min(1.0, max_K / (c.K_max + 1e-12))
    features = ["curvature_positive", f"max_K={max_K:.4f}", f"mean_K={mean_K:.4f}"]

    # D3 密度增强置信度
    if projected.D3.available and projected.D3.density is not None:
        mean_density = float(np.mean(projected.D3.density))
        features.append(f"density={mean_density:.4f}")
        confidence = min(1.0, confidence * 1.2)

    return IntuitionRule(
        rule_type="gravitational_well",
        confidence=float(confidence),
        triggered_features=features,
    )


def _infer_periodic(projected: ProjectedState, c: UniversalConstants) -> IntuitionRule | None:
    """周期直觉：D4 FFT 谱峰占比。

    触发条件：D4 可用且存在显著周期峰。
    """
    # 伪时间场景直接跳过
    if projected.D4.is_fake_time:
        return IntuitionRule(
            rule_type="periodic_dynamics",
            confidence=0.0,
            triggered_features=["fake_time_skipped"],
        )

    if not projected.D4.available or projected.D4.evolution is None:
        return None

    evolution = projected.D4.evolution
    if evolution.size < 4:
        return None

    # FFT 谱峰占比（简化版）
    fft = np.abs(np.fft.fft(evolution.flatten()))
    fft = fft[1:len(fft)//2]  # 去掉 DC 和共轭对称
    if len(fft) == 0:
        return None

    total_power = float(np.sum(fft ** 2))
    if total_power < 1e-12:
        return IntuitionRule(
            rule_type="periodic_dynamics",
            confidence=0.0,
            triggered_features=["flat_evolution"],
        )

    peak_power = float(np.max(fft ** 2))
    peak_ratio = peak_power / total_power

    threshold = c.periodicity_threshold
    confidence = min(1.0, peak_ratio / (threshold + 1e-12))

    triggered = [f"peak_ratio={peak_ratio:.4f}", f"threshold={threshold:.4f}"]

    return IntuitionRule(
        rule_type="periodic_dynamics",
        confidence=float(confidence),
        triggered_features=triggered,
    )


def _infer_conservation(projected: ProjectedState, c: UniversalConstants) -> IntuitionRule | None:
    """守恒直觉：D1 全局散度为零。

    触发条件：D1 散度接近零，且 conservation_stats 中有守恒律。
    """
    if not projected.D1.available or projected.D1.divergence is None:
        return None

    divergence = projected.D1.divergence
    mean_div = float(np.mean(np.abs(divergence)))

    # 散度越小，守恒置信度越高
    threshold = c.epsilon_gate
    confidence = max(0.0, 1.0 - mean_div / (threshold * 10 + 1e-12))

    triggered = [f"mean_abs_div={mean_div:.6f}"]

    # 已识别的守恒律增强置信度
    if c.laws_set:
        triggered.append(f"laws={c.laws_set}")
        confidence = min(1.0, confidence * 1.1)

    return IntuitionRule(
        rule_type="conservation",
        confidence=float(confidence),
        triggered_features=triggered,
    )


def _infer_topological(projected: ProjectedState, c: UniversalConstants) -> IntuitionRule | None:
    """拓扑直觉：D3 欧拉示性数 χ≠2。

    触发条件：D3 可用，且 χ 偏离 χ_ref（2）显著。
    """
    if not projected.D3.available:
        return None

    chi = projected.D3.euler_char
    chi_ref = c.chi_ref

    deviation = abs(chi - chi_ref)
    if deviation < 0.5:
        return IntuitionRule(
            rule_type="topological",
            confidence=0.0,
            triggered_features=[f"chi={chi}", "near_trivial"],
        )

    # 偏差越大，拓扑直觉越强
    confidence = min(1.0, deviation / 5.0)

    triggered = [
        f"chi={chi}",
        f"chi_ref={chi_ref}",
        f"deviation={deviation:.2f}",
    ]

    # 判断拓扑类型
    if chi == 0:
        triggered.append("type=torus")
    elif chi < 0:
        triggered.append("type=hyperbolic")
    elif chi > 2:
        triggered.append("type=spheres_with_handles")

    return IntuitionRule(
        rule_type="topological",
        confidence=float(confidence),
        triggered_features=triggered,
    )
