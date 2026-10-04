"""自适应几何门禁：check 接口（附录 B B.5 + 附录 D）。"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .types import (
    ProjectedState, UniversalConstants, Modification, GateResult,
)


def check(
    projected: ProjectedState,
    constants: UniversalConstants,
    modification: Modification,
) -> GateResult:
    """自适应几何门禁校验主接口。

    按 effective_dim 动态激活校验项：
    - 2D: 信息守恒 + 曲率有界 + 自指稳定
    - 2.5D: 2D三项 + 涡旋校验（条件激活）
    - 3D: 再加拓扑合法 + 跨维隔离
    - 4D+: 再加因果一致
    """
    dim = projected.dim_info.effective_dim
    spatial = projected.dim_info.spatial

    violations: List[str] = []
    scores: List[float] = []
    action_priority: List[Tuple[str, str]] = []
    all_observed: dict = {}
    all_threshold: dict = {}

    # 1. 信息守恒
    v, score, obs, thr = _check_conservation(projected, constants)
    if v: violations.append("conservation_violation")
    scores.append(score)
    all_observed["conservation"] = obs
    all_threshold["conservation"] = thr
    if v: action_priority.append(("block", "conservation"))

    # 2. 曲率有界
    v, score, obs, thr = _check_curvature_bounded(projected, constants)
    if v: violations.append("curvature_violation")
    scores.append(score)
    all_observed["curvature"] = obs
    all_threshold["curvature"] = thr
    if v: action_priority.append(("dimension_lift", "curvature"))

    # 3. 自指稳定
    v, score, obs, thr = _check_self_reference_stable(projected, constants, modification)
    if v: violations.append("self_reference_violation")
    scores.append(score)
    all_observed["self_reference"] = obs
    all_threshold["self_reference"] = thr
    if v: action_priority.append(("block", "self_reference"))

    # 2.5D+：涡旋校验（条件激活）
    if spatial >= 2.5 and projected.D2.available:
        v, score, obs, thr = _check_vorticity(projected, constants)
        if v: violations.append("vorticity_violation")
        scores.append(score)
        all_observed["vorticity"] = obs
        all_threshold["vorticity"] = thr
        if v: action_priority.append(("block", "vorticity"))

    # 3D+：拓扑合法 + 跨维隔离
    if spatial >= 3.0:
        v, score, obs, thr = _check_topology(projected, constants)
        if v: violations.append("topology_violation")
        scores.append(score)
        all_observed["topology"] = obs
        all_threshold["topology"] = thr
        if v: action_priority.append(("block", "topology"))

        v, score, obs, thr = _check_cross_dim_isolation(projected, constants)
        if v: violations.append("cross_dim_violation")
        scores.append(score)
        all_observed["cross_dim"] = obs
        all_threshold["cross_dim"] = thr
        if v: action_priority.append(("dimension_reduce", "cross_dim"))

    # 4D+：因果一致
    if dim >= 4 and projected.D4.available:
        v, score, obs, thr = _check_causal_consistency(projected, constants)
        if v: violations.append("causal_violation")
        scores.append(score)
        all_observed["causal"] = obs
        all_threshold["causal"] = thr
        if v: action_priority.append(("block", "causal"))

    # 综合决策
    passed = len(violations) == 0
    avg_score = float(np.mean(scores)) if scores else 1.0

    if passed:
        action = "pass"
    else:
        actions = [a for a, _ in action_priority]
        if "dimension_lift" in actions:
            action = "dimension_lift"
        elif "dimension_reduce" in actions:
            action = "dimension_reduce"
        else:
            action = "block"

    return GateResult(
        passed=passed,
        violations=violations,
        score=avg_score,
        action=action,
        threshold=all_threshold,
        observed=all_observed,
        evidence={
            "effective_dim": dim,
            "spatial": spatial,
            "activated_gates": len(scores),
        },
    )


def _check_conservation(projected, c):
    """门禁 1：信息守恒。"""
    if not projected.D1.available:
        return True, 1.0, {}, {}
    div = projected.D1.divergence
    mean_div = float(np.mean(np.abs(div)))
    threshold = c.epsilon_gate * c.I_0 / (c.l_0 ** 2)
    violated = mean_div >= threshold
    score = max(0.0, 1.0 - mean_div / (threshold + 1e-12))
    return violated, score, {"mean_abs_divergence": mean_div}, {"threshold": threshold}


def _check_curvature_bounded(projected, c):
    """门禁 2：曲率有界。"""
    if not projected.D2.available or projected.D2.curvature is None:
        return True, 1.0, {}, {}
    K = projected.D2.curvature
    max_abs_K = float(np.max(np.abs(K)))
    threshold = c.K_max
    violated = max_abs_K >= threshold
    score = max(0.0, 1.0 - max_abs_K / (threshold + 1e-12))
    return violated, score, {"max_abs_K": max_abs_K}, {"threshold": threshold}


def _check_vorticity(projected, c):
    """门禁 3：涡旋校验。"""
    if not projected.D2.available or projected.D2.vorticity is None:
        return True, 1.0, {}, {}
    omega = projected.D2.vorticity
    max_abs_omega = float(np.max(np.abs(omega)))
    threshold = c.omega_max
    omega_std = float(np.std(omega))
    if omega_std < 1e-6:
        return True, 1.0, {"omega_std": omega_std, "skipped": "uniform"}, {}
    violated = max_abs_omega >= threshold
    score = max(0.0, 1.0 - max_abs_omega / (threshold + 1e-12))
    return violated, score, {"max_abs_omega": max_abs_omega, "omega_std": omega_std}, {"threshold": threshold}


def _check_topology(projected, c):
    """门禁 4：拓扑合法。"""
    if not projected.D3.available:
        return True, 1.0, {}, {}
    chi = projected.D3.euler_char
    deviation = abs(chi - c.chi_ref)
    threshold = 5.0
    violated = deviation > threshold
    score = max(0.0, 1.0 - deviation / threshold)
    return violated, score, {"euler_char": chi, "deviation": deviation}, {"threshold": threshold, "chi_ref": c.chi_ref}


def _check_self_reference_stable(projected, c, modification):
    """门禁 5：自指稳定。"""
    info_delta = abs(modification.info_after - modification.info_before)
    params_values = list(modification.params.values()) if modification.params else [0.0]
    param_delta = float(np.linalg.norm(params_values)) if params_values else 1e-12
    if param_delta < 1e-12:
        return True, 1.0, {"param_delta": 0.0}, {}
    lipschitz = info_delta / param_delta
    threshold = 1.0
    violated = lipschitz >= threshold
    score = max(0.0, 1.0 - lipschitz)
    return violated, score, {"lipschitz": lipschitz, "info_delta": info_delta, "param_delta": param_delta}, {"threshold": threshold}


def _check_cross_dim_isolation(projected, c):
    """门禁 6：跨维隔离。"""
    if not projected.D1.available or not projected.D2.available:
        return True, 1.0, {}, {}
    d1_flat = projected.D1.gradient.flatten()
    d2_flat = projected.D2.curvature.flatten() if projected.D2.curvature is not None else np.zeros_like(d1_flat)
    if len(d1_flat) != len(d2_flat):
        return True, 1.0, {}, {}
    d1_norm = np.linalg.norm(d1_flat)
    d2_norm = np.linalg.norm(d2_flat)
    if d1_norm < 1e-12 or d2_norm < 1e-12:
        return True, 1.0, {}, {}
    cos_sim = float(np.dot(d1_flat, d2_flat) / (d1_norm * d2_norm))
    overlap = abs(cos_sim)
    threshold = c.delta_overlap + 0.5
    violated = overlap > threshold
    score = max(0.0, 1.0 - overlap / (threshold + 1e-12))
    return violated, score, {"cosine_similarity": cos_sim, "overlap": overlap}, {"threshold": threshold}


def _check_causal_consistency(projected, c):
    """门禁 7：因果一致。"""
    if not projected.D4.available:
        return True, 1.0, {}, {}
    time_arrow = c.time_arrow
    if projected.D4.is_fake_time:
        return True, 0.0, {"is_fake_time": True, "time_arrow": time_arrow}, {"required_time_arrow": 0.0}
    violated = time_arrow <= 0
    score = max(0.0, min(1.0, time_arrow))
    return violated, score, {"time_arrow": time_arrow}, {"threshold": 0.0}
