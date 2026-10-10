"""DPL → SICA 桥接层（附录 F F.2）。

将 DPL 五接口输出转换为三类 SICA 对接结构：
- project → GeometricInvariant（几何不变式，候选规则）
- check → GateCounterexample（门禁反例，负样本）
- infer → IntuitionHypothesis（直觉假设，候选规则）

桥接只进入候选流程，不直接写生产规则池；复用现有 CandidateRegistry 的安全门和生命周期。
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np

from ..dpl.types import (
    ProjectedState, GateResult, Modification,
    GeometricInvariant, GateCounterexample, IntuitionHypothesis,
    IntuitionResult,
)
from ..sica.candidate import CandidateModification, CandidateRegistry


# ============================================================
# 三个转换函数
# ============================================================

def project_to_invariant(
    projected: ProjectedState,
    source_family: str = "unknown",
) -> GeometricInvariant:
    """将 project 输出转换为 GeometricInvariant。

    提取 D1-D4 中的稳定几何特征作为候选不变式。
    """
    features: dict[str, float] = {}
    pattern_type = "unknown"
    dim_layer = ""
    stability_score = 0.0
    evidence_count = 1

    # D1 特征：梯度能量、散度均值
    if projected.D1.available:
        dim_layer = "D1"
        features["mean_gradient"] = float(np.mean(projected.D1.gradient))
        features["max_gradient"] = float(np.max(projected.D1.gradient))
        features["mean_divergence"] = float(np.mean(projected.D1.divergence))
        stability_score = 0.5  # D1 基础稳定性

        # 如果梯度场稳定（方差小），稳定性更高
        grad_std = float(np.std(projected.D1.gradient))
        if grad_std < 0.1:
            pattern_type = "stable_flow"
            stability_score = 0.7

    # D2 特征：曲率、涡旋
    if projected.D2.available and projected.D2.curvature is not None:
        dim_layer = "D2"
        features["mean_curvature"] = float(np.mean(projected.D2.curvature))
        features["max_curvature"] = float(np.max(projected.D2.curvature))
        if projected.D2.vorticity is not None:
            features["mean_vorticity"] = float(np.mean(projected.D2.vorticity))

        # 持续正曲率 → 稳定曲率模式
        if features["mean_curvature"] > 0.01:
            pattern_type = "stable_curvature"
            stability_score = 0.8

    # D3 特征：密度、欧拉示性数
    if projected.D3.available:
        dim_layer = "D3"
        features["euler_char"] = float(projected.D3.euler_char)
        if projected.D3.density is not None:
            features["mean_density"] = float(np.mean(projected.D3.density))
        pattern_type = "topological_structure"
        stability_score = 0.9

    return GeometricInvariant(
        pattern_type=pattern_type,
        dim_layer=dim_layer,
        stability_score=stability_score,
        evidence_count=evidence_count,
        source_family=source_family,
        features=features,
    )


def check_to_counterexample(
    gate_result: GateResult,
    projected: ProjectedState,
    modification: Modification,
) -> GateCounterexample:
    """将 check 违规结果转换为 GateCounterexample。

    门禁反例作为 SICA 负样本，用于避免重复违规。
    """
    # 计算 sample hash（对输入状态和修改提案的哈希）
    hash_input = f"{projected.input_shape}_{modification.type}_{modification.target}_{modification.params}"
    sample_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:64]

    # 违反幅度：1 - score（score 越低，违反越严重）
    violation_magnitude = 1.0 - gate_result.score

    return GateCounterexample(
        gate_type=gate_result.violations[0] if gate_result.violations else "unknown",
        violation=float(violation_magnitude),
        raw_dim=projected.dim_info.spatial,
        effective_dim=projected.dim_info.effective_dim,
        sample_hash=sample_hash,
        timestamp=time.time_ns(),
    )


def infer_to_hypothesis(intuition: IntuitionResult) -> IntuitionHypothesis:
    """将 infer 输出转换为 IntuitionHypothesis。

    直觉假设作为 SICA 候选规则，需要经过 holdout 验证。
    """
    # 选置信度最高的规则作为主假设
    if not intuition.rules:
        return IntuitionHypothesis(
            rule_type="no_intuition",
            confidence=0.0,
            triggered_features=[],
            holdout_delta=0.0,
        )

    main_rule = max(intuition.rules, key=lambda r: r.confidence)

    return IntuitionHypothesis(
        rule_type=main_rule.rule_type,
        confidence=float(intuition.confidence),
        triggered_features=list(main_rule.triggered_features),
        holdout_delta=0.0,  # 待 SICA holdout 评估后回填
    )


# ============================================================
# 桥接到 SICA CandidateRegistry
# ============================================================

def submit_invariant_to_sica(
    registry: CandidateRegistry,
    invariant: GeometricInvariant,
    candidate_id: str,
    tick: int = 0,
) -> CandidateModification:
    """将 GeometricInvariant 包装为 CandidateModification 并提交到 SICA。

    Args:
        registry: SICA CandidateRegistry 实例
        invariant: 几何不变式
        candidate_id: 唯一候选 ID
        tick: 当前时间步

    Returns:
        CandidateModification: 提交后的候选
    """
    content = {
        "type": "geometric_invariant",
        "pattern_type": invariant.pattern_type,
        "dim_layer": invariant.dim_layer,
        "stability_score": invariant.stability_score,
        "source_family": invariant.source_family,
        "features": invariant.features,
    }

    candidate = CandidateModification(
        candidate_id=candidate_id,
        target="rule_set",
        content=content,
        evidence_count=invariant.evidence_count,
        expected_gain=invariant.stability_score,
    )

    return registry.submit(candidate, tick=tick)


def submit_hypothesis_to_sica(
    registry: CandidateRegistry,
    hypothesis: IntuitionHypothesis,
    candidate_id: str,
    tick: int = 0,
) -> CandidateModification:
    """将 IntuitionHypothesis 包装为 CandidateModification 并提交到 SICA。

    Args:
        registry: SICA CandidateRegistry 实例
        hypothesis: 直觉假设
        candidate_id: 唯一候选 ID
        tick: 当前时间步

    Returns:
        CandidateModification: 提交后的候选
    """
    content = {
        "type": "intuition_hypothesis",
        "rule_type": hypothesis.rule_type,
        "confidence": hypothesis.confidence,
        "triggered_features": hypothesis.triggered_features,
        "holdout_delta": hypothesis.holdout_delta,
    }

    candidate = CandidateModification(
        candidate_id=candidate_id,
        target="rule_set",
        content=content,
        evidence_count=1,
        expected_gain=hypothesis.confidence,
    )

    return registry.submit(candidate, tick=tick)


def submit_counterexample_to_sica(
    registry: CandidateRegistry,
    counterexample: GateCounterexample,
    candidate_id: str,
    tick: int = 0,
) -> CandidateModification:
    """将 GateCounterexample 作为负样本提交到 SICA。

    注意：反例是负样本，expected_gain 为负。
    """
    content = {
        "type": "gate_counterexample",
        "gate_type": counterexample.gate_type,
        "violation": counterexample.violation,
        "raw_dim": counterexample.raw_dim,
        "effective_dim": counterexample.effective_dim,
        "sample_hash": counterexample.sample_hash,
    }

    candidate = CandidateModification(
        candidate_id=candidate_id,
        target="rule_set",
        content=content,
        evidence_count=1,
        expected_gain=-counterexample.violation,  # 负样本
    )

    return registry.submit(candidate, tick=tick)
