"""AOP (Action Outcome Predictor) — R3 规划层神经加速器（附录 H）。

输入：ProjectedState 的 D1-D4 低维特征 + Modification
输出：GateResult 通过/拦截初筛 + 置信度

设计原则（附录 H H.3）：
- AOP 只做排序/初筛，最终动作必须由确定性 check 决定
- fail-closed：未训练 / 低置信 / 输入错误 → 回退纯因果图搜索
- model_version 与场版本 V_N 独立管理，禁止混用
- mismatch 记录作为训练数据，不能吞掉
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..dpl.types import (
    ProjectedState, Modification, GateResult, UniversalConstants,
)
from .mlp import OnlineMLP


@dataclass
class OutcomePrediction:
    """AOP 输出：动作结果预测。"""
    predicted_gate_action: str = "unknown"    # pass / block / dimension_lift
    predicted_gate_score: float = 0.0        # 0-1
    predicted_next_stats: dict = field(default_factory=dict)
    confidence: float = 0.0                   # 模型置信度
    model_version: str = "untrained"
    used_aop: bool = False                    # 是否实际使用了 AOP 预测


class ActionOutcomePredictor:
    """AOP 小模型：预测修改提案的门禁结果。

    用法：
        aop = ActionOutcomePredictor()
        pred = aop.predict(projected, modification, constants)
        if pred.used_aop and pred.confidence > 0.8:
            # 高置信，用 AOP 结果排序
            ...
        else:
            # 低置信或未训练，回退纯因果图
            ...
    """

    # 输入特征维度：D1(3) + D2(3) + D3(2) + D4(2) + Modification(4) = 14
    INPUT_DIM = 14
    OUTPUT_DIM = 3  # [pass_score, block_score, lift_score]

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 0.01,
        confidence_threshold: float = 0.7,
        model_version: str = "aop_v0.1",
    ):
        self.confidence_threshold = confidence_threshold
        self.model_version = model_version
        self.mlp = OnlineMLP(
            in_dim=self.INPUT_DIM,
            hidden=hidden,
            out_dim=self.OUTPUT_DIM,
            lr=lr,
        )
        self.trained_steps = 0
        self.mismatch_log: List[dict] = []

    def predict(
        self,
        projected: ProjectedState,
        modification: Modification,
        constants: UniversalConstants,
    ) -> OutcomePrediction:
        """预测修改提案的门禁结果。

        Args:
            projected: DPL project 输出
            modification: 修改提案
            constants: 宇宙常数

        Returns:
            OutcomePrediction: 预测结果，used_aop=False 表示回退纯搜索
        """
        # 输入编码
        try:
            x = self._encode_input(projected, modification, constants)
        except (ValueError, TypeError):
            return OutcomePrediction(
                predicted_gate_action="unknown",
                confidence=0.0,
                model_version=self.model_version,
                used_aop=False,
            )

        # 未训练 → 回退
        if self.trained_steps < 10:
            return OutcomePrediction(
                predicted_gate_action="unknown",
                confidence=0.0,
                model_version=self.model_version,
                used_aop=False,
            )

        # 前向推理
        out, _, _ = self.mlp.forward(x)
        out = np.atleast_1d(out).flatten()
        # 确保输出维度正确，不够补零，多了截断
        if len(out) < self.OUTPUT_DIM:
            out = np.pad(out, (0, self.OUTPUT_DIM - len(out)))
        elif len(out) > self.OUTPUT_DIM:
            out = out[:self.OUTPUT_DIM]
        out = self._softmax(out)

        # 置信度 = max 概率
        confidence = float(np.max(out))

        # 预测动作
        action_idx = int(np.argmax(out))
        actions = ["pass", "block", "dimension_lift"]
        predicted_action = actions[action_idx]
        predicted_score = float(out[0])  # pass 的概率

        return OutcomePrediction(
            predicted_gate_action=predicted_action,
            predicted_gate_score=predicted_score,
            predicted_next_stats={},
            confidence=confidence,
            model_version=self.model_version,
            used_aop=confidence >= self.confidence_threshold,
        )

    def record_mismatch(
        self,
        projected: ProjectedState,
        modification: Modification,
        predicted: OutcomePrediction,
        actual: GateResult,
    ) -> None:
        """记录 AOP 与实际 check 的 mismatch，作为训练数据。

        不能吞掉 mismatch，必须记录。
        """
        mismatch = {
            "predicted_action": predicted.predicted_gate_action,
            "actual_action": actual.action,
            "predicted_score": predicted.predicted_gate_score,
            "actual_passed": actual.passed,
            "model_version": predicted.model_version,
            "effective_dim": projected.dim_info.effective_dim,
        }
        self.mismatch_log.append(mismatch)

        # 如果 mismatch，用实际结果在线训练一步
        if predicted.predicted_gate_action != actual.action:
            try:
                x = self._encode_input(projected, modification, UniversalConstants())
                target = self._encode_target(actual)
                self.mlp.fit(x, target)
                self.trained_steps += 1
            except Exception:
                pass  # 训练失败不阻塞主流程

    def _encode_input(
        self,
        projected: ProjectedState,
        modification: Modification,
        constants: UniversalConstants,
    ) -> np.ndarray:
        """将 DPL 状态 + 修改提案编码为特征向量。

        特征顺序：
        D1: mean_gradient, mean_divergence, gradient_std
        D2: mean_curvature, mean_vorticity, curvature_max
        D3: euler_char, density_mean（不可用则 0）
        D4: time_arrow, fake_time（不可用则 0）
        Modification: info_delta, param_norm, mod_type_onehot(2)
        """
        features = []

        # D1
        if projected.D1.available:
            features.extend([
                float(np.mean(projected.D1.gradient)),
                float(np.mean(projected.D1.divergence)),
                float(np.std(projected.D1.gradient)),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # D2
        if projected.D2.available:
            mean_k = float(np.mean(projected.D2.curvature)) if projected.D2.curvature is not None else 0.0
            max_k = float(np.max(projected.D2.curvature)) if projected.D2.curvature is not None else 0.0
            mean_omega = float(np.mean(projected.D2.vorticity)) if projected.D2.vorticity is not None else 0.0
            features.extend([mean_k, mean_omega, max_k])
        else:
            features.extend([0.0, 0.0, 0.0])

        # D3
        if projected.D3.available:
            density_mean = float(np.mean(projected.D3.density)) if projected.D3.density is not None else 0.0
            features.extend([float(projected.D3.euler_char), density_mean])
        else:
            features.extend([0.0, 0.0])

        # D4
        if projected.D4.available:
            features.extend([constants.time_arrow, 1.0 if projected.D4.is_fake_time else 0.0])
        else:
            features.extend([0.0, 0.0])

        # Modification
        info_delta = abs(modification.info_after - modification.info_before)
        params = list(modification.params.values()) if modification.params else [0.0]
        param_norm = float(np.linalg.norm(params))

        # mod_type one-hot (简化：add_source=0, modify_topology=1, other=2)
        type_map = {"add_source": [1, 0], "modify_topology": [0, 1]}
        type_oh = type_map.get(modification.type, [0, 0])

        features.extend([info_delta, param_norm] + type_oh)

        # 确保维度正确
        arr = np.array(features, dtype=np.float32)
        if len(arr) != self.INPUT_DIM:
            raise ValueError(f"Input feature dim mismatch: got {len(arr)}, expected {self.INPUT_DIM}")

        return arr

    def _encode_target(self, gate_result: GateResult) -> np.ndarray:
        """将 GateResult 编码为训练目标。"""
        target = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        if gate_result.action == "pass":
            target[0] = 1.0
        elif gate_result.action == "block":
            target[1] = 1.0
        elif gate_result.action == "dimension_lift":
            target[2] = 1.0
        else:
            target[1] = 1.0  # 默认 block
        return target

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax。"""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


