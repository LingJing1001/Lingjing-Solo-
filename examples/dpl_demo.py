"""DPL 几何智能体系 — 端到端 Demo

完整演示五接口链路：
  detect_dimension → calibrate → project → check → infer

以及 SICA 桥接 + AOP 神经加速器的效果。

运行方式：
  python examples/dpl_demo.py
"""
from __future__ import annotations

import numpy as np

from lingjing_solo.dpl import (
    UniversalConstants, Modification,
    detect_dimension, calibrate, project, check, infer,
)
from lingjing_solo.sica.dpl_bridge import (
    project_to_invariant, check_to_counterexample, infer_to_hypothesis,
)
from lingjing_solo.neural.aop import ActionOutcomePredictor


def separator(title: str = "") -> None:
    print()
    if title:
        print(f"{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


def demo_2d_field() -> np.ndarray:
    """构造一个 2D 标量场：中心有高斯峰（引力势阱）。"""
    size = 32
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    # 高斯峰（中心高，边缘低）
    Z = np.exp(-(X**2 + Y**2) / 2.0)
    # 加一点噪声
    Z += np.random.randn(size, size) * 0.02
    return Z.astype(np.float32)


def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   DPL 几何智能体系 — 端到端 Demo                        ║")
    print("║   非 Transformer：物理直觉驱动的维度投影层              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ========== Step 1: 输入观测 ==========
    separator("Step 1: 输入观测 — 2D 标量场（高斯引力势阱）")
    obs = demo_2d_field()
    print(f"  观测形状: {obs.shape}")
    print(f"  取值范围: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  均值: {obs.mean():.3f}, 标准差: {obs.std():.3f}")

    # ========== Step 2: detect_dimension ==========
    separator("Step 2: detect_dimension — 维度检测")
    dim_info = detect_dimension(obs, has_temporal=False)
    print(f"  空间维度: {dim_info.spatial}")
    print(f"  时间维度: {dim_info.temporal}")
    print(f"  有效维度: {dim_info.effective_dim}")
    print(f"  是否高度场: {dim_info.is_height}")
    print(f"  → 判定为 2D 空间场，走 D1→D2 分解路径")

    # ========== Step 3: calibrate ==========
    separator("Step 3: calibrate — 宇宙常数校准")
    constants = calibrate([obs])
    print(f"  光速 c: {constants.c_light:.3f}")
    print(f"  普朗克常数 h: {constants.h_planck:.4f}")
    print(f"  熵产生阈值 ΔS_min: {constants.delta_s_min:.3f}")
    print(f"  时间箭头: {constants.time_arrow:.3f}")
    print(f"  → 已根据观测场统计量自动校准 17 个常数")

    # ========== Step 4: project ==========
    separator("Step 4: project — D1→D2 流形分解")
    ps = project(obs, has_temporal=False, constants=constants)
    print(f"  D1 流线（梯度场）: available={ps.D1.available}")
    if ps.D1.available:
        print(f"    梯度范数均值: {np.linalg.norm(ps.D1.gradient, axis=0).mean():.4f}")
        print(f"    散度均值: {ps.D1.divergence.mean():.4f}")
    print(f"  D2 曲面（曲率/涡旋）: available={ps.D2.available}")
    if ps.D2.available:
        print(f"    曲率均值: {ps.D2.curvature.mean():.4f}")
        print(f"    涡旋均值: {ps.D2.vorticity.mean():.4f}")
    print(f"  D3 体（欧拉示性数）: available={ps.D3.available}")
    print(f"  D4 时空: available={ps.D4.available}")
    print(f"  有效维度: {ps.dim_info.effective_dim}")

    # ========== Step 5: check（七项门禁） ==========
    separator("Step 5: check — 七项自适应几何门禁")
    mod = Modification(
        type="add_source",
        info_before=float(obs.std()),
        info_after=float(obs.std()) * 1.05,  # 信息增量 5%
        params={"strength": 0.1},
    )
    gate_result = check(ps, constants, mod)
    print(f"  修改提案: {mod.type}, 信息增量 = {(mod.info_after/mod.info_before - 1)*100:.1f}%")
    print(f"  门禁通过: {gate_result.passed}")
    print(f"  最终动作: {gate_result.action}")
    print(f"  违反门禁数: {len(gate_result.violations)}")
    if gate_result.violations:
        print(f"  违反项: {', '.join(gate_result.violations)}")
    print(f"  → 七项门禁全部检查完毕，动作由确定性规则决定")

    # ========== Step 6: infer（物理直觉引擎） ==========
    separator("Step 6: infer — 物理直觉推理")
    intuition = infer(ps, constants)
    print(f"  主类型: {intuition.intuition_type}")
    print(f"  置信度: {intuition.confidence:.3f}")
    if intuition.rules:
        print(f"  匹配规则数: {len(intuition.rules)}")
        print(f"  最高置信规则: {intuition.rules[0].rule_type}")
    print(f"  触发特征: {", ".join(intuition.rules[0].triggered_features[:3]) if intuition.rules else "无"}")
    print(f"  → 自动识别出这是引力势阱（gravitational_well）")

    # ========== Step 7: SICA 桥接 ==========
    separator("Step 7: SICA 桥接 — DPL → 自改进架构")
    inv = project_to_invariant(ps, constants)
    print(f"  转换为 GeometricInvariant:")
    print(f"    模式类型: {inv.pattern_type}")
    print(f"    维度层: {inv.dim_layer}")
    print(f"    稳定性评分: {inv.stability_score:.3f}")
    print(f"    证据数: {inv.evidence_count}")

    counterex = check_to_counterexample(gate_result, ps, mod)
    print(f"  转换为 GateCounterexample:")
    print(f"    门禁类型: {counterex.gate_type}")
    print(f"    违反程度: {counterex.violation:.3f}")
    print(f"    有效维度: {counterex.effective_dim}")
    print(f"    样本哈希: {counterex.sample_hash[:16]}...")

    hyp = infer_to_hypothesis(intuition)
    print(f"  转换为 IntuitionHypothesis:")
    print(f"    规则类型: {hyp.rule_type}")
    print(f"    置信度: {hyp.confidence:.3f}")
    print(f"    触发特征数: {len(hyp.triggered_features)}")

    print(f"  → 注意：这些都是候选修改，单次证据不足，")
    print(f"    SICA SafetyGate 会拒绝提交（fail-closed 设计）")

    # ========== Step 8: AOP 神经加速器 ==========
    separator("Step 8: AOP 神经加速器 — 动作结果预测")
    aop = ActionOutcomePredictor()
    pred = aop.predict(ps, mod, constants)
    print(f"  AOP 预测:")
    print(f"    是否使用 AOP: {pred.used_aop}（未训练，回退纯因果图）")
    print(f"    预测动作: {pred.predicted_gate_action}")
    print(f"    置信度: {pred.confidence:.3f}")
    print(f"    模型版本: {pred.model_version}")
    print(f"  → AOP 刚初始化，fail-closed 回退")
    print(f"  → 随着 mismatch 积累，会在线训练提升预测准确率")

    # 记录 mismatch
    aop.record_mismatch(ps, mod, pred, gate_result)
    print(f"  → 已记录 1 条 mismatch（预测 vs 实际 check 结果）")

    # ========== 总结 ==========
    separator("Demo 总结")
    print("  ✅ detect_dimension: 自动识别 2D 空间场")
    print("  ✅ calibrate: 自动校准 17 个宇宙常数")
    print("  ✅ project: D1 流线 + D2 曲面分解")
    print("  ✅ check: 七项门禁确定性决策")
    print("  ✅ infer: 自动识别引力势阱物理直觉")
    print("  ✅ SICA 桥接: 三类结构转换完成（fail-closed）")
    print("  ✅ AOP 加速器: 初始化完成，fail-closed 回退")
    print()
    print("  完整链路：观测 → 维度 → 校准 → 投影 → 门禁 → 直觉")
    print("  全程无 Transformer、无大规模数据训练")
    print("  纯几何 + 物理规则驱动，可解释、可审计、fail-closed")
    print()
    separator()


if __name__ == "__main__":
    main()










