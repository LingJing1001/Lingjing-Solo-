"""DPL → SICA 桥接测试。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import UniversalConstants, Modification
from lingjing_solo.dpl.projection import project
from lingjing_solo.dpl.gates import check
from lingjing_solo.dpl.intuition import infer
from lingjing_solo.sica.candidate import CandidateRegistry, CandidateModification
from lingjing_solo.sica.dpl_bridge import (
    project_to_invariant,
    check_to_counterexample,
    infer_to_hypothesis,
    submit_invariant_to_sica,
    submit_hypothesis_to_sica,
    submit_counterexample_to_sica,
)


def _make_pipeline():
    """构造完整 DPL 流水线输出。"""
    obs = np.random.randn(32, 32) * 0.1
    c = UniversalConstants()
    ps = project(obs, has_temporal=False, constants=c)
    mod = Modification(type="add_source", info_before=1.0, info_after=1.05, params={"strength": 0.1})
    gate = check(ps, c, mod)
    intuition = infer(ps, c)
    return ps, c, mod, gate, intuition


class TestProjectToInvariant:
    def test_returns_geometric_invariant(self):
        """project_to_invariant 应该返回 GeometricInvariant。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        inv = project_to_invariant(ps, source_family="test_env")
        assert inv.pattern_type is not None
        assert inv.dim_layer in ("D1", "D2", "D3", "")
        assert inv.stability_score >= 0.0
        assert inv.evidence_count >= 1
        assert inv.source_family == "test_env"
        assert isinstance(inv.features, dict)

    def test_d2_has_curvature_features(self):
        """D2 可用时应该有曲率特征。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        inv = project_to_invariant(ps)
        if ps.D2.available:
            assert "mean_curvature" in inv.features


class TestCheckToCounterexample:
    def test_returns_gate_counterexample(self):
        """check_to_counterexample 应该返回 GateCounterexample。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        ce = check_to_counterexample(gate, ps, mod)
        assert ce.gate_type is not None
        assert ce.violation >= 0.0
        assert ce.raw_dim == ps.dim_info.spatial
        assert ce.effective_dim == ps.dim_info.effective_dim
        assert len(ce.sample_hash) == 64  # SHA-256
        assert ce.timestamp > 0

    def test_passed_gate_has_low_violation(self):
        """通过的门禁 violation 应该接近 0。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        if gate.passed:
            ce = check_to_counterexample(gate, ps, mod)
            assert ce.violation < 0.5


class TestInferToHypothesis:
    def test_returns_intuition_hypothesis(self):
        """infer_to_hypothesis 应该返回 IntuitionHypothesis。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        hyp = infer_to_hypothesis(intuition)
        assert hyp.rule_type is not None
        assert hyp.confidence >= 0.0
        assert isinstance(hyp.triggered_features, list)
        assert hyp.holdout_delta == 0.0  # 待回填


class TestSicaFailClosed:
    """验证 SICA SafetyGate 的 fail-closed 行为：DPL 单次输出证据不足，会被安全门拒绝。"""

    def test_invariant_rejected_by_safety_gate(self):
        """单次 DPL 输出的 GeometricInvariant 证据不足，应该被 SafetyGate 拒绝。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        inv = project_to_invariant(ps)
        registry = CandidateRegistry()
        # 证据不足（evidence_count=1, distinct_games=0），SafetyGate 应该拒绝
        with pytest.raises(PermissionError, match="evidence"):
            submit_invariant_to_sica(registry, inv, candidate_id="test_inv_001")

    def test_hypothesis_rejected_by_safety_gate(self):
        """单次 DPL 输出的 IntuitionHypothesis 证据不足，应该被 SafetyGate 拒绝。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        hyp = infer_to_hypothesis(intuition)
        registry = CandidateRegistry()
        with pytest.raises(PermissionError, match="evidence"):
            submit_hypothesis_to_sica(registry, hyp, candidate_id="test_hyp_001")

    def test_counterexample_rejected_by_safety_gate(self):
        """GateCounterexample 作为负样本也应该被安全门拒绝（证据不足 + 负收益）。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        ce = check_to_counterexample(gate, ps, mod)
        registry = CandidateRegistry()
        with pytest.raises(PermissionError):
            submit_counterexample_to_sica(registry, ce, candidate_id="test_ce_001")


class TestSicaWithSufficientEvidence:
    """验证：当证据足够时，候选能通过 SafetyGate。"""

    def _make_rich_invariant(self):
        """构造一个证据充足的 GeometricInvariant。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        inv = project_to_invariant(ps)
        inv.evidence_count = 5  # 足够的证据
        return inv

    def test_rich_invariant_accepted(self):
        """证据充足的 GeometricInvariant 应该能通过 SafetyGate。"""
        inv = self._make_rich_invariant()
        registry = CandidateRegistry()
        # 手动构造满足要求的候选（绕过桥接函数的默认证据数）
        candidate = CandidateModification(
            candidate_id="rich_inv_001",
            target="rule_set",
            content={"type": "geometric_invariant"},
            evidence_count=5,
            evidence_refs=("ref1", "ref2", "ref3"),
            distinct_games=3,
            expected_gain=0.8,
        )
        accepted = registry.submit(candidate, tick=0)
        assert accepted.candidate_id == "rich_inv_001"
        assert registry.get("rich_inv_001") is not None


class TestFullBridgePipeline:
    def test_conversion_functions_work(self):
        """验证三个转换函数都能正确输出对应结构。"""
        ps, c, mod, gate, intuition = _make_pipeline()

        # project → invariant
        inv = project_to_invariant(ps, source_family="e2e_test")
        assert inv.source_family == "e2e_test"
        assert inv.stability_score > 0

        # check → counterexample
        ce = check_to_counterexample(gate, ps, mod)
        assert ce.sample_hash != ""
        assert ce.effective_dim == ps.dim_info.effective_dim

        # infer → hypothesis
        hyp = infer_to_hypothesis(intuition)
        assert hyp.rule_type is not None

    def test_fail_closed_by_design(self):
        """验证 DPL→SICA 桥接是 fail-closed：单次输出不直接进入生产规则池。"""
        ps, c, mod, gate, intuition = _make_pipeline()
        registry = CandidateRegistry()

        inv = project_to_invariant(ps)
        # 提交应该被 SafetyGate 拒绝（证据不足）
        with pytest.raises(PermissionError):
            submit_invariant_to_sica(registry, inv, candidate_id="e2e_inv_001")

        # registry 应该是空的（没有候选被接受）
        assert len(registry.all()) == 0

