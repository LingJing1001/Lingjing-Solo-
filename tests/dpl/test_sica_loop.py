"""SICA 提交闭环测试：多轮 DPL 输出积累证据，成功提交候选修改。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import UniversalConstants, Modification
from lingjing_solo.dpl.projection import project
from lingjing_solo.sica.candidate import CandidateModification, CandidateRegistry


class TestSICACommitLoop:
    def test_single_dpl_output_rejected(self):
        """单次 DPL 输出证据不足，应该被 SafetyGate 拒绝。"""
        registry = CandidateRegistry()
        candidate = CandidateModification(
            candidate_id="test_001",
            target="rule_set",
            content={"type": "geometric_invariant"},
            evidence_count=1,
            evidence_refs=("ref1",),
            distinct_games=1,
            expected_gain=0.5,
        )
        with pytest.raises(PermissionError, match="evidence"):
            registry.submit(candidate, tick=0)

    def test_sufficient_evidence_accepted(self):
        """证据充足的候选应该被 SafetyGate 接受并提交。"""
        registry = CandidateRegistry()
        candidate = CandidateModification(
            candidate_id="test_002",
            target="rule_set",
            content={"type": "geometric_invariant", "pattern": "stable_flow"},
            evidence_count=5,
            evidence_refs=("ref1", "ref2", "ref3", "ref4", "ref5"),
            distinct_games=3,
            expected_gain=0.8,
        )
        accepted = registry.submit(candidate, tick=0)
        assert accepted.candidate_id == "test_002"
        assert accepted.target == "rule_set"

    def test_multi_round_dpl_accumulates_evidence(self):
        """多轮 DPL 输出积累证据，最终成功提交完整闭环。"""
        registry = CandidateRegistry()
        accepted_count = 0
        evidence_rounds = 0

        # 模拟 5 轮 DPL 输出，每轮积累证据
        for round_idx in range(5):
            # 每轮生成稍微不同的观测
            obs = np.random.randn(32, 32) * 0.2 + 0.5  # 加偏置让它更稳定
            c = UniversalConstants()
            ps = project(obs, has_temporal=False, constants=c)

            # 用这轮 DPL 输出作为证据
            evidence_rounds += 1

            # 当积累了足够证据，尝试提交
            if evidence_rounds >= 3:
                candidate = CandidateModification(
                    candidate_id=f"dpl_invariant_round_{round_idx}",
                    target="rule_set",
                    content={
                        "type": "geometric_invariant",
                        "pattern": "stable_flow",
                        "effective_dim": ps.dim_info.effective_dim,
                    },
                    evidence_count=evidence_rounds,
                    evidence_refs=tuple(f"round_{i}" for i in range(evidence_rounds)),
                    distinct_games=min(evidence_rounds, 3),  # 模拟跨游戏
                    expected_gain=0.6 + 0.05 * round_idx,
                    complexity_delta=0.1,
                )

                try:
                    registry.submit(candidate, tick=round_idx)
                    accepted_count += 1
                except PermissionError:
                    pass  # 还不够证据，继续积累

        # 至少应该有一次成功提交（第 3 轮以后应该满足门槛）
        assert accepted_count >= 1, f"Expected at least 1 accepted candidate, got {accepted_count}"

    def test_fail_closed_never_accepts_insufficient_evidence(self):
        """fail-closed：证据永远不足时，永远不会提交。"""
        registry = CandidateRegistry()
        rejected = 0

        for i in range(10):
            candidate = CandidateModification(
                candidate_id=f"weak_{i}",
                target="rule_set",
                content={},
                evidence_count=1,  # 永远只有 1 个证据
                evidence_refs=("only_one",),
                distinct_games=1,
                expected_gain=0.1,
            )
            try:
                registry.submit(candidate, tick=i)
            except PermissionError:
                rejected += 1

        # 全部应该被拒绝
        assert rejected == 10, "All weak candidates should be rejected"
