"""物理直觉引擎测试。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import UniversalConstants, Modification
from lingjing_solo.dpl.projection import project
from lingjing_solo.dpl.gates import check
from lingjing_solo.dpl.intuition import infer


def _make_2d_projected(obs=None):
    """构造一个 2D 投影状态用于测试。"""
    if obs is None:
        obs = np.random.randn(32, 32) * 0.1
    c = UniversalConstants()
    ps = project(obs, has_temporal=False, constants=c)
    return ps, c


class TestInferBasic:
    def test_infer_returns_result(self):
        """infer 应该返回 IntuitionResult。"""
        ps, c = _make_2d_projected()
        result = infer(ps, c)
        assert hasattr(result, "rules")
        assert hasattr(result, "confidence")
        assert hasattr(result, "intuition_type")
        assert result.intuition_type in (
            "geometric_gravity", "temporal_rhythm",
            "topological_navigation", "conservation_law", "no_intuition",
        )

    def test_2d_has_no_topological_intuition(self):
        """2D D3 不可用，不应该有拓扑直觉。"""
        ps, c = _make_2d_projected()
        result = infer(ps, c)
        topo_rules = [r for r in result.rules if r.rule_type == "topological"]
        # D3 不可用，topological 规则应该是 None，不在 rules 里
        assert len(topo_rules) == 0 or topo_rules[0].confidence == 0.0


class TestGravitationalIntuition:
    def test_flat_field_no_gravity(self):
        """平坦场不应该有引力直觉。"""
        obs = np.ones((16, 16)) * 0.5
        c = UniversalConstants(K_max=1.0)
        ps = project(obs, has_temporal=False, constants=c)
        result = infer(ps, c)
        g_rules = [r for r in result.rules if r.rule_type == "gravitational_well"]
        if g_rules:
            assert g_rules[0].confidence < 0.1  # 平坦场置信度很低

    def test_peaked_field_has_gravity(self):
        """中心尖峰场应该有引力直觉。"""
        obs = np.zeros((16, 16))
        obs[8, 8] = 5.0  # 中心尖峰
        c = UniversalConstants(K_max=10.0, gravity_threshold_ratio=0.01)
        ps = project(obs, has_temporal=False, constants=c)
        result = infer(ps, c)
        g_rules = [r for r in result.rules if r.rule_type == "gravitational_well"]
        assert len(g_rules) > 0
        # 尖峰场曲率大，应该有引力置信度
        assert g_rules[0].confidence > 0.1


class TestConservationIntuition:
    def test_uniform_field_conservation(self):
        """均匀场散度为零，应该有守恒直觉。"""
        obs = np.ones((16, 16)) * 0.5
        c = UniversalConstants(epsilon_gate=1.0)  # 很大阈值
        ps = project(obs, has_temporal=False, constants=c)
        result = infer(ps, c)
        c_rules = [r for r in result.rules if r.rule_type == "conservation"]
        assert len(c_rules) > 0
        # 均匀场散度接近 0，守恒置信度应该很高
        assert c_rules[0].confidence > 0.5


class TestPeriodicIntuition:
    def test_fake_time_skipped(self):
        """伪时间场景周期直觉应该被跳过。"""
        ps, c = _make_2d_projected()
        result = infer(ps, c)
        p_rules = [r for r in result.rules if r.rule_type == "periodic_dynamics"]
        # D4.is_fake_time=True，应该跳过或置信度为 0
        if p_rules:
            assert p_rules[0].confidence == 0.0
            assert "fake_time_skipped" in p_rules[0].triggered_features or not p_rules[0].triggered_features


class TestPriorityOrder:
    def test_gravity_has_highest_priority(self):
        """引力直觉优先级最高。"""
        # 构造一个同时有引力和守恒特征的场
        obs = np.zeros((16, 16))
        obs[8, 8] = 3.0  # 中心尖峰 → 引力
        c = UniversalConstants(K_max=10.0, gravity_threshold_ratio=0.01, epsilon_gate=1.0)
        ps = project(obs, has_temporal=False, constants=c)
        result = infer(ps, c)
        # 引力优先级最高，主类型应该是 geometric_gravity
        # （前提是引力置信度 > 0.1）
        g_rules = [r for r in result.rules if r.rule_type == "gravitational_well"]
        if g_rules and g_rules[0].confidence > 0.1:
            assert result.intuition_type == "geometric_gravity"


class TestFullPipeline:
    def test_five_interface_pipeline(self):
        """五接口完整链路：detect → calibrate → project → check → infer。"""
        obs_seq = [np.random.randn(16, 16) * 0.1 for _ in range(5)]

        # calibrate
        c = __import__("lingjing_solo.dpl", fromlist=["calibrate"]).calibrate(obs_seq)

        # project
        ps = project(obs_seq[-1], has_temporal=False, constants=c)

        # check
        mod = Modification(type="add_source", info_before=1.0, info_after=1.05, params={"strength": 0.1})
        gate_result = check(ps, c, mod)

        # infer
        intuition = infer(ps, c)

        # 验证所有输出都存在
        assert ps.dim_info is not None
        assert ps.D1.available is True
        assert gate_result.passed is not None
        assert intuition.intuition_type is not None
        assert intuition.confidence >= 0.0
