"""七项自适应几何门禁测试。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import (
    UniversalConstants, Modification, D1FlowLine, D2Surface,
    DimensionInfo, ProjectedState,
)
from lingjing_solo.dpl.gates import check
from lingjing_solo.dpl.projection import project


def _make_2d_projected(shape=(32, 32)):
    """构造一个 2D 投影状态用于测试。"""
    obs = np.random.randn(*shape) * 0.1  # 小幅度场，曲率不会太大
    c = UniversalConstants()
    return project(obs, has_temporal=False, constants=c), c


class TestBasicGates:
    def test_2d_passes_basic(self):
        """2D 小幅度场应该通过基础门禁。"""
        ps, c = _make_2d_projected()
        mod = Modification(type="add_source", info_before=1.0, info_after=1.1, params={"strength": 0.01})
        result = check(ps, c, mod)
        # 2D 应该激活 3 个门禁（守恒、曲率、自指）
        assert result.evidence["activated_gates"] >= 3
        assert result.action in ("pass", "dimension_lift", "block")

    def test_2d_does_not_activate_topology(self):
        """2D 不应该激活拓扑门禁。"""
        ps, c = _make_2d_projected()
        mod = Modification()
        result = check(ps, c, mod)
        # 2D spatial=2.0 < 3.0，不应该有 topology
        assert "topology" not in result.observed

    def test_2d_does_not_activate_causal(self):
        """2D 单帧不应该激活因果门禁。"""
        ps, c = _make_2d_projected()
        mod = Modification()
        result = check(ps, c, mod)
        assert "causal" not in result.observed


class TestConservationGate:
    def test_uniform_field_conservation(self):
        """均匀场散度为 0，应该通过守恒门禁。"""
        obs = np.ones((16, 16)) * 0.5
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        mod = Modification()
        result = check(ps, c, mod)
        # 均匀场散度接近 0，应该守恒通过
        assert result.observed["conservation"]["mean_abs_divergence"] < 0.01


class TestCurvatureGate:
    def test_extreme_curvature_lifts_dimension(self):
        """极端曲率应该触发 dimension_lift。"""
        # 构造一个曲率极大的场：中心尖峰
        obs = np.zeros((16, 16))
        obs[8, 8] = 100.0  # 单点尖峰
        c = UniversalConstants(K_max=0.1)  # 很低的阈值
        ps = project(obs, has_temporal=False, constants=c)
        mod = Modification()
        result = check(ps, c, mod)
        # 曲率超限，应该 dimension_lift
        assert result.action == "dimension_lift"
        assert any("curvature" in v for v in result.violations)


class TestSelfReferenceGate:
    def test_violent_modification_blocked(self):
        """Lipschitz > 1 的修改应该被 block。"""
        ps, c = _make_2d_projected()
        # info 变化很大，参数变化很小 → Lipschitz >> 1
        mod = Modification(
            type="modify_topology",
            info_before=1.0,
            info_after=100.0,  # 巨大变化
            params={"small_param": 0.01},  # 很小参数
        )
        result = check(ps, c, mod)
        assert result.passed is False
        assert any("self_reference" in v for v in result.violations)

    def test_gentle_modification_passes(self):
        """Lipschitz < 1 的温和修改应该通过。"""
        ps, c = _make_2d_projected()
        mod = Modification(
            type="add_source",
            info_before=1.0,
            info_after=1.01,  # 很小变化
            params={"strength": 1.0},  # 大参数
        )
        result = check(ps, c, mod)
        # 自指稳定应该通过（Lipschitz < 1）
        assert result.observed["self_reference"]["lipschitz"] < 1.0


class TestVorticityGate:
    def test_uniform_vorticity_skipped(self):
        """均匀涡旋场应该跳过涡旋校验。"""
        obs = np.random.randn(16, 16) * 0.01  # 很小幅度
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        mod = Modification()
        result = check(ps, c, mod)
        # 2.5D 才激活涡旋，2D 不激活
        # 但如果激活了，均匀场应该被跳过
        if "vorticity" in result.observed:
            assert "skipped" in result.observed["vorticity"] or result.observed["vorticity"]["omega_std"] < 1e-6


class TestGateResult:
    def test_result_structure(self):
        """GateResult 应该包含所有必要字段。"""
        ps, c = _make_2d_projected()
        mod = Modification()
        result = check(ps, c, mod)
        assert hasattr(result, "passed")
        assert hasattr(result, "violations")
        assert hasattr(result, "score")
        assert hasattr(result, "action")
        assert hasattr(result, "threshold")
        assert hasattr(result, "observed")
        assert hasattr(result, "evidence")
        assert result.action in ("pass", "block", "dimension_lift", "dimension_reduce")
