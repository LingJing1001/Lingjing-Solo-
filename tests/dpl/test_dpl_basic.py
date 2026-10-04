"""DPL 基本测试：类型 + 维度检测 + 校准 + 投影。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import (
    DimensionInfo, D1FlowLine, D2Surface, D3Volume, D4Spacetime,
    ProjectedState, UniversalConstants, Modification, GateResult,
)
from lingjing_solo.dpl.dimension import detect_dimension
from lingjing_solo.dpl.calibration import calibrate
from lingjing_solo.dpl.projection import project


class TestTypes:
    def test_universal_constants_defaults(self):
        c = UniversalConstants()
        assert c.c_light == 1.0
        assert c.h_planck == 1e-3
        assert c.K_max == 1.0
        assert c.epsilon_gate == 1e-3
        assert c.I_0 == 1.0
        assert len(c.laws_set) == 0
        assert len(c.conservation_stats) == 0

    def test_gate_result_default(self):
        g = GateResult()
        assert g.passed is True
        assert g.action == "pass"
        assert g.score == 1.0


class TestDimension:
    def test_2d_grid(self):
        obs = np.random.randn(64, 64)
        info = detect_dimension(obs, has_temporal=False)
        assert info.spatial == 2.0
        assert info.temporal is False
        assert info.effective_dim == 2
        assert info.is_height is False

    def test_2_5d_height(self):
        obs = np.random.randn(64, 64, 1)
        info = detect_dimension(obs, has_temporal=False)
        assert info.spatial == 2.5
        assert info.is_height is True
        assert info.effective_dim == 3

    def test_3d_voxel(self):
        obs = np.random.randn(32, 32, 3)
        info = detect_dimension(obs, has_temporal=False)
        assert info.spatial == 3.0
        assert info.is_height is False

    def test_scalar_raises(self):
        with pytest.raises(ValueError):
            detect_dimension(np.array(1.0), has_temporal=False)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            detect_dimension(np.random.randn(10), has_temporal=False)


class TestCalibrate:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calibrate([])

    def test_single_observation(self):
        obs = [np.random.randn(16, 16)]
        c = calibrate(obs)
        assert c.provenance["sample_size"] == "insufficient"
        assert c.c_light == 1.0  # 默认值

    def test_two_observations(self):
        obs1 = np.zeros((16, 16))
        obs2 = np.ones((16, 16))
        c = calibrate([obs1, obs2])
        assert c.c_light > 0
        assert c.h_planck > 0
        assert "c_light" in c.provenance
        assert c.provenance["c_light"] == "calibrated"


class TestProject:
    def test_2d_project(self):
        obs = np.random.randn(32, 32)
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        assert ps.dim_info.spatial == 2.0
        assert ps.D1.available is True
        assert ps.D1.gradient.shape == (32, 32)
        assert ps.D1.divergence.shape == (32, 32)
        assert ps.D1.gradient_vector.shape == (32, 32, 2)
        assert ps.D2.available is True
        assert ps.D2.curvature.shape == (32, 32)
        assert ps.D3.available is False  # 2D 不可用
        assert ps.D4.available is False  # 单帧不可用
        assert ps.D4.is_fake_time is True

    def test_flat_field(self):
        obs = np.ones((16, 16)) * 0.5
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        assert np.max(ps.D1.gradient) < 1e-6  # 平坦场梯度接近 0
