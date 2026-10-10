"""D4 时空维测试。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import UniversalConstants
from lingjing_solo.dpl.projection import project


class TestD4Spacetime:
    def test_2d_field_has_d4(self):
        """2D 场景 D4 应该可用（赝时间映射）。"""
        obs = np.random.randn(32, 32) * 0.2
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        assert ps.D4.available is True
        assert ps.D4.is_fake_time is True

    def test_d4_has_evolution_sequence(self):
        """D4 应该有赝时间演化序列。"""
        obs = np.random.randn(32, 32) * 0.2
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        assert ps.D4.evolution is not None
        assert ps.D4.evolution.shape[0] == 5  # 5 步演化
        assert ps.D4.evolution.shape[1:] == obs.shape

    def test_d4_has_causal_link(self):
        """D4 应该有因果连接场（散度）。"""
        obs = np.random.randn(32, 32) * 0.2
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        assert ps.D4.causal_link is not None
        assert ps.D4.causal_link.shape == obs.shape

    def test_flat_field_d4_is_static(self):
        """平坦场 D4 应该接近静态。"""
        obs = np.ones((16, 16)) * 0.5
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        assert ps.D4.available is True
        # 平坦场梯度小，演化变化也小
        if ps.D4.evolution is not None:
            change = np.mean(np.abs(ps.D4.evolution[-1] - ps.D4.evolution[0]))
            assert change < 0.01  # 平坦场几乎不演化

    def test_d4_available_means_d1_d2_ready(self):
        """D4 可用的前提是 D1/D2 都可用。"""
        obs = np.random.randn(32, 32) * 0.2
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        if ps.D4.available:
            assert ps.D1.available is True
            assert ps.D2.available is True
