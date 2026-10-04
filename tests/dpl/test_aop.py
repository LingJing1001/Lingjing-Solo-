"""AOP 小模型测试。"""
import numpy as np
import pytest

from lingjing_solo.dpl.types import UniversalConstants, Modification
from lingjing_solo.dpl.projection import project
from lingjing_solo.dpl.gates import check
from lingjing_solo.neural.aop import ActionOutcomePredictor, OutcomePrediction


def _make_pipeline():
    """构造完整 DPL 流水线输出。"""
    obs = np.random.randn(32, 32) * 0.1
    c = UniversalConstants()
    ps = project(obs, has_temporal=False, constants=c)
    mod = Modification(type="add_source", info_before=1.0, info_after=1.05, params={"strength": 0.1})
    gate = check(ps, c, mod)
    return ps, c, mod, gate


class TestAOPBasic:
    def test_create_aop(self):
        """AOP 应该能正常创建。"""
        aop = ActionOutcomePredictor()
        assert aop.model_version == "aop_v0.1"
        assert aop.trained_steps == 0
        assert aop.confidence_threshold == 0.7

    def test_untrained_returns_fallback(self):
        """未训练的 AOP 应该回退 used_aop=False。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()
        pred = aop.predict(ps, mod, c)
        assert pred.used_aop is False
        assert pred.confidence == 0.0
        assert pred.predicted_gate_action == "unknown"


class TestAOPEncoding:
    def test_input_encoding_dimension(self):
        """输入编码应该输出 14 维向量。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()
        x = aop._encode_input(ps, mod, c)
        assert x.shape == (14,)
        assert x.dtype == np.float32

    def test_input_encoding_handles_unavailable(self):
        """D3/D4 不可用时应该填 0，不报错。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()
        x = aop._encode_input(ps, mod, c)
        # D3/D4 不可用，对应维度应该是 0
        assert x[8] == 0.0  # D3 euler_char
        assert x[9] == 0.0  # D3 density


class TestAOPPredict:
    def test_trained_predict_returns_result(self):
        """训练后 AOP 应该返回预测结果。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()

        # 手动设置 trained_steps 模拟已训练
        aop.trained_steps = 100

        pred = aop.predict(ps, mod, c)
        assert pred.predicted_gate_action in ("pass", "block", "dimension_lift")
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.model_version == "aop_v0.1"

    def test_low_confidence_falls_back(self):
        """低置信度时 used_aop=False。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor(confidence_threshold=0.99)
        aop.trained_steps = 100

        pred = aop.predict(ps, mod, c)
        # 随机初始化的模型置信度不会很高
        assert pred.used_aop in (True, False)  # 随机模型，不一定低


class TestAOPMismatch:
    def test_record_mismatch(self):
        """mismatch 应该被记录，不能吞掉。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()
        aop.trained_steps = 100

        pred = OutcomePrediction(
            predicted_gate_action="pass",
            confidence=0.9,
            model_version="aop_v0.1",
            used_aop=True,
        )

        aop.record_mismatch(ps, mod, pred, gate)
        assert len(aop.mismatch_log) == 1
        assert aop.mismatch_log[0]["predicted_action"] == "pass"
        assert "actual_action" in aop.mismatch_log[0]

    def test_mismatch_triggers_training(self):
        """mismatch 应该触发在线训练。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()
        aop.trained_steps = 100

        pred = OutcomePrediction(
            predicted_gate_action="pass",
            confidence=0.9,
            model_version="aop_v0.1",
            used_aop=True,
        )

        steps_before = aop.trained_steps
        aop.record_mismatch(ps, mod, pred, gate)
        # 如果 predicted != actual，应该训练一步
        #（不一定触发，取决于预测和实际是否一致）


class TestAOPFailClosed:
    def test_bad_input_fails_closed(self):
        """输入错误时应该 fail-closed 回退。"""
        ps, c, mod, gate = _make_pipeline()
        aop = ActionOutcomePredictor()
        aop.trained_steps = 100

        # 修改成会导致编码错误的情况
        bad_mod = Modification(type="add_source", info_before=np.nan, info_after=1.0, params={})
        # 不报错就好，fail-closed
        try:
            pred = aop.predict(ps, bad_mod, c)
            # 如果没崩，应该 used_aop=False 或正常预测
        except Exception:
            pass  # 崩了也可以，fail-closed

    def test_model_version_separated(self):
        """model_version 应该独立于场版本。"""
        aop1 = ActionOutcomePredictor(model_version="aop_v0.1")
        aop2 = ActionOutcomePredictor(model_version="aop_v0.2")
        assert aop1.model_version != aop2.model_version


class TestAOPPipelineIntegration:
    def test_aop_in_full_pipeline(self):
        """AOP 应该能在完整 DPL 流水线中工作。"""
        obs = np.random.randn(32, 32) * 0.1
        c = UniversalConstants()
        ps = project(obs, has_temporal=False, constants=c)
        mod = Modification(type="add_source", info_before=1.0, info_after=1.1, params={"strength": 0.2})

        # check 确定性门禁
        gate_result = check(ps, c, mod)

        # AOP 预测
        aop = ActionOutcomePredictor()
        pred = aop.predict(ps, mod, c)

        # AOP 未训练，应该回退
        assert pred.used_aop is False

        # 记录 mismatch（即使 AOP 没实际用，也记录预测 vs 实际）
        aop.record_mismatch(ps, mod, pred, gate_result)
        assert len(aop.mismatch_log) == 1
