"""tests/test_exploration_rha_penalty.py — RHA 惩罚方法测试。"""
from __future__ import annotations

from types import SimpleNamespace

from lingjing_solo.core import SoloConfig
from lingjing_solo.exploration.explorer import ExplorationEngine


def _engine() -> ExplorationEngine:
    return ExplorationEngine(SoloConfig(), SimpleNamespace())


def test_zero_failures_zero_penalty() -> None:
    e = _engine()
    assert e.rha_penalty("ACTION1") == 0.0


def test_penalty_grows_with_failures() -> None:
    e = _engine()
    e._failure_counts["ACTION1"] = 1
    p1 = e.rha_penalty("ACTION1")
    e._failure_counts["ACTION1"] = 3
    p3 = e.rha_penalty("ACTION1")
    base = SoloConfig().rha_penalty_base
    assert 0.0 < p1 < p3 < base  # 递增，趋近 base


def test_penalty_approaches_base() -> None:
    e = _engine()
    e._failure_counts["ACTION1"] = 100
    base = SoloConfig().rha_penalty_base
    p = e.rha_penalty("ACTION1")
    assert base - 0.01 < p < base  # 大失败次数趋近 base


def test_record_outcome_drives_penalty() -> None:
    e = _engine()
    for _ in range(3):
        e.record_action_outcome("ACTION1", delta_pixels=0, progressed=False)
    assert e.rha_penalty("ACTION1") > 0.0
    # 成功一次 → 清零
    e.record_action_outcome("ACTION1", delta_pixels=1, progressed=True)
    assert e.rha_penalty("ACTION1") == 0.0


def test_different_actions_independent() -> None:
    e = _engine()
    e._failure_counts["ACTION1"] = 5
    assert e.rha_penalty("ACTION1") > 0.0
    assert e.rha_penalty("ACTION2") == 0.0  # 未失败
