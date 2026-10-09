"""tests/test_aop_controller.py — AOP 高置信预测接入控制路径测试。

覆盖:
  - aop_controller=None → 回归不变
  - 高置信 → 覆盖 fallback，used_for_control=True
  - 低置信 → 保留 fallback，used_for_control=False
  - 异常 → fail-closed 回退，不抛
  - 控制日志写入 + stats 统计
  - 游戏影响: mock controller 高置信预测 → agent 实际选该动作
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lingjing_solo.agent import LingjingSoloAgent
from lingjing_solo.neural.aop_controller import AOPController
from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
FRAMES = 20


def _make_checkpoint(path: Path) -> None:
    model = AOPModel(16, len(ACTIONS))
    save_checkpoint(path, model, {"action_names": ACTIONS, "input_width": 16, "action_classes": len(ACTIONS)})


def _stub_frame(tick: int) -> dict:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[0, 0] = (tick % 4) + 1
    return {"grid": grid, "state": "NOT_FINISHED", "levels_completed": 0, "available_actions": ACTIONS}


def _run_agent(aop_controller=None) -> list[str]:
    random.seed(0)
    np.random.seed(0)
    agent = LingjingSoloAgent(aop_controller=aop_controller)
    agent.reset(env=SimpleNamespace(game_id="ctrl-test"))
    if not hasattr(agent.explorer, "rha_penalty"):
        agent.explorer.rha_penalty = lambda a: 0.0
    actions = []
    for tick in range(FRAMES):
        f = _stub_frame(tick)
        actions.append(agent.choose_action([f], f, valid_actions=ACTIONS))
    return actions


# ── 回归: None 不影响 ─────────────────────────────────────────────────────
def test_none_controller_no_regression() -> None:
    base = _run_agent(None)
    # 再跑一次确认确定性
    again = _run_agent(None)
    assert base == again


# ── AOPController 单元 ────────────────────────────────────────────────────
def test_high_threshold_never_overrides(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.999)
    action, meta = ctrl.advise(
        "ACTION1", state="NOT_FINISHED", levels_completed=0,
        legal_actions=ACTIONS, tick=0, step_id=0,
    )
    assert meta["used_for_control"] is False
    assert action == "ACTION1"  # 保留 fallback


def test_low_threshold_always_overrides(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.001)  # 任何置信都通过
    action, meta = ctrl.advise(
        "ACTION1", state="NOT_FINISHED", levels_completed=0,
        legal_actions=ACTIONS, tick=0, step_id=0,
    )
    assert meta["used_for_control"] is True
    assert action in ACTIONS


def test_exception_falls_back(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.5)

    def boom(x):
        raise RuntimeError("simulated")

    ctrl.model.forward = boom  # type: ignore[method-assign]
    action, meta = ctrl.advise(
        "ACTION1", state="NOT_FINISHED", levels_completed=0,
        legal_actions=ACTIONS, tick=0, step_id=0,
    )
    assert action == "ACTION1"
    assert meta["used_for_control"] is False
    assert meta["exception"] is not None
    assert ctrl.stats()["exceptions"] == 1


def test_empty_legal_actions(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.001)
    action, meta = ctrl.advise(
        "ACTION1", state="NOT_FINISHED", levels_completed=0,
        legal_actions=[], tick=0, step_id=0,
    )
    assert action == "ACTION1"
    assert meta["used_for_control"] is False


def test_control_log_written(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    log = tmp_path / "control.jsonl"
    ctrl = AOPController(ckpt, threshold=0.001, control_log_path=log)
    ctrl.advise("ACTION1", state="NOT_FINISHED", levels_completed=0,
                legal_actions=ACTIONS, tick=0, step_id=0, episode_id="ep1")
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert "used_for_control" in rows[0]
    assert "latency_ms" in rows[0]


def test_stats_accumulate(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.001)
    for i in range(5):
        ctrl.advise("ACTION1", state="NOT_FINISHED", levels_completed=0,
                    legal_actions=ACTIONS, tick=i, step_id=i)
    s = ctrl.stats()
    assert s["calls"] == 5
    assert s["overrides"] == 5  # threshold 极低总是覆盖


# ── 游戏影响: mock controller 覆盖动作 ────────────────────────────────────
class _FakeController:
    """高置信总是返回 ACTION3 的 mock controller。"""
    def __init__(self, target: str = "ACTION3"):
        self.target = target
        self.calls = 0

    def advise(self, fallback_action, **kwargs):
        self.calls += 1
        overrode = fallback_action != self.target
        return self.target, {
            "used_for_control": True, "confidence": 0.95,
            "fallback_action": fallback_action, "chosen_action": self.target,
            "overrode": overrode, "reason": "mock_high_confidence", "exception": None,
        }


def test_controller_overrides_agent_action() -> None:
    """mock controller 高置信预测 ACTION3 → agent 实际选 ACTION3。"""
    ctrl = _FakeController("ACTION3")
    actions = _run_agent(ctrl)
    assert ctrl.calls == FRAMES
    # 每一步都应被 controller 覆盖为 ACTION3
    assert all(a == "ACTION3" for a in actions), f"存在未被覆盖的动作: {actions}"


def test_real_controller_does_not_crash_agent(tmp_path: Path) -> None:
    """真实 AOPController 接入 agent 不崩溃，动作合法。"""
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.5)
    actions = _run_agent(ctrl)
    assert len(actions) == FRAMES
    assert all(a in ACTIONS for a in actions)
    s = ctrl.stats()
    assert s["calls"] == FRAMES
    assert s["exceptions"] == 0


def test_override_changes_rationale(tmp_path: Path) -> None:
    """controller 覆盖时 rationale 应变为 aop_control。"""
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = _FakeController("ACTION3")
    agent = LingjingSoloAgent(aop_controller=ctrl)
    agent.reset(env=SimpleNamespace(game_id="rationale-test"))
    if not hasattr(agent.explorer, "rha_penalty"):
        agent.explorer.rha_penalty = lambda a: 0.0
    f = _stub_frame(0)
    action = agent.choose_action([f], f, valid_actions=ACTIONS)
    assert action == "ACTION3"
    assert agent.last_rationale == "aop_control"
    assert agent.last_aop_meta["used_for_control"] is True


def test_batch_forward_scores_all_actions(tmp_path: Path) -> None:
    """批量前向：confidence 是所有合法动作中最高 progressed 概率，应在 (0,1)。"""
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.99)
    action, meta = ctrl.advise(
        "ACTION1", state="NOT_FINISHED", levels_completed=0,
        legal_actions=ACTIONS, tick=0, step_id=0,
    )
    assert 0.0 < meta["confidence"] < 1.0
    assert action == "ACTION1"  # threshold=0.99 不覆盖


def test_batch_forward_consistent_across_legal_sets(tmp_path: Path) -> None:
    """不同合法动作集都应正常打分不崩溃。"""
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.5)
    for legal in [["ACTION1"], ACTIONS, ["ACTION2", "ACTION4"]]:
        action, meta = ctrl.advise(
            legal[0], state="NOT_FINISHED", levels_completed=0,
            legal_actions=legal, tick=0, step_id=0,
        )
        assert action in legal
        assert 0.0 <= meta["confidence"] <= 1.0


def test_batch_forward_scores_all_actions(tmp_path: Path) -> None:
    """批量前向：confidence 是所有合法动作中最高 progressed 概率，应在 (0,1)。"""
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.99)
    action, meta = ctrl.advise(
        "ACTION1", state="NOT_FINISHED", levels_completed=0,
        legal_actions=ACTIONS, tick=0, step_id=0,
    )
    assert 0.0 < meta["confidence"] < 1.0
    assert action == "ACTION1"  # threshold=0.99 不覆盖


def test_batch_forward_consistent_across_legal_sets(tmp_path: Path) -> None:
    """不同合法动作集都应正常打分不崩溃。"""
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    ctrl = AOPController(ckpt, threshold=0.5)
    for legal in [["ACTION1"], ACTIONS, ["ACTION2", "ACTION4"]]:
        action, meta = ctrl.advise(
            legal[0], state="NOT_FINISHED", levels_completed=0,
            legal_actions=legal, tick=0, step_id=0,
        )
        assert action in legal
        assert 0.0 <= meta["confidence"] <= 1.0
