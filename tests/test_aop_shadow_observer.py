"""tests/test_aop_shadow_observer.py — Shadow Observer 旁路记录测试。"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint
from lingjing_solo.neural.aop_shadow import AOPShadowObserver, build_label_for_step

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def _make_checkpoint(path: Path) -> None:
    model = AOPModel(16, len(ACTIONS))
    save_checkpoint(path, model, {"action_names": ACTIONS, "input_width": 16, "action_classes": len(ACTIONS)})


def _label(step: int, action: str = "ACTION1") -> dict:
    return build_label_for_step(
        state="NOT_FINISHED", levels_completed=0, legal_actions=ACTIONS,
        tick=step, step_id=step, action=action,
    )


def test_used_for_control_always_false(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    out = tmp_path / "shadow.jsonl"
    obs = AOPShadowObserver(ckpt, out)
    obs.on_episode_reset("ep1")
    for s in range(5):
        obs.observe_predict("ep1", s, _label(s), "ACTION1")
        obs.observe_outcome("ep1", s, {"action_effective": True, "progressed": False})
    obs.flush()
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 5
    assert all(r["used_for_control"] is False for r in rows)


def test_prediction_and_actual_recorded(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    out = tmp_path / "shadow.jsonl"
    obs = AOPShadowObserver(ckpt, out)
    obs.on_episode_reset("ep1")
    obs.observe_predict("ep1", 0, _label(0), "ACTION1")
    obs.observe_outcome("ep1", 0, {"action_effective": True, "progressed": True})
    obs.flush()
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["prediction"] is not None
    assert "action_effective" in row["prediction"]
    assert row["actual"] == {"action_effective": True, "progressed": True}
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0


def test_exception_does_not_raise(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    out = tmp_path / "shadow.jsonl"
    obs = AOPShadowObserver(ckpt, out)
    obs.on_episode_reset("ep1")
    # 空输入 / 缺字段 → encode_label 抛 ValueError → 记 exception，不抛
    obs.observe_predict("ep1", 0, {}, "ACTION1")
    obs.observe_outcome("ep1", 0, {"action_effective": False, "progressed": False})
    obs.flush()
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["prediction"] is None
    assert row["exception"] is not None
    assert row["used_for_control"] is False


def test_flush_writes_unresolved_pending(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    out = tmp_path / "shadow.jsonl"
    obs = AOPShadowObserver(ckpt, out)
    obs.on_episode_reset("ep1")
    obs.observe_predict("ep1", 0, _label(0), "ACTION1")
    obs.observe_predict("ep1", 1, _label(1), "ACTION2")
    # 只回填 step 0，step 1 未回填
    obs.observe_outcome("ep1", 0, {"action_effective": True, "progressed": False})
    obs.flush()
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["actual"] is not None
    assert rows[1]["actual"] is None  # 未回填


def test_episode_reset_flushes_previous(tmp_path: Path) -> None:
    ckpt = tmp_path / "aop.pt"
    _make_checkpoint(ckpt)
    out = tmp_path / "shadow.jsonl"
    obs = AOPShadowObserver(ckpt, out)
    obs.on_episode_reset("ep1")
    obs.observe_predict("ep1", 0, _label(0), "ACTION1")
    obs.on_episode_reset("ep2")  # 触发 ep1 flush
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep1"
