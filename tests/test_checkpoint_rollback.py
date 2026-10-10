"""tests/test_checkpoint_rollback.py — 重启与回滚演练测试。

验证:
  - 新进程 load 目标 checkpoint → verify 通过
  - 回滚到上一份 verified checkpoint 成功
  - 权威动作路径不受 shadow 回滚影响（used_for_control=False）
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lingjing_solo.agent import LingjingSoloAgent
from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint
from lingjing_solo.neural.aop_shadow import AOPShadowObserver
from tools.aop.checkpoint_ops import (
    list_verified,
    rollback_to_previous,
    verify_checkpoint,
    write_rollback_report,
)

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
FRAMES = 30


def _make_checkpoint(path: Path, version: str) -> None:
    model = AOPModel(16, len(ACTIONS))
    save_checkpoint(
        path, model,
        {"action_names": ACTIONS, "input_width": 16, "action_classes": len(ACTIONS), "version": version},
    )


def _stub_frame(tick: int) -> dict:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[0, 0] = (tick % 4) + 1
    return {"grid": grid, "state": "NOT_FINISHED", "levels_completed": 0, "available_actions": ACTIONS}


def _run_agent(shadow_observer=None) -> list[str]:
    """跑固定 frame 序列，返回权威动作序列。每次重置随机种子保证确定性。"""
    random.seed(0)
    np.random.seed(0)
    agent = LingjingSoloAgent(shadow_observer=shadow_observer)
    agent.reset(env=SimpleNamespace(game_id="rollback-test"))
    # 现有 ExplorationEngine 缺 rha_penalty（advisor.py:112 会调），打最小 stub
    if not hasattr(agent.explorer, "rha_penalty"):
        agent.explorer.rha_penalty = lambda a: 0.0
    actions: list[str] = []
    for tick in range(FRAMES):
        f = _stub_frame(tick)
        a = agent.choose_action([f], f, valid_actions=ACTIONS)
        actions.append(a)
    if shadow_observer is not None:
        shadow_observer.flush()
    return actions


# ── checkpoint 验证 ───────────────────────────────────────────────────────
def test_verify_checkpoint_writes_verified_marker(tmp_path: Path) -> None:
    ckpt = tmp_path / "v1.pt"
    _make_checkpoint(ckpt, "v1")
    assert verify_checkpoint(ckpt) is True
    assert ckpt.with_suffix(".pt.verified").exists()


def test_verify_rejects_corrupt_checkpoint(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    assert verify_checkpoint(bad) is False
    assert not bad.with_suffix(".pt.verified").exists()


# ── 回滚 ──────────────────────────────────────────────────────────────────
def test_rollback_to_previous_verified(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path / "v1.pt", "v1")
    _make_checkpoint(tmp_path / "v2.pt", "v2")
    assert verify_checkpoint(tmp_path / "v1.pt")
    assert verify_checkpoint(tmp_path / "v2.pt")
    prev = rollback_to_previous(tmp_path / "v2.pt", tmp_path)
    assert prev == tmp_path / "v1.pt"


def test_rollback_none_when_only_one_verified(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path / "v1.pt", "v1")
    assert verify_checkpoint(tmp_path / "v1.pt")
    assert rollback_to_previous(tmp_path / "v1.pt", tmp_path) is None


# ── 权威动作路径不受影响 ──────────────────────────────────────────────────
def test_authoritative_path_unchanged_across_rollback(tmp_path: Path) -> None:
    """shadow 回滚（换 checkpoint）前后，agent 的权威动作序列逐位相等。"""
    ckpt_v1 = tmp_path / "v1.pt"
    ckpt_v2 = tmp_path / "v2.pt"
    _make_checkpoint(ckpt_v1, "v1")
    _make_checkpoint(ckpt_v2, "v2")
    verify_checkpoint(ckpt_v1)
    verify_checkpoint(ckpt_v2)

    out_v1 = tmp_path / "shadow_v1.jsonl"
    out_v2 = tmp_path / "shadow_v2.jsonl"
    obs_v1 = AOPShadowObserver(ckpt_v1, out_v1)
    obs_v2 = AOPShadowObserver(ckpt_v2, out_v2)

    base = _run_agent(None)          # 无 shadow 基线
    actions_v1 = _run_agent(obs_v1)  # shadow v1
    actions_v2 = _run_agent(obs_v2)  # shadow v2

    assert base == actions_v1, "shadow v1 改变了权威动作路径"
    assert base == actions_v2, "shadow v2 改变了权威动作路径"

    # 回滚报告
    write_rollback_report(
        ckpt_v2, ckpt_v1, tmp_path / "rollback_report.json",
        authoritative_unchanged=(base == actions_v1 == actions_v2),
    )
    report = json.loads((tmp_path / "rollback_report.json").read_text(encoding="utf-8"))
    assert report["authoritative_action_path_unchanged"] is True


def test_new_process_loads_target_checkpoint(tmp_path: Path) -> None:
    """模拟新进程：load 目标 checkpoint 并 verify 通过。"""
    ckpt = tmp_path / "target.pt"
    _make_checkpoint(ckpt, "target")
    # 新进程语义 = 重新 import + load
    from lingjing_solo.neural.aop_model import load_checkpoint as fresh_load

    model, config = fresh_load(ckpt)
    assert config["input_width"] == 16
    assert verify_checkpoint(ckpt) is True


def test_list_verified_orders_by_name(tmp_path: Path) -> None:
    for v in ("v1", "v2", "v3"):
        p = tmp_path / f"{v}.pt"
        _make_checkpoint(p, v)
        verify_checkpoint(p)
    listed = list_verified(tmp_path)
    assert [p.name for p in listed] == ["v1.pt", "v2.pt", "v3.pt"]
