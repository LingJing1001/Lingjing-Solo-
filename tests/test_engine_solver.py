"""tests/test_engine_solver.py — 引擎解法器测试。

验证: 有引擎代码 → 搜解法; 没有 → None（回退原流程）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lingjing_solo.engine_solver import EngineSolver


def test_no_engine_code_returns_none(tmp_path: Path) -> None:
    """没有引擎代码 → None（回退原流程）。"""
    solver = EngineSolver(environments_dir=str(tmp_path))
    assert solver.get_action("nonexistent-game") is None
    assert solver.has_solution("nonexistent-game") is False


def test_engine_solver_none_default() -> None:
    """默认 environments_dir 不存在 → None。"""
    solver = EngineSolver(environments_dir="nonexistent_dir")
    assert solver.get_action("any-game") is None


def test_reset_episode() -> None:
    """reset_episode 重置步数索引。"""
    solver = EngineSolver(environments_dir="nonexistent")
    solver._solutions["test"] = ["ACTION6", "ACTION6"]
    solver._step_idx["test"] = 2  # 用完
    solver.reset_episode("test")
    assert solver._step_idx["test"] == 0


def test_solution_exhausted_returns_none() -> None:
    """解法用完 → None。"""
    solver = EngineSolver(environments_dir="nonexistent")
    solver._solutions["test"] = ["ACTION6"]
    solver._step_idx["test"] = 0
    assert solver.get_action("test") == "ACTION6"
    assert solver.get_action("test") is None  # 用完


def test_agent_with_engine_solver_none() -> None:
    """agent + engine_solver=None → 原流程不变。"""
    from lingjing_solo.agent import LingjingSoloAgent
    agent = LingjingSoloAgent(engine_solver=None)
    assert agent.engine_solver is None


def test_agent_with_engine_solver() -> None:
    """agent + engine_solver → 有解法用解法，没解法回退。"""
    from lingjing_solo.agent import LingjingSoloAgent
    solver = EngineSolver(environments_dir="nonexistent")
    agent = LingjingSoloAgent(engine_solver=solver)
    assert agent.engine_solver is solver
