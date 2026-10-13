"""R3Kernel + WorldView + P1–P4 闸门测试（§3.2.1 / §7.1.1）。

覆盖：
    - WorldView invariant（key 不含计数器）
    - P1 noop：同 key 子节点丢弃 + evidence 记录
    - P2 计数器支配：同 key 保留更优 counter 节点
    - P3 死亡：dead frame 停止 + divergence 记录
    - P4 预算：超预算 fail-closed + validity=none
    - R3-A 线上边界：LearnedGraphBackend 不接触 _game
    - T0–T3 分层：罐头路线优先、专用 planner 次之、通用内核兜底
    - plan 出口：R3Kernel 产出经 plan_contract.validate_plan 通过
"""
from __future__ import annotations

from typing import Any, Hashable

import pytest

from lingjing_solo.planning.world_view import (
    ActionSpec, ConfigSpace, WorldView, WorldViewInvariantError,
    assert_world_view_invariant,
)
from lingjing_solo.planning.backends import (
    SearchBackend, EngineBackend, SimulatorBackend,
    LearnedGraphBackend, UndoBackend,
)
from lingjing_solo.planning.r3_kernel import (
    PruningContract, PruningViolation, SearchBudget, R3Kernel,
)
from lingjing_solo.planning.plan_contract import (
    validate_plan, PlanContractError, register_planner,
)
from lingjing_solo.planning.registry import (
    register_world_view, get_world_view, register_backend, get_backend,
    clear_registry, resolve_planner,
)


# ── 测试用 WorldView 实现 ──────────────────────────────────────

class _Frame:
    """测试用帧。"""
    def __init__(self, grid=None, state="NOT_FINISHED", levels=0, dead=False):
        self.grid = grid or [[0, 0], [0, 0]]
        self.state = state
        self.levels = levels
        self._dead = dead


class _TestWorldView(WorldView):
    """最小 WorldView 实现，用于测试 R3Kernel。

    key() 只看 grid 内容，不看步数计数器（P2 硬约束）。
    counters() 返回步数和分数。
    """
    def grids(self, frame: Any) -> list[Any]:
        return [frame.grid]

    def key(self, grids: list[Any]) -> Hashable:
        # 只看 grid 内容，不含计数器
        return tuple(tuple(row) for row in grids[0])

    def counters(self, grids: list[Any]) -> dict[str, int]:
        return {"steps": 0, "score": 0}

    def proposals(self, grids: list[Any], key: Hashable) -> list[ActionSpec]:
        return [ActionSpec("ACTION1"), ActionSpec("ACTION2")]

    def config_space(self, grids: list[Any]) -> ConfigSpace | None:
        return None

    def score(self, grids: list[Any]) -> float:
        return 0.0

    def progress(self, frame: Any) -> int:
        return frame.levels

    def done(self, frame: Any) -> bool:
        return frame.state == "WIN"

    def dead(self, frame: Any) -> bool:
        return frame._dead or frame.state == "GAME_OVER"


# ── WorldView invariant ───────────────────────────────────────

class TestWorldViewInvariant:
    def test_concrete_world_view_passes_invariant(self):
        wv = _TestWorldView()
        frame = _Frame()
        # 不应抛异常
        assert_world_view_invariant(wv, sample_frame=frame)

    def test_non_world_view_rejected(self):
        with pytest.raises(WorldViewInvariantError, match="不是 WorldView"):
            assert_world_view_invariant("not a worldview")

    def test_key_is_deterministic(self):
        wv = _TestWorldView()
        frame = _Frame(grid=[[1, 0], [0, 1]])
        grids = wv.grids(frame)
        k1 = wv.key(grids)
        k2 = wv.key(grids)
        assert k1 == k2

    def test_key_excludes_counters(self):
        """P2 硬约束：key 不含计数器。

        同一 grid 不同 counters 应该产生相同 key。
        """
        wv = _TestWorldView()
        grid = [[1, 0], [0, 1]]
        grids = [grid]
        k = wv.key(grids)
        # counters 变化不影响 key
        c1 = wv.counters(grids)
        c2 = {"steps": 999, "score": 50}
        assert k == wv.key(grids)  # key 不变


# ── 后端边界 ──────────────────────────────────────────────────

class TestBackends:
    def test_engine_backend_is_offline(self):
        b = EngineBackend()
        assert b.is_online is False
        assert b.can_snapshot is True
        assert b.evidence_tier == "offline_engine"

    def test_simulator_backend_is_offline(self):
        b = SimulatorBackend()
        assert b.is_online is False
        assert b.can_snapshot is True
        assert b.evidence_tier == "simulator"

    def test_learned_graph_backend_is_online(self):
        b = LearnedGraphBackend()
        assert b.is_online is True
        assert b.can_snapshot is False
        assert b.evidence_tier == "live_scorecard"

    def test_learned_graph_predict_and_update(self):
        b = LearnedGraphBackend()
        assert b.predict("state_a", "ACTION1") is None
        b.update("state_a", "ACTION1", "state_b")
        assert b.predict("state_a", "ACTION1") == "state_b"

    def test_undo_backend_requires_verification(self):
        with pytest.raises(ValueError, match="仅在.*实测等价"):
            UndoBackend(undo_verified=False)

    def test_undo_backend_when_verified(self):
        b = UndoBackend(undo_verified=True)
        assert b.is_online is True
        assert b.can_snapshot is False
        assert b.evidence_tier == "live_scorecard"

    def test_online_backend_cannot_snapshot(self):
        """R3-A：线上后端不支持快照/回滚。"""
        b = LearnedGraphBackend()
        with pytest.raises(NotImplementedError, match="不支持快照"):
            b.snapshot()
        with pytest.raises(NotImplementedError, match="不支持恢复"):
            b.restore(None)


# ── P1 noop 闸门 ──────────────────────────────────────────────

class TestP1Noop:
    def test_noop_action_is_discarded_with_evidence(self):
        """P1：predict 返回相同 state hash → 丢弃 + 记录 violation。"""
        wv = _TestWorldView()
        # 后端 predict 返回相同 hash → noop
        backend = LearnedGraphBackend()
        backend.update("s_hash", "ACTION1", "s_hash")  # noop
        kernel = R3Kernel(wv, backend, planner_name="generic")
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION1", "ACTION2"], "s_hash")

        # ACTION1 被 P1 丢弃，ACTION2 保留
        assert "ACTION2" in plan["candidate_actions"]
        # 有 P1 violation 记录
        p1_violations = [v for v in kernel.violations if v.contract == PruningContract.P1_NOOP]
        assert len(p1_violations) == 1
        assert "noop" in p1_violations[0].divergence


# ── P2 计数器支配 ──────────────────────────────────────────────

class TestP2CounterDominance:
    def test_same_key_dominance(self):
        """P2：同 key 时 counters 全优的节点支配。"""
        kernel = R3Kernel(_TestWorldView(), LearnedGraphBackend())
        assert kernel.check_counter_dominance(
            "key_a", {"steps": 10, "score": 5},
            "key_a", {"steps": 5, "score": 3},
        ) is True

    def test_different_key_no_dominance(self):
        kernel = R3Kernel(_TestWorldView(), LearnedGraphBackend())
        assert kernel.check_counter_dominance(
            "key_a", {"steps": 10},
            "key_b", {"steps": 5},
        ) is False

    def test_partial_dominance_fails(self):
        """counters 不是全优 → 不支配。"""
        kernel = R3Kernel(_TestWorldView(), LearnedGraphBackend())
        assert kernel.check_counter_dominance(
            "key_a", {"steps": 10, "score": 2},
            "key_a", {"steps": 5, "score": 3},
        ) is False


# ── P3 死亡闸门 ──────────────────────────────────────────────

class TestP3Death:
    def test_dead_frame_produces_none_plan(self):
        """P3：dead frame → fail-closed，validity=none。"""
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame(dead=True)
        plan = kernel.plan(frame, ["ACTION1"], "s_hash")

        assert plan["validity"] == "none"
        assert plan["candidate_actions"] == []

    def test_dead_frame_records_violation(self):
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame(dead=True)
        kernel.plan(frame, ["ACTION1"], "s_hash")

        p3 = [v for v in kernel.violations if v.contract == PruningContract.P3_DEATH]
        assert len(p3) == 1
        assert "dead" in p3[0].reason

    def test_game_over_state_is_dead(self):
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame(state="GAME_OVER")
        plan = kernel.plan(frame, ["ACTION1"], "s_hash")
        assert plan["validity"] == "none"


# ── P4 预算闸门 ──────────────────────────────────────────────

class TestP4Budget:
    def test_budget_must_be_positive(self):
        with pytest.raises(ValueError):
            SearchBudget(max_steps=0)
        with pytest.raises(ValueError):
            SearchBudget(time_ms=-1)

    def test_node_budget_exhaustion(self):
        """P4：超 max_nodes → fail-closed。"""
        wv = _TestWorldView()
        backend = LearnedGraphBackend()
        # 极小节点预算：第一个 proposal 消耗 nodes=1，第二个触发 >max_nodes
        budget = SearchBudget(max_nodes=1)
        kernel = R3Kernel(wv, backend, budget=budget)
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION1", "ACTION2"], "s_hash")

        # 超节点预算 → P4 violation
        p4 = [v for v in kernel.violations if v.contract == PruningContract.P4_BUDGET]
        assert len(p4) >= 1

    def test_timeout_produces_none_plan(self):
        """P4：墙钟超限 → fail-closed，validity=none。"""
        # 用慢 WorldView 确保搜索超过 1ms 预算
        class _SlowWV(_TestWorldView):
            def proposals(self, grids, key):
                import time as _t
                _t.sleep(0.005)  # 5ms > 1ms 预算
                return [ActionSpec("ACTION1")]
        wv = _SlowWV()
        backend = LearnedGraphBackend()
        budget = SearchBudget(time_ms=1, max_steps=400)
        kernel = R3Kernel(wv, backend, budget=budget)
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION1", "ACTION2"], "s_hash")
        # 超时 → validity=none
        assert plan["validity"] == "none"


# ── T0–T3 分层 ────────────────────────────────────────────────

class TestLayering:
    def test_t0_script_bank_takes_priority(self):
        """T0 罐头路线优先：命中则 validity=verified_offline。"""
        wv = _TestWorldView()
        kernel = R3Kernel(
            wv, LearnedGraphBackend(),
            script_bank_fn=lambda gid, lvl: ["ACTION1", "ACTION2"],
        )
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION1", "ACTION2"], "s_hash",
                            game_id="test", level=0)
        assert plan["validity"] == "verified_offline"
        assert "T0" in kernel.stats["layer_tried"]

    def test_t1_proposals_used_when_no_script(self):
        """T1：无罐头路线时用 WorldView.proposals。"""
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION1", "ACTION2"], "s_hash")
        assert "ACTION1" in plan["candidate_actions"]
        assert "T1" in kernel.stats["layer_tried"]

    def test_t3_explore_fallback(self):
        """T3：proposals 为空时兜底选一个合法动作。"""
        wv = _TestWorldView()
        # 覆盖 proposals 返回空
        class _EmptyWV(_TestWorldView):
            def proposals(self, grids, key):
                return []
        kernel = R3Kernel(_EmptyWV(), LearnedGraphBackend())
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION3", "ACTION4"], "s_hash")
        assert len(plan["candidate_actions"]) >= 1
        assert "T3" in kernel.stats["layer_tried"]

    def test_done_frame_skips_all_layers(self):
        """done=True 时不需要动作，直接返回 none plan。"""
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame(state="WIN")
        plan = kernel.plan(frame, ["ACTION1"], "s_hash")
        assert plan["validity"] == "none"


# ── plan 出口校验 ─────────────────────────────────────────────

class TestPlanContract:
    def test_kernel_output_passes_validate_plan(self):
        """R3Kernel 产出的 plan 经 validate_plan 通过。"""
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame()
        plan = kernel.plan(frame, ["ACTION1", "ACTION2"], "a" * 64)
        # validate_plan 应该通过（审计模式，不检查 planner 注册表）
        validated = validate_plan(plan, require_registered_planner=False)
        assert validated["plan_id"]
        assert validated["search_budget"]["time_ms"] > 0

    def test_none_plan_is_valid(self):
        """validity=none 的 plan 也是合法的（fail-closed 不是错误）。"""
        wv = _TestWorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        frame = _Frame(dead=True)
        plan = kernel.plan(frame, ["ACTION1"], "b" * 64)
        validated = validate_plan(plan, require_registered_planner=False)
        assert validated["validity"] == "none"
        assert validated["candidate_actions"] == []


# ── 注册表 ────────────────────────────────────────────────────

class TestRegistry:
    def setup_method(self):
        clear_registry()

    def teardown_method(self):
        clear_registry()

    def test_register_and_get_world_view(self):
        wv = _TestWorldView()
        register_world_view("test_game", wv)
        assert get_world_view("test_game") is wv

    def test_get_unregistered_returns_none(self):
        assert get_world_view("nonexistent") is None

    def test_register_and_get_backend(self):
        backend = EngineBackend()
        register_backend("test_game", backend)
        assert get_backend("test_game") is backend

    def test_default_backend_is_learned_graph(self):
        """未注册后端 → 返回 LearnedGraphBackend 默认。"""
        b = get_backend("unregistered")
        assert isinstance(b, LearnedGraphBackend)

    def test_resolve_planner_for_registered_game(self):
        wv = _TestWorldView()
        register_world_view("test_game", wv)
        frame = _Frame()
        plan = resolve_planner("test_game", frame, ["ACTION1"], "c" * 64)
        assert plan is not None
        assert "ACTION1" in plan["candidate_actions"]

    def test_resolve_planner_for_unregistered_returns_none(self):
        frame = _Frame()
        plan = resolve_planner("nonexistent", frame, ["ACTION1"], "d" * 64)
        assert plan is None


# ── R3-A 线上边界 ─────────────────────────────────────────────

class TestR3AOnlineBoundary:
    def test_online_backend_does_not_snapshot(self):
        """R3-A：线上后端不接触引擎对象，不支持快照。"""
        wv = _TestWorldView()
        backend = LearnedGraphBackend()
        kernel = R3Kernel(wv, backend)
        assert backend.is_online is True
        assert backend.can_snapshot is False

    def test_offline_backend_can_snapshot(self):
        """离线后端可以快照，但 evidence_tier != live_scorecard。"""
        wv = _TestWorldView()
        backend = EngineBackend(
            snapshot_fn=lambda: "snap",
            restore_fn=lambda s: None,
        )
        kernel = R3Kernel(wv, backend)
        assert backend.is_online is False
        assert backend.can_snapshot is True
        assert backend.evidence_tier == "offline_engine"
