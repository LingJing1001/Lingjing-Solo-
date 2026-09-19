"""统一 R3 plan 出口（`lingjing_solo/planning/plan_contract.py`）的契约单测。

只依赖 stdlib + 被测模块，因此同一个文件既能在本仓库根目录跑，也能被
`arc_adaptor/sync_to_arc.sh` 复制进 ARC checkout 的 `tests/unit/` 跑（规则 7）。
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

from lingjing_solo.planning import plan_contract as pc


def _plan(**overrides):
    """一份合法的 plan；用 `overrides` 逐项破坏契约，验证每条断言真的会拦。"""
    args = dict(
        planner="astar",
        input_state_hash="a" * 64,
        candidate_actions=["ACTION3"],
        expected_goal="astar:remaining_goals=1",
        cost=4,
        search_budget={"time_s": 30, "max_nodes": 400, "max_depth": 8},
        validity="candidate",
        evidence_refs=["obs-1"],
    )
    args.update(overrides)
    return pc.build_plan(**args)


# ── 九字段与 happy path ────────────────────────────────────────

def test_build_plan_emits_all_nine_required_fields():
    plan = _plan()
    assert set(pc.REQUIRED_FIELDS) <= set(plan)
    assert plan["schema_version"] == pc.PLAN_SCHEMA
    # plan_id 由 state hash + planner 派生：同一局面同一次规划可复现，不需要外部计数器
    assert plan["plan_id"] == f"{'a' * 12}-astar"


def test_expected_goal_must_be_a_real_product_not_a_placeholder():
    with pytest.raises(pc.PlanContractError, match="expected_goal"):
        _plan(expected_goal="")


# ── §7.1 门槛：state hash / 合法动作集 / 预算有界 ───────────────

def test_state_hash_must_be_present():
    with pytest.raises(pc.PlanContractError, match="input_state_hash"):
        _plan(input_state_hash="")


def test_illegal_candidate_action_is_rejected():
    with pytest.raises(pc.PlanContractError, match="不在当帧合法集合"):
        _plan(legal_actions=["ACTION1", "ACTION2"], candidate_actions=["ACTION6"])


def test_legal_actions_are_kept_in_the_plan_for_replay_checks():
    plan = _plan(legal_actions=["ACTION1", "ACTION3"])
    assert plan["legal_actions"] == ["ACTION1", "ACTION3"]
    # 落盘后从 dict 重读：不需要调用方再传一次合法集，闸门依然生效
    with pytest.raises(pc.PlanContractError, match="不在当帧合法集合"):
        pc.validate_plan({**plan, "candidate_actions": ["ACTION2"]})


def test_budget_is_normalized_and_must_be_positive():
    assert _plan(search_budget={"time_s": 1.5, "max_nodes": 10})["search_budget"] == \
        {"time_ms": 1500, "max_nodes": 10}
    with pytest.raises(pc.PlanContractError, match="search_budget"):
        _plan(search_budget={})
    with pytest.raises(pc.PlanContractError, match="search_budget"):
        _plan(search_budget={"time_s": 30, "max_nodes": 0})


# ── validity 语义：不许把没验过的说成验过（§10）────────────────

def test_verified_offline_requires_evidence_refs():
    with pytest.raises(pc.PlanContractError, match="evidence_refs"):
        _plan(validity="verified_offline", evidence_refs=[])


def test_none_validity_cannot_carry_candidate_actions():
    with pytest.raises(pc.PlanContractError, match="自相矛盾"):
        _plan(validity="none", candidate_actions=["ACTION3"], cost=None)


def test_unknown_validity_is_rejected():
    with pytest.raises(pc.PlanContractError, match="validity"):
        _plan(validity="probably_fine")


# ── planner 注册表：统计口径不能各写各的（§7.5 命名漂移）────────

def test_unregistered_planner_is_rejected():
    with pytest.raises(pc.PlanContractError, match="未登记的 planner"):
        _plan(planner="my_homemade_search")


def test_register_planner_allows_it_but_keeps_one_description_per_name():
    pc.register_planner("astar", "网格/位置 A*（抽象动作，只做寻路）")   # 同描述：幂等
    _plan(planner="astar")
    with pytest.raises(pc.PlanContractError, match="不能改描述"):
        pc.register_planner("astar", "另一种解释")


def test_planner_registry_only_gates_generation_not_audit():
    """审计旧 recording 的进程没 import 过运行器，注册表是空的——那时只能放松⑤。"""
    with pytest.raises(pc.PlanContractError, match="未登记的 planner"):
        _plan(planner="r3_ls20_solver")             # 生成时：未登记就拒
    plan = {**_plan(), "planner": "r3_ls20_solver"}  # 落盘后再读：注册表可能已经空了
    pc.validate_plan(plan, require_registered_planner=False)
    # 放松⑤不等于放松全部：九字段/预算/合法集照旧
    with pytest.raises(pc.PlanContractError, match="search_budget"):
        pc.validate_plan({**plan, "search_budget": {}}, require_registered_planner=False)


# ── 落盘链路：plan 进 tick 后，schema 校验要真的看它 ────────────

def _evidence_compat():
    """`arc_adaptor/evidence_compat.py` 不在包里，ARC checkout 也没有 arc_adaptor —— 取不到就 skip。"""
    root = pathlib.Path(__file__).resolve().parents[1] / "arc_adaptor"
    if not (root / "evidence_compat.py").exists():
        pytest.skip("arc_adaptor/evidence_compat.py 不在当前 checkout 内")
    sys.path.insert(0, str(root))
    return importlib.import_module("evidence_compat")


def test_tick_validation_checks_plan_and_tolerates_recordings_without_one():
    ev = _evidence_compat()
    plan = _plan(legal_actions=["ACTION1", "ACTION3"])
    tick = ev.build_tick(run_id="r", episode_id="e", tick=1, frame=[[0]], state="NOT_FINISHED",
                         levels_completed=0, legal_actions=["ACTION1", "ACTION3"],
                         state_hash="b" * 64, plan_id=plan["plan_id"], plan=plan)
    assert tick["plan"]["plan_id"] == plan["plan_id"]
    assert ev.validate_tick(tick)["plan"]["validity"] == "candidate"
    # ⑤ 之外全生效：候选动作越界要拦
    with pytest.raises(ev.EvidenceValidationError, match="plan 不符合"):
        ev.validate_tick({**tick, "plan": {**plan, "candidate_actions": ["ACTION9"]}})
    # 旧 recording 不带 plan —— 必须仍然可校验（§7.2「既有 recording 不被覆盖」）
    legacy = {k: v for k, v in tick.items() if k != "plan"}
    assert ev.validate_tick(legacy)["plan_id"] == plan["plan_id"]


# ── 形状与类型 ─────────────────────────────────────────────────

def test_candidate_actions_must_be_names_not_engine_enums():
    with pytest.raises(pc.PlanContractError, match="list\\[str\\]"):
        _plan(candidate_actions=[3])          # type: ignore[list-item]


def test_cost_must_be_non_negative_or_none():
    _plan(cost=None)                          # 决策失败时 cost 未知，合法
    with pytest.raises(pc.PlanContractError, match="cost"):
        _plan(cost=-1)


def test_none_plan_is_rejected_rather_than_degraded():
    with pytest.raises(pc.PlanContractError, match="plan 为 None"):
        pc.validate_plan(None)


def test_extra_diagnostics_are_carried_through():
    assert _plan(solver_phase="script")["solver_phase"] == "script"
