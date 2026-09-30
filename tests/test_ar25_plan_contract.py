"""AR25 运行器的 R3 出口必须符合 §3.2 契约（设计文档 §8.8 的回归闸门）。

② 之前这里返回的是**裸 int 序列**，而且在 planner 函数体内就转成引擎枚举，所以本测试盯三件事：

1. planner 的出口是 `list[str]` 的 abstract action name——`validate_plan` ②会拒 int，
   于是"改回 int 就红"是机械的，不靠 review 记性；
2. 每份 plan 九字段齐全、能独立重放校验（`require_registered_planner=False` 的审计口径）；
3. `validity` 不越级：只有罐头解能声明 `verified_offline`，搜索当场算出来的一律 `candidate`
   （团队规范 §10 禁止把没验过的说成验过）。
4. ③ 之后 `input_state_hash` 是 **正式 R2 状态哈希**（`r2_ar25.state_hash`），不再是
   `sha256(repr(_state_key(g)))` 那个搜索去重键；`tick_trail`/`r2_ar25` 自身的逐 tick 行为
   在只依赖 stdlib 的 `tests/test_ar25_recording.py` 里测，本文件仍受"要真引擎"的限制。

被测模块 import 时要真引擎（`arc_agi`/`arcengine`），拿不到就 skip——本仓库的 CI 与
ARC checkout 的 `tests/unit/` 环境不同，规则 7 要求复制过去也不能炸。
"""
from __future__ import annotations

import contextlib
import pathlib
import sys

import pytest

from lingjing_solo.planning import plan_contract as pc

_RUNNER_NAME = "run_ar25_r234.py"
#: 运行器在两种 checkout 里的落点：本仓库走 `arc_adaptor/agents/strategies/`，
#: ARC checkout 里没有 `arc_adaptor/`，它是被 `sync_to_arc.sh` 按声明清单（`SYNC_PAIRS`，
#: 逐文件复制，已不再有 `cp -R`）放进 `agents/strategies/` 的。规则 ① 那处越界仍登记在
#: §7.3：清单目前把整个 strategies 目录展开了，缩到线上真正需要的那几份归用户定。
_LAYOUTS = (("arc_adaptor", "agents", "strategies"), ("agents", "strategies"))


def _runner_candidates() -> tuple[pathlib.Path, ...]:
    """两种 checkout 布局下的运行器落点，按离测试文件由近到远展开。

    仓库根不能写死成 `parents[1]`：本仓库测试在 `tests/`，ARC 在 `tests/unit/`，写死会让
    ARC 里的根指到 `tests/`，于是明明存在的运行器被报成"找不到文件"，把真原因——它 import
    不了，因为 ARC 没有 `arc_adaptor/paths.py`——盖掉。skip 的原因必须是真的那种。
    """
    here = pathlib.Path(__file__).resolve()
    return tuple(
        root.joinpath(*parts, _RUNNER_NAME)
        for root in here.parents
        for parts in _LAYOUTS
    )


def _runner_path() -> pathlib.Path:
    """本轮实际被测的那份运行器文件（两种 checkout 布局都认）。"""
    candidates = _runner_candidates()
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        pytest.skip(f"运行器不在当前 checkout 内，试过: {[str(p) for p in candidates]}")
    return found


@pytest.fixture(scope="module")
def run():
    """import AR25 运行器（引擎不在就 skip），并把 reconfigure 的副作用挡在真 stdout 上。"""
    pytest.importorskip("arc_agi")
    pytest.importorskip("arcengine")
    runner = _runner_path()
    sys.path.insert(0, str(runner.parent))
    # 运行器 import 时执行 `sys.stdout.reconfigure(...)`；pytest 的捕获对象不一定有这个方法，
    # 所以在真 stdout 上完成 import，再把 pytest 的捕获对象放回去。
    captured, prev = sys.stdout, sys.__stdout__
    try:
        sys.stdout = prev
        import run_ar25_r234 as module
        return module
    except Exception as exc:                    # noqa: BLE001 - 拿不到就是拿不到，如实报
        # ARC checkout 里没有 `arc_adaptor/paths.py`，运行器 import 必然失败：
        # 这里 skip 而不是 error，是为了让"复制过去的测试不影响他人"，同时把原因摊开。
        pytest.skip(f"运行器不可 import（{type(exc).__name__}: {exc}）")
    finally:
        sys.stdout = captured


class _StubEngine:
    """`r2_ar25.enrich` 只读这几个字段：给个最小替身，让单测不依赖真引擎也能出正式哈希。"""

    _state = "PLAYING"
    ovoizfolxfq: dict = {}
    ouurgkpbbjj: list = []
    hsiusrsrdkswnt = 0
    qehjebksqcm = False
    hujpxmlafgh = False
    xukxeewuexo = False
    xjwpeqpcxav = False


@pytest.fixture
def perc(run):
    """一份最小可用的 R2 观测：单 h 轴 + 2 拼块 + 4 目标，字段与 `r2_perceive` 一致。

    末尾过一遍 `r2_ar25.enrich`：③ 起 `input_state_hash` 是正式的状态哈希，它要求观测里带上
    只有引擎句柄读得到的三个字段（`state`/`rotation_distances`/`engine_flags`）。缺这些时
    `state_hash` 抛错而不是算个更粗的哈希，所以伪造观测也得按同一契约装配。
    """
    axis = {"x": -6, "y": 3, "type": "h", "cells": [(0, 0)]}
    moves = [{"idx": 1, "is_axis": False, "x": 0, "y": 0, "cells": [(0, 0)]},
             {"idx": 2, "is_axis": False, "x": 1, "y": 0, "cells": [(0, 0)]}]
    obs = {
        "axes": [axis], "switch_order": [dict(axis, idx=0, is_axis=True, cells=None), *moves],
        "sel_idx": 0, "targets": {(3, 4), (3, 5), (4, 4), (4, 5)}, "budget": 128,
        "steps_left": 128, "n_switch": 3, "movable": moves, "n_axes": 1, "atype": "h",
        "covered": 1, "total_targets": 4, "uncovered": [(3, 4), (3, 5), (4, 4)],
        "won": False, "level": 3, "state_key": ("k", 1, b"\x00"),
    }
    run.r2_ar25.enrich(_StubEngine(), obs)
    return obs


# ── 边界：只有这里能把 name 变成引擎枚举 ─────────────────────
def test_boundary_converts_names_to_enum(run):
    from arcengine import ActionInput, GameAction

    assert run.to_enum("ACTION3") is GameAction.ACTION3
    assert isinstance(run.act_input("ACTION4"), ActionInput)
    assert run.act_input("ACTION4").id is GameAction.ACTION4


def test_boundary_rejects_unknown_names_instead_of_degrading(run):
    """转不动就抛错：静默把 "ACTION9" 当合法动作，等于把非法动作送进引擎。"""
    with pytest.raises(pc.PlanContractError):
        run.to_enum("ACTION9")
    with pytest.raises(pc.PlanContractError):
        run.to_enum("jump")


def test_ar25_action_names_cover_only_the_five_real_moves(run):
    """AR25 只用 ACTION1..5，且实测全部非复杂动作——§8.4 的 payload 条目因此不适用。"""
    from arcengine import GameAction

    assert run.AR25_ACTION_NAMES == ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
    for name in run.AR25_ACTION_NAMES:
        assert GameAction[name].is_complex() is False


def test_nums_to_names_converts_legacy_int_sequences(run):
    assert run.nums_to_names([1, 5, 5, 2]) == ["ACTION1", "ACTION5", "ACTION5", "ACTION2"]
    with pytest.raises(KeyError):
        run.nums_to_names([0])          # 编号 0 在 AR25 里不存在，不能悄悄当成合法动作


# ── planner 出口：abstract name，不是裸 int ──────────────────
def test_single_axis_generator_returns_names(run):
    ax = {"x": -6, "y": 3, "type": "h"}
    movable = [{"idx": 1, "x": 0, "y": 0, "cells": [(0, 0)]}]
    combo = [(2, 4, frozenset({(2, 4)}))]
    path = run._generate_actions(ax, 9, movable, combo, "h", 0, 0, 2)
    assert path and all(isinstance(a, str) and a in run.AR25_ACTION_NAMES for a in path)


def test_dual_axis_generator_returns_names(run):
    h_axis, v_axis = {"x": 0, "y": 5}, {"x": 3, "y": 0}
    move = {"idx": 2, "x": 0, "y": 0}
    p = run._dual_generate_actions(h_axis, v_axis, 9, 8, [(2, 4, 5)], 0, 1,
                                   {"movable": [move]}, 3, 0)
    assert p and all(isinstance(a, str) and a in run.AR25_ACTION_NAMES for a in p)


def test_r4_scorer_is_keyed_by_action_names(run):
    scorer = run.R4Scorer()
    assert set(scorer.stats) == set(run.AR25_ACTION_NAMES)
    assert all(isinstance(k, str) for k in scorer.stats)


# ── 九字段 plan ─────────────────────────────────────────────
def test_search_plan_has_nine_fields_and_validates(run, perc):
    plan = run.ar25_plan(perc, "R3-目标分解", ["ACTION5", "ACTION2", "ACTION4"],
                         {"time_ms": 60000, "max_nodes": run.GOAL_DECOMB_COMBO_CAP,
                          "max_depth": run.BOARD})
    for field in pc.REQUIRED_FIELDS:
        assert field in plan, f"缺 §3.2 字段 {field}"
    assert plan["schema_version"] == pc.PLAN_SCHEMA == "lingjing-r3-plan-v1"
    assert plan["planner"] == "ar25_goal_decomp"
    assert plan["candidate_actions"] == ["ACTION5", "ACTION2", "ACTION4"]
    assert plan["cost"] == 3
    assert len(plan["input_state_hash"]) == 16
    # 审计口径：换进程读 recording 时注册表是空的，所以 validate 只能不查注册表
    assert pc.validate_plan(plan, require_registered_planner=False) is plan


def test_input_state_hash_is_the_formal_r2_hash_not_the_search_key(run, perc):
    """③ 的替换要防"改回临时实现"：plan 的哈希 = `r2_ar25.state_hash`，不再是 `_state_key` 的 repr。

    旧实现是**搜索去重键**，`arc_shadow._snapshot` 里可变的旋转距离/标志位它没全收，所以两者
    一般不同值；这里同时钉住"等于正式哈希"和"不等于占位哈希"，退回 ② 会立刻红。
    """
    import hashlib

    plan = run.ar25_plan(perc, "R3-目标分解", ["ACTION5", "ACTION2"], {"max_nodes": 8})
    placeholder = hashlib.sha256(repr(perc["state_key"]).encode("utf-8")).hexdigest()[:16]
    assert plan["input_state_hash"] == run.r2_ar25.state_hash(perc)
    assert plan["input_state_hash"] != placeholder


def test_goal_and_subgoals_are_decomposition_products(run, perc):
    """`expected_goal` 不得是常量占位；`subgoals` 要把「全覆盖」拆到逐目标点。"""
    goal = run.ar25_expected_goal(perc)
    assert goal.startswith("cover_all_targets:1->4")
    assert "axes=1(h)" in goal and "steps_left=128/128" in goal
    subs = run.ar25_subgoals(perc)
    assert subs[-1] == "all_targets_covered"
    assert set(subs[:-1]) == {f"cover({x},{y})" for x, y in perc["uncovered"]}
    assert "cover(4,5)" not in subs            # 输入态里已覆盖的目标不再列进子目标


def test_dual_axis_goal_reports_real_axis_types_not_legacy_placeholder(run):
    """`perc["atype"]` 只在单轴关卡有值；双轴的 expected_goal 不能读成 `axes=2(?)`。"""
    goal = run.ar25_expected_goal(_perc_dual(run))
    assert "axes=2(h+v)" in goal


def _perc_dual(run):
    """双轴观测（h+v 各一根，类型标签都解析出来了，但 atype 按老约定是 '?'）。

    和 `perc` fixture 同一套装配契约：③ 起 `ar25_plan` 要算正式 `input_state_hash`，
    未 enrich 的观测会被 `state_hash` 拒掉，所以这里也过一遍 stub 引擎。
    """
    axes = [{"x": 0, "y": 5, "type": "h", "cells": [(0, 0)]},
            {"x": 3, "y": 0, "type": "v", "cells": [(0, 0)]}]
    move = {"idx": 2, "is_axis": False, "x": 0, "y": 0, "cells": [(0, 0)]}
    obs = {"axes": axes, "switch_order": [], "sel_idx": 0, "targets": {(1, 2)},
           "budget": 320, "steps_left": 320, "n_switch": 3, "movable": [move],
           "n_axes": 2, "atype": "?", "covered": 0, "total_targets": 1,
           "uncovered": [(1, 2)], "won": False, "level": 5, "state_key": ("k", 5, b"")}
    run.r2_ar25.enrich(_StubEngine(), obs)
    return obs


def test_canned_plan_is_the_only_layer_may_claim_verified_offline(run, perc):
    plan = run.ar25_plan(perc, "R0-已知解法", ["ACTION2"] * 4,
                         {"max_nodes": 1, "max_depth": 4})
    assert plan["planner"] == "t0_known_solution"
    assert plan["validity"] == "verified_offline"
    assert any("KNOWN_SOLUTIONS" in ref for ref in plan["evidence_refs"])
    # 搜索产物不许升级：同输入下 beam 出来的那份仍是 candidate
    search = run.ar25_plan(perc, "R3-beam", ["ACTION2"] * 4, {"time_ms": 1000, "max_nodes": 8})
    assert search["validity"] == "candidate"


def test_every_method_string_maps_to_a_registered_planner(run):
    """method 串留住是为了跟历史 report.json 对账；planner 名必须已登记（§7.5 命名漂移条）。"""
    assert set(run.AR25_METHOD_TO_PLANNER) == {
        "R0-已知解法", "R3-目标分解", "R3-双轴分解", "R3-beam",
        "R3-arc_shadow", "R3-搜索", "R4-贪心"}
    for method, planner in run.AR25_METHOD_TO_PLANNER.items():
        assert planner in pc.PLANNER_NAMES, f"{method} → {planner} 未登记"


def test_unknown_method_string_fails_closed(run, perc):
    with pytest.raises(KeyError):
        run.ar25_plan(perc, "R9-新加的层", ["ACTION2"], {"time_ms": 1000, "max_nodes": 8})


def test_contract_rejects_the_legacy_int_output(run, perc):
    """②之前的出口形态：裸 int。这条测试就是防止有人改回去还"看起来是契约"。"""
    with pytest.raises(pc.PlanContractError):
        pc.build_plan(planner="ar25_goal_decomp", input_state_hash="a" * 16,
                      candidate_actions=[3, 4], expected_goal="cover_all_targets",
                      cost=2, search_budget={"time_ms": 1000}, validity="candidate")


def test_contract_rejects_illegal_action_and_unbounded_budget(run, perc):
    plan = run.ar25_plan(perc, "R3-目标分解", ["ACTION2"], {"time_ms": 1000, "max_nodes": 8})
    with pytest.raises(pc.PlanContractError):
        pc.validate_plan({**plan, "candidate_actions": ["ACTION6"]})      # 不在当帧合法集
    with pytest.raises(pc.PlanContractError):
        pc.validate_plan({**plan, "search_budget": {}})                   # 预算无界 = 不可验收
    with pytest.raises(pc.PlanContractError):
        pc.validate_plan({**plan, "search_budget": {"max_nodes": 0}})     # 非正
    with pytest.raises(pc.PlanContractError):
        pc.validate_plan({**plan, "validity": "verified_offline", "evidence_refs": []})


def test_budget_fields_are_positive_or_clamped(run):
    """`_pos_int` 兜的是 0/None：报 1 而不是让 validate 放行一个 0 预算。"""
    assert run._pos_int(0) == 1 and run._pos_int(None) == 1 and run._pos_int("12") == 12
    assert run._pos_int(-5) == 1
    with pytest.raises(pc.PlanContractError):
        pc.validate_plan(run.ar25_plan(_perc_dual(run), "R3-双轴分解",
                                       ["ACTION2"], {"time_ms": 0}))      # 0 预算进不去


def test_failure_path_carries_no_plan(run, perc):
    """决策失败就记 None：不拿 validity=none 的空壳冒充"套过契约"。"""
    assert run.r234_solve.__doc__ is not None
    src = _runner_path().read_text(encoding="utf-8")
    assert 'return None, "失败", None' in src
    assert '"plan": None' in src
