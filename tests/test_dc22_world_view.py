"""DC22WorldView 的 Tier A 单测（团队规范《最小可运行闭环》§3.2.1 / §7.1.1）。

Tier A 口径：只用**合成帧**，不启动引擎、不读 `state/` 产物，因此换机器、
换目录都必须照样跑通。引擎真值（`environment_files/dc22/fdcac232/dc22.py`）
在这里只作为**注释里的事实来源**，断言本身不依赖它。

覆盖三组东西：

1. 接口与三条硬约束
   - §3.2.1-1 单调量（剩余步数条）只进 `counters()`，`key()` 里遮掉 row 63；
   - §3.2.1-2 `progress()/done()/dead()` 只读官方 `FrameData` 字段，
     视觉变化不许冒充进度、步数耗尽不许自己判死；
   - §3.2.1-3 `proposals()` 只返回 abstract name，坐标进 `payload_hint`。
   另加 R3-A 线上边界：profile 只许碰 5 个公开字段（用严格帧当场证明）。
2. 闸门可触发证据（R3-C）：经 **R3Kernel** 时 P1 / P3 / P4 各至少 fire 一次，
   并且 fire 之后 plan 出口仍然合法（`validate_plan` 不抛）。
3. 诚实记账：P2 当前**没有**接进 `plan()`，所以用真实 profile 跑任何帧都
   不该出现 P2 violation —— 这条测试的作用是"哪天接上了、它就会提醒改断言"。
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Hashable

import numpy as np
import pytest

from lingjing_solo.planning.backends import LearnedGraphBackend
from lingjing_solo.planning.dc22_world_view import (
    BUDGET_BAR_ROW, DC22WorldView, KEYBOARD_ACTIONS, MAX_CLICK_CANDIDATES,
)
from lingjing_solo.planning.plan_contract import validate_plan
from lingjing_solo.planning.r3_kernel import PruningContract, R3Kernel, SearchBudget
from lingjing_solo.planning.registry import (
    clear_registry, register_backend, register_world_view, resolve_planner,
)
from lingjing_solo.planning.world_view import (
    ActionSpec, assert_world_view_invariant,
)

# 引擎帧恒 64×64（camera 尺寸），步数条画在最后一行 row 63。
H, W = 64, 64

#: profile 在 dc22 上唯一允许读取的 FrameData 字段（R3-A 线上边界）。
PUBLIC_FIELDS = ("frame", "state", "levels_completed", "win_levels", "available_actions")

#: 合法的 dc22 动作集（引擎 `available_actions=[1,2,3,4,6]`，ACTION5 是 `pass`）。
LEGAL = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]


class _GameState(Enum):
    """模仿 arc_agi 的 `GameState`：profile 经 `.value/.name` 读它。"""
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


def bar_grid(*, consumed: int, playfield: dict[tuple[int, int], int] | None = None
             ) -> list[list[int]]:
    """造一张 dc22 形状的帧：背景 0 + 若干前景格 + 最后一行步数条。

    步数条按引擎画法建模：头部 `consumed` 格是"已消耗"色(1)，
    尾部 `64-consumed` 格是"剩余"色(2)；`consumed` 越大代表越接近耗尽。
    """
    g = [[0] * W for _ in range(H)]
    # 一道横墙（4-连通域，面积 20 ≤ cap 64 → 会进点击候选）
    for c in range(10, 30):
        g[20][c] = 5
    for (r, c), v in (playfield or {}).items():
        g[r][c] = v
    g[BUDGET_BAR_ROW] = [1] * consumed + [2] * (W - consumed)
    return g


class _Frame:
    """最小可用的 FrameData 替身：只带公开字段。"""

    def __init__(self, grid, *, state=_GameState.NOT_FINISHED, levels=0, win_levels=5):
        self.frame = [grid]
        self.state = state
        self.levels_completed = levels
        self.win_levels = win_levels
        self.available_actions = LEGAL


class _StrictFrame(_Frame):
    """R3-A 探针：允许字段之外的任何属性访问当场报错。

    只要 profile 试图读 `_game`、`room_grid`、`step_count` 这类引擎私有对象，
    这里就会 AttributeError，测试立刻失败——比 grep 源码更硬。
    """

    def __getattr__(self, name: str) -> Any:  # 只有"常规查找失败"才会走到这里
        raise AttributeError(
            f"R3-A 越界：profile 试图读取非公开字段 {name!r}"
            f"（白名单={PUBLIC_FIELDS}）"
        )


def _same_playfield_vary_bar():
    """两份只有步数条消耗不同的帧（局面逐格相同）。"""
    low = _Frame(bar_grid(consumed=4), levels=0)     # 剩余 60 格
    high = _Frame(bar_grid(consumed=62), levels=0)   # 剩余 2 格
    return low, high


# ── 1a. 接口完整性 ────────────────────────────────────────────

class TestInterface:
    def test_profile_satisfies_invariant(self):
        assert_world_view_invariant(DC22WorldView(), sample_frame=_Frame(bar_grid(consumed=8)))

    def test_grids_returns_raw_and_masked(self):
        """`grids()` 给两层：第 0 层给 counters 读条，第 1 层是遮掉条行的主网格。"""
        grids = DC22WorldView().grids(_Frame(bar_grid(consumed=8)))
        assert len(grids) == 2
        assert np.count_nonzero(grids[0][BUDGET_BAR_ROW]) > 0            # 原始条行有内容
        assert np.count_nonzero(grids[1][BUDGET_BAR_ROW]) == 0  # 主网格那条被抹平

    def test_grids_on_unreadable_frame_returns_empty(self):
        """读不到网格就返回空，不猜一个假局面出来。"""
        assert DC22WorldView().grids(None) == []
        assert DC22WorldView().key([]) == ""

    def test_config_space_is_none(self):
        """dc22 无配置层（§3.2.2 的"通常无配置层"一类）→ T2 配置搜索必须整层跳过。"""
        assert DC22WorldView().config_space(
            DC22WorldView().grids(_Frame(bar_grid(consumed=8)))
        ) is None

    def test_score_refuses_visual_proxy(self):
        """帧上读不出 dc22 得分 → 恒 0，不许拿视觉代理冒充启发式。"""
        wv = DC22WorldView()
        assert wv.score(wv.grids(_Frame(bar_grid(consumed=8)))) == 0.0
        assert wv.score(wv.grids(_Frame(bar_grid(consumed=8, playfield={(30, 5): 3})))) == 0.0


# ── 1b. 硬约束 1：key() 不含单调计数器 ─────────────────────────

class TestKeyExcludesCounters:
    def test_key_stable_while_budget_bar_shrinks(self):
        low, high = _same_playfield_vary_bar()
        wv = DC22WorldView()
        assert wv.key(wv.grids(low)) == wv.key(wv.grids(high))

    def test_counters_do_shrink_with_the_same_key(self):
        """key 相同的同时 counters 必须真的在变（否则上一条是空断言）。"""
        low, high = _same_playfield_vary_bar()
        wv = DC22WorldView()
        c_low = wv.counters(wv.grids(low))
        c_high = wv.counters(wv.grids(high))
        assert c_low["step_budget_cells_approx"] > c_high["step_budget_cells_approx"]
        assert wv.key(wv.grids(low)) == wv.key(wv.grids(high))

    def test_key_changes_when_playfield_changes(self):
        wv = DC22WorldView()
        a = _Frame(bar_grid(consumed=8))
        b = _Frame(bar_grid(consumed=8, playfield={(30, 5): 3}))
        assert wv.key(wv.grids(a)) != wv.key(wv.grids(b))

    def test_key_is_stable_hash_str(self):
        wv = DC22WorldView()
        k = wv.key(wv.grids(_Frame(bar_grid(consumed=8))))
        assert isinstance(k, str) and len(k) == 16 and k == wv.key(wv.grids(
            _Frame(bar_grid(consumed=8))))


# ── 1c. 硬约束 2：终局证据只来自官方字段 ──────────────────────

class TestOfficialFieldsNotForged:
    def test_progress_reads_levels_completed(self):
        wv = DC22WorldView()
        assert wv.progress(_Frame(bar_grid(consumed=8), levels=3)) == 3

    def test_progress_ignores_visual_change(self):
        """画面变了但官方 levels 没涨 → 进度不许自己涨（不拿视觉变化冒充）。"""
        wv = DC22WorldView()
        before = _Frame(bar_grid(consumed=8), levels=0)
        after = _Frame(bar_grid(consumed=8, playfield={(30, 5): 3}), levels=0)
        assert wv.key(wv.grids(before)) != wv.key(wv.grids(after))
        assert wv.progress(before) == wv.progress(after) == 0

    @pytest.mark.parametrize("value,expected", [(None, 0), ("x", 0), (2.7, 2)])
    def test_progress_never_raises_on_garbage_field(self, value, expected):
        """官方字段脏值只能降级，不许把异常带进决策路径。"""
        frame = _Frame(bar_grid(consumed=8))
        frame.levels_completed = value
        assert DC22WorldView().progress(frame) == expected

    def test_dead_only_from_game_over(self):
        wv = DC22WorldView()
        assert wv.dead(_Frame(bar_grid(consumed=8), state=_GameState.GAME_OVER)) is True
        assert wv.dead(_Frame(bar_grid(consumed=8), state=_GameState.NOT_FINISHED)) is False
        assert wv.dead(_Frame(bar_grid(consumed=8), state=_GameState.WIN)) is False

    def test_dead_not_forged_from_empty_budget_bar(self):
        """步数条快见底 ≠ 已死：死亡只认引擎给的 GAME_OVER（§3.2.1 硬约束 2）。"""
        wv = DC22WorldView()
        nearly_out = _Frame(bar_grid(consumed=63), state=_GameState.NOT_FINISHED)
        assert wv.counters(wv.grids(nearly_out))["step_budget_cells_approx"] <= 1
        assert wv.dead(nearly_out) is False

    def test_done_from_win_state_or_official_levels(self):
        wv = DC22WorldView()
        assert wv.done(_Frame(bar_grid(consumed=8), state=_GameState.WIN)) is True
        assert wv.done(_Frame(bar_grid(consumed=8), levels=5, win_levels=5)) is True
        assert wv.done(_Frame(bar_grid(consumed=8), levels=4, win_levels=5)) is False

    def test_done_does_not_fire_on_intermediate_level(self):
        """过了一关不等于通关：levels 只追平 win_levels 才算 done。"""
        wv = DC22WorldView()
        assert wv.done(_Frame(bar_grid(consumed=8), levels=1, win_levels=8)) is False


# ── 1d. 硬约束 3 + R3-A：动作方言与线上边界 ───────────────────

class TestActionDialectAndOnlineBoundary:
    def test_proposals_are_abstract_names_only(self):
        specs = DC22WorldView().proposals(
            DC22WorldView().grids(_Frame(bar_grid(consumed=8))), "k"
        )
        assert specs and all(isinstance(s, ActionSpec) for s in specs)
        assert {s.name for s in specs} <= {*KEYBOARD_ACTIONS, "ACTION6"}
        assert "ACTION5" not in {s.name for s in specs}  # 引擎里 ACTION5 是 pass

    def test_four_directions_come_first(self):
        specs = DC22WorldView().proposals(
            DC22WorldView().grids(_Frame(bar_grid(consumed=8))), "k"
        )
        assert [s.name for s in specs][:4] == list(KEYBOARD_ACTIONS)

    def test_click_coordinates_live_only_in_payload_hint(self):
        """坐标只能在 `payload_hint` 里；转换归 boundary adapter，profile 不塞枚举。"""
        specs = DC22WorldView().proposals(
            DC22WorldView().grids(_Frame(bar_grid(consumed=8))), "k"
        )
        clicks = [s for s in specs if s.name == "ACTION6"]
        assert 0 < len(clicks) <= MAX_CLICK_CANDIDATES
        for s in clicks:
            assert isinstance(s.payload_hint.get("x"), int)
            assert isinstance(s.payload_hint.get("y"), int)
            assert "," not in s.name and "(" not in s.name

    def test_proposals_exclude_the_budget_bar_cells(self):
        """条行不许被当成"可点的东西"，否则每个状态都会提议点 UI。"""
        wv = DC22WorldView()
        specs = wv.proposals(wv.grids(_Frame(bar_grid(consumed=32))), "k")
        for s in specs:
            if s.name == "ACTION6":
                assert s.payload_hint["y"] != BUDGET_BAR_ROW

    def test_profile_touches_only_public_frame_fields(self):
        """R3-A：profile 逐方法调用一遍，越界字段访问会直接 AttributeError。"""
        frame = _StrictFrame(bar_grid(consumed=8), levels=1)
        wv = DC22WorldView()
        grids = wv.grids(frame)
        wv.key(grids)
        wv.counters(grids)
        wv.proposals(grids, wv.key(grids))
        wv.config_space(grids)
        wv.score(grids)
        wv.progress(frame)
        wv.done(frame)
        wv.dead(frame)
        # 反证：私有字段确实读不到（证明上面的"没报错"不是因为兜住了异常）
        with pytest.raises(AttributeError, match="R3-A 越界"):
            frame._game  # noqa: B018 — 就是要它炸

    def test_kernel_plan_on_strict_frame(self):
        """整条内核路径（含 plan 出口）都不需要引擎私有对象。"""
        frame = _StrictFrame(bar_grid(consumed=8), levels=0)
        wv = DC22WorldView()
        plan = R3Kernel(wv, LearnedGraphBackend()).plan(
            frame, LEGAL, wv.key(wv.grids(frame))
        )
        assert plan["candidate_actions"]
        validate_plan(plan, require_registered_planner=False)


# ── 2. 闸门可触发证据（R3-C）：经 R3Kernel 各 fire 一次 ────────

def _kernel(frame, *, backend=None, budget=None, planner_name="generic"):
    """`planner_name` 必须用已登记的名字：`build_plan` 在生成时就查注册表。"""
    wv = DC22WorldView()
    return R3Kernel(wv, backend or LearnedGraphBackend(),
                    budget=budget, planner_name=planner_name), wv.key(wv.grids(frame))


class TestP1NoopFiresWithRealProfile:
    def test_p1_discards_self_looping_action_and_keeps_the_rest(self):
        frame = _Frame(bar_grid(consumed=8))
        kernel, key = _kernel(frame)
        kernel.backend.update(key, "ACTION1", key)  # 学到的自环 = noop
        plan = kernel.plan(frame, LEGAL, key)

        fired = [v for v in kernel.violations if v.contract is PruningContract.P1_NOOP]
        assert fired and fired[0].action == "ACTION1"
        assert "ACTION1" not in plan["candidate_actions"]
        assert "ACTION2" in plan["candidate_actions"]
        assert fired[0].divergence.startswith("noop:")
        assert validate_plan(plan, require_registered_planner=False)["plan_id"]

    def test_p1_does_not_starve_the_candidate_set(self):
        """把四向全判成 noop → 剩下的点击候选仍在，plan 非空（不是 fail-closed）。"""
        frame = _Frame(bar_grid(consumed=8))
        kernel, key = _kernel(frame)
        for name in KEYBOARD_ACTIONS:
            kernel.backend.update(key, name, key)
        plan = kernel.plan(frame, LEGAL, key)

        click_specs = [s for s in DC22WorldView().proposals(
            DC22WorldView().grids(frame), key) if s.name == "ACTION6"]
        assert click_specs, "合成帧应至少给出一个点击候选，否则这条测试是空断言"
        assert plan["candidate_actions"] == ["ACTION6"] * len(click_specs)


class TestP3DeathFiresWithRealProfile:
    def test_p3_fail_closed_on_game_over(self):
        frame = _Frame(bar_grid(consumed=63), state=_GameState.GAME_OVER)
        kernel, key = _kernel(frame)
        plan = kernel.plan(frame, LEGAL, key)

        fired = [v for v in kernel.violations if v.contract is PruningContract.P3_DEATH]
        assert len(fired) == 1
        assert "dead" in fired[0].reason
        assert plan["validity"] == "none"
        assert plan["expected_goal"] == "none:p3_death"
        assert plan["candidate_actions"] == []

    def test_p3_fires_before_any_layer_is_tried(self):
        """死亡检查在最前：不许白烧 T1/T2/T3。"""
        frame = _Frame(bar_grid(consumed=63), state=_GameState.GAME_OVER)
        kernel, key = _kernel(frame)
        kernel.plan(frame, LEGAL, key)
        assert kernel.stats["layer_tried"] == []

    def test_win_frame_is_not_death_and_still_fail_closed(self):
        frame = _Frame(bar_grid(consumed=8), state=_GameState.WIN)
        kernel, key = _kernel(frame)
        plan = kernel.plan(frame, LEGAL, key)
        assert not [v for v in kernel.violations
                    if v.contract is PruningContract.P3_DEATH]
        assert plan["validity"] == "none"
        assert plan["expected_goal"] == "none:already_done"


class TestP4BudgetFiresWithRealProfile:
    def test_p4_node_budget(self):
        """dc22 一个 tick 最多能提议 4+8=12 个候选 → 预算 1 时必然撞 P4 节点闸门。"""
        frame = _Frame(bar_grid(consumed=8))
        kernel, key = _kernel(frame, budget=SearchBudget(max_nodes=1))
        plan = kernel.plan(frame, LEGAL, key)
        fired = [v for v in kernel.violations if v.contract is PruningContract.P4_BUDGET]
        assert fired and f"> max_nodes {kernel.budget.max_nodes}" in fired[0].reason
        assert len(plan["candidate_actions"]) == 2  # 撞闸门前只收到 2 个候选

    def test_p4_wall_clock(self):
        """墙钟超预算 → fail-closed，validity=none。

        用慢 proposals 触发（dc22 的 `_components` 本身够快，没法在真 profile 上
        自然超时，所以这里只替换耗时、不替换候选内容）。
        """
        class _SlowDC22(DC22WorldView):
            def proposals(self, grids: list[Any], key: Hashable) -> list[ActionSpec]:
                time.sleep(0.02)  # 20ms > 1ms 预算
                return super().proposals(grids, key)

        frame = _Frame(bar_grid(consumed=8))
        wv = _SlowDC22()
        kernel = R3Kernel(wv, LearnedGraphBackend(), budget=SearchBudget(time_ms=1))
        plan = kernel.plan(frame, LEGAL, wv.key(wv.grids(frame)))
        fired = [v for v in kernel.violations if v.contract is PruningContract.P4_BUDGET]
        assert fired and "wall clock" in fired[0].reason
        assert plan["validity"] == "none"
        assert plan["expected_goal"] == "none:p4_timeout"


class TestP2NotWiredYetHonestAccounting:
    """P2 的计数器比较本身可用，但 `plan()` 没调用它 —— 旁路实测过：线上 fire 恒 0。

    断言写成"不许 fire"，是为了让**将来接 P2 的人必须显式改这条测试**，
    而不是让"P2 有效"变成一个没人验过的口头结论。
    """

    def test_dominance_comparison_works_on_dc22_counters(self):
        low, high = _same_playfield_vary_bar()
        wv = DC22WorldView()
        kernel = R3Kernel(wv, LearnedGraphBackend())
        k_low, k_high = wv.key(wv.grids(low)), wv.key(wv.grids(high))
        c_low, c_high = wv.counters(wv.grids(low)), wv.counters(wv.grids(high))
        assert k_low == k_high  # 同局面
        assert kernel.check_counter_dominance(k_low, c_low, k_high, c_high) is True
        assert kernel.check_counter_dominance(k_high, c_high, k_low, c_low) is False

    def test_plan_never_emits_p2_violation(self):
        frame = _Frame(bar_grid(consumed=8))
        kernel, key = _kernel(frame)
        kernel.backend.update(key, "ACTION1", key)
        kernel.plan(frame, LEGAL, key)
        assert not [v for v in kernel.violations
                    if v.contract is PruningContract.P2_COUNTER_DOMINANCE]


# ── 3. R3-D：真实 profile 经统一路由给出同一结论 ──────────────

class TestUnifiedRoutingWithRealProfile:
    def setup_method(self):
        clear_registry()

    def teardown_method(self):
        clear_registry()

    def test_resolve_planner_matches_kernel_for_dc22(self):
        frame = _Frame(bar_grid(consumed=8))
        wv = DC22WorldView()
        key = wv.key(wv.grids(frame))
        register_world_view("dc22", wv)
        backend = LearnedGraphBackend()
        backend.update(key, "ACTION1", key)  # 让 P1 真的参与一次
        register_backend("dc22", backend)

        kernel = R3Kernel(wv, backend, planner_name="generic")
        mine = kernel.plan(frame, LEGAL, key, game_id="dc22", level=0)
        routed = resolve_planner("dc22", frame, LEGAL, key, level=0)

        assert "ACTION1" not in mine["candidate_actions"]
        for field_name in ("validity", "expected_goal", "candidate_actions", "planner"):
            assert mine[field_name] == routed[field_name], field_name

    def test_unregistered_game_returns_none_from_router(self):
        """未注册 profile → 路由给 None（不许拿通用猜测冒充游戏知识）。"""
        frame = _Frame(bar_grid(consumed=8))
        assert resolve_planner("zzzz", frame, LEGAL, "deadbeef") is None
