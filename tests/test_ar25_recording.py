"""AR25 的逐 tick 证据层（`r2_ar25` + `tick_trail`）只依赖 stdlib 的回归闸门（设计文档 §8 ③）。

为什么单开一个文件：`run_ar25_r234.py` 的测试要真引擎（`arc_agi`/`arcengine`），在
`.venv`（3.12，无 arc_agi）和 ARC checkout（无 `arc_adaptor/`）里一律 skip，于是"哈希怎么算、
回滚要不要落盘"这些**判据本身**反而没人守。本文件被测的三个模块刻意做成 stdlib-only，
所以在任何解释器里都是真 assert 而不是 skip。

盯住的五件事：
  1. `state_hash` 是**语义态**的哈希：跨调用稳定、与列表原序无关、逐字段敏感，而步数条不进
     （ACTION5 只扣一步那种局面必须仍判为"同一状态"，否则 R4 反循环与证据都会错判）；
  2. 未经 `enrich` 的观测**拒绝**出哈希——少收一类可变状态比跑不起来严重；
  3. 回滚的搜索段落一行都不进 recording（§10：「搜索尝试过」≠「环境发生过」）；
  4. 每行都过 `evidence_compat.validate_tick`，且 `replay_recording` 能把它读成写入器自报的
     那条数（判据来自回读，不是自报 True）；
  5. 整关一份的 plan 只盖在本关第一行、其余行只带 `plan_id`；plan 声明的 `input_state_hash`
     对不上本关入口态时要在审计里现形。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

_ADAPTOR = pathlib.Path(__file__).resolve().parents[1] / "arc_adaptor"
if not _ADAPTOR.is_dir():                    # ARC checkout 里没有 arc_adaptor/（规则 ① 的已知越界）
    pytest.skip(f"被测模块不在当前 checkout 内: {_ADAPTOR}", allow_module_level=True)
sys.path.insert(0, str(_ADAPTOR))

import evidence_compat as ev                 # noqa: E402
import r2_ar25 as r2                         # noqa: E402
import tick_trail as tt                      # noqa: E402
from lingjing_solo.planning import plan_contract as pc  # noqa: E402

LEGAL = list(r2.AR25_LEGAL_ACTIONS)
SIZE = 21                                   # AR25 渲染网格是 21×21（引擎 naxbskjmlg()）
FRAME = [[0] * SIZE for _ in range(SIZE)]


class FakeEngine:
    """只实现 `r2_ar25.enrich` 会读的那几个字段；多给的名字一律拒绝，防"改了没被读"的假测试。"""

    def __init__(self, **flags):
        self._state = "PLAYING"
        self.ovoizfolxfq: dict = {}
        self.ouurgkpbbjj: list = []
        self.hsiusrsrdkswnt = 0
        self.qehjebksqcm = False
        self.hujpxmlafgh = False
        self.xukxeewuexo = False
        self.xjwpeqpcxav = False
        for name, value in flags.items():
            if not hasattr(self, name):
                raise AttributeError(f"不是 r2_ar25.enrich 会读的字段: {name}")
            setattr(self, name, value)


#: 引擎标志位的默认值（`make_perc` 直接引用，避免每个用例再穿一遍混淆名）。
DEFAULT_FLAGS = r2.engine_flags(FakeEngine())


def make_perc(**over):
    """一份与 `r2_perceive` 同形状的观测（1 轴 + 2 拼块 + 4 目标），并按契约 enrich 过。"""
    axis = {"x": -6, "y": 3, "type": "h", "cells": [(0, 0)]}
    moves = [{"idx": 1, "is_axis": False, "x": 0, "y": 0, "cells": [(0, 0)]},
             {"idx": 2, "is_axis": False, "x": 1, "y": 0, "cells": [(0, 1)]}]
    perc = {
        "axes": [axis], "switch_order": [dict(axis, idx=0, is_axis=True, cells=None), *moves],
        "sel_idx": 0, "targets": {(3, 4), (3, 5), (4, 4), (4, 5)}, "budget": 128,
        "steps_left": 128, "n_switch": 3, "movable": moves, "n_axes": 1, "atype": "h",
        "covered": 1, "total_targets": 4, "uncovered": [(3, 4), (3, 5), (4, 4)],
        "won": False, "level": 0, "state_key": ("k", 1, b"\x00"),
    }
    # 先 enrich 再套 override：`enrich` 会无条件写 state/rotation_distances/engine_flags，
    # 放在 update 之后就把用例想改的那三个字段悄悄盖回去了（哈希敏感用例于是变成空 assert）。
    r2.enrich(FakeEngine(), perc)
    return perc | over


def make_obs(tick=0, **over):
    """现成 observation：`over` 先并进底层观测再算哈希，保证 `state_hash` 与内容自洽。"""
    return r2.observe(make_perc(**over), tick=tick, frame=FRAME, run_id="unit",
                      episode_id="ar25-test", legal_actions=LEGAL)


def plan_for(input_state_hash, planner="beam", **over):
    """九字段合法的 plan（`tick_trail` 落盘时会当场过 §3.2 契约，随手编的 dict 进不去）。"""
    return pc.build_plan(planner=planner, input_state_hash=input_state_hash,
                         candidate_actions=["ACTION2", "ACTION5"],
                         expected_goal="cover_all_targets:1->4", cost=2,
                         search_budget={"time_ms": 1000, "max_nodes": 8},
                         validity="candidate", evidence_refs=["unit:test"], **over)


def rows(path):
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ── 1. 状态哈希：稳定、与顺序无关、逐字段敏感、不含步数条 ──────
def test_hash_is_stable_and_ignores_container_order():
    """同一语义态不论列表原序如何，哈希一致——否则搜索侧与证据侧无法对账。"""
    base = r2.state_hash(make_perc())
    shuffled = make_perc()
    shuffled["movable"] = list(reversed(shuffled["movable"]))
    shuffled["uncovered"] = list(reversed(shuffled["uncovered"]))
    shuffled["targets"] = set(reversed(sorted(shuffled["targets"])))
    assert r2.state_hash(shuffled) == base
    assert r2.state_hash(make_perc()) == base                     # 跨对象稳定


def test_hash_sees_every_state_field_but_not_the_step_bar():
    """每个语义字段单独变一点，哈希就得变；只有步数条例外（HUD 不进状态身份）。"""
    base = r2.state_hash(make_perc())
    moved = make_perc(movable=[{"idx": 1, "is_axis": False, "x": 0, "y": 0, "cells": [(0, 0)]},
                               {"idx": 2, "is_axis": False, "x": 9, "y": 0, "cells": [(0, 1)]}])
    assert r2.state_hash(moved) != base
    assert r2.state_hash(make_perc(sel_idx=1)) != base
    assert r2.state_hash(make_perc(uncovered=[(3, 4)])) != base
    assert r2.state_hash(make_perc(level=1)) != base
    assert r2.state_hash(make_perc(state="GAME_OVER")) != base
    assert r2.state_hash(make_perc(rotation_distances=[[0, 3]])) != base
    flagged = dict(DEFAULT_FLAGS, hujpxmlafgh=True)
    assert r2.state_hash(make_perc(engine_flags=flagged)) != base
    # 只扣一步、别的都没变（探针实测 ACTION5 就是这个形状）→ 同一个状态
    assert r2.state_hash(make_perc(steps_left=127)) == base


def test_enrich_reads_the_engine_fields_the_search_key_omits():
    """`enrich` 要把 `_state_key` 没收的那些可变全局状态抄进观测——这是换哈希的全部理由。"""
    g = FakeEngine(ovoizfolxfq={object(): 7}, xukxeewuexo=True)
    perc = r2.enrich(g, {"level": 0, "axes": [], "movable": [], "sel_idx": 0,
                         "total_targets": 0, "uncovered": [], "targets": set()})
    assert perc["state"] == "PLAYING"
    assert perc["engine_flags"]["xukxeewuexo"] is True
    assert perc["rotation_distances"] == [[-1, 7]]        # -1 = 不在 ouurgkpbbjj 里，照实记
    json.dumps(r2.semantic_state(perc), allow_nan=False)  # 序列化不出错才算能落盘


def test_unenriched_perc_is_refused_instead_of_hashing_less():
    """缺 enrich 字段的观测必须抛错，而不是算出一个少收了一类可变状态的哈希。"""
    bare = {"axes": [], "movable": [], "sel_idx": 0, "uncovered": [], "targets": set(),
            "total_targets": 0, "level": 0, "steps_left": 1, "budget": 1, "won": False,
            "n_axes": 0}
    with pytest.raises(r2.R2PerceptionError):
        r2.state_hash(bare)
    with pytest.raises(r2.R2PerceptionError):
        r2.observe(bare, tick=0, frame=FRAME, run_id="t", episode_id="e")


def test_observe_refuses_a_placeholder_frame():
    """§5.2 每行要带真帧：frame 缺省时宁可抛错，也不造一个尺寸对不上的占位帧。"""
    with pytest.raises(r2.R2PerceptionError):
        r2.observe(make_perc(), tick=0, frame=None, run_id="t", episode_id="e")


def test_semantic_field_table_matches_the_hashed_content():
    """manifest 发布的是 `SEMANTIC_STATE_FIELDS`；它和真正进哈希的字段不一致就是假声明。"""
    semantic = r2.semantic_state(make_perc())
    assert tuple(semantic) == r2.SEMANTIC_STATE_FIELDS
    assert "steps_left" not in semantic


def test_engine_key_hash_is_only_a_comparison_and_none_is_honest():
    """对照哈希按 ② 的公式复算；观测没有 state_key 时返回 None，不编一个假哈希。"""
    perc = make_perc()
    assert r2.engine_key_hash(perc) == \
        hashlib.sha256(repr(perc["state_key"]).encode("utf-8")).hexdigest()[:16]
    assert r2.engine_key_hash({}) is None
    assert r2.observe(perc, tick=0, frame=FRAME, run_id="t",
                      episode_id="e")["game_specific"]["engine_key_hash"] is not None


def test_observation_carries_the_public_fields_and_recomputes():
    """§3.1 的 observation 字段齐、`hash_of` 能从观测本身复算 state_hash（审计路径不读引擎）。"""
    obs = make_obs()
    for field in ("schema", "observation_id", "game_id", "episode_id", "tick", "grid_shape",
                  "state_hash", "state", "levels_completed", "legal_actions",
                  "objects_or_features", "visibility", "game_specific"):
        assert field in obs, f"缺 §3.1 字段 {field}"
    assert obs["schema"] == r2.OBSERVATION_SCHEMA
    assert obs["legal_actions"] == LEGAL
    assert obs["grid_shape"] == [SIZE, SIZE]
    assert obs["visibility"] == "full"                      # AR25 完整可观测
    assert obs["game_specific"]["steps_left"] == 128        # 步数条另存，不进哈希
    assert r2.hash_of(obs) == obs["state_hash"]
    json.dumps(obs, allow_nan=False)                        # 能原样进 recording


# ── 2. transition：因果三元组，moved 与 steps 分开 ─────────────
def test_transition_reports_steps_only_moves_as_not_moved():
    before, after = make_obs(tick=0), make_obs(tick=1, steps_left=127)
    trans = r2.transition(before, "ACTION5", after)
    assert trans["moved"] is False and trans["steps_delta"] == -1
    assert trans["before_hash"] == before["state_hash"]
    assert trans["after_hash"] == after["state_hash"]
    assert trans["levels_delta"] == 0 and trans["covered_delta"] == 0


def test_transition_counts_coverage_gain_and_level_advance():
    before = make_obs(tick=0)
    after = make_obs(tick=1, uncovered=[(3, 4), (3, 5)], steps_left=127)
    assert r2.transition(before, "ACTION2", after)["covered_delta"] == 1
    next_level = make_obs(tick=2, level=1, uncovered=[(3, 4), (3, 5), (4, 4), (4, 5)])
    assert r2.transition(after, "ACTION3", next_level)["levels_delta"] == 1


# ── 3. TickTrail：回滚不入证、提交才落盘 ───────────────────────
@pytest.fixture
def trail(tmp_path):
    """开一条已写基线行的 recording；用例结束后 close() 收尾（句柄别留给 Windows 的 tmp 清理）。"""
    t = tt.TickTrail(tmp_path / "recording.jsonl", run_id="unit",
                     episode_id="ar25-test", legal_actions=LEGAL)
    t.baseline(make_obs(tick=0), FRAME)
    yield t
    t.close()


def _commit_two_steps(t):
    """两条已提交转移：第一条只扣一步（moved=False），第二条换了选中块。"""
    t.begin_attempt()
    t.record(t.last_obs, "ACTION2", make_obs(tick=t.next_tick, steps_left=127), FRAME)
    t.record(t.last_obs, "ACTION5",
             make_obs(tick=t.next_tick, sel_idx=1, steps_left=126), FRAME)
    assert t.commit_attempt() == 2


def test_baseline_is_written_once_and_is_the_reset_row(trail):
    with pytest.raises(tt.TickTrailError):
        trail.baseline(make_obs(tick=0), FRAME)
    row = rows(trail.path)[0]["data"]
    assert row["tick"] == 0 and row["requested_action"]["name"] == "RESET"
    assert row["state"] == "RESET" and row["plan"] is None
    assert row["state_hash"] == trail.last_obs["state_hash"]


def test_aborted_attempt_leaves_no_rows(tmp_path):
    """搜索失败会 `_restore` 回入口帧：这些帧是"尝试过"，不是"发生过"（§10）。"""
    path = tmp_path / "recording.jsonl"
    t = tt.TickTrail(path, run_id="unit", episode_id="ar25-test", legal_actions=LEGAL)
    t.baseline(make_obs(tick=0), FRAME)
    t.begin_attempt()
    t.record(t.last_obs, "ACTION2", make_obs(tick=t.next_tick, steps_left=127), FRAME)
    assert t.abort_attempt() == 1
    summary = t.close()
    assert summary["actions"] == 0 and summary["discarded_rows"] == 1
    assert summary["final_state_hash"] == r2.state_hash(make_perc())
    assert len(rows(path)) == 1                                   # 只剩基线行
    # 回读侧的硬证据：协议层认定"一条转移都没有"的 recording 不可用（基线行不算转移）。
    with pytest.raises(ev.EvidenceValidationError, match="no action transition"):
        ev.replay_recording(path, legal_actions=LEGAL)


def test_record_requires_an_open_attempt_and_a_legal_action(trail):
    with pytest.raises(tt.TickTrailError):                        # 没 begin 就 record
        trail.record(trail.last_obs, "ACTION2", make_obs(tick=1), FRAME)
    trail.begin_attempt()
    with pytest.raises(ev.EvidenceValidationError):               # 非法动作当场拒，不等回读
        trail.record(trail.last_obs, "ACTION9", make_obs(tick=1), FRAME)
    trail.commit_attempt()
    assert trail.actions_committed == 0                           # 抛错那条没进缓冲


def test_last_obs_follows_the_engine_through_commit_and_rollback(tmp_path):
    """`last_obs` 必须始终等于"引擎现在停在哪儿"：尝试中看尝试末尾，回滚后退回缓冲末尾。"""
    t = tt.TickTrail(tmp_path / "recording.jsonl", run_id="unit",
                     episode_id="ar25-test", legal_actions=LEGAL)
    t.baseline(make_obs(tick=0), FRAME)
    entry = t.last_obs
    t.begin_attempt()
    t.record(t.last_obs, "ACTION2", make_obs(tick=1, steps_left=127), FRAME)
    assert t.last_obs is not entry and t.last_obs["game_specific"]["steps_left"] == 127
    t.abort_attempt()
    assert t.last_obs is entry                                    # 跟着 _restore 回到入口态
    t.begin_attempt()
    t.record(t.last_obs, "ACTION3", make_obs(tick=1, sel_idx=1), FRAME)
    committed = t.last_obs
    t.commit_attempt()
    assert t.last_obs is committed
    t.flush_level()
    assert t.last_obs is committed                                # flush 只编号，不改状态


def test_flush_numbers_rows_and_stamps_the_plan_only_on_the_first_one(trail):
    _commit_two_steps(trail)
    plan = plan_for(r2.state_hash(make_perc()))                   # 入口态 = 基线那条观测
    trail.attach_plan(plan)
    assert trail.flush_level() == 2
    written = rows(trail.path)
    assert [r["data"]["tick"] for r in written] == [0, 1, 2]
    assert written[1]["plan"] == plan and written[1]["data"]["plan"] == plan
    assert written[2]["data"]["plan"] is None                     # 其余行只带 join 键
    assert written[1]["data"]["plan_id"] == written[2]["data"]["plan_id"] == plan["plan_id"]
    assert written[2]["data"]["requested_action"]["name"] == "ACTION5"
    assert written[2]["data"]["requested_action"]["id"] == 5
    # 每行的终态身份来自动作之后：steps_left/level_index 跟着走，不复用入口态
    assert written[1]["data"]["game_specific"]["steps_left"] == 127
    assert written[2]["data"]["game_specific"]["steps_left"] == 126
    assert written[2]["data"]["game_specific"]["level_index"] == 0
    assert written[2]["observation"]["tick"] == 1                 # 输入态 = 上一条的终态
    for r in written:
        ev.validate_tick(r["data"])                               # 落盘即合法（含内嵌 plan）
    assert len(ev.replay_recording(trail.path, legal_actions=LEGAL).transitions) == 2
    assert trail.plan_hash_linked is True
    assert (trail.audit["plan_link"]["plans"], trail.audit["plan_link"]["linked"]) == (1, 1)
    # 第一条只扣一步（moved=False），第二条换了选中块而搜索键没变（新哈希更细）
    assert trail.audit["steps_only"] == 1
    assert trail.audit["hash_split_semantic_only"] == 1
    assert trail.audit["hash_split_engine_key_only"] == 0


def test_plan_that_does_not_match_the_entry_state_is_caught(trail):
    """§3.2 字段③ 的闭环要有闸门：plan 声明的输入态对不上，就得在审计里现形。"""
    _commit_two_steps(trail)
    trail.attach_plan(plan_for("deadbeef" * 2))
    trail.flush_level()
    link = trail.audit["plan_link"]
    assert (link["plans"], link["linked"]) == (1, 0)
    assert trail.plan_hash_linked is False
    example = link["examples"][0]
    assert example["input_state_hash"] == "deadbeef" * 2
    assert example["entry_state_hash"] == r2.state_hash(make_perc())


def test_continuity_check_flags_engine_moves_that_were_not_recorded(trail):
    """每关入口调一次：引擎状态与最后一条证据不符 = 有动作漏记，记进 chain_breaks。"""
    assert trail.check_continuity(r2.state_hash(make_perc())) is True
    assert trail.audit["chain_breaks"] == 0
    assert trail.check_continuity(r2.state_hash(make_perc(sel_idx=1))) is False
    assert trail.audit["chain_breaks"] == 1
    assert trail.audit["examples"], "分叉要留样本，否则报告里只剩一个数字"


def test_hash_audit_separates_the_two_divergence_directions(trail):
    """双向审计：搜索键变而语义不变 = 新哈希欠收（要能进报告）；反之是"更细"，不算缺陷。"""
    trail.begin_attempt()                                          # 只改搜索键的那一帧
    trail.record(trail.last_obs, "ACTION5",
                 make_obs(tick=trail.next_tick, state_key=("k", 2, b"\x00")), FRAME)
    trail.commit_attempt()
    assert trail.audit["hash_split_engine_key_only"] == 1
    assert trail.audit["hash_split_semantic_only"] == 0
    assert any(x.get("why", "").startswith("搜索键变了") for x in trail.audit["examples"])

    before = trail.last_obs                                        # 语义变、搜索键不变
    same_key = ("k", 2, b"\x00")                                   # 与 before 同一把搜索键
    trail.begin_attempt()
    trail.record(before, "ACTION5",
                 make_obs(tick=trail.next_tick, sel_idx=1, state_key=same_key), FRAME)
    trail.commit_attempt()
    assert trail.audit["hash_split_engine_key_only"] == 1           # 没被误记成欠收
    assert trail.audit["hash_split_semantic_only"] == 1
    assert trail.audit["steps_only"] == 0                           # 两条要么变了语义要么没扣步
    trail.flush_level()


def test_close_flushes_the_level_that_failed_and_discards_an_open_attempt(trail):
    """最后一关没解出来也要留证据：close 落盘已提交行，未 settle 的尝试整段丢。"""
    _commit_two_steps(trail)
    trail.begin_attempt()                                          # 没 commit 的一段
    trail.record(trail.last_obs, "ACTION1", make_obs(tick=trail.next_tick, level=1), FRAME)
    summary = trail.close()
    assert summary["flushed_at_close"] == 2 and summary["ticks"] == 3
    assert summary["discarded_rows"] == 1
    assert [r["data"]["tick"] for r in rows(trail.path)] == [0, 1, 2]


def test_replay_reads_the_recording_as_transitions(trail):
    """判据来自协议层回读：动作序列、终态、通关数都要跟写入器自报的一致。"""
    _commit_two_steps(trail)
    trail.flush_level()
    res = ev.replay_recording(trail.path, legal_actions=LEGAL)
    assert [t["action"]["name"] for t in res.transitions] == ["ACTION2", "ACTION5"]
    assert len(res.transitions) == trail.actions_committed == 2
    assert res.final_state == "PLAYING" and res.levels_completed == 0 and res.reset_count == 0
    summary = trail.close()
    assert summary["ticks"] == len(rows(trail.path)) == len(res.transitions) + 1
    assert summary["backend"] == ev.BACKEND
