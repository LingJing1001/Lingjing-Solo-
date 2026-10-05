"""trajectory / affordance / miner 纯逻辑单测（不依赖 arc_agi/arcengine）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lingjing_solo.transfer.affordance import (
    AffordanceRanker, click_feature_dict, color_prior_train, logo_eval, record_label,
)
from lingjing_solo.transfer.miner import (
    extract_agent_plans, counter_free_digest, plan_steps, route_json_steps,
    solution_json_steps,
)
from lingjing_solo.transfer.trajectory import (
    StepRecord, classify_effect_kind, classify_outcome, click_context,
    connected_components, grid_digest, grid_of, keyboard_context,
    outcome_distribution, read_jsonl, screen_delta, write_jsonl,
)


# ---------------------------------------------------------------- trajectory

def test_grid_of_squeezes_channel_dim():
    g = grid_of([[[5, 5], [0, 1]]])
    assert g.shape == (2, 2) and g[1, 1] == 1


def test_screen_delta_and_digest():
    a = np.zeros((3, 3), dtype=int)
    b = a.copy()
    b[1, 2] = 3
    d = screen_delta(a, b)
    assert d == [{"x": 2, "y": 1, "old": 0, "new": 3}]
    assert grid_digest(a) == grid_digest(a.copy())
    assert grid_digest(a) != grid_digest(b)


def test_classify_outcome_priority():
    kw = dict(levels_before=0, levels_after=0, delta_count=0,
              hidden_changed=False, sig_changed=False)

    def call(**over):
        return classify_outcome(state_after=over.pop("state_after", "NOT_FINISHED"),
                                **{**kw, **over})

    assert call(state_after="WIN") == "win"
    assert call(levels_after=1) == "advance"
    assert call(state_after="GAME_OVER") == "game_over"
    assert call(delta_count=2) == "screen"
    assert call(hidden_changed=True) == "state_only"
    assert call(sig_changed=True) == "state_only"
    assert call() == "no_op"


def _grids(kind):
    a = np.zeros((4, 4), dtype=int)
    if kind == "recolor":
        a[1:3, 1:3] = 3
        b = a.copy()
        b[1:3, 1:3] = 5
    elif kind == "spawn":
        b = a.copy()
        b[2, 2] = 7
    elif kind == "despawn":
        b = a.copy()
        a[2, 2] = 7
    elif kind == "move":
        a[1, 1:3] = 4
        b = a.copy()
        b[1, 1:3] = 0
        b[2, 1:3] = 4
    else:  # mixed
        a[0, 0] = 2
        b = a.copy()
        b[3, 3] = 6
        b[0, 0] = 0
    return a, b


@pytest.mark.parametrize("kind,expected", [
    ("recolor", "recolor"), ("spawn", "spawn"), ("despawn", "despawn"),
    ("move", "move"), ("mixed", "mixed"),
])
def test_classify_effect_kind(kind, expected):
    a, b = _grids(kind)
    assert classify_effect_kind(a, b) == expected


def test_connected_components_and_click_context():
    g = np.zeros((6, 6), dtype=int)
    g[0:2, 0:2] = 3          # 大块
    g[4, 4] = 3              # 同色对角 = 4-连通下另一个域
    g[0, 5] = 9
    comps = connected_components(g)
    by_color = {}
    for c in comps:
        by_color.setdefault(c["color"], []).append(c)
    assert len(by_color[3]) == 2 and len(by_color[9]) == 1
    big = max(by_color[3], key=lambda c: c["area"])
    assert big["area"] == 4 and (big["y0"], big["x0"]) == (0, 0)
    ctx = click_context(g, 1, 1, comps)
    assert ctx["kind"] == "click" and ctx["click_color"] == 3
    assert ctx["obj"]["area"] == 4 and ctx["obj"]["n_comps_of_color"] == 2
    assert len(ctx["hood3x3"]) == 9 and ctx["hood3x3"][4] == 3
    kctx = keyboard_context(g)
    assert kctx["kind"] == "keyboard" and kctx["fg_cells"] == 6


def test_step_record_json_roundtrip(tmp_path: Path):
    rec = StepRecord(
        game_id="vc33-5430563c", gid="vc33", level_idx=0, step_idx=0,
        action={"id": 6, "x": 12, "y": 56}, source="probe", reliability=0.5,
        outcome="screen", effect_kind="recolor", is_probe=True, base_sig="ab12",
        grid_hash_before="c1", grid_hash_after="c2", delta_count=1,
        screen_delta=[{"x": 12, "y": 56, "old": 0, "new": 8}],
        context={"kind": "click", "click_color": 8},
    )
    p = tmp_path / "rec.jsonl"
    assert write_jsonl([rec], p) == 1
    back = read_jsonl(p)
    assert len(back) == 1 and back[0] == rec
    assert outcome_distribution(back)["screen"] == 1


# ---------------------------------------------------------------- miner 纯函数

def test_counter_free_digest_ignores_counters_only():
    snap_a = {"grid": [[1, 2]], "action_count": 3}
    snap_b = {"grid": [[1, 2]], "action_count": 9}   # 计数器 → 应忽略
    snap_c = {"grid": [[1, 3]], "action_count": 3}   # 真状态变化 → 应变
    assert counter_free_digest(snap_a) == counter_free_digest(snap_b)
    assert counter_free_digest(snap_a) != counter_free_digest(snap_c)


def test_plan_steps_formats():
    assert plan_steps("kbd", [1, 6]) == [
        {"aid": 1, "x": None, "y": None}, {"aid": 6, "x": None, "y": None}]
    assert plan_steps("click", [(12, 56)]) == [{"aid": 6, "x": 12, "y": 56}]
    assert plan_steps("axy", [(4, 0, 0), (6, 45, 33)]) == [
        {"aid": 4, "x": None, "y": None}, {"aid": 6, "x": 45, "y": 33}]


def test_route_and_solution_steps():
    assert route_json_steps([{"a": "ACTION6", "x": 1, "y": 2}, {"a": "ACTION3"}]) == [
        {"aid": 6, "x": 1, "y": 2}, {"aid": 3, "x": None, "y": None}]
    assert solution_json_steps([{"action": 6, "data": {"x": 5, "y": 6}, "why": "w"},
                                {"action": 5, "data": None, "why": "s"}]) == [
        {"aid": 6, "x": 5, "y": 6}, {"aid": 5, "x": None, "y": None}]


def test_extract_agent_plans_real_file():
    plans = extract_agent_plans()  # 仓库内的 my_agent.py，受信源码
    for var in ("_BP35_PLANS", "_VC33_PLANS", "_LS20_LEVEL_ACTIONS",
                "_AR25_LEVEL_ACTIONS", "_FT09_HARDCODED"):
        assert var in plans, f"{var} 未提取到"
    assert plans["_BP35_PLANS"][0][0] == (4, 0, 0)
    assert plans["_AR25_LEVEL_ACTIONS"][0] == [3] * 5 + [2] * 10  # 表达式兜底


# ---------------------------------------------------------------- affordance

def _click_rec(gid, color, outcome, area=4, rel=(0.5, 0.5), grid="h1"):
    ctx = {"kind": "click", "click_color": color, "bg": 0,
           "hood3x3": [0, color, 0, color, color, color, 0, 0, 0],
           "n_objects": 2, "n_colors": 3, "grid_h": 8, "grid_w": 8,
           "rel_x": rel[0], "rel_y": rel[1], "on_bg": color == 0,
           "obj": {"color": color, "area": area, "bw": 2, "bh": 2,
                   "n_comps_of_color": 1, "color_cells_total": area}}
    return StepRecord(game_id=f"{gid}-x", gid=gid, level_idx=0, step_idx=0,
                      action={"id": 6, "x": 1, "y": 1}, source="agent-plan",
                      reliability=1.0, outcome=outcome, effect_kind="recolor",
                      grid_hash_before=grid, context=ctx)


def test_click_feature_dict_shape_and_none():
    fd = click_feature_dict(_click_rec("g1", 3, "screen").context)
    assert fd is not None and fd["color3"] == 1.0 and fd["has_obj"] == 1.0
    assert click_feature_dict({"kind": "keyboard"}) is None
    assert click_feature_dict(None) is None


def test_ranker_learns_separable_synthetic():
    recs = []
    # 色 1 小孤立对象 → 触及机制；色 2 背景大块 → 只动像素（可分合成数据，
    # productive 正类 = state_only/advance/win，screen 是负类）
    for i in range(30):
        recs.append(_click_rec("ga", 1, "state_only", area=4))
        recs.append(_click_rec("ga", 0, "no_op", area=40))
        recs.append(_click_rec("gb", 1, "state_only", area=6))
        recs.append(_click_rec("gb", 0, "no_op", area=60))
    model = AffordanceRanker(epochs=20).fit(recs)
    assert model.ready
    p_good = model.score(_click_rec("gc", 1, "state_only").context)
    p_bad = model.score(_click_rec("gc", 0, "no_op", area=50).context)
    assert p_good > p_bad + 0.2
    assert record_label(recs[0], "productive") == 1
    assert record_label(recs[1], "productive") == 0
    assert record_label(_click_rec("ga", 2, "screen"), "has_effect") == 1
    assert record_label(_click_rec("ga", 2, "screen"), "productive") == 0


def test_logo_eval_returns_per_game_and_pooled():
    recs = []
    for gid in ("ga", "gb", "gc"):
        for i in range(12):
            recs.append(_click_rec(gid, 1, "state_only", grid=f"st{i % 3}"))
            recs.append(_click_rec(gid, 0, "no_op", area=50, grid=f"st{i % 3}"))
    per = logo_eval(recs, min_test_clicks=5, epochs=8)
    assert {"ga", "gb", "gc", "pooled"} <= set(per)
    for gid in ("ga", "gb", "gc"):
        assert per[gid]["auc_model"] >= 0.9
    assert per["pooled"]["auc_model"] >= 0.9
    assert per["pooled"]["n_groups"] > 0


def test_color_prior_baseline():
    recs = [_click_rec("ga", 3, "state_only"), _click_rec("ga", 3, "no_op"),
            _click_rec("gb", 4, "state_only")]
    prior = color_prior_train(recs)
    assert abs(prior[3] - 0.5) < 1e-9 and abs(prior[4] - 2 / 3) < 1e-9


def test_adaptation_fixes_flipped_game():
    """gd 是特征翻转局：别的局"小对象=机制"，它"大对象=机制"。

    零样本在 gd 上应低于随机；喂几发本局观测（adapt）后应翻正——
    这就是 ft09/tu93 的合成缩影，也是部署形态（live 有本局观测）的依据。
    """
    recs = []
    for gid in ("ga", "gb"):
        for i in range(10):
            recs.append(_click_rec(gid, 1, "state_only", area=4, grid=f"st{i % 4}"))
            recs.append(_click_rec(gid, 0, "no_op", area=60, grid=f"st{i % 4}"))
    flipped = []
    for i in range(10):
        flipped.append(_click_rec("gd", 2, "state_only", area=60, grid=f"st{i % 4}"))
        flipped.append(_click_rec("gd", 0, "no_op", area=4, grid=f"st{i % 4}"))
    all_recs = recs + flipped

    zero = logo_eval(all_recs, min_test_clicks=5, epochs=10, adapt_shots=0)
    k = logo_eval(all_recs, min_test_clicks=5, epochs=10, adapt_shots=8)
    assert zero["gd"]["auc_model"] < 0.5
    assert k["gd"]["auc_model"] > 0.5
    assert k["gd"]["n_shots"] == 8 and k["gd"]["n_clicks"] == 20
    # 校准的 shot 不允许混进评测集
    assert k["gd"]["n_clicks"] == zero["gd"]["n_clicks"]


def test_ranking_loss_keeps_fit_and_adapt_stable():
    recs = []
    for gid in ("ga", "gb"):
        for i in range(8):
            recs.append(_click_rec(gid, 1, "state_only", area=4, grid=f"st{i}"))
            recs.append(_click_rec(gid, 0, "no_op", area=40, grid=f"st{i}"))
    model = AffordanceRanker(epochs=10, rank_weight=1.0).fit(recs)
    assert model.ready
    # adapt 返回的是副本：原模型不被 shot 污染
    shots = [_click_rec("gc", 1, "state_only", area=50),
             _click_rec("gc", 0, "no_op", area=5)]
    before = model.score(shots[0].context)
    adapted = model.adapt(shots, epochs=3)
    assert adapted is not model
    assert model.score(shots[0].context) == before
    # 全同标签 shot：无信号，共享原头
    same = [_click_rec("gc", 1, "state_only"), _click_rec("gc", 2, "state_only")]
    assert model.adapt(same).score(shots[0].context) == before
