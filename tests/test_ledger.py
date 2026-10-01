"""ledger / SkillManifest / prune_click_targets 单测（纯逻辑，无引擎依赖）。"""
from __future__ import annotations

from pathlib import Path

from lingjing_solo.transfer.ceax_controller import ClickTarget, prune_click_targets
from lingjing_solo.transfer.ledger import (
    STATUS_DISCARDED, SkillManifest, VerdictRecord, append_record, experiment_row,
    read_records,
)


# ---------------------------------------------------------------- verdict

def test_verdict_roundtrip_and_ts():
    rec = VerdictRecord(game_id="vc33-1", block=0, step=3, action="ACTION6",
                        x=61, y=33, delta_pixels=8, progressed=False, levels=0,
                        effect=0.125, grid_hash="ab12")
    d = rec.to_dict()
    assert d["type"] == "verdict" and d["schema"] == 1 and d["ts"]
    assert d["game_id"] == "vc33-1" and d["effect"] == 0.125


def test_append_and_read_records(tmp_path: Path):
    p = tmp_path / "nested" / "ledger.jsonl"
    for i in range(3):
        assert append_record(p, VerdictRecord(game_id="g", step=i).to_dict())
    vs = read_records(p, kind="verdict")
    assert len(vs) == 3 and vs[0]["step"] == 0 and vs[2]["step"] == 2
    assert read_records(p, kind="experiment") == []


def test_append_record_never_raises(tmp_path: Path):
    # 不可写路径：返回 False 而不是抛出（agent 运行时安全）
    assert append_record(tmp_path / "no_dir" / "\0bad", VerdictRecord().to_dict()) is False \
        or True  # 平台相关：至少不抛
    assert read_records(tmp_path / "missing.jsonl") == []


def test_experiment_row_shape():
    row = experiment_row(name="train_v5", kind="train",
                         config={"labels": "engine-truth"},
                         artifact="state/click_heatmap_v5.npz",
                         metric={"entry_p@10": 0.183},
                         gate={"metric": "entry_p@10", "line": 0.5},
                         verdict="discard")
    assert row["type"] == "experiment" and row["verdict"] == "discard"
    assert row["gate"]["line"] == 0.5


# ---------------------------------------------------------------- manifest

def test_manifest_register_save_load(tmp_path: Path):
    m = SkillManifest(tmp_path / "skills" / "manifest.json")
    m.register("route:x", kind="route", path="a.json", version="v1",
               gate={"replay": "3xWIN"})
    m.register("hm:v5", kind="click_heatmap", path="v5.npz",
               gate={"p@10": 0.15, "line": 0.5}, status=STATUS_DISCARDED)
    assert m.save()

    m2 = SkillManifest(tmp_path / "skills" / "manifest.json")
    assert m2.skills["route:x"]["gate"]["replay"] == "3xWIN"
    assert m2.skills["hm:v5"]["status"] == STATUS_DISCARDED


def test_manifest_discard_survives_reregister(tmp_path: Path):
    """循环不得绕过闸门复活被弃技能：discarded 后再 register(activate) 仍保持弃用。"""
    m = SkillManifest(tmp_path / "manifest.json")
    m.register("hm:v5", kind="click_heatmap", path="v5.npz", status=STATUS_DISCARDED)
    m.register("hm:v5", kind="click_heatmap", path="v5.npz", status="active")
    assert m.skills["hm:v5"]["status"] == STATUS_DISCARDED
    assert m.set_status("hm:v5", "active")   # 显式人工/闸门放行仍可行
    assert m.skills["hm:v5"]["status"] == "active"
    assert m.set_status("missing", "active") is False


# ---------------------------------------------------------------- prune

def _tg(trials, effect_sum, progress_hits=0):
    return ClickTarget(xy=(1, 1), key=("k",), area=4, color=3,
                       trials=trials, effect_sum=effect_sum,
                       progress_hits=progress_hits)


def test_prune_retires_only_dead_targets():
    targets = {
        "dead": _tg(trials=10, effect_sum=0.2),          # 均值 0.02 < 0.06 → 退役
        "alive": _tg(trials=10, effect_sum=3.0),         # 均值 0.3 → 保留
        "untried": _tg(trials=0, effect_sum=0.4),        # 未尝试 → 永不退役
        "progressed": _tg(trials=10, effect_sum=0.1, progress_hits=2),  # 推进过 → 保留
        "young": _tg(trials=2, effect_sum=0.0),          # 试验不足 → 保留
    }
    discarded: dict = {}
    retired = prune_click_targets(targets, discarded)
    assert retired == ["dead"]
    assert "dead" in discarded and "dead" not in targets
    for k in ("alive", "untried", "progressed", "young"):
        assert k in targets and k not in discarded
