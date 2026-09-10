"""AR25 Layer0+1: parse engine source, cover closure, config search."""
from pathlib import Path

import pytest

from lingjing_solo.core.types import Ar25Config
from lingjing_solo.perception.ar25_encoder import Ar25Encoder, DEFAULT_SOURCE
from lingjing_solo.world_model.ar25_field import Ar25Field

ENGINE = Path(DEFAULT_SOURCE)


@pytest.fixture(scope="module")
def encoder() -> Ar25Encoder:
    assert ENGINE.is_file(), f"missing engine source: {ENGINE}"
    return Ar25Encoder(ENGINE).load()


@pytest.fixture(scope="module")
def field() -> Ar25Field:
    return Ar25Field()


def test_parse_eight_levels(encoder: Ar25Encoder):
    levels = encoder.levels
    assert len(levels) == 8
    l1 = encoder.encode_level(1)
    assert l1.grid_w == 21 and l1.grid_h == 21
    assert len(l1.targets) == 5
    assert len(l1.pieces) == 1
    assert len(l1.axes) == 1
    assert l1.axes[0].kind == "V"
    assert l1.axes[0].fixed is True
    assert l1.steps_left == 64


def test_l1_known_config_covers(encoder: Ar25Encoder, field: Ar25Field):
    """Report: piece → (1,15) with fixed V-axis at x=10 covers all targets."""
    obs = encoder.encode_level(1)
    piece_id = obs.pieces[0].id
    cfg = Ar25Config(piece_xy={piece_id: (1, 15)})
    applied = field.apply_config(obs, cfg)
    report = field.cover_report(applied)
    assert report.ok, f"uncovered={report.uncovered}"
    assert field.estimate_path_cost(obs, cfg) == 15


def test_l1_config_search_finds_cover(encoder: Ar25Encoder, field: Ar25Field):
    obs = encoder.encode_level(1)
    assert not field.check_cover(obs)
    sols = field.enumerate_covering_configs(obs, max_solutions=3, joint_pieces=False)
    assert sols, "expected at least one covering config on L1"
    best_cfg, cost = sols[0]
    assert cost <= 15
    assert field.check_cover(field.apply_config(obs, best_cfg))


def test_l7_arcsage_anchor_geometry(encoder: Ar25Encoder, field: Ar25Field):
    """ARC-SAGE L7 geometric plan: H y=7, V x=12, p1=(7,7), p2=(8,1).

    Action replay may stop at (8,11) due to collision; cover math uses (8,1).
    """
    obs = encoder.encode_level(7)
    assert len(obs.targets) == 42
    assert len(obs.pieces) == 2
    assert {a.kind for a in obs.axes} == {"V", "H"}

    h = next(a for a in obs.axes if a.kind == "H")
    v = next(a for a in obs.axes if a.kind == "V")
    p1 = next(p for p in obs.pieces if p.x == 17 and p.y == 13)
    p2 = next(p for p in obs.pieces if p.x == 5 and p.y == 16)

    cfg = Ar25Config(
        axis_coord={h.id: 7, v.id: 12},
        piece_xy={p1.id: (7, 7), p2.id: (8, 1)},
    )
    assert not field.is_futile(obs, cfg)
    report = field.cover_report(field.apply_config(obs, cfg))
    assert report.ok, f"uncovered={report.uncovered}"


def test_facades_on_encoder_and_field(encoder: Ar25Encoder):
    from lingjing_solo.core import SoloConfig
    from lingjing_solo.perception import PerceptionEncoder
    from lingjing_solo.world_model import WorldModelField

    pe = PerceptionEncoder(SoloConfig())
    pe._ar25 = encoder
    assert pe.ar25.encode_level(1).level_index == 1

    wm = WorldModelField(SoloConfig())
    assert wm.ar25.bounce_limit == 12
