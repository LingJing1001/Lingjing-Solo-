"""AR25 Layer 0/1: cover closure + config enumeration."""
from lingjing_solo.core import SoloConfig
from lingjing_solo.core.types import Ar25Config
from lingjing_solo.perception.ar25_encoder import DEFAULT_SOURCE, Ar25Encoder
from lingjing_solo.world_model.ar25_field import BOUNCE_LIMIT, Ar25Field
from lingjing_solo.world_model.field import WorldModelField


def test_bounce_limit_matches_engine():
    assert BOUNCE_LIMIT == 12


def test_encoder_parses_eight_levels():
    enc = Ar25Encoder(DEFAULT_SOURCE).load()
    assert len(enc.levels) == 8
    l7 = enc.encode_level(7)
    assert l7.grid_w == 21 and l7.grid_h == 21
    assert len(l7.targets) == 42
    assert len(l7.pieces) == 2
    assert {a.kind for a in l7.axes} == {"H", "V"}
    assert l7.steps_left == 320


def test_l1_known_config_covers_and_enum_finds_it():
    obs = Ar25Encoder(DEFAULT_SOURCE).encode_level(1)
    field = Ar25Field()
    assert not field.check_cover(obs)

    piece = obs.pieces[0]
    cfg = Ar25Config(piece_xy={piece.id: (1, 15)})
    assert field.check_cover(field.apply_config(obs, cfg))
    assert field.estimate_path_cost(obs, cfg) == 15

    sols = field.enumerate_covering_configs(obs, max_solutions=3, joint_pieces=False)
    assert sols
    best_cfg, best_cost = sols[0]
    assert best_cost == 15
    assert best_cfg.piece_xy[piece.id] == (1, 15)


def test_l7_arcsage_anchor_covers_geometrically():
    obs = Ar25Encoder(DEFAULT_SOURCE).encode_level(7)
    field = Ar25Field()
    h = next(a for a in obs.axes if a.kind == "H")
    v = next(a for a in obs.axes if a.kind == "V")
    p1 = next(p for p in obs.pieces if p.x == 17 and p.y == 13)
    p2 = next(p for p in obs.pieces if p.x == 5 and p.y == 16)
    cfg = Ar25Config(
        axis_coord={h.id: 7, v.id: 12},
        piece_xy={p1.id: (7, 7), p2.id: (8, 1)},
    )
    report = field.cover_report(field.apply_config(obs, cfg))
    assert report.ok
    assert report.uncovered == []
    assert field.estimate_path_cost(obs, cfg) <= obs.steps_left
    assert not field.is_futile(obs, cfg)


def test_world_model_field_exposes_ar25():
    wm = WorldModelField(SoloConfig())
    assert isinstance(wm.ar25, Ar25Field)
