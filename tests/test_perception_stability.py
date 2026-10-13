"""Unit tests for lingjing_solo.perception.stability.

The 23 recorded episodes contain NO self-animating HUD pixel (max per-pixel change
frequency 0.75 on the layer delta_px reads, 0.833 anywhere) -- so the real corpus
cannot exercise the animated-mask path.  These tests cover it synthetically, so
LevelStability is verified rather than merely written.
"""

import numpy as np
import pytest

from lingjing_solo.perception.stability import (
    ANIM_RATIO,
    LevelStability,
    object_hud_score,
    interior_point,
    rerank_click_candidates,
)

SHAPE = (8, 8)


class _Obj:
    """Stand-in for GameObject: bbox is (min_x, min_y, max_x, max_y)."""

    def __init__(self, pixels, bbox):
        self.pixels = pixels
        self.bbox = bbox


def _ring(side=5, at=(1, 1)):
    r0, c0 = at
    pix = []
    for i in range(side):
        for j in range(side):
            if i in (0, side - 1) or j in (0, side - 1):
                pix.append((r0 + i, c0 + j))
    return pix


# ------------------------------------------------------------------ LevelStability


def test_self_animating_hud_does_not_count_as_board_change():
    """A cell that toggles every single step is chrome: raw delta > 0, board delta 0."""
    base = np.zeros(SHAPE, dtype=np.int16)
    base[0, :] = 3                      # a HUD bar
    st = LevelStability(SHAPE, warmup=4)
    st.observe(base)

    for k in range(24):
        f = base.copy()
        f[0, 0] = 3 + (k % 2)           # toggles on every step, independent of any action
        info = st.observe(f)

    assert info["settled"] is True
    assert info["animated_pixels"] >= 1
    # the last step is after warmup: raw says "changed", board says "nothing happened"
    assert info["raw_delta"] == 1
    assert info["board_delta"] == 0
    assert info["board_changed"] is False
    assert info["hud_delta"] == 1


def test_agent_driven_motion_does_count_as_board_change():
    """A pixel the agent moves changes only some steps -> it is playfield, not chrome."""
    st = LevelStability(SHAPE, warmup=4)
    grid = np.zeros(SHAPE, dtype=np.int16)
    grid[4, 0] = 7
    st.observe(grid)

    for k in range(1, 8):
        grid = np.zeros(SHAPE, dtype=np.int16)
        grid[4, k] = 7
        info = st.observe(grid)

    assert info["board_changed"] is True
    assert info["board_delta"] == 2     # one cell vacated, one entered
    assert info["hud_delta"] == 0


def test_no_history_before_warmup_never_suppresses_a_change():
    st = LevelStability(SHAPE, warmup=6)
    assert st.settled is False
    assert st.animated_mask().sum() == 0
    a = np.zeros(SHAPE, dtype=np.int16)
    b = np.zeros(SHAPE, dtype=np.int16)
    b[0, 0] = 1
    st.observe(a)
    info = st.observe(b)
    assert info["settled"] is False
    assert info["board_changed"] is True
    assert info["hud_delta"] == 0


def test_reset_clears_history():
    st = LevelStability(SHAPE, warmup=2)
    g = np.zeros(SHAPE, dtype=np.int16)
    st.observe(g)
    g2 = np.ones(SHAPE, dtype=np.int16)
    st.observe(g2)
    assert st.steps == 2
    st.reset()
    assert st.steps == 0
    assert st.settled is False
    assert st.changed_ever().sum() == 0


def test_observe_tolerates_wrong_shape_and_none():
    st = LevelStability(SHAPE, warmup=2)
    assert st.observe(None)["raw_delta"] == 0
    assert st.observe(np.zeros((4, 4), dtype=np.int16))["raw_delta"] == 0
    assert st.steps == 0


def test_anim_ratio_is_the_documented_default():
    assert 0.0 < ANIM_RATIO <= 1.0


# ------------------------------------------------------------------ geometry


def test_interior_point_of_a_ring_is_a_real_cell():
    pix = _ring(5, (1, 1))
    pt = interior_point(pix)
    assert pt in set(pix)
    # the bbox centre of that ring is the hole -> NOT a cell of the object
    assert (3, 3) not in set(pix)


def test_interior_point_empty_is_none():
    assert interior_point([]) is None
    assert interior_point(None) is None


def test_full_frame_band_scores_as_chrome():
    pix = [(r, c) for r in range(8) for c in range(8)]
    s = object_hud_score((0, 0, 7, 7), SHAPE, pixels=pix, npix=len(pix))
    assert s >= 0.5


def test_small_central_object_is_not_chrome():
    pix = [(3, 3), (3, 4), (4, 3), (4, 4)]
    s = object_hud_score((3, 3, 4, 4), SHAPE, pixels=pix, npix=len(pix))
    assert s < 0.5


def test_edge_hugging_thin_bar_is_chrome():
    pix = [(0, c) for c in range(8)]
    s = object_hud_score((0, 0, 7, 0), SHAPE, pixels=pix, npix=len(pix))
    assert s >= 0.5


def test_animated_evidence_pushes_score_over_the_line():
    pix = [(3, 3), (3, 4), (4, 3), (4, 4)]
    anim = np.zeros(SHAPE, dtype=bool)
    anim[3, 3] = anim[3, 4] = anim[4, 3] = anim[4, 4] = True
    base = object_hud_score((3, 3, 4, 4), SHAPE, pixels=pix, npix=len(pix))
    hot = object_hud_score((3, 3, 4, 4), SHAPE, pixels=pix, npix=len(pix), animated=anim)
    assert hot > base
    assert hot >= 0.5


def test_frozen_term_needs_settled():
    """Before warmup 'never changed' means nothing -- at frame 0 nothing has changed."""
    pix = [(3, 3), (3, 4), (4, 3), (4, 4)]
    changed = np.zeros(SHAPE, dtype=bool)      # nothing has ever changed yet
    unscored = object_hud_score((3, 3, 4, 4), SHAPE, pixels=pix, npix=len(pix),
                                changed=changed, settled=False)
    scored = object_hud_score((3, 3, 4, 4), SHAPE, pixels=pix, npix=len(pix),
                              changed=changed, settled=True)
    assert scored > unscored


# ------------------------------------------------------------------ rerank


def test_rerank_demotes_chrome_but_keeps_it_as_a_fallback():
    big_chrome = _Obj([(r, c) for r in range(8) for c in range(8)], (0, 0, 7, 7))
    small = _Obj([(3, 3), (3, 4)], (3, 3, 4, 3))
    out = rerank_click_candidates([big_chrome, small], SHAPE, k=8)
    assert len(out) == 2
    assert out[0] in set(small.pixels)          # the real object comes first
    assert out[-1] in set(big_chrome.pixels)    # chrome is demoted, not dropped


def test_rerank_orders_by_size_within_the_non_chrome_group():
    a = _Obj([(3, 3), (3, 4), (4, 3)], (3, 3, 4, 3))
    b = _Obj([(6, 6)], (6, 6, 6, 6))
    out = rerank_click_candidates([b, a], SHAPE, k=8)
    assert out[0] in set(a.pixels)


def test_rerank_points_always_lie_on_the_object():
    ring = _Obj(_ring(5, (1, 1)), (1, 1, 5, 5))
    for pt in rerank_click_candidates([ring], SHAPE, k=4):
        assert pt in set(ring.pixels)


def test_rerank_respects_k_and_skips_empty_objects():
    objs = [_Obj([(3, 3)], (3, 3, 3, 3)), _Obj([], (0, 0, 1, 1)), _Obj([(5, 5)], (5, 5, 5, 5))]
    assert len(rerank_click_candidates(objs, SHAPE, k=1)) == 1
    assert len(rerank_click_candidates(objs, SHAPE, k=8)) == 2


def test_rerank_accepts_dicts_and_returns_ints():
    objs = [{"pixels": [(2, 2), (2, 3)], "bbox": (2, 2, 3, 2)}]
    out = rerank_click_candidates(objs, SHAPE, k=4)
    assert out and all(isinstance(v, int) for v in out[0])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
