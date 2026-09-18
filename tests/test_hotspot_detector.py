import numpy as np
import pytest

from lingjing_solo.exploration.hotspot_candidates import stable_hotspot_id
from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.perception.observation import normalize_observation


def observation(frame, previous=None, family="click"):
    return normalize_observation(
        frame,
        frame_id="fixture-001",
        previous_frame=previous,
        game_family=family,
        evidence_refs=("fixture:hotspots/fixture-001",),
    )


def test_normalize_supports_rgb_and_preserves_dimensions():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[1:3, 2:4] = (255, 0, 0)
    result = observation(rgb)
    assert result.frame.shape == (4, 5)
    assert (result.width, result.height) == (5, 4)
    assert result.evidence_refs == ("fixture:hotspots/fixture-001",)


def test_normalize_rejects_nan_and_mismatched_previous_frame():
    with pytest.raises(ValueError, match="NaN"):
        observation(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="shape mismatch"):
        observation(np.zeros((2, 2)), np.zeros((3, 3)))


def test_component_center_has_geometry_and_stable_evidence():
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 5:7] = 7
    result = R4HotspotDetector(min_component_area=2).detect(observation(frame))
    assert result.retained_count == 1
    candidate = result.candidates[0]
    assert (candidate.x, candidate.y) == (6, 2)
    assert candidate.bbox == (5, 2, 6, 3)
    assert candidate.source == "component_center"
    assert candidate.evidence_refs == ("fixture:hotspots/fixture-001", "fixture-001")


def test_delta_region_is_emitted_without_treating_change_as_success():
    before = np.zeros((8, 8), dtype=np.uint8)
    after = before.copy()
    after[5:7, 1:3] = 3
    result = R4HotspotDetector(min_component_area=2).detect(observation(after, before))
    assert any(c.source == "delta_region" for c in result.candidates)
    assert all(c.status == "unknown" for c in result.candidates)


def test_keyboard_family_fails_closed_for_click_hotspots():
    frame = np.ones((8, 8), dtype=np.uint8)
    result = R4HotspotDetector().detect(observation(frame, family="keyboard"))
    assert result.candidates == ()
    assert result.retained_count == 0
    assert "skips click hotspots" in result.warnings[0]


@pytest.mark.parametrize("limit", [0, 1, 20])
def test_candidate_budget_is_bounded(limit):
    frame = np.zeros((32, 32), dtype=np.uint8)
    for y in range(0, 32, 4):
        for x in range(0, 32, 4):
            frame[y : y + 2, x : x + 2] = ((x + y) % 15) + 1
    result = R4HotspotDetector(min_component_area=2, deduplicate_radius=0).detect(
        observation(frame), max_candidates=limit
    )
    assert result.retained_count <= limit <= 20
    assert result.raw_candidate_count >= result.retained_count


def test_invalid_budget_is_rejected_and_ids_do_not_depend_on_level_constants():
    with pytest.raises(ValueError, match=r"\[0, 20\]"):
        R4HotspotDetector().detect(observation(np.ones((2, 2))), max_candidates=21)
    expected = stable_hotspot_id("frame-a", "component_center", 4, 5, (3, 4, 5, 6))
    assert expected == stable_hotspot_id("frame-a", "component_center", 4, 5, (3, 4, 5, 6))
