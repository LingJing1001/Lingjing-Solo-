import numpy as np
import pytest

from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.exploration.probe_planner import ProbePlanner
from lingjing_solo.perception.observation import normalize_observation


def detected_observation(*, family="click", budget=None):
    frame = np.zeros((12, 12), dtype=np.uint8)
    frame[2:4, 2:4] = 10
    frame[8:10, 8:10] = 20
    observation = normalize_observation(frame, frame_id="planner-fixture", game_family=family,
                                        remaining_budget=budget)
    detection = R4HotspotDetector(min_component_area=2, deduplicate_radius=0).detect(
        observation
    )
    return observation, detection


def test_planner_uses_only_detector_coordinates_and_caps_three_probes():
    observation, detection = detected_observation()
    plan = ProbePlanner().plan(observation, detection, max_probes=3)
    assert len(plan.actions) == 2
    assert {(a.x, a.y) for a in plan.actions} == {(2, 2), (8, 8)}
    assert all(0 <= a.x < 12 and 0 <= a.y < 12 for a in plan.actions)


def test_planner_filters_tabu_and_remaining_budget():
    observation, detection = detected_observation(budget=1)
    plan = ProbePlanner().plan(observation, detection, max_probes=3,
                               tabu_hotspot_ids=(detection.candidates[0].hotspot_id,))
    assert len(plan.actions) == 1
    assert plan.remaining_budget == 0
    assert detection.candidates[0].hotspot_id in plan.skipped_hotspot_ids


def test_planner_fails_closed_for_keyboard_and_exhausted_budget():
    keyboard_obs, detection = detected_observation(family="keyboard")
    assert ProbePlanner().plan(keyboard_obs, detection).actions == ()
    exhausted_obs, detection = detected_observation(budget=0)
    assert ProbePlanner().plan(exhausted_obs, detection).actions == ()


def test_low_confidence_is_observe_only():
    observation, detection = detected_observation()
    planner = ProbePlanner(min_click_confidence=1.0)
    plan = planner.plan(observation, detection)
    assert plan.actions
    assert all(a.mode == "observe_only" and a.action == "observe" for a in plan.actions)


def test_planner_rejects_more_than_three_probes():
    observation, detection = detected_observation()
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        ProbePlanner().plan(observation, detection, max_probes=4)
