import numpy as np

from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.exploration.probe_gate import R4ProbeGate
from lingjing_solo.exploration.probe_planner import ProbePlan, ProbePlanner
from lingjing_solo.perception.observation import normalize_observation


def _plan(family="click"):
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 2:4] = 9
    observation = normalize_observation(frame, frame_id="gate", game_family=family)
    detection = R4HotspotDetector(min_component_area=2).detect(observation)
    return observation, ProbePlanner().plan(observation, detection)


def test_disabled_gate_is_fail_closed_and_preserves_input_plan():
    observation, plan = _plan()
    gated = R4ProbeGate().apply(observation, plan)
    assert gated.actions == ()
    assert "R4 probe feature flag is disabled" in gated.warnings
    assert plan.actions


def test_enabled_gate_allows_click_family_without_mutating_plan():
    observation, plan = _plan()
    gated = R4ProbeGate(enabled=True).apply(observation, plan)
    assert gated == plan


def test_enabled_gate_rejects_keyboard_probe_consumption():
    observation, plan = _plan("keyboard")
    gated = R4ProbeGate(enabled=True).apply(observation, plan)
    assert gated.actions == ()
    assert "keyboard observations cannot consume click probes" in gated.warnings


def test_empty_plan_is_idempotent_through_gate():
    observation, _ = _plan()
    empty = ProbePlan((), (), None)
    assert R4ProbeGate(enabled=True).apply(observation, empty) == empty
