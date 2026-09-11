import numpy as np

from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.exploration.probe_planner import ProbePlanner
from lingjing_solo.perception.observation import (
    HypothesisContext,
    normalize_observation,
)


def test_planner_carries_hypothesis_context_into_phi_evidence():
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 2:4] = 9
    observation = normalize_observation(
        frame,
        frame_id="hypothesis-fixture",
        current_hypothesis=HypothesisContext(
            hypothesis_space_type="object_interaction",
            relevance_by_source=(("component_center", 0.8),),
        ),
    )
    detection = R4HotspotDetector(min_component_area=2).detect(observation)

    action = ProbePlanner().plan(observation, detection).actions[0]

    assert action.hypothesis_space_type == "object_interaction"
    assert action.evidence_role == "phi_interactive_hotspot"


def test_observation_rejects_invalid_hypothesis_context():
    try:
        HypothesisContext(hypothesis_space_type=123)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "hypothesis_space_type" in str(exc)
    else:
        raise AssertionError("invalid hypothesis context was accepted")
