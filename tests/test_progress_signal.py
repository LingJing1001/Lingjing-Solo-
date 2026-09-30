import numpy as np

from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.exploration.probe_evidence import run_synthetic_probe_loop
from lingjing_solo.exploration.probe_feedback import (
    ProbeFeedback,
    TabuContext,
    attribute_feedback,
    update_candidate_after_feedback,
)
from lingjing_solo.exploration.probe_planner import ProbePlanner
from lingjing_solo.perception.observation import normalize_observation


def test_feedback_does_not_call_visual_change_success():
    outcome = attribute_feedback(ProbeFeedback(state_changed=True, evidence_refs=("after:1",)))
    assert outcome.outcome == "changed_no_progress"
    assert outcome.terminal is False


def test_feedback_records_level_and_score_progress_and_terminal():
    assert attribute_feedback(ProbeFeedback(levels_after=1)).outcome == "progress"
    assert attribute_feedback(ProbeFeedback(score_after=2.5)).outcome == "score_progress"
    assert attribute_feedback(ProbeFeedback(terminal=True)).outcome == "terminal"


def test_feedback_distinguishes_no_effect_and_noise():
    assert attribute_feedback(ProbeFeedback()).outcome == "no_effect"
    feedback = ProbeFeedback(state_changed=True, dynamic_noise=True)
    assert attribute_feedback(feedback).outcome == "noisy"


def test_failed_candidate_confidence_decreases_and_tabu_is_recorded():
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 2:4] = 5
    observation = normalize_observation(frame, frame_id="feedback-fixture")
    detection = R4HotspotDetector(min_component_area=2).detect(observation)
    candidate = detection.candidates[0]
    outcome = attribute_feedback(ProbeFeedback(evidence_refs=("after:failure",)))
    updated = update_candidate_after_feedback(candidate, outcome)
    assert updated.confidence < candidate.confidence
    plan = ProbePlanner().plan(observation, detection)
    tabu = TabuContext().record(plan.actions[0], outcome)
    assert tabu.ids() == (candidate.hotspot_id,)
    assert tabu.entries[0].failures == 1


def test_fixture_detector_planner_feedback_loop_is_auditable():
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 2:4] = 5
    observation = normalize_observation(frame, frame_id="loop-fixture", evidence_refs=("frame:1",))
    detection = R4HotspotDetector(min_component_area=2).detect(observation)
    hotspot_id = detection.candidates[0].hotspot_id
    evidence = run_synthetic_probe_loop(
        observation,
        detection,
        {hotspot_id: ProbeFeedback(levels_after=1, evidence_refs=("after:1",))},
        run_id="r4-loop-001",
    )
    assert evidence.run_id == "r4-loop-001"
    assert evidence.raw_candidate_count == 1
    assert evidence.probe_plan.actions[0].hotspot_id == hotspot_id
    assert evidence.outcomes[0].outcome == "progress"
    assert evidence.outcomes[0].evidence_refs == ("after:1",)
