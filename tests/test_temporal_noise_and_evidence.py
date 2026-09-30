import json

import numpy as np

from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.exploration.probe_evidence import (
    run_synthetic_probe_loop,
    write_probe_evidence_jsonl,
)
from lingjing_solo.exploration.probe_feedback import ProbeFeedback, calculate_frame_delta
from lingjing_solo.exploration.temporal_noise import analyze_temporal_window
from lingjing_solo.perception.observation import normalize_observation


def _obs(frame, frame_id, previous=None):
    return normalize_observation(frame, frame_id=frame_id, previous_frame=previous)


def test_periodic_window_is_classified_as_dynamic_noise():
    base = np.zeros((6, 6), dtype=np.uint8)
    flash = base.copy()
    flash[2:4, 2:4] = 9
    summary = analyze_temporal_window(
        [_obs(base, "f0"), _obs(flash, "f1"), _obs(base, "f2"), _obs(flash, "f3")]
    )
    assert summary.dynamic_noise is True
    assert summary.periodicity_score == 1.0


def test_non_periodic_change_is_not_promoted_to_noise():
    base = np.zeros((6, 6), dtype=np.uint8)
    moved = base.copy()
    moved[2:4, 2:4] = 9
    final = moved.copy()
    final[2:4, 3:5] = 9
    summary = analyze_temporal_window(
        [_obs(base, "f0"), _obs(moved, "f1"), _obs(final, "f2")]
    )
    assert summary.dynamic_noise is False


def test_detector_window_penalizes_periodic_delta_candidate():
    base = np.zeros((8, 8), dtype=np.uint8)
    flash = base.copy()
    flash[2:4, 2:4] = 9
    frames = [
        _obs(base, "f0"),
        _obs(flash, "f1", base),
        _obs(base, "f2", flash),
        _obs(flash, "f3", base),
    ]
    result = R4HotspotDetector(min_component_area=2).detect_window(frames)
    delta_candidates = [
        candidate for candidate in result.candidates if candidate.source == "delta_region"
    ]
    assert delta_candidates
    assert delta_candidates[0].features.dynamic_noise > 0
    assert "periodic_dynamic_noise" in delta_candidates[0].score.explanation


def test_frame_delta_ratio_is_computed_and_validated():
    before = np.zeros((2, 2), dtype=np.uint8)
    after = before.copy()
    after[0, 0] = 1
    assert calculate_frame_delta(before, after) == 0.25


def test_probe_evidence_jsonl_contains_frame_refs_and_requested_action(tmp_path):
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 2:4] = 5
    observation = _obs(frame, "before-1")
    detection = R4HotspotDetector(min_component_area=2).detect(observation)
    hotspot_id = detection.candidates[0].hotspot_id
    evidence = run_synthetic_probe_loop(
        observation,
        detection,
        {hotspot_id: ProbeFeedback(
            state_changed=True,
            before_frame_ref="frame:before",
            after_frame_ref="frame:after",
            evidence_refs=("artifact:after",),
        )},
        run_id="jsonl-run-1",
    )
    path = tmp_path / "probe.jsonl"
    assert write_probe_evidence_jsonl(path, evidence) == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["record_type"] == "probe_run"
    assert records[1]["requested_action"] == "observe"
    assert records[1]["before_frame_ref"] == "frame:before"
    assert records[1]["after_frame_ref"] == "frame:after"
    assert records[1]["outcome"] == "changed_no_progress"
