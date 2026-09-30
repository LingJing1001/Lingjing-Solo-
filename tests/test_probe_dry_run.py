import json

import numpy as np
import pytest

from lingjing_solo.exploration.hotspot_detector import R4HotspotDetector
from lingjing_solo.exploration.probe_dry_run import (
    prepare_probe_dry_run,
    write_probe_dry_run_jsonl,
)
from lingjing_solo.exploration.probe_planner import ProbeAction, ProbePlan, ProbePlanner
from lingjing_solo.perception.observation import normalize_observation


def test_dry_run_prepares_bounded_requests_without_execution(tmp_path):
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:4, 2:4] = 7
    observation = normalize_observation(frame, frame_id="dry-frame")
    detection = R4HotspotDetector(min_component_area=2).detect(observation)
    plan = ProbePlanner().plan(observation, detection, max_probes=1)

    requests = prepare_probe_dry_run(observation, plan, run_id="dry-1")
    assert len(requests) == 1
    assert requests[0].executed is False
    assert requests[0].status == "dry_run"

    path = tmp_path / "dry-run.jsonl"
    assert write_probe_dry_run_jsonl(path, requests, observation=observation) == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["record_type"] == "probe_dry_run"
    assert records[0]["executed"] is False
    assert records[1]["record_type"] == "probe_request"
    assert records[1]["requested_action"] in {"click", "observe"}


def test_empty_plan_writes_header_without_requests(tmp_path):
    observation = normalize_observation(np.zeros((3, 3), dtype=np.uint8), frame_id="empty")
    path = tmp_path / "empty.jsonl"
    assert write_probe_dry_run_jsonl(path, (), observation=observation) == 1
    record = json.loads(path.read_text())
    assert record["request_count"] == 0
    assert record["executed"] is False


def test_same_plan_produces_stable_dry_run_payload():
    frame = np.zeros((4, 4), dtype=np.uint8)
    observation = normalize_observation(frame, frame_id="stable")
    plan = ProbePlan((ProbeAction("h", "observe", 1, 1, "observe_only", ("e",), "test"),), (), 2)
    first = prepare_probe_dry_run(observation, plan, run_id="stable-run")
    second = prepare_probe_dry_run(observation, plan, run_id="stable-run")
    assert first == second


def test_dry_run_rejects_invalid_coordinates_and_modes():
    frame = np.zeros((4, 4), dtype=np.uint8)
    observation = normalize_observation(frame, frame_id="dry-invalid")
    invalid = ProbePlan((ProbeAction("h", "click", 4, 1, "click", (), "test"),), (), 0)
    with pytest.raises(ValueError, match="outside"):
        prepare_probe_dry_run(observation, invalid)

    invalid_mode = ProbePlan((ProbeAction("h", "click", 1, 1, "observe_only", (), "test"),), (), 0)
    with pytest.raises(ValueError, match="requires click mode"):
        prepare_probe_dry_run(observation, invalid_mode)


def test_dry_run_rejects_unknown_action():
    frame = np.zeros((4, 4), dtype=np.uint8)
    observation = normalize_observation(frame, frame_id="dry-unknown")
    invalid = ProbePlan((ProbeAction("h", "drag", 1, 1, "click", (), "test"),), (), 0)
    with pytest.raises(ValueError, match="unsupported probe action"):
        prepare_probe_dry_run(observation, invalid)
