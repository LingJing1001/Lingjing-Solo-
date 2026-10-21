import json
from pathlib import Path

import pytest

from lingjing_solo.evidence.adapter import RecordingAdapter, SandboxAdapter
from lingjing_solo.evidence.dpl_projection import evaluate_recording_frame, frame_to_array, project_recording_frame


FIXTURE = Path("data/aop/episodes/ls20_success_rep0/recording.jsonl")


def test_formal_projection_projects_real_recording_frame():
    raw = json.loads(next(FIXTURE.open(encoding="utf-8")))
    frame = raw["data"]["frame"]
    array = frame_to_array(frame)
    projected = project_recording_frame(frame)
    assert array.shape == (64, 64)
    assert projected.input_shape == (64, 64)
    assert projected.D1.available is True
    result = evaluate_recording_frame({"frame": frame, "levels_completed": 0}, {"frame": frame, "levels_completed": 0}, {"name": "RESET", "id": "RESET"})
    assert result["projection"]["frame_shape"] == [64, 64]
    assert result["gate"]["evidence"]["effective_dim"] == 2


def test_formal_projection_rejects_non_numeric_frame():
    with pytest.raises(ValueError, match="numeric"):
        frame_to_array([["not-a-frame"]])


def test_recording_adapter_is_deterministic_and_preserves_source(tmp_path):
    out1 = tmp_path / "one"
    out2 = tmp_path / "two"
    result1 = RecordingAdapter(FIXTURE, environment_id="arc_ls20", environment_version="local-fixed-v1", seed=0, source_commit="abc").build(out1)
    result2 = RecordingAdapter(FIXTURE, environment_id="arc_ls20", environment_version="local-fixed-v1", seed=0, source_commit="abc").build(out2)
    assert result1.event_count > 0
    assert (out1 / "episode_events.v1.jsonl").read_bytes() == (out2 / "episode_events.v1.jsonl").read_bytes()
    assert (out1 / "manifest.json").read_bytes() == (out2 / "manifest.json").read_bytes()
    assert FIXTURE.read_bytes() == FIXTURE.read_bytes()
    report = json.loads((out1 / "validation_report.json").read_text())
    assert report["authority_status"] == "shadow"
    assert report["recording_rows"] >= report["event_rows"]


def test_adapter_rejects_empty_or_malformed_recording(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        RecordingAdapter(bad, environment_id="x", environment_version="v", seed=1, source_commit="abc").build(tmp_path / "out")


def test_sandbox_adapter_preserves_reset_as_new_segment(tmp_path):
    source = tmp_path / "reset.jsonl"
    rows = [
        {"observation": {"tick": 0, "legal_actions": [{"id": "RESET"}], "state": "before-reset", "frame": [[0, 1], [1, 0]]}, "data": {"episode_id": "fixture", "tick": 1, "action_input": {"id": "RESET"}, "state": "after-reset", "frame": [[0, 1], [1, 0]]}},
        {"observation": {"tick": 0, "legal_actions": [{"id": 1}], "state": "reset-state", "frame": [[1, 0], [0, 1]]}, "data": {"episode_id": "fixture", "tick": 1, "action_input": {"id": "ACTION1"}, "state": "after-action", "frame": [[1, 0], [0, 1]]}},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result_dir = tmp_path / "reset-out"
    SandboxAdapter(source, environment_id="fixture", environment_version="v1", seed=0, source_commit="abc").build(result_dir)
    events = [json.loads(line) for line in (result_dir / "episode_events.v1.jsonl").read_text().splitlines()]
    assert [event["action"]["name"] for event in events] == ["RESET", "ACTION1"]
    assert [event["segment"] for event in events] == ["0", "1"]


def test_adapter_rejects_non_finite_dpl_result(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        SandboxAdapter(FIXTURE, environment_id="x", environment_version="v", seed=1, source_commit="abc", dpl_evaluator=lambda *_: {"score": float("nan")}).build(tmp_path / "out")


def test_sandbox_adapter_calls_dpl_and_records_real_action_boundary(tmp_path):
    calls = []

    def evaluate(before, after, action):
        calls.append((before["tick"], after["tick"], action["name"]))
        return {"status": "passed", "passed": True, "score": 0.75}

    source_hash = __import__("hashlib").sha256(FIXTURE.read_bytes()).hexdigest()
    result = SandboxAdapter(FIXTURE, environment_id="arc_ls20", environment_version="local-fixed-v1", seed=0, source_commit="abc", dpl_evaluator=evaluate).build(tmp_path / "out")
    assert result.dpl_status == "configured"
    assert len(calls) == result.event_count
    assert calls[0][2].startswith("ACTION")
    assert calls[0][0] + 1 == calls[0][1]
    assert __import__("hashlib").sha256(FIXTURE.read_bytes()).hexdigest() == source_hash
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["dpl_status"] == "configured"
    event = json.loads((tmp_path / "out" / "episode_events.v1.jsonl").read_text().splitlines()[0])
    assert event["gate_results"]["dpl"]["passed"] is True
