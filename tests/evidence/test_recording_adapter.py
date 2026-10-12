import json
from pathlib import Path

import pytest

from lingjing_solo.evidence.adapter import RecordingAdapter


FIXTURE = Path("data/aop/episodes/ls20_success_rep0/recording.jsonl")


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
