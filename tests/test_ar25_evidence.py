import json

import pytest

from lingjing_solo.evidence import run_ar25_level
from lingjing_solo.profiles import AR25_PROFILE


def test_ar25_profile_rejects_mismatched_action_payload():
    with pytest.raises(ValueError, match="payload id"):
        AR25_PROFILE.validate_tick({"requested_action": {"name": "ACTION2", "payload": {"id": 1}}, "settled_frame": True})


def test_ar25_candidate_replay_report(tmp_path):
    result = run_ar25_level(level=1, output_dir=tmp_path, t_limit=5)
    report = result["report_data"]
    assert report["verdict"] == "PASS"
    assert report["criteria"] == {"candidate_present": True, "offline_replay": True, "won": True}
    assert report["metrics"]["replayed_transitions"] == report["metrics"]["candidate_steps"]
    recording = tmp_path / "ar25-l1-offline.jsonl"
    assert recording.is_file()
    assert (tmp_path / "ar25-l1-offline.report.json").is_file()

    ticks = [json.loads(line) for line in recording.read_text(encoding="utf-8").splitlines()]
    assert ticks
    assert all(tick["reflection_id"] is None for tick in ticks)
    assert all(tick["reflection_reasons"] == [] for tick in ticks)
    assert all(tick["hypotheses"] == [] for tick in ticks)
    assert all(tick["skill_context"] == {} for tick in ticks)
    assert all(tick["reflection_accepted"] is None for tick in ticks)
