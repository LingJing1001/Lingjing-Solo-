from pathlib import Path

from lingjing_solo.evidence import convert_ls20_recording


RECORDING = Path(
    "/srv/agent-platform/projects/ARC-AGI-3-Agents/recordings/"
    "ls20-9607627b.lingjingsolo.800.bbc5baae-3c02-44f1-9a96-70159143d4b2.recording.jsonl"
)


def test_ls20_real_recording_unified_report(tmp_path):
    result = convert_ls20_recording(RECORDING, tmp_path)
    report = result["report"]
    assert report["verdict"] == "PASS"
    assert report["metrics"] == {
        "records": 309,
        "transitions": 308,
        "field_transitions": 308,
        "levels_completed": 7,
        "field_version": 308,
        "win_hash_count": 1,
    }
    assert (tmp_path / "ls20-9607627b.verification.json").is_file()
