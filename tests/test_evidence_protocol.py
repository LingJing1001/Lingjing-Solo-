import json

import numpy as np
import pytest

from lingjing_solo.evidence.protocol import (
    EvidenceValidationError,
    build_manifest,
    build_tick,
    build_verification_report,
    replay_recording,
    validate_manifest,
    validate_tick,
    validate_verification_report,
)


def frame(x=0):
    grid = np.zeros((3, 3), dtype=int)
    grid[1, x] = 1
    return grid.tolist()


def test_protocol_builders_validate_public_and_game_specific_fields():
    manifest = build_manifest(
        run_id="run-1",
        game_id="ar25-demo",
        branch="feat/test",
        commit="abc123",
        module_versions={"r2": "v1", "r3": "v1", "r4": "v1", "r5": "v1", "ceax": "v1"},
        evidence_tier="offline_engine",
        mode="replay",
        game_specific={"profile": "ar25-v1"},
    )
    tick = build_tick(
        run_id="run-1", episode_id="ep-1", tick=0, frame=frame(),
        state="NOT_FINISHED", levels_completed=0, legal_actions=["ACTION1"],
        state_hash="h0", game_specific={"profile": "ar25-v1"},
    )
    report = build_verification_report(
        run_id="run-1", tier="offline_engine", verdict="PASS",
        criteria={"schema_valid": True, "actions_legal": True, "replay_complete": True,
                  "terminal_verified": True, "goal_verified": True, "levels_completed": 1},
        metrics={"actions": 1, "transitions": 1, "resets": 0},
        evidence_refs=["manifest.json", "recording.jsonl"],
    )
    assert validate_manifest(manifest)["schema_version"] == "lingjing-evidence-v1"
    assert validate_tick(tick)["game_specific"]["profile"] == "ar25-v1"
    assert validate_verification_report(report)["verdict"] == "PASS"


def test_replay_resets_baseline_and_returns_transitions(tmp_path):
    path = tmp_path / "recording.jsonl"
    records = [
        {"frame": frame(0), "state": "NOT_FINISHED", "levels_completed": 0},
        {"frame": frame(1), "requested_action": {"name": "ACTION1", "id": 1},
         "server_action_input": {"name": "ACTION1", "id": 1}, "state": "NOT_FINISHED", "levels_completed": 0},
        {"frame": frame(2), "requested_action": {"name": "RESET", "id": 0}, "state": "RESET", "levels_completed": 0},
        {"frame": frame(1), "requested_action": {"name": "ACTION2", "id": 2},
         "server_action_input": {"name": "ACTION2", "id": 2}, "state": "WIN", "levels_completed": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    result = replay_recording(path, legal_actions=["ACTION1", "ACTION2"])
    assert [t["action"]["name"] for t in result.transitions] == ["ACTION1", "ACTION2"]
    assert result.transitions[1]["before_frame"] == frame(2)
    assert result.final_state == "WIN"
    assert result.levels_completed == 1


def test_replay_rejects_empty_missing_baseline_missing_action_and_bad_payload(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="no baseline"):
        replay_recording(empty, legal_actions=["ACTION1"])

    missing_baseline = tmp_path / "missing-baseline.jsonl"
    missing_baseline.write_text(json.dumps({"requested_action": "ACTION1"}) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="no baseline"):
        replay_recording(missing_baseline, legal_actions=["ACTION1"])

    missing_action = tmp_path / "missing-action.jsonl"
    missing_action.write_text(json.dumps({"frame": frame(0)}) + "\n" + json.dumps({"frame": frame(1)}) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="requested_action"):
        replay_recording(missing_action, legal_actions=["ACTION1"])

    bad_payload = tmp_path / "bad-payload.jsonl"
    bad_payload.write_text(json.dumps({"frame": frame(0)}) + "\n" + json.dumps({"frame": frame(1), "requested_action": {"name": "ACTION1", "payload": {"x": -1}}}) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="payload"):
        replay_recording(bad_payload, legal_actions=["ACTION1"])


def test_replay_rejects_illegal_action_and_does_not_overwrite_existing_file(tmp_path):
    path = tmp_path / "recording.jsonl"
    original = json.dumps({"frame": frame(0)}) + "\n"
    path.write_text(original, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"frame": frame(1), "requested_action": "UNKNOWN"}) + "\n")
    with pytest.raises(EvidenceValidationError, match="illegal action"):
        replay_recording(path, legal_actions=["ACTION1"])
    assert path.read_text(encoding="utf-8") == original + json.dumps({"frame": frame(1), "requested_action": "UNKNOWN"}) + "\n"


def test_validators_fail_closed_on_missing_required_fields():
    with pytest.raises(EvidenceValidationError):
        validate_manifest({"schema_version": "lingjing-evidence-v1"})
    with pytest.raises(EvidenceValidationError):
        validate_tick({"run_id": "x"})
    with pytest.raises(EvidenceValidationError):
        validate_verification_report({"run_id": "x", "verdict": "PASS"})


def test_tick_records_r5_decision_context_and_acceptance():
    tick = build_tick(
        run_id="run-r5",
        episode_id="ep-1",
        tick=3,
        frame=frame(1),
        state="NOT_FINISHED",
        levels_completed=0,
        legal_actions=["ACTION1"],
        state_hash="h3",
        reflection_id="reflection-3",
        reflection_reasons=["loop_trapped", "budget_warning"],
        hypotheses=[{"id": "h-1", "text": "ACTION1 changes the marker", "confidence": 0.8}],
        skill_context={"family": "click", "candidates": ["marker_click"]},
        reflection_accepted=False,
    )

    assert tick["reflection_id"] == "reflection-3"
    assert tick["reflection_reasons"] == ["loop_trapped", "budget_warning"]
    assert tick["hypotheses"][0]["id"] == "h-1"
    assert tick["skill_context"]["family"] == "click"
    assert tick["reflection_accepted"] is False
    assert validate_tick(tick) == tick


def test_tick_rejects_invalid_r5_evidence_types():
    base = build_tick(
        run_id="run-r5",
        episode_id="ep-1",
        tick=3,
        frame=frame(1),
        state="NOT_FINISHED",
        levels_completed=0,
        legal_actions=["ACTION1"],
        state_hash="h3",
    )
    for field, value in (
        ("reflection_reasons", "loop_trapped"),
        ("hypotheses", {"id": "h-1"}),
        ("skill_context", ["not-an-object"]),
        ("reflection_accepted", "false"),
    ):
        invalid = dict(base)
        invalid[field] = value
        with pytest.raises(EvidenceValidationError, match=field):
            validate_tick(invalid)
