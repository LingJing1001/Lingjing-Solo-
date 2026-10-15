import json
from pathlib import Path

import pytest

from lingjing_solo.evidence.canonical_projection import project_canonical_rows, project_canonical_jsonl
from lingjing_solo.evidence.schema import EventValidationError


def canonical_row(action_name="ACTION1"):
    return {
        "schema_version": "aop-canonical-v2",
        "thread": "model",
        "episode_id": "shard:ar25_episode",
        "tick": 3,
        "requested_action": {"name": action_name, "id": action_name, "payload": {}},
        "observation": {"frame": [[0, 1]], "state": "PLAYING"},
        "data": {"frame": [[1, 1]], "state": "PLAYING"},
        "state_delta": {
            "before_hash": "before",
            "after_hash": "after",
            "action_effective": True,
            "progressed": True,
        },
        "replay_state_hash": "before",
        "provenance": {"kind": "remote_counterfactual", "shard_id": "shard", "sha256": "a" * 64},
    }


def test_projection_preserves_action_delta_provenance_and_shadow_boundary():
    projected = project_canonical_rows([canonical_row()], environment_id="arc_ar25", environment_version="canonical-v2", seed=7, source_commit="abc")
    assert len(projected) == 1
    event = projected[0]
    assert event["schema_version"] == "episode_event.v1"
    assert event["action"]["name"] == "ACTION1"
    assert event["transition"]["state_before"] == "PLAYING"
    assert event["transition"]["state_after"] == "PLAYING"
    assert event["provenance"]["input_hash"] == "a" * 64
    assert event["provenance"]["canonical_shard"] == "shard"
    assert event["authority_status"] == "shadow"


def test_projection_rejects_missing_canonical_provenance():
    row = canonical_row()
    row["provenance"] = {"kind": "remote_counterfactual"}
    with pytest.raises(EventValidationError, match="provenance"):
        project_canonical_rows([row], environment_id="arc_ar25", environment_version="canonical-v2", seed=7, source_commit="abc")


def test_projection_preserves_reset_segment_and_writes_manifest(tmp_path: Path):
    rows = [canonical_row(), canonical_row("RESET")]
    output = tmp_path / "events"
    result = project_canonical_jsonl(rows, output, environment_id="arc_ar25", environment_version="canonical-v2", seed=7, source_commit="abc")
    assert result["events"] == 2
    saved = [json.loads(line) for line in (output / "episode_events.v1.jsonl").read_text().splitlines()]
    assert saved[1]["action"]["name"] == "RESET"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["authority_status"] == "shadow"
    assert manifest["source_schema_version"] == "aop-canonical-v2"
