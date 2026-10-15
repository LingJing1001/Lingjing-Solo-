import hashlib
import json
from pathlib import Path

import pytest

from tools.aop.canonical_ingest import ingest_counterfactual_shard
from tools.aop.collection_runner import semantic_key
from tools.aop.freeze_stage3_dataset import stratified_split


def row(action: int, group: str = "g0") -> dict:
    return {
        "source": "remote_counterfactual_v1",
        "replay_verified": True,
        "episode_id": "ar25_episode",
        "step_id": 1,
        "counterfactual_group_id": group,
        "candidate_action_id": action,
        "legal_actions": ["ACTION1", "ACTION7"],
        "requested_action": {
            "id": f"ACTION{action}",
            "name": f"ACTION{action}",
            **({"payload": {"game_id": "ar25-example", "x": 15, "y": 31}} if action == 6 else {}),
        },
        "observation_before": {"available_actions": [1, 7], "legal_actions": [1, 7]},
        "observation_after": {"available_actions": [1, 7], "legal_actions": [1, 7]},
        "state_delta": {"before_hash": "before", "after_hash": f"after-{action}"},
        "replay_state_hash": "before",
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_for(digest: str, rows: int) -> dict:
    return {
        "schema_version": "aop-counterfactual-shard-v2",
        "thread": "model",
        "shard_id": "fixture-shard",
        "source": "remote_counterfactual_v1",
        "created_at": "2026-10-14T00:00:00Z",
        "collector_exit": 0,
        "rows": rows,
        "labels_sha256": digest,
        "source_sha256": digest,
        "provenance_complete": True,
    }


def test_counterfactual_ingest_requires_manifest_provenance(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    write_jsonl(source, [row(1), row(7)])
    with pytest.raises(ValueError, match="provenance"):
        ingest_counterfactual_shard(source, tmp_path / "out.jsonl", None, "model")


def test_counterfactual_ingest_canonicalizes_and_emits_manifest(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [row(1), row(7)])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 2)))
    output = tmp_path / "canonical.jsonl"
    result = ingest_counterfactual_shard(source, output, manifest, "model")
    assert result["rows_written"] == 2
    assert result["groups"] == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(item["schema_version"] == "aop-canonical-v2" for item in rows)
    assert all(item["provenance"]["kind"] == "remote_counterfactual" for item in rows)
    assert all(item["thread"] == "model" for item in rows)
    emitted = json.loads(Path(result["manifest_path"]).read_text())
    assert emitted["labels_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_counterfactual_ingest_preserves_payload_and_authoritative_delta(tmp_path: Path) -> None:
    action6 = row(6)
    action6["legal_actions"] = [1, 6]
    action6["observation_before"]["legal_actions"] = [1, 6]
    action6["observation_after"]["legal_actions"] = [1, 6]
    action6["requested_action"] = {
        "id": 6,
        "name": "ACTION6",
        "payload": {"game_id": "ar25-example", "x": 15, "y": 31},
    }
    action6["state_delta"] = {
        "before_hash": "before",
        "after_hash": "after-6",
        "state_changed": True,
        "action_effective": True,
    }
    comparator = row(1)
    comparator["legal_actions"] = [1, 6]
    comparator["observation_before"]["legal_actions"] = [1, 6]
    comparator["observation_after"]["legal_actions"] = [1, 6]
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [action6, comparator])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 2)))
    output = tmp_path / "canonical.jsonl"

    ingest_counterfactual_shard(source, output, manifest, "model")

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    saved = next(item for item in rows if item["candidate_action_id"] == 6)
    assert saved["requested_action"]["payload"] == action6["requested_action"]["payload"]
    assert saved["state_delta"] == action6["state_delta"]
    assert saved["state_delta"]["before_hash"] == saved["replay_state_hash"]


def test_counterfactual_ingest_rejects_action6_without_payload(tmp_path: Path) -> None:
    action6 = row(6)
    action6["legal_actions"] = [1, 6]
    action6["observation_before"]["legal_actions"] = [1, 6]
    action6["observation_after"]["legal_actions"] = [1, 6]
    action6["requested_action"] = {"id": 6, "name": "ACTION6"}
    comparator = row(1)
    comparator["legal_actions"] = [1, 6]
    comparator["observation_before"]["legal_actions"] = [1, 6]
    comparator["observation_after"]["legal_actions"] = [1, 6]
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [action6, comparator])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 2)))
    with pytest.raises(ValueError, match="ACTION6 payload"):
        ingest_counterfactual_shard(source, tmp_path / "out.jsonl", manifest, "model")


def test_counterfactual_ingest_rejects_before_hash_mismatch(tmp_path: Path) -> None:
    first = row(1)
    first["state_delta"]["before_hash"] = "not-replay-state"
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [first, row(7)])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 2)))
    with pytest.raises(ValueError, match="before_hash"):
        ingest_counterfactual_shard(source, tmp_path / "out.jsonl", manifest, "model")


def test_counterfactual_ingest_rejects_mixed_replay_states_in_group(tmp_path: Path) -> None:
    second = row(7)
    second["replay_state_hash"] = "different"
    second["state_delta"]["before_hash"] = "different"
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [row(1), second])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 2)))
    with pytest.raises(ValueError, match="replay state"):
        ingest_counterfactual_shard(source, tmp_path / "out.jsonl", manifest, "model")


def test_counterfactual_ingest_deduplicates_existing_semantic_keys(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [row(1), row(7)])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 2)))
    output = tmp_path / "canonical.jsonl"
    first = ingest_counterfactual_shard(source, output, manifest, "model")
    second = ingest_counterfactual_shard(source, output, manifest, "model")
    assert first["rows_written"] == 2
    assert second["rows_written"] == 0
    assert second["duplicates"] == 2
    assert len(output.read_text().splitlines()) == 2


def test_group_is_part_of_dedupe_key() -> None:
    left = row(7, "group-a")
    right = row(7, "group-b")
    assert semantic_key(left) != semantic_key(right)


def test_ingest_rejects_manifest_hash_and_row_count_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [row(1), row(7)])
    manifest = tmp_path / "source-manifest.json"
    payload = manifest_for(digest, 2)
    payload["labels_sha256"] = "wrong"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="labels_sha256"):
        ingest_counterfactual_shard(source, tmp_path / "out.jsonl", manifest, "model")


def test_ingest_rejects_incomplete_counterfactual_group(tmp_path: Path) -> None:
    source = tmp_path / "labels.jsonl"
    digest = write_jsonl(source, [row(7)])
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(manifest_for(digest, 1)))
    with pytest.raises(ValueError, match="at least two"):
        ingest_counterfactual_shard(source, tmp_path / "out.jsonl", manifest, "model")


def test_split_is_episode_disjoint_and_keeps_profile_recording_support() -> None:
    labels = []
    for profile in ("ar25", "ls20"):
        for index in (1, 2):
            labels.append({
                "episode_id": f"{profile}_episode_{index}",
                "requested_action": {"name": "ACTION1"},
            })
    train, holdout = stratified_split(labels, 0.5)
    train_keys = {row["episode_id"] for row in train}
    holdout_keys = {row["episode_id"] for row in holdout}
    assert train and holdout
    assert train_keys.isdisjoint(holdout_keys)
    assert {key.split("_", 1)[0] for key in train_keys} == {"ar25", "ls20"}
    assert {key.split("_", 1)[0] for key in holdout_keys} == {"ar25", "ls20"}
