import hashlib
import json
from pathlib import Path

import pytest

from tools.aop.collection_runner import shard_status
from tools.aop.freeze_stage3_dataset import build_packet


def write_episode(root: Path, name: str, game: str, before: dict, after: dict) -> None:
    episode = root / name
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text(json.dumps({"game_id": game}), encoding="utf-8")
    events = [{"data": before}, {"data": after}]
    (episode / "recording.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def event(frame, *, state="NOT_FINISHED", levels=0, action=None):
    value = {
        "guid": f"guid-{frame}-{state}-{levels}",
        "state": state,
        "levels_completed": levels,
        "available_actions": [1, 2],
        "frame": [[frame]],
    }
    if action is not None:
        value["action_input"] = {"id": f"ACTION{action}", "reasoning": {"ignored": True}}
    return value


def fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    episodes = tmp_path / "episodes"
    shards = tmp_path / "shards"
    shards.mkdir()
    write_episode(episodes, "ar25_progress_rep0", "ar25", event(0), event(1, state="WIN", levels=1, action=1))
    write_episode(episodes, "ar25_progress_rep1", "ar25", event(0), event(1, state="WIN", levels=1, action=1))
    write_episode(episodes, "ar25_noop_rep0", "ar25", event(0), event(0, action=1))
    write_episode(episodes, "dc22_fail_rep0", "dc22", event(0), event(1, state="GAME_OVER", action=2))
    write_episode(episodes, "dc22_effective_rep0", "dc22", event(0), event(1, action=2))
    return episodes, shards


def test_shard_status_never_reports_collecting_after_process_exit():
    assert shard_status(12, 0) == "shard_complete"
    assert shard_status(0, 1) == "no_rows_retry_with_new_target"
    assert shard_status(0, 2) == "no_rows_target_quarantined"


def test_packet_is_complete_deterministic_and_leakage_safe(tmp_path):
    episodes, shards = fixture_tree(tmp_path)
    first = tmp_path / "packet-a"
    second = tmp_path / "packet-b"
    manifest = build_packet(episodes, shards, first, validation_fraction=0.5)
    repeated = build_packet(episodes, shards, second, validation_fraction=0.5)

    assert manifest["status"] == "complete"
    assert manifest == repeated
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (first / "holdout.jsonl").read_bytes() == (second / "holdout.jsonl").read_bytes()
    coverage = json.loads((first / "coverage.json").read_text())
    assert all(coverage["completion_checks"].values())
    train = {json.loads(line)["episode_id"] for line in (first / "train.jsonl").read_text().splitlines()}
    holdout = {json.loads(line)["episode_id"] for line in (first / "holdout.jsonl").read_text().splitlines()}
    assert train.isdisjoint(holdout)
    for name, metadata in manifest["artifacts"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == metadata["sha256"]


def test_empty_sources_are_rejected(tmp_path):
    episodes = tmp_path / "episodes"
    shards = tmp_path / "shards"
    episodes.mkdir()
    shards.mkdir()
    with pytest.raises(ValueError, match="no labels admitted"):
        build_packet(episodes, shards, tmp_path / "packet")


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_validation_fraction_is_rejected(tmp_path, fraction):
    episodes, shards = fixture_tree(tmp_path)
    with pytest.raises(ValueError, match="validation_fraction"):
        build_packet(episodes, shards, tmp_path / "packet", validation_fraction=fraction)


def test_partial_counterfactual_shard_is_rejected(tmp_path):
    episodes, shards = fixture_tree(tmp_path)
    shard = shards / "partial"
    shard.mkdir()
    (shard / "counterfactual_labels.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partial shard"):
        build_packet(episodes, shards, tmp_path / "packet")


def test_missing_category_fails_closed_but_writes_audit_coverage(tmp_path):
    episodes = tmp_path / "episodes"
    shards = tmp_path / "shards"
    shards.mkdir()
    write_episode(episodes, "ar25_a_rep0", "ar25", event(0), event(1, action=1))
    write_episode(episodes, "ar25_a_rep1", "ar25", event(0), event(1, action=1))
    write_episode(episodes, "dc22_b_rep0", "dc22", event(0), event(2, action=2))
    write_episode(episodes, "dc22_b_rep1", "dc22", event(0), event(2, action=2))
    output = tmp_path / "packet"
    with pytest.raises(ValueError, match="completion checks failed"):
        build_packet(episodes, shards, output, validation_fraction=0.5)
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["completion_checks"]["noop_present"] is False
    assert coverage["completion_checks"]["progress_present"] is False
