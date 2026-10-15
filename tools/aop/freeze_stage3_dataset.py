#!/usr/bin/env python3
"""Freeze audited real ARC recordings into a leakage-safe Stage 3 AOP packet."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aop.labels import row_to_label  # noqa: E402
from tools.aop.normalize_arc_recording import normalize  # noqa: E402

SCHEMA_VERSION = "aop-stage3-dataset-v1"
PROFILES = {"ar25", "vc33", "ls20", "dc22", "re86"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(value)
    return rows


def compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "state",
        "levels_completed",
        "score",
        "tick",
        "legal_actions",
        "available_actions",
        "steps_left",
    )
    return {key: observation[key] for key in keys if key in observation}


def compact_label(label: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    label = dict(label)
    label["observation_before"] = compact_observation(label["observation_before"])
    label["observation_after"] = compact_observation(label["observation_after"])
    label["provenance"] = provenance
    return label


def _action_id(value: Any) -> int | None:
    if isinstance(value, str) and value.startswith("ACTION"):
        value = value.removeprefix("ACTION")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_counterfactual_row(row: dict[str, Any], shard_id: str) -> None:
    """Validate replay identity and action semantics before normalization."""
    if row.get("source") != "remote_counterfactual_v1" or row.get("replay_verified") is not True:
        raise ValueError(f"{shard_id}: counterfactual row lacks verified remote provenance")
    action = row.get("requested_action")
    legal = row.get("legal_actions")
    if not isinstance(action, dict):
        raise ValueError(f"{shard_id}: requested_action must be an object")
    action_id = _action_id(action.get("id"))
    candidate_id = _action_id(row.get("candidate_action_id"))
    legal_ids = {_action_id(value) for value in (legal or [])}
    if action_id is None or action_id not in legal_ids:
        raise ValueError(f"{shard_id}: counterfactual action is not legal")
    if candidate_id != action_id or action.get("name") != f"ACTION{action_id}":
        raise ValueError(f"{shard_id}: candidate/requested action mismatch")
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError(f"{shard_id}: action payload must be an object")
    if action_id == 6:
        if (
            not isinstance(payload.get("game_id"), str)
            or not payload.get("game_id")
            or not isinstance(payload.get("x"), int)
            or isinstance(payload.get("x"), bool)
            or not isinstance(payload.get("y"), int)
            or isinstance(payload.get("y"), bool)
        ):
            raise ValueError(f"{shard_id}: ACTION6 payload requires game_id and integer x,y")
    recorded_action = (((row.get("observation_after") or {}).get("frame") or {}).get("action_input"))
    if isinstance(recorded_action, dict):
        if _action_id(recorded_action.get("id")) != action_id or recorded_action.get("data") != payload:
            raise ValueError(f"{shard_id}: requested action does not match recorded action payload")
    replay_hash = row.get("replay_state_hash")
    delta = row.get("state_delta")
    if not isinstance(replay_hash, str) or not replay_hash:
        raise ValueError(f"{shard_id}: replay_state_hash is missing")
    if not isinstance(delta, dict) or delta.get("before_hash") != replay_hash:
        raise ValueError(f"{shard_id}: state_delta.before_hash must equal replay_state_hash")
    if not isinstance(delta.get("after_hash"), str) or not delta.get("after_hash"):
        raise ValueError(f"{shard_id}: state_delta.after_hash is missing")
    group = row.get("counterfactual_group_id")
    episode = row.get("episode_id")
    if not isinstance(group, str) or not group or not isinstance(episode, str) or not episode:
        raise ValueError(f"{shard_id}: counterfactual identity is missing")


def validate_counterfactual_groups(rows: list[dict[str, Any]], shard_id: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        validate_counterfactual_row(row, shard_id)
        groups[str(row["counterfactual_group_id"])].append(row)
    if not groups:
        raise ValueError(f"{shard_id}: no counterfactual groups")
    for group_id, candidates in groups.items():
        if len(candidates) < 2:
            raise ValueError(f"{shard_id}:{group_id}: counterfactual group must contain at least two candidates")
        replay_hashes = {row["replay_state_hash"] for row in candidates}
        candidate_ids = {_action_id(row.get("candidate_action_id")) for row in candidates}
        legal_sets = {
            tuple(sorted(value for value in (_action_id(item) for item in row.get("legal_actions", [])) if value is not None))
            for row in candidates
        }
        prefix_sequences = {
            tuple(row.get("prefix_action_ids") or [])
            for row in candidates
            if "prefix_action_ids" in row
        }
        if len(replay_hashes) != 1:
            raise ValueError(f"{shard_id}:{group_id}: counterfactual group has mixed replay states")
        if len(legal_sets) != 1:
            raise ValueError(f"{shard_id}:{group_id}: counterfactual group has mixed legal actions")
        if len(prefix_sequences) > 1:
            raise ValueError(f"{shard_id}:{group_id}: counterfactual group has mixed replay prefixes")
        if len(candidate_ids) != len(candidates):
            raise ValueError(f"{shard_id}:{group_id}: duplicate candidate action")


def refresh_counterfactual(row: dict[str, Any], shard_id: str, source_hash: str, source_commit: str | None = None) -> dict[str, Any]:
    validate_counterfactual_row(row, shard_id)
    action = row["requested_action"]
    group = row["counterfactual_group_id"]
    episode = row["episode_id"]
    normalized = row_to_label(
        {
            "episode_id": f"{shard_id}:{episode}",
            "tick": row.get("step_id"),
            "requested_action": action,
            "observation": row.get("observation_before"),
            "data": row.get("observation_after"),
        }
    ).to_dict()
    normalized["counterfactual_group_id"] = f"{shard_id}:{group}"
    normalized["candidate_action_id"] = row.get("candidate_action_id")
    normalized["replay_state_hash"] = row.get("replay_state_hash")
    normalized["replay_verified"] = True
    normalized["source"] = "remote_counterfactual_v1"
    # Preserve hashes/effect labels computed from raw replay frames. Recomputing
    # after observation compaction would discard the visual state boundary.
    normalized["state_delta"] = dict(row["state_delta"])
    provenance = {"kind": "remote_counterfactual", "shard_id": shard_id, "sha256": source_hash}
    if source_commit:
        provenance["source_commit"] = source_commit
    return compact_label(
        normalized,
        provenance,
    )


def profile_of(row: dict[str, Any]) -> str:
    episode = row.get("episode_id")
    if not isinstance(episode, str):
        raise ValueError("label missing episode_id")
    value = episode.split(":")[-1].split("_", 1)[0]
    if value not in PROFILES:
        raise ValueError(f"unknown profile in episode_id: {episode}")
    return value


def coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    profiles: collections.Counter[str] = collections.Counter()
    actions: collections.Counter[str] = collections.Counter()
    categories: collections.Counter[str] = collections.Counter()
    episodes: set[str] = set()
    groups: set[str] = set()
    for row in rows:
        profiles[profile_of(row)] += 1
        episodes.add(row["episode_id"])
        if row.get("counterfactual_group_id"):
            groups.add(row["counterfactual_group_id"])
        action = row.get("requested_action") or {}
        actions[str(action.get("name"))] += 1
        delta = row.get("state_delta") or {}
        categories["effective" if delta.get("action_effective") else "noop"] += 1
        categories["progressed" if delta.get("progressed") else "not_progressed"] += 1
        terminal = delta.get("state_after") in {"GAME_OVER", "FAIL", "LOST"}
        categories["terminal_failure" if terminal else "non_failure"] += 1
    return {
        "rows": sum(profiles.values()),
        "episodes": len(episodes),
        "counterfactual_groups": len(groups),
        "profiles": dict(sorted(profiles.items())),
        "actions": dict(sorted(actions.items())),
        "categories": dict(sorted(categories.items())),
    }


def stratified_split(
    labels: list[dict[str, Any]], validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    buckets: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in labels:
        profile = profile_of(row)
        kind = "counterfactual" if row.get("counterfactual_group_id") else "recording"
        key = str(row.get("counterfactual_group_id") or row["episode_id"])
        buckets[(profile, kind)].add(key)

    holdout_keys: set[str] = set()
    for (_profile, kind), keys in sorted(buckets.items()):
        ordered = sorted(keys, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
        if kind == "recording" and len(ordered) < 2:
            raise ValueError("each profile requires at least two recording episodes")
        if len(ordered) < 2:
            continue
        count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
        holdout_keys.update(ordered[:count])

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in labels:
        key = str(row.get("counterfactual_group_id") or row["episode_id"])
        (holdout if key in holdout_keys else train).append(row)
    return train, holdout


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def build_packet(episodes_dir: Path, shards_dir: Path, output: Path, validation_fraction: float = 0.2) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="aop-stage3-") as temporary:
        temporary_root = Path(temporary)
        for episode_dir in sorted(path for path in episodes_dir.iterdir() if path.is_dir() and path.name != "quarantine"):
            recording = episode_dir / "recording.jsonl"
            episode_manifest = episode_dir / "manifest.json"
            if not recording.exists() or not episode_manifest.exists():
                raise ValueError(f"{episode_dir}: recording or manifest missing")
            manifest = json.loads(episode_manifest.read_text(encoding="utf-8"))
            game = manifest.get("game_id")
            if game not in PROFILES or not episode_dir.name.startswith(f"{game}_"):
                raise ValueError(f"{episode_dir}: manifest/profile mismatch")
            normalized_path = temporary_root / f"{episode_dir.name}.jsonl"
            transitions = normalize(recording, normalized_path, episode_dir.name)
            if transitions < 1:
                raise ValueError(f"{episode_dir}: no normalized transitions")
            recording_hash = sha256(recording)
            admitted = 0
            skipped_reset = 0
            for transition in read_jsonl(normalized_path):
                action = transition.get("requested_action") or {}
                if action.get("name") == "RESET":
                    skipped_reset += 1
                    continue
                label = row_to_label(transition).to_dict()
                label["source"] = "remote_arc_recording_v1"
                labels.append(
                    compact_label(
                        label,
                        {"kind": "remote_recording", "path": str(recording), "sha256": recording_hash},
                    )
                )
                admitted += 1
            if admitted < 1:
                raise ValueError(f"{episode_dir}: no non-RESET labels admitted")
            inputs.append(
                {
                    "kind": "remote_recording",
                    "path": str(recording),
                    "sha256": recording_hash,
                    "transitions": transitions,
                    "rows": admitted,
                    "skipped_reset": skipped_reset,
                }
            )

    for shard in sorted(path for path in shards_dir.iterdir() if path.is_dir()):
        manifest_path = shard / "manifest.json"
        labels_path = shard / "counterfactual_labels.jsonl"
        if not manifest_path.exists() and not labels_path.exists():
            continue
        if not manifest_path.exists() or not labels_path.exists():
            raise ValueError(f"{shard}: partial shard must not be admitted")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hash = sha256(labels_path)
        rows = read_jsonl(labels_path)
        if (
            manifest.get("collector_exit") != 0
            or manifest.get("rows") != len(rows)
            or manifest.get("labels_sha256") != source_hash
        ):
            raise ValueError(f"{shard}: manifest integrity check failed")
        validate_counterfactual_groups(rows, shard.name)
        refreshed = [refresh_counterfactual(row, shard.name, source_hash) for row in rows]
        counts = collections.Counter(row["counterfactual_group_id"] for row in refreshed)
        if not counts or any(count < 2 for count in counts.values()):
            raise ValueError(f"{shard}: incomplete counterfactual group")
        labels.extend(refreshed)
        inputs.append({"kind": "remote_counterfactual", "path": str(labels_path), "sha256": source_hash, "rows": len(rows)})

    if not labels:
        raise ValueError("no labels admitted")
    train, holdout = stratified_split(labels, validation_fraction)
    train_keys = {row.get("counterfactual_group_id") or row["episode_id"] for row in train}
    holdout_keys = {row.get("counterfactual_group_id") or row["episode_id"] for row in holdout}
    if not train or not holdout or not train_keys.isdisjoint(holdout_keys):
        raise ValueError("train/holdout leakage or empty split")

    packet_coverage = coverage(labels)
    train_coverage = coverage(train)
    holdout_coverage = coverage(holdout)
    category = packet_coverage["categories"]
    holdout_category = holdout_coverage["categories"]
    expected_profiles = set(packet_coverage["profiles"])
    completion_checks = {
        "at_least_two_profiles": len(expected_profiles) >= 2,
        "effective_present": category.get("effective", 0) > 0,
        "noop_present": category.get("noop", 0) > 0,
        "progress_present": category.get("progressed", 0) > 0,
        "terminal_failure_present": category.get("terminal_failure", 0) > 0,
        "train_holdout_nonempty": bool(train and holdout),
        "split_keys_disjoint": train_keys.isdisjoint(holdout_keys),
        "all_profiles_in_train": set(train_coverage["profiles"]) == expected_profiles,
        "all_profiles_in_holdout": set(holdout_coverage["profiles"]) == expected_profiles,
        "all_actions_in_holdout": set(holdout_coverage["actions"]) == set(packet_coverage["actions"]),
        "holdout_effective_present": holdout_category.get("effective", 0) > 0,
        "holdout_noop_present": holdout_category.get("noop", 0) > 0,
        "holdout_progress_present": holdout_category.get("progressed", 0) > 0,
        "holdout_terminal_failure_present": holdout_category.get("terminal_failure", 0) > 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    holdout_path = output / "holdout.jsonl"
    coverage_path = output / "coverage.json"
    write_jsonl(train_path, train)
    write_jsonl(holdout_path, holdout)
    coverage_payload = {
        "all": packet_coverage,
        "train": train_coverage,
        "holdout": holdout_coverage,
        "completion_checks": completion_checks,
    }
    coverage_path.write_text(json.dumps(coverage_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if all(completion_checks.values()) else "incomplete",
        "validation_fraction": validation_fraction,
        "inputs": inputs,
        "artifacts": {
            "train.jsonl": {"rows": len(train), "sha256": sha256(train_path)},
            "holdout.jsonl": {"rows": len(holdout), "sha256": sha256(holdout_path)},
            "coverage.json": {"sha256": sha256(coverage_path)},
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["status"] != "complete":
        missing = sorted(name for name, passed in completion_checks.items() if not passed)
        raise ValueError(f"stage 3 completion checks failed: {missing}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/aop/episodes"))
    parser.add_argument("--shards", type=Path, default=Path("data/aop/collection/shards"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    try:
        manifest = build_packet(args.episodes, args.shards, args.output, args.validation_fraction)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REJECTED: {exc}")
        return 1
    print(json.dumps({"status": manifest["status"], "artifacts": manifest["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
