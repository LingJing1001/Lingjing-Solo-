#!/usr/bin/env python3
"""Repair legacy AOP counterfactual shards into the strict v2 contract.

The migration reconstructs each candidate transition from its marked remote
recording, preserves the exact SDK action payload, aligns before_hash with the
authoritative replay hash, updates source manifests, and rebuilds canonical
shards. It is dry-run unless --apply is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aop.canonical_ingest import ingest_counterfactual_shard
from tools.aop.counterfactual_collect import obs_from_record, recording_action_id, stable_hash
from tools.aop.freeze_stage3_dataset import validate_counterfactual_groups

COLLECTION = ROOT / "data/aop/collection"
CANONICAL = ROOT / "data/aop/canonical/model/shards"
SOURCE_SCHEMA_VERSION = "aop-counterfactual-shard-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def recording_index(shard: Path) -> dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]]:
    index: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    recordings = shard / "raw" / "remote_recordings"
    for path in sorted(recordings.glob("*/*.jsonl")):
        records = [json.loads(line)["data"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for position in range(1, len(records)):
            after = records[position]
            action_input = after.get("action_input") or {}
            reasoning = action_input.get("reasoning") or {}
            if reasoning.get("probe") != "counterfactual candidate":
                continue
            group = reasoning.get("group")
            candidate = recording_action_id(action_input)
            if not isinstance(group, str) or candidate is None or reasoning.get("candidate") != candidate:
                continue
            index.setdefault((group, candidate), []).append((records[position - 1], after))
    return index


def repair_row(
    row: dict[str, Any],
    transitions: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]],
) -> dict[str, Any]:
    group = str(row.get("counterfactual_group_id") or "")
    candidate = recording_action_id(row.get("candidate_action_id"))
    matches = transitions.get((group, candidate if candidate is not None else -1), [])
    if len(matches) != 1:
        raise ValueError(f"{group}: candidate {candidate} has {len(matches)} marked recording transitions")
    before_record, after_record = matches[0]
    step = row.get("step_id")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"{group}: invalid step_id")
    before = obs_from_record(before_record, step)
    after = obs_from_record(after_record, step + 1)
    action_input = after_record.get("action_input") or {}
    payload = action_input.get("data")
    if not isinstance(payload, dict):
        raise ValueError(f"{group}: candidate {candidate} recording payload is missing")
    replay_hash = row.get("replay_state_hash")
    if not isinstance(replay_hash, str) or not replay_hash:
        raise ValueError(f"{group}: replay_state_hash is missing")
    after_hash = stable_hash(after_record)
    before_levels = before.get("levels_completed", 0)
    after_levels = after.get("levels_completed", 0)
    repaired = dict(row)
    repaired["requested_action"] = {
        "name": f"ACTION{candidate}",
        "id": candidate,
        "payload": dict(payload),
    }
    repaired["observation_before"] = before
    repaired["observation_after"] = after
    repaired["legal_actions"] = list(before.get("legal_actions") or [])
    repaired["state_delta"] = {
        "before_hash": replay_hash,
        "after_hash": after_hash,
        "state_changed": replay_hash != after_hash,
        "visual_changed": before_record.get("frame") != after_record.get("frame"),
        "visual_delta": sum(
            left != right
            for left_row, right_row in zip(before_record.get("frame") or [], after_record.get("frame") or [])
            for left, right in zip(left_row, right_row)
        ) if isinstance(before_record.get("frame"), list) and isinstance(after_record.get("frame"), list) else None,
        "action_effective": replay_hash != after_hash or before.get("state") != after.get("state"),
        "progressed": after_levels > before_levels or after.get("state") == "WIN",
        "state_before": before.get("state"),
        "state_after": after.get("state"),
        "levels_delta": after_levels - before_levels,
        "score_delta": 0.0,
    }
    repaired["reward"] = 1.0 if repaired["state_delta"]["progressed"] else 0.0
    repaired["level_completed"] = after_levels
    return repaired


def repair_rows(shard: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions = recording_index(shard)
    repaired = [repair_row(row, transitions) for row in rows]
    validate_counterfactual_groups(repaired, shard.name)
    return repaired


def backup_file(path: Path, backup_root: Path) -> None:
    if path.exists():
        destination = backup_root / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def migrate_shard(shard: Path, backup_root: Path, apply: bool) -> dict[str, Any]:
    labels = shard / "counterfactual_labels.jsonl"
    raw_labels = shard / "raw" / "counterfactual_labels.jsonl"
    manifest_path = shard / "manifest.json"
    if not labels.exists() or not manifest_path.exists():
        raise ValueError("labels or manifest missing")
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_rows = read_jsonl(labels)
    canonical_dir = CANONICAL / shard.name
    canonical_labels = canonical_dir / "counterfactual_labels.jsonl"
    canonical_manifest_path = canonical_dir / "manifest.json"
    if (
        existing_manifest.get("schema_version") == SOURCE_SCHEMA_VERSION
        and existing_manifest.get("labels_sha256") == sha256(labels)
        and canonical_labels.exists()
        and canonical_manifest_path.exists()
    ):
        canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
        if (
            canonical_manifest.get("schema_version") == "aop-canonical-v2"
            and canonical_manifest.get("labels_sha256") == sha256(canonical_labels)
        ):
            validate_counterfactual_groups(existing_rows, shard.name)
            canonical_rows = read_jsonl(canonical_labels)
            validate_counterfactual_groups(canonical_rows, shard.name)
            if len(canonical_rows) == len(existing_rows):
                return {
                    "shard": shard.name,
                    "rows": len(existing_rows),
                    "action6": sum(1 for row in existing_rows if row.get("candidate_action_id") == 6),
                    "status": "already_current",
                }
    authoritative = raw_labels if raw_labels.exists() else labels
    repaired = repair_rows(shard, read_jsonl(authoritative))
    raw_repaired = repaired if raw_labels.exists() else None
    action6 = sum(1 for row in repaired if row.get("candidate_action_id") == 6)
    if not apply:
        return {"shard": shard.name, "rows": len(repaired), "action6": action6, "status": "would_migrate"}

    backup_file(labels, backup_root)
    backup_file(raw_labels, backup_root)
    backup_file(manifest_path, backup_root)
    canonical_dir = CANONICAL / shard.name
    if canonical_dir.exists():
        for path in canonical_dir.iterdir():
            if path.is_file():
                backup_file(path, backup_root)

    write_jsonl(labels, repaired)
    if raw_repaired is not None:
        write_jsonl(raw_labels, raw_repaired)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_digest = manifest.get("labels_sha256")
    digest = sha256(labels)
    manifest.update({
        "schema_version": SOURCE_SCHEMA_VERSION,
        "shard_id": shard.name,
        "thread": "model",
        "source": "remote_counterfactual_v1",
        "created_at": manifest.get("created_at") or shard.stat().st_mtime,
        "collector_exit": 0,
        "rows": len(repaired),
        "unique_rows": len(repaired),
        "labels_sha256": digest,
        "source_sha256": digest,
        "provenance_complete": True,
        "training_performed": False,
        "migration": {
            "name": "counterfactual-contract-v2",
            "previous_labels_sha256": previous_digest,
            "reconstructed_from_marked_remote_recordings": True,
        },
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix=f"aop-{shard.name}-") as temporary:
        temporary_dir = Path(temporary)
        result = ingest_counterfactual_shard(
            labels,
            temporary_dir / "counterfactual_labels.jsonl",
            manifest_path,
            "model",
        )
        canonical_dir.mkdir(parents=True, exist_ok=True)
        for name in ("counterfactual_labels.jsonl", "manifest.json"):
            os.replace(temporary_dir / name, canonical_dir / name)
    return {
        "shard": shard.name,
        "rows": len(repaired),
        "action6": action6,
        "canonical_rows": result["rows_written"],
        "status": "migrated",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--shard", action="append", default=[])
    args = parser.parse_args()
    shards = sorted(path for path in (COLLECTION / "shards").iterdir() if path.is_dir())
    if args.shard:
        selected = set(args.shard)
        shards = [path for path in shards if path.name in selected]
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    backup_root = COLLECTION / "migration-backups" / stamp
    results = []
    failures = []
    for shard in shards:
        if not (shard / "counterfactual_labels.jsonl").exists():
            continue
        try:
            results.append(migrate_shard(shard, backup_root, args.apply))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append({"shard": shard.name, "error": str(exc)})
    migrated_count = sum(item["status"] == "migrated" for item in results)
    payload = {
        "mode": "apply" if args.apply else "dry-run",
        "shards": len(results),
        "migrated_shards": migrated_count,
        "already_current_shards": sum(item["status"] == "already_current" for item in results),
        "rows": sum(item["rows"] for item in results),
        "action6_rows": sum(item["action6"] for item in results),
        "failures": failures,
        "backup": str(backup_root) if args.apply and migrated_count else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
